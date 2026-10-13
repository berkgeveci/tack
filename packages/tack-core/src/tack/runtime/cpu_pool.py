"""A persistent thread pool for CPU fan-outs, JIT-compiled with llvmlite.

A fan-out through ``ThreadPoolExecutor`` costs about 60 µs per worker on a
2 GHz Xeon (490 µs for eight, 1.2 ms after an idle gap): every worker
wakes, takes the GIL to reach its ctypes call, and takes it again to
return, and the handoffs serialize. Tack's launches are short -- a contour
at a million cells is 37 of them, 0.1 to 14 ms each -- so the policy
rightly kept most of them serial and the rest gained little.

Here the fan-out is one ctypes call. The pool is a small program in LLVM
IR, built with ``llvmlite.ir`` (no C compiler, no binary wheel) and JIT'd
beside the kernels: ``N - 1`` detached pthreads wait for a generation
counter to change, spinning for a while and then sleeping -- on a futex
on Linux and on ``os_sync_wait_on_address`` on macOS 14.4+, woken by
``pool_run`` only when one of them sleeps; elsewhere in ``nanosleep``
polls that back off to a millisecond; ``pool_run`` splits
``[start, end)`` into ``N`` chunks
exactly as the Python path does, bumps the counter, runs the last chunk on
the calling thread, and waits for the others. The workers never touch
Python. Measured on the Xeon: 4 µs hot, about 100 µs after an idle gap.

Chunks are claimed, not assigned: a run is cut into several chunks per
worker (`GRAIN`, each at least `MIN_CHUNK` elements) and every worker takes
the next one from a counter until none is left. A worker that runs slow --
on the sibling hyperthread of a busy core, or over the dear part of an
uneven range -- simply takes fewer. With one equal chunk per worker the
dispatch waited for the slowest: a compute-bound kernel over a million
cells took 29 ms on a worker with a core to itself and 50 on one sharing
its core, run to run, as the scheduler placed them.

A kernel is called through a wrapper JIT'd once per compiled kernel
(``chunk_wrapper``): ``void __tack_chunk(pack, start, end)`` loads the
kernel's arguments from 8-byte slots in ``pack`` (``pack_args`` fills them
from the ``bind()`` prefix) and calls the kernel at its address, so the
pool needs one calling convention for every kernel.

``TACK_CPU_POOL=python`` keeps the ThreadPoolExecutor path. The pool is
also left alone where it cannot be built (Windows, or a JIT failure), so
the Python path is the fallback everywhere.
"""
import ctypes
import ctypes.util
import os
import platform
import threading
import time

import llvmlite.binding as llvm
import llvmlite.ir as ir

MAX_WORKERS = 256
# How long a worker spins (in pause iterations) before it sleeps between
# polls, and how long each sleep is. Spinning keeps back-to-back launches
# at a few microseconds; a worker left idle polls every SLEEP_NS.
SPIN_ITERATIONS = 20_000
# Chunks per worker, and the smallest chunk worth claiming.
GRAIN = 4
MIN_CHUNK = 256
SLEEP_NS = 50_000
# ...doubling each poll up to this, so a pool idle for seconds costs a
# thousand wake-ups a second per worker, not twenty thousand.
SLEEP_MAX_NS = 1_000_000

# Linux's futex syscall, where the pool can block without polling. The
# number is the architecture's; elsewhere the workers poll.
_SYS_FUTEX = {"x86_64": 202, "amd64": 202, "aarch64": 98, "arm64": 98}
_FUTEX_WAIT_PRIVATE, _FUTEX_WAKE_PRIVATE = 128, 129
# macOS has no futex; since 14.4 it has the equivalent, public
# `os_sync_wait_on_address` / `os_sync_wake_by_address_all`. Polling
# instead cost up to a millisecond per launch on an M1 Max: an idle
# worker's `nanosleep` had backed off to 1 ms, every launch waited for
# the slowest sleeper, and the policy, measuring a 1 ms fan-out, kept
# mid-size launches serial. The pool program calls one or the other
# (it is built per process); `futex_nr` nonzero selects blocking at run
# time, zero the polling fallback.
_DARWIN = platform.system() == "Darwin"
_WAIT_ON_ADDRESS_NONE = 0

_i8, _i32, _i64 = ir.IntType(8), ir.IntType(32), ir.IntType(64)
_ptr = _i8.as_pointer()
_chunk_fn_type = ir.FunctionType(ir.VoidType(), [_ptr, _i64, _i64])


def enabled() -> bool:
    """Whether dispatches should try the pool at all."""
    choice = os.environ.get("TACK_CPU_POOL", "").strip().lower()
    if choice in ("python", "0", "off", "no", "false"):
        return False
    return platform.system() != "Windows"


# ── The pool program ───────────────────────────────────────────────

def _pause(builder, module):
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        fn = module.globals.get("llvm.x86.sse2.pause")
        if fn is None:
            fn = ir.Function(module, ir.FunctionType(ir.VoidType(), []), "llvm.x86.sse2.pause")
        builder.call(fn, [])
    elif machine in ("aarch64", "arm64"):
        builder.asm(ir.FunctionType(ir.VoidType(), []), "yield", "~{memory}", [], True)


def _pool_module() -> ir.Module:
    m = ir.Module(name="tack_cpu_pool")
    m.triple = llvm.get_default_triple()

    def glob(name, typ, init=None):
        g = ir.GlobalVariable(m, typ, name)
        g.initializer = ir.Constant(typ, init if init is not None else None)
        g.linkage = "internal"
        return g

    n = glob("n", _i32, 0)
    active = glob("active", _i32, 0)       # workers this run uses, <= n
    spin = glob("spin", _i32, 0)
    sleep_ns = glob("sleep_ns", _i64, 0)
    sleep_max = glob("sleep_max", _i64, 0)
    clock_id = glob("clock_id", _i32, 0)
    gen = glob("gen", _i32, 0)
    done = glob("done", _i32, 0)
    futex_nr = glob("futex_nr", _i64, 0)       # 0: poll instead
    sleepers = glob("sleepers", _i32, 0)
    cur_fn = glob("cur_fn", _ptr)
    cur_pack = glob("cur_pack", _ptr)
    cur_start = glob("cur_start", _i64, 0)
    cur_end = glob("cur_end", _i64, 0)
    cur_chunk = glob("cur_chunk", _i64, 0)
    spans = glob("spans", ir.ArrayType(_i64, MAX_WORKERS))
    elems = glob("elems", ir.ArrayType(_i64, MAX_WORKERS))
    n_chunks = glob("n_chunks", _i32, 0)
    next_chunk = glob("next_chunk", _i32, 0)
    grain = glob("grain", _i32, 0)
    timespec = ir.LiteralStructType([_i64, _i64])

    pthread_create = ir.Function(m, ir.FunctionType(_i32, [_ptr, _ptr, _ptr, _ptr]), "pthread_create")
    pthread_detach = ir.Function(m, ir.FunctionType(_i32, [_i64]), "pthread_detach")
    nanosleep = ir.Function(m, ir.FunctionType(_i32, [_ptr, _ptr]), "nanosleep")
    clock_gettime = ir.Function(m, ir.FunctionType(_i32, [_i32, _ptr]), "clock_gettime")
    syscall = ir.Function(m, ir.FunctionType(_i64, [_i64], var_arg=True), "syscall")
    if _DARWIN:
        # int os_sync_wait_on_address(void *addr, uint64_t value, size_t size, flags)
        # int os_sync_wake_by_address_all(void *addr, size_t size, flags)
        wait_on_address = ir.Function(m, ir.FunctionType(_i32, [_ptr, _i64, _i64, _i32]),
                                      "os_sync_wait_on_address")
        wake_by_address = ir.Function(m, ir.FunctionType(_i32, [_ptr, _i64, _i32]),
                                      "os_sync_wake_by_address_all")

    def block_on(b, nr, last):
        """Block until `gen` changes from `last` (the caller has checked it
        still equals `last` under a fence)."""
        if _DARWIN:
            b.call(wait_on_address, [b.bitcast(gen, _ptr), b.zext(last, _i64), _i64(4),
                                     _i32(_WAIT_ON_ADDRESS_NONE)])
        else:
            b.call(syscall, [nr, b.ptrtoint(gen, _i64), _i64(_FUTEX_WAIT_PRIVATE),
                             b.sext(last, _i64), _i64(0), _i64(0), _i64(0)])

    def wake_all(b, nr):
        if _DARWIN:
            b.call(wake_by_address, [b.bitcast(gen, _ptr), _i64(4), _i32(_WAIT_ON_ADDRESS_NONE)])
        else:
            b.call(syscall, [nr, b.ptrtoint(gen, _i64), _i64(_FUTEX_WAKE_PRIVATE),
                             _i64(0x7fffffff), _i64(0), _i64(0), _i64(0)])
    memset = ir.Function(m, ir.FunctionType(ir.VoidType(), [_ptr, _i8, _i64, ir.IntType(1)]),
                         "llvm.memset.p0.i64")

    # now_ns(): CLOCK_MONOTONIC in nanoseconds.
    now = ir.Function(m, ir.FunctionType(_i64, []), "now_ns")
    now.linkage = "internal"
    b = ir.IRBuilder(now.append_basic_block("entry"))
    ts = b.alloca(timespec)
    b.call(clock_gettime, [b.load(clock_id), b.bitcast(ts, _ptr)])
    sec = b.load(b.gep(ts, [_i32(0), _i32(0)]))
    nsec = b.load(b.gep(ts, [_i32(0), _i32(1)]))
    b.ret(b.add(b.mul(sec, _i64(1_000_000_000)), nsec))

    # run_chunks(id): claim chunks of the current range until none is left;
    # worker id's time and elements over them are recorded.
    run_chunk = ir.Function(m, ir.FunctionType(ir.VoidType(), [_i32]), "run_chunks")
    run_chunk.linkage = "internal"
    entry = run_chunk.append_basic_block("entry")
    claim = run_chunk.append_basic_block("claim")
    body = run_chunk.append_basic_block("body")
    finish = run_chunk.append_basic_block("finish")
    b = ir.IRBuilder(entry)
    wid = run_chunk.args[0]
    span = b.alloca(_i64)
    count = b.alloca(_i64)
    b.store(_i64(0), span)
    b.store(_i64(0), count)
    b.branch(claim)
    b = ir.IRBuilder(claim)
    c = b.atomic_rmw("add", next_chunk, _i32(1), "acq_rel")
    b.cbranch(b.icmp_signed("<", c, b.load(n_chunks)), body, finish)
    b = ir.IRBuilder(body)
    chunk = b.load(cur_chunk)
    a = b.add(b.load(cur_start), b.mul(b.sext(c, _i64), chunk))
    end = b.load(cur_end)
    bb = b.add(a, chunk)
    bb = b.select(b.icmp_signed(">", bb, end), end, bb)
    t0 = b.call(now, [])
    fn = b.bitcast(b.load(cur_fn), _chunk_fn_type.as_pointer())
    b.call(fn, [b.load(cur_pack), a, bb])
    b.store(b.add(b.load(span), b.sub(b.call(now, []), t0)), span)
    b.store(b.add(b.load(count), b.sub(bb, a)), count)
    b.branch(claim)
    b = ir.IRBuilder(finish)
    b.store(b.load(span), b.gep(spans, [_i32(0), wid]))
    b.store(b.load(count), b.gep(elems, [_i32(0), wid]))
    b.ret_void()

    # worker(arg): wait for the generation to change, run the chunk, count in.
    worker = ir.Function(m, ir.FunctionType(_ptr, [_ptr]), "tack_pool_worker")
    worker.linkage = "internal"
    entry = worker.append_basic_block("entry")
    wait = worker.append_basic_block("wait")
    idle = worker.append_basic_block("idle")
    sleep = worker.append_basic_block("sleep")
    spin_bb = worker.append_basic_block("spin")
    work = worker.append_basic_block("work")
    b = ir.IRBuilder(entry)
    wid = b.trunc(b.ptrtoint(worker.args[0], _i64), _i32)
    seen = b.alloca(_i32)
    b.store(_i32(0), seen)
    count = b.alloca(_i32)
    ts = b.alloca(timespec)
    b.store(_i64(0), b.gep(ts, [_i32(0), _i32(0)]))
    b.branch(wait)
    b = ir.IRBuilder(wait)
    b.store(_i32(0), count)
    b.store(b.load(sleep_ns), b.gep(ts, [_i32(0), _i32(1)]))
    b.branch(idle)
    b = ir.IRBuilder(idle)
    current = b.load_atomic(gen, "acquire", 4)
    b.cbranch(b.icmp_signed("!=", current, b.load(seen)), work, sleep)
    b = ir.IRBuilder(sleep)
    c = b.add(b.load(count), _i32(1))
    b.store(c, count)
    with b.if_then(b.icmp_signed(">", c, b.load(spin))):
        b.store(b.load(spin), count)        # keep sleeping until woken
        nr = b.load(futex_nr)
        with b.if_else(b.icmp_signed("!=", nr, _i64(0))) as (futex, poll):
            with futex:
                # Count in as a sleeper, then re-check the generation under
                # a full fence before blocking: pool_run bumps it, fences,
                # and wakes only if it sees a sleeper, so no wake is lost.
                b.atomic_rmw("add", sleepers, _i32(1), "seq_cst")
                b.fence("seq_cst")
                last = b.load(seen)
                with b.if_then(b.icmp_signed("==", b.load_atomic(gen, "seq_cst", 4), last)):
                    block_on(b, nr, last)
                b.atomic_rmw("sub", sleepers, _i32(1), "seq_cst")
            with poll:
                b.call(nanosleep, [b.bitcast(ts, _ptr), ir.Constant(_ptr, None)])
                nsec = b.gep(ts, [_i32(0), _i32(1)])
                longer = b.mul(b.load(nsec), _i64(2))
                cap = b.load(sleep_max)
                b.store(b.select(b.icmp_signed(">", longer, cap), cap, longer), nsec)
        b.branch(idle)
    b.branch(spin_bb)
    b = ir.IRBuilder(spin_bb)
    _pause(b, m)
    b.branch(idle)
    b = ir.IRBuilder(work)
    b.store(b.load_atomic(gen, "acquire", 4), seen)
    # Workers 0 .. active - 2 claim chunks; the caller is worker active - 1.
    # A worker past that sits the run out, its slots cleared so nothing
    # stale is read.
    with b.if_else(b.icmp_signed("<", wid, b.sub(b.load(active), _i32(1)))) as (mine, idle_run):
        with mine:
            b.call(run_chunk, [wid])
        with idle_run:
            with b.if_then(b.icmp_signed(">=", wid, b.load(active))):
                b.store(_i64(0), b.gep(spans, [_i32(0), wid]))
                b.store(_i64(0), b.gep(elems, [_i32(0), wid]))
    b.atomic_rmw("add", done, _i32(1), "release")
    b.branch(wait)

    # pool_start(n, spin, clock_id, sleep_ns): 0, or -1 if it cannot start.
    start = ir.Function(m, ir.FunctionType(_i32, [_i32, _i32, _i32, _i64, _i64, _i64, _i32]),
                        "tack_pool_start")
    b = ir.IRBuilder(start.append_basic_block("entry"))
    want, spin_arg, clock_arg, sleep_arg, sleep_max_arg, futex_arg, grain_arg = start.args
    have = b.load(n)
    with b.if_then(b.icmp_signed("!=", have, _i32(0))):
        b.ret(b.select(b.icmp_signed("==", have, want), _i32(0), _i32(-1)))
    bad = b.or_(b.icmp_signed("<", want, _i32(1)), b.icmp_signed(">", want, _i32(MAX_WORKERS)))
    with b.if_then(bad):
        b.ret(_i32(-1))
    b.store(spin_arg, spin)
    b.store(clock_arg, clock_id)
    b.store(sleep_arg, sleep_ns)
    b.store(sleep_max_arg, sleep_max)
    b.store(futex_arg, futex_nr)
    b.store(grain_arg, grain)
    b.store(want, n)
    i = b.alloca(_i32)
    b.store(_i32(0), i)
    loop = start.append_basic_block("loop")
    body = start.append_basic_block("body")
    out = start.append_basic_block("out")
    b.branch(loop)
    b = ir.IRBuilder(loop)
    b.cbranch(b.icmp_signed("<", b.load(i), b.sub(want, _i32(1))), body, out)
    b = ir.IRBuilder(body)
    tid = b.alloca(_i64)
    arg = b.inttoptr(b.sext(b.load(i), _i64), _ptr)
    rc = b.call(pthread_create, [b.bitcast(tid, _ptr), ir.Constant(_ptr, None),
                                 b.bitcast(worker, _ptr), arg])
    with b.if_then(b.icmp_signed("!=", rc, _i32(0))):
        b.ret(_i32(-1))
    b.call(pthread_detach, [b.load(tid)])
    b.store(b.add(b.load(i), _i32(1)), i)
    b.branch(loop)
    b = ir.IRBuilder(out)
    b.ret(_i32(0))

    # pool_run(fn, pack, start, end): every chunk, this thread's included.
    run = ir.Function(m, ir.FunctionType(ir.VoidType(), [_ptr, _ptr, _i64, _i64, _i32]),
                      "tack_pool_run")
    b = ir.IRBuilder(run.append_basic_block("entry"))
    fn_arg, pack_arg, s_arg, e_arg, k_arg = run.args
    b.store(fn_arg, cur_fn)
    b.store(pack_arg, cur_pack)
    b.store(s_arg, cur_start)
    b.store(e_arg, cur_end)
    b.store(k_arg, active)
    k64 = b.sext(k_arg, _i64)
    total = b.sub(e_arg, s_arg)
    # GRAIN chunks per worker, each at least MIN_CHUNK elements unless the
    # range is too short even for one such chunk per worker.
    parts = b.mul(k64, b.sext(b.load(grain), _i64))
    fine = b.sdiv(b.add(total, b.sub(parts, _i64(1))), parts)
    per_worker = b.sdiv(b.add(total, b.sub(k64, _i64(1))), k64)
    floor = b.select(b.icmp_signed("<", per_worker, _i64(MIN_CHUNK)), per_worker, _i64(MIN_CHUNK))
    chunk = b.select(b.icmp_signed("<", fine, floor), floor, fine)
    chunk = b.select(b.icmp_signed("<", chunk, _i64(1)), _i64(1), chunk)
    b.store(chunk, cur_chunk)
    b.store(b.trunc(b.sdiv(b.add(total, b.sub(chunk, _i64(1))), chunk), _i32), n_chunks)
    # Workers past `active` clear their own slots; slot n - 1 belongs to the
    # caller only when every thread is active, so clear it here otherwise.
    last = b.sub(b.load(n), _i32(1))
    with b.if_then(b.icmp_signed("<", k_arg, b.load(n))):
        b.store(_i64(0), b.gep(spans, [_i32(0), last]))
        b.store(_i64(0), b.gep(elems, [_i32(0), last]))
    b.store_atomic(_i32(0), next_chunk, "monotonic", 4)
    b.store_atomic(_i32(0), done, "monotonic", 4)
    b.atomic_rmw("add", gen, _i32(1), "seq_cst")
    b.fence("seq_cst")
    nr = b.load(futex_nr)
    asleep = b.icmp_signed(">", b.load_atomic(sleepers, "seq_cst", 4), _i32(0))
    with b.if_then(b.and_(b.icmp_signed("!=", nr, _i64(0)), asleep)):
        wake_all(b, nr)
    b.call(run_chunk, [b.sub(k_arg, _i32(1))])
    wait = run.append_basic_block("wait")
    spin_bb = run.append_basic_block("spin")
    out = run.append_basic_block("out")
    b.branch(wait)
    b = ir.IRBuilder(wait)
    finished = b.load_atomic(done, "acquire", 4)
    b.cbranch(b.icmp_signed("<", finished, b.sub(b.load(n), _i32(1))), spin_bb, out)
    b = ir.IRBuilder(spin_bb)
    _pause(b, m)
    b.branch(wait)
    b = ir.IRBuilder(out)
    b.ret_void()

    # pool_spans(out): the workers' spans of the last run, n of them.
    get_spans = ir.Function(m, ir.FunctionType(ir.VoidType(), [_ptr]), "tack_pool_spans")
    b = ir.IRBuilder(get_spans.append_basic_block("entry"))
    memcpy = ir.Function(m, ir.FunctionType(ir.VoidType(), [_ptr, _ptr, _i64, ir.IntType(1)]),
                         "llvm.memcpy.p0.p0.i64")
    b.call(memcpy, [get_spans.args[0], b.bitcast(spans, _ptr),
                    b.mul(b.sext(b.load(n), _i64), _i64(8)), ir.Constant(ir.IntType(1), 0)])
    b.ret_void()
    get_elems = ir.Function(m, ir.FunctionType(ir.VoidType(), [_ptr]), "tack_pool_elems")
    b = ir.IRBuilder(get_elems.append_basic_block("entry"))
    b.call(memcpy, [get_elems.args[0], b.bitcast(elems, _ptr),
                    b.mul(b.sext(b.load(n), _i64), _i64(8)), ir.Constant(ir.IntType(1), 0)])
    b.ret_void()

    # zero_chunk(base, a, b): a chunk function that zeroes bytes [a, b).
    zero = ir.Function(m, _chunk_fn_type, "tack_pool_zero_chunk")
    b = ir.IRBuilder(zero.append_basic_block("entry"))
    base, a, bb = zero.args
    b.call(memset, [b.gep(base, [a]), _i8(0), b.sub(bb, a), ir.Constant(ir.IntType(1), 0)])
    b.ret_void()
    return m


def _target_machine():
    llvm.initialize_native_target()      # idempotent; cpu.py does it too
    llvm.initialize_native_asmprinter()
    # The arm64 pause (`_pause`) is inline asm, which the JIT parses at
    # code generation: without the parser LLVM aborts the process (not an
    # exception) on Apple silicon.
    llvm.initialize_native_asmparser()
    target = llvm.Target.from_default_triple()
    return target.create_target_machine(cpu=llvm.get_host_cpu_name(),
                                        features=llvm.get_host_cpu_features().flatten(), opt=2)


def _available_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:                 # not Linux
        return _performance_cores() or os.cpu_count() or 1


def _performance_cores() -> int | None:
    """Apple silicon's performance cores (`hw.perflevel0.physicalcpu`), the
    count the CPU backend defaults to; None elsewhere. A pool of
    `os.cpu_count()` on an 8+2 machine had two workers more than any launch
    used, which every launch still woke and waited for."""
    if not _DARWIN:
        return None
    try:
        libc = ctypes.CDLL(None)
        value = ctypes.c_int(0)
        length = ctypes.c_size_t(ctypes.sizeof(value))
        if libc.sysctlbyname(b"hw.perflevel0.physicalcpu", ctypes.byref(value),
                             ctypes.byref(length), None, ctypes.c_size_t(0)) == 0:
            return value.value or None
    except (OSError, AttributeError):
        pass
    return None


def _blocking_available() -> bool:
    """Whether the pool can block rather than poll on this host."""
    if platform.system() == "Linux":
        return platform.machine().lower() in _SYS_FUTEX
    if _DARWIN:
        try:
            libc = ctypes.CDLL(None)
            return bool(libc.os_sync_wait_on_address and libc.os_sync_wake_by_address_all)
        except (OSError, AttributeError):
            return False
    return False


def _register_libc_symbols():
    """MCJIT resolves the pool's libc calls through these, so it does not
    depend on how the process exports them."""
    libc = ctypes.CDLL(None)
    names = ["pthread_create", "pthread_detach", "nanosleep", "clock_gettime", "syscall"]
    if _DARWIN:
        names += ["os_sync_wait_on_address", "os_sync_wake_by_address_all"]
    for name in names:
        try:
            address = ctypes.cast(getattr(libc, name), ctypes.c_void_p).value
        except AttributeError:
            # macOS before 14.4: the pool program still names them, so give
            # the JIT an address; `futex_nr` stays 0 and they are never called.
            address = ctypes.cast(libc.getpid, ctypes.c_void_p).value
        llvm.add_symbol(name, address)


class CPool:
    """``threads`` workers, the calling thread one of them. One per process:
    its threads are detached and its state is a set of JIT'd globals."""

    _instance = None
    _lock = threading.Lock()

    def __init__(self, threads: int):
        _register_libc_symbols()
        module = llvm.parse_assembly(str(_pool_module()))
        module.verify()
        self._engine = llvm.create_mcjit_compiler(module, _target_machine())
        self._engine.finalize_object()
        addr = self._engine.get_function_address
        self._start = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                       ctypes.c_int64, ctypes.c_int64,
                                       ctypes.c_int64, ctypes.c_int)(addr("tack_pool_start"))
        self._run = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
                                     ctypes.c_int64, ctypes.c_int)(addr("tack_pool_run"))
        self._get_spans = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(addr("tack_pool_spans"))
        self._get_elems = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(addr("tack_pool_elems"))
        self.zero_chunk = addr("tack_pool_zero_chunk")
        spin = int(os.environ.get("TACK_CPU_POOL_SPIN", SPIN_ITERATIONS))
        if not _blocking_available():
            futex, self.waiter = 0, "poll"
        elif _DARWIN:
            futex, self.waiter = 1, "wait_on_address"   # any nonzero: block
        else:
            futex, self.waiter = _SYS_FUTEX[platform.machine().lower()], "futex"
        if self._start(threads, spin, time.CLOCK_MONOTONIC, SLEEP_NS, SLEEP_MAX_NS,
                       futex, GRAIN) != 0:
            raise RuntimeError(f"could not start a pool of {threads} threads")
        self.threads = threads
        self._spans = (ctypes.c_int64 * threads)()
        self._elems = (ctypes.c_int64 * threads)()

    @classmethod
    def get(cls, threads: int):
        """The process's pool, made on first use with a thread for every
        CPU this process may run on (or `threads`, if more); None if it
        cannot be made or has fewer threads than asked for. A run uses as
        many of its threads as the caller's backend has, so backends with
        different thread counts share it."""
        with cls._lock:
            if cls._instance is None:
                try:
                    cls._instance = cls(min(MAX_WORKERS, max(threads, _available_cpus())))
                except Exception:          # no pool: the Python path stays
                    cls._instance = False
            pool = cls._instance
        return pool if pool and threads <= pool.threads else None

    def run(self, wrapper: int, pack, start: int, end: int, threads: int):
        """[start, end) in chunks claimed by `threads` workers, this thread
        one of them."""
        self._run(wrapper, ctypes.addressof(pack), start, end, threads)

    def spans(self) -> list:
        """Per worker, the ns it spent in the last run's chunks."""
        self._get_spans(ctypes.addressof(self._spans))
        return list(self._spans)

    @staticmethod
    def chunk_size(total: int, threads: int) -> int:
        """The chunk a run of `total` elements over `threads` workers uses:
        GRAIN per worker, at least MIN_CHUNK unless the range is shorter."""
        parts = threads * GRAIN
        fine = (total + parts - 1) // parts
        per_worker = (total + threads - 1) // threads
        return max(1, max(fine, min(per_worker, MIN_CHUNK)))

    def elems(self) -> list:
        """Per worker, the elements it ran in the last run."""
        self._get_elems(ctypes.addressof(self._elems))
        return list(self._elems)

    def zero(self, address: int, nbytes: int, threads: int):
        """memset(address, 0, nbytes), one chunk per worker: a new buffer's
        pages are first touched by the workers, not the calling thread."""
        self._run(self.zero_chunk, address, 0, nbytes, threads)


# ── Per-kernel wrappers and argument packs ──────────────────────────

_wrapper_lock = threading.Lock()


def chunk_wrapper(compiled) -> int:
    """Address of ``void __tack_chunk(pack, start, end)`` for this kernel,
    JIT'd once and kept with it."""
    if compiled._chunk:
        return compiled._chunk
    with _wrapper_lock:
        # Two threads dispatching the same kernel must not each build one:
        # the engine the loser dropped could still be running a launch.
        if compiled._chunk:
            return compiled._chunk
        return _build_chunk_wrapper(compiled)


def _build_chunk_wrapper(compiled) -> int:
    from tack.codegen.llvm_gen import _llvm_type

    ptypes = []
    for ptype, is_field in zip(compiled._param_types, compiled._param_is_field):
        t = _llvm_type(ptype)
        ptypes.append(t.as_pointer() if is_field else t)
    ptypes.extend([_i64] * compiled._num_dims)
    kernel_type = ir.FunctionType(ir.VoidType(), ptypes + [_i64, _i64])
    module = ir.Module(name=f"chunk_{compiled._func_name}")
    module.triple = llvm.get_default_triple()
    wrapper = ir.Function(module, _chunk_fn_type, name="__tack_chunk")
    b = ir.IRBuilder(wrapper.append_basic_block("entry"))
    pack, start, end = wrapper.args
    args = []
    for i, t in enumerate(ptypes):
        slot = b.gep(pack, [_i64(8 * i)])
        args.append(b.load(b.bitcast(slot, t.as_pointer())))
    kernel = b.inttoptr(_i64(compiled._func_ptr), kernel_type.as_pointer())
    b.call(kernel, args + [start, end])
    b.ret_void()
    mod = llvm.parse_assembly(str(module))
    mod.verify()
    engine = llvm.create_mcjit_compiler(mod, _target_machine())
    engine.finalize_object()
    compiled._chunk_engine = engine
    compiled._chunk = engine.get_function_address("__tack_chunk")
    return compiled._chunk


def pack_args(compiled, prefix: tuple):
    """The ``bind()`` prefix as 8-byte slots: a field's address, a scalar
    stored with its own ctypes type at the slot's start, an extent as i64."""
    pack = (ctypes.c_uint64 * max(1, len(prefix)))()
    fields = compiled._param_is_field
    for i, a in enumerate(prefix):
        if i >= len(fields):
            ctypes.c_int64.from_buffer(pack, 8 * i).value = a
        elif fields[i]:
            ctypes.c_void_p.from_buffer(pack, 8 * i).value = a
        else:
            type(a).from_buffer(pack, 8 * i).value = a.value
    return pack
