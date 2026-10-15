"""A persistent thread pool for CPU fan-outs, JIT-compiled with llvmlite.

A fan-out through ``ThreadPoolExecutor`` costs about 60 µs per worker on a
2 GHz Xeon (490 µs for eight, 1.2 ms after an idle gap): every worker
wakes, takes the GIL to reach its ctypes call, and takes it again to
return, and the handoffs serialize. Tack's launches are short -- a contour
at a million cells is 37 of them, 0.1 to 14 ms each -- so the policy
rightly kept most of them serial and the rest gained little.

Here the fan-out is one ctypes call. The pool is a small program in LLVM
IR, built with ``llvmlite.ir`` (no C compiler, no binary wheel) and JIT'd
beside the kernels: ``N - 1`` detached pthreads each wait on their own
mailbox, a launch count, spinning for a while and then sleeping -- on a
futex on Linux and on ``os_sync_wait_on_address`` on macOS 14.4+, woken
by ``pool_run`` only when they sleep; elsewhere in ``nanosleep`` polls
that back off to a millisecond. ``pool_run`` with ``k`` threads posts to
workers ``0 .. k - 2`` and no other, runs chunks on the calling thread as
worker ``k - 1``, and waits for those ``k - 1``; a worker past that is
not woken, so a launch of four on a pool of twenty-eight costs four
wake-ups, and the ``k`` workers of every launch are the same ``k``.
Workers are not pinned: pinning them to cores lost on both hosts it was
measured on (``docs/design/cpu-threading.md``). The workers never touch
Python. Measured on the Xeon: 4 µs hot,
about 100 µs after an idle gap.

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
import struct
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
    # Each worker's mailbox: the launch count posted to it, which it waits
    # on, and whether it sleeps on that word. A run posts to the workers it
    # uses and wakes only those asleep; the rest never stir.
    mail = glob("mail", ir.ArrayType(_i32, MAX_WORKERS))
    asleep = glob("asleep", ir.ArrayType(_i32, MAX_WORKERS))
    done = glob("done", _i32, 0)
    futex_nr = glob("futex_nr", _i64, 0)       # 0: poll instead
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
    # Set when a start fails part way: the workers already running leave
    # instead of waiting for work, and pool_start joins them before it
    # returns, so none is left running code the caller is about to free.
    quit = glob("quit", _i32, 0)
    tids = glob("tids", ir.ArrayType(_i64, MAX_WORKERS))
    timespec = ir.LiteralStructType([_i64, _i64])

    pthread_create = ir.Function(m, ir.FunctionType(_i32, [_ptr, _ptr, _ptr, _ptr]), "pthread_create")
    pthread_detach = ir.Function(m, ir.FunctionType(_i32, [_i64]), "pthread_detach")
    pthread_join = ir.Function(m, ir.FunctionType(_i32, [_i64, _ptr]), "pthread_join")
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

    def block_on(b, nr, word, last):
        """Block until the i32 at `word` changes from `last` (the caller has
        checked it still equals `last` under a fence)."""
        if _DARWIN:
            b.call(wait_on_address, [b.bitcast(word, _ptr), b.zext(last, _i64), _i64(4),
                                     _i32(_WAIT_ON_ADDRESS_NONE)])
        else:
            b.call(syscall, [nr, b.ptrtoint(word, _i64), _i64(_FUTEX_WAIT_PRIVATE),
                             b.sext(last, _i64), _i64(0), _i64(0), _i64(0)])

    def wake(b, nr, word):
        if _DARWIN:
            b.call(wake_by_address, [b.bitcast(word, _ptr), _i64(4), _i32(_WAIT_ON_ADDRESS_NONE)])
        else:
            b.call(syscall, [nr, b.ptrtoint(word, _i64), _i64(_FUTEX_WAKE_PRIVATE),
                             _i64(0x7fffffff), _i64(0), _i64(0), _i64(0)])

    def post(b, nr, w):
        """Post a launch to worker `w`'s mailbox and wake it if it sleeps:
        bump under seq_cst, fence, then read its sleep flag -- the mirror
        of the worker's flag-fence-recheck, so no wake-up is lost."""
        word = b.gep(mail, [_i32(0), w])
        b.atomic_rmw("add", word, _i32(1), "seq_cst")
        b.fence("seq_cst")
        flag = b.load_atomic(b.gep(asleep, [_i32(0), w]), "seq_cst", 4)
        with b.if_then(b.and_(b.icmp_signed("!=", nr, _i64(0)),
                              b.icmp_signed("!=", flag, _i32(0)))):
            wake(b, nr, word)

    def post_each(b, fn, nr, count):
        """`post` to workers 0 .. count - 1."""
        wi = b.alloca(_i32)
        b.store(_i32(0), wi)
        loop = fn.append_basic_block("post_loop")
        body = fn.append_basic_block("post_body")
        out = fn.append_basic_block("post_out")
        b.branch(loop)
        b = ir.IRBuilder(loop)
        b.cbranch(b.icmp_signed("<", b.load(wi), count), body, out)
        b = ir.IRBuilder(body)
        post(b, nr, b.load(wi))
        b.store(b.add(b.load(wi), _i32(1)), wi)
        b.branch(loop)
        return ir.IRBuilder(out)

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

    # worker(arg): wait for a post to its mailbox, run the chunks, count in.
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
    my_mail = b.gep(mail, [_i32(0), wid])
    my_sleep = b.gep(asleep, [_i32(0), wid])
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
    current = b.load_atomic(my_mail, "acquire", 4)
    b.cbranch(b.icmp_signed("!=", current, b.load(seen)), work, sleep)
    b = ir.IRBuilder(sleep)
    c = b.add(b.load(count), _i32(1))
    b.store(c, count)
    with b.if_then(b.icmp_signed(">", c, b.load(spin))):
        b.store(b.load(spin), count)        # keep sleeping until woken
        nr = b.load(futex_nr)
        with b.if_else(b.icmp_signed("!=", nr, _i64(0))) as (futex, poll):
            with futex:
                # Raise the sleep flag, then re-check the mailbox under a
                # full fence before blocking: a post bumps it, fences, and
                # wakes only if it sees the flag, so no wake is lost.
                b.store_atomic(_i32(1), my_sleep, "seq_cst", 4)
                b.fence("seq_cst")
                last = b.load(seen)
                with b.if_then(b.icmp_signed("==", b.load_atomic(my_mail, "seq_cst", 4), last)):
                    block_on(b, nr, my_mail, last)
                b.store_atomic(_i32(0), my_sleep, "seq_cst", 4)
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
    b.store(b.load_atomic(my_mail, "acquire", 4), seen)
    # A start that failed after this worker was made: leave, to be joined.
    with b.if_then(b.icmp_signed("!=", b.load_atomic(quit, "acquire", 4), _i32(0))):
        b.ret(ir.Constant(_ptr, None))
    # Only workers 0 .. active - 2 are posted to; the caller is worker
    # active - 1, and the rest sleep on, their slots cleared by pool_run.
    b.call(run_chunk, [wid])
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
    fail = start.append_basic_block("fail")
    out = start.append_basic_block("out")
    b.branch(loop)
    b = ir.IRBuilder(loop)
    b.cbranch(b.icmp_signed("<", b.load(i), b.sub(want, _i32(1))), body, out)
    b = ir.IRBuilder(body)
    tid = b.gep(tids, [_i32(0), b.load(i)])
    arg = b.inttoptr(b.sext(b.load(i), _i64), _ptr)
    rc = b.call(pthread_create, [b.bitcast(tid, _ptr), ir.Constant(_ptr, None),
                                 b.bitcast(worker, _ptr), arg])
    with b.if_then(b.icmp_signed("!=", rc, _i32(0))):
        b.branch(fail)
    b.store(b.add(b.load(i), _i32(1)), i)
    b.branch(loop)
    # A thread could not be made: the `i` already running wait on their
    # mailboxes, in the JIT'd code the caller will free when this returns
    # -1. Tell them to leave, post to each, and join each before returning.
    b = ir.IRBuilder(fail)
    b.store_atomic(_i32(1), quit, "seq_cst", 4)
    b = post_each(b, start, b.load(futex_nr), b.load(i))
    j = b.alloca(_i32)
    b.store(_i32(0), j)
    join_loop = start.append_basic_block("join_loop")
    join_body = start.append_basic_block("join_out")
    join_out = start.append_basic_block("join_done")
    b.branch(join_loop)
    b = ir.IRBuilder(join_loop)
    b.cbranch(b.icmp_signed("<", b.load(j), b.load(i)), join_body, join_out)
    b = ir.IRBuilder(join_body)
    b.call(pthread_join, [b.load(b.gep(tids, [_i32(0), b.load(j)])), ir.Constant(_ptr, None)])
    b.store(b.add(b.load(j), _i32(1)), j)
    b.branch(join_loop)
    b = ir.IRBuilder(join_out)
    b.store(_i32(0), n)
    b.store_atomic(_i32(0), quit, "seq_cst", 4)
    b.ret(_i32(-1))
    # Every worker made: let them go.
    b = ir.IRBuilder(out)
    b.store(_i32(0), i)
    detach_loop = start.append_basic_block("detach_loop")
    detach_body = start.append_basic_block("detach_body")
    detach_out = start.append_basic_block("detach_out")
    b.branch(detach_loop)
    b = ir.IRBuilder(detach_loop)
    b.cbranch(b.icmp_signed("<", b.load(i), b.sub(want, _i32(1))), detach_body, detach_out)
    b = ir.IRBuilder(detach_body)
    b.call(pthread_detach, [b.load(b.gep(tids, [_i32(0), b.load(i)]))])
    b.store(b.add(b.load(i), _i32(1)), i)
    b.branch(detach_loop)
    b = ir.IRBuilder(detach_out)
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
    # The slots of the workers this run does not use, the caller's (k - 1)
    # included: cleared here, so nothing stale is read from them.
    ci = b.alloca(_i32)
    b.store(b.sub(k_arg, _i32(1)), ci)
    clear_loop = run.append_basic_block("clear_loop")
    clear_body = run.append_basic_block("clear_body")
    clear_out = run.append_basic_block("clear_out")
    b.branch(clear_loop)
    b = ir.IRBuilder(clear_loop)
    b.cbranch(b.icmp_signed("<", b.load(ci), b.load(n)), clear_body, clear_out)
    b = ir.IRBuilder(clear_body)
    b.store(_i64(0), b.gep(spans, [_i32(0), b.load(ci)]))
    b.store(_i64(0), b.gep(elems, [_i32(0), b.load(ci)]))
    b.store(b.add(b.load(ci), _i32(1)), ci)
    b.branch(clear_loop)
    b = ir.IRBuilder(clear_out)
    b.store_atomic(_i32(0), next_chunk, "monotonic", 4)
    b.store_atomic(_i32(0), done, "monotonic", 4)
    # Post to workers 0 .. k - 2 only; the others sleep through this run.
    b = post_each(b, run, b.load(futex_nr), b.sub(k_arg, _i32(1)))
    b.call(run_chunk, [b.sub(k_arg, _i32(1))])
    wait = run.append_basic_block("wait")
    spin_bb = run.append_basic_block("spin")
    out = run.append_basic_block("out")
    b.branch(wait)
    b = ir.IRBuilder(wait)
    finished = b.load_atomic(done, "acquire", 4)
    b.cbranch(b.icmp_signed("<", finished, b.sub(k_arg, _i32(1))), spin_bb, out)
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
    names = ["pthread_create", "pthread_detach", "pthread_join", "nanosleep", "clock_gettime",
             "syscall"]
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
    its threads are detached and its state is a set of JIT'd globals.

    That state -- the function, pack, range and counters of the current run,
    and the spans and counts it leaves -- is one set for the process, so
    a run is a transaction under ``_txn``: two Python threads dispatching
    at once would otherwise each write the globals and read the other's
    results (the ctypes call releases the GIL). Zeroing goes through the
    same lock; the empty probes do too, through ``run``."""

    _instance = None
    _lock = threading.Lock()

    def __init__(self, threads: int):
        self._txn = threading.Lock()
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
        self._report_fmt = f"{threads}q"

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

    def run(self, wrapper: int, pack, start: int, end: int, threads: int,
            measure: bool = False):
        """[start, end) in chunks claimed by `threads` workers, this thread
        one of them. With `measure`, the run's per-worker spans (ns) and
        element counts, read before another thread's run can replace them."""
        with self._txn:
            self._run(wrapper, ctypes.addressof(pack), start, end, threads)
            if measure:
                return self._spans_and_elems()
        return None

    def _spans_and_elems(self):
        self._get_spans(ctypes.addressof(self._spans))
        self._get_elems(ctypes.addressof(self._elems))
        # One unpack each; `list(ctypes_array)` was 7 µs for the pair.
        return (struct.unpack_from(self._report_fmt, self._spans),
                struct.unpack_from(self._report_fmt, self._elems))

    def spans(self) -> list:
        """Per worker, the ns it spent in the last run's chunks."""
        with self._txn:
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
        with self._txn:
            self._get_elems(ctypes.addressof(self._elems))
            return list(self._elems)


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
    for ptype, is_array in zip(compiled._param_types, compiled._param_is_array):
        t = _llvm_type(ptype)
        ptypes.append(t.as_pointer() if is_array else t)
    ptypes.extend([_i64] * compiled._num_dims)
    kernel_type = ir.FunctionType(ir.VoidType(), [*ptypes, _i64, _i64])
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
    b.call(kernel, [*args, start, end])
    b.ret_void()
    mod = llvm.parse_assembly(str(module))
    mod.verify()
    engine = llvm.create_mcjit_compiler(mod, _target_machine())
    engine.finalize_object()
    compiled._chunk_engine = engine
    compiled._chunk = engine.get_function_address("__tack_chunk")
    return compiled._chunk


# A `bind()` scalar's ctypes type as a struct code padded to its 8-byte slot.
_SLOT_CODES = {ctypes.c_int8: "b7x", ctypes.c_uint8: "B7x", ctypes.c_int16: "h6x",
               ctypes.c_uint16: "H6x", ctypes.c_int32: "i4x", ctypes.c_uint32: "I4x",
               ctypes.c_int64: "q", ctypes.c_uint64: "Q", ctypes.c_float: "f4x",
               ctypes.c_double: "d"}


def pack_args(compiled, prefix: tuple):
    """The ``bind()`` prefix as 8-byte slots: an array's address, a scalar
    stored with its own ctypes type at the slot's start, an extent as i64.
    One `struct.pack_into` with a format built once per kernel, instead of
    a ctypes `from_buffer` per slot (6 µs for nine slots on a 2 GHz Xeon)."""
    arrays = compiled._param_is_array
    fmt = compiled._pack_formats.get(len(prefix))
    if fmt is None:
        codes = ["Q" if arrays[i] else _SLOT_CODES[type(prefix[i])]
                 for i in range(min(len(prefix), len(arrays)))]
        codes.extend("q" for _ in range(len(arrays), len(prefix)))
        fmt = compiled._pack_formats[len(prefix)] = "=" + "".join(codes)
    pack = (ctypes.c_uint64 * max(1, len(prefix)))()
    struct.pack_into(fmt, pack, 0,
                     *(a if f else a.value for a, f in zip(prefix, arrays)),
                     *prefix[len(arrays):])
    return pack
