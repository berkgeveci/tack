"""The JIT'd thread pool (`tack.runtime.cpu_pool`): a fan-out as one ctypes
call. The policy that decides *whether* to fan out is tested through the
Python path in test_cpu_threading.py; here the pool itself is: that it is
chosen, that it runs every chunk once with the kernel's own arguments, that
backends with different thread counts share it, that it zeroes new arrays,
and that it can be switched off."""
import os
import platform
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

import tack
from tack.runtime import cpu as cpu_mod
from tack.runtime import cpu_pool
from tack.runtime.cpu import CPUBackend
from tack.runtime.kernel_utils import resolve_variant

pytestmark = pytest.mark.skipif(platform.system() == "Windows",
                                reason="the pool is pthreads")


@pytest.fixture(autouse=True)
def _on_the_cpu():
    """The arrays these tests make must be CPU arrays whatever backend the
    session last initialized."""
    tack.set_default_device("cpu")


@tack.kernel
def _mixed(x, y, out, a: tack.f32, b: tack.f64, k: tack.i32, m: tack.i64):
    for i in range(out.shape[0]):
        out[i] = x[i] * a + y[i] * b + k + m


@tack.kernel
def _number(out):
    for i in range(out.shape[0]):
        out[i] = out[i] + i


@tack.kernel
def _grid(out, scale: tack.i32):
    for i, j in tack.ndindex(300, 257):
        out[i * 257 + j] = i * scale + j


def _compiled(backend, kernel, args):
    """The compiled variant and its bound prefix, the way `execute` gets them."""
    variant, effective = resolve_variant(backend, kernel, tuple(args), {},
                                         build=backend._build_variant,
                                         specialize_disjoint=cpu_mod._SPECIALIZE_DISJOINT)
    loop_end, extents = cpu_mod._get_launch(variant.ir, list(effective))
    return variant.payload, variant.payload.bind(list(effective), extents), loop_end


def _fresh_backend(threads, monkeypatch=None):
    if monkeypatch is not None:
        monkeypatch.delenv("TACK_CPU_POOL", raising=False)
    return CPUBackend(num_threads=threads)


def test_the_pool_is_the_default_and_can_be_switched_off(monkeypatch):
    monkeypatch.delenv("TACK_CPU_POOL", raising=False)
    assert cpu_pool.enabled()
    assert CPUBackend(num_threads=2).pool_kind == "jit"
    for off in ("python", "0", "off"):
        monkeypatch.setenv("TACK_CPU_POOL", off)
        assert not cpu_pool.enabled()
        assert CPUBackend(num_threads=2).pool_kind == "python"


def test_one_thread_never_makes_a_pool(monkeypatch):
    backend = _fresh_backend(1, monkeypatch)
    assert backend._jit_pool() is None and backend._cpool is None


def test_a_fan_out_runs_every_element_once_with_every_argument_kind(monkeypatch):
    """Pointers of three widths, f32, f64, i32 and i64 scalars, through the
    pack and the wrapper, on a range the thread count does not divide."""
    backend = _fresh_backend(4, monkeypatch)
    n = 100_003
    x = tack.empty((n,)); tack.copyto(x, np.arange(n, dtype=np.float32))
    y = tack.empty((n,), tack.f64); tack.copyto(y, np.full(n, 0.5))
    out = tack.zeros((n,), tack.f64)
    compiled, prefix, loop_end = _compiled(backend, _mixed, [x, y, out, 2.0, 4.0, 7, 1 << 40])
    assert loop_end == n
    backend._parallel_execute(compiled, prefix, 0, n, whole_range=True)
    assert backend._cpool is not None, "the fan-out did not go through the pool"
    expected = np.arange(n, dtype=np.float32).astype(np.float64) * 2.0 + 2.0 + 7 + (1 << 40)
    np.testing.assert_array_equal(out.numpy(), expected)
    # Every element ran once: the workers' counts add up to the range and
    # the result says each element was touched exactly once.
    assert sum(backend._cpool.elems()[:4]) == n
    assert any(s > 0 for s in backend._cpool.spans()[:4])


def test_each_element_is_visited_exactly_once(monkeypatch):
    backend = _fresh_backend(8, monkeypatch)
    n = 1_000_001
    out = tack.zeros((n,), tack.i64)
    compiled, prefix, _ = _compiled(backend, _number, [out])
    backend._parallel_execute(compiled, prefix, 0, n, whole_range=True)
    backend._parallel_execute(compiled, prefix, 0, n, whole_range=True)
    np.testing.assert_array_equal(out.numpy(), 2 * np.arange(n, dtype=np.int64))


def test_a_multi_dimensional_range_carries_its_extents(monkeypatch):
    backend = _fresh_backend(7, monkeypatch)
    out = tack.zeros((300 * 257,), tack.i32)
    compiled, prefix, loop_end = _compiled(backend, _grid, [out, 1000])
    assert loop_end == 300 * 257
    backend._parallel_execute(compiled, prefix, 0, loop_end, whole_range=True)
    i, j = np.indices((300, 257))
    np.testing.assert_array_equal(out.numpy().reshape(300, 257), i * 1000 + j)


def test_backends_with_different_thread_counts_share_the_pool(monkeypatch):
    two, three = _fresh_backend(2, monkeypatch), _fresh_backend(3, monkeypatch)
    n = 50_000
    out = tack.zeros((n,), tack.i64)
    for backend, chunks in ((two, 2), (three, 3)):
        compiled, prefix, _ = _compiled(backend, _number, [out])
        backend._parallel_execute(compiled, prefix, 0, n, whole_range=True)
        counts = backend._cpool.elems()
        assert sum(counts[:chunks]) == n
        assert all(c == 0 for c in counts[chunks:])
    assert two._cpool is three._cpool
    np.testing.assert_array_equal(out.numpy(), 2 * np.arange(n, dtype=np.int64))


def test_an_empty_range_is_a_fan_out_of_nothing(monkeypatch):
    """The calibration probes fan out empty ranges: no element may run."""
    backend = _fresh_backend(4, monkeypatch)
    n = 4096
    out = tack.zeros((n,), tack.i64)
    compiled, prefix, _ = _compiled(backend, _number, [out])
    assert backend._empty_fan_out_ns(compiled, prefix) > 0
    backend._parallel_execute(compiled, prefix, 0, 0, whole_range=True)
    assert not out.numpy().any()


def test_the_policy_measures_the_pool(monkeypatch):
    """The fan-out the policy calibrates is the pool's, not the executor's."""
    backend = _fresh_backend(4, monkeypatch)
    n = 4096
    out = tack.zeros((n,), tack.i64)
    compiled, prefix, _ = _compiled(backend, _number, [out])
    backend._calibrate_fan_out(compiled, prefix)
    assert backend._pool is None, "the ThreadPoolExecutor was made"
    assert 0 < backend._fan_out_ns < 5_000_000


def test_chunks_are_claimed_and_cover_the_range(monkeypatch):
    """Chunks are claimed, not assigned: a range long enough for several per
    worker is cut finer than one share each, every element is claimed once
    in whole chunks, and no worker is bound to a share. Which workers claim
    is the scheduler's: a late one may claim nothing (on a loaded host the
    counts have come out as [437500, 0, 0, 562500]), so nothing here needs
    every worker to take part."""
    backend = _fresh_backend(4, monkeypatch)
    n = 1_000_000
    out = tack.zeros((n,), tack.i64)
    compiled, prefix, _ = _compiled(backend, _number, [out])
    backend._parallel_execute(compiled, prefix, 0, n, whole_range=True)
    counts = backend._cpool.elems()[:4]
    chunk = backend._cpool.chunk_size(n, 4)
    assert chunk < n // 4, "one chunk per worker"
    assert sum(counts) == n
    assert all(c % chunk == 0 or c % chunk == n % chunk for c in counts)
    assert sum(-(-c // chunk) for c in counts) == -(-n // chunk), "a chunk claimed twice or not at all"
    np.testing.assert_array_equal(out.numpy(), np.arange(n, dtype=np.int64))


def test_concurrent_fan_outs_keep_their_own_launch_state(monkeypatch):
    """The pool's launch state is one set of globals for the process, so
    two threads fanning out at once must take turns: before they did,
    each wrote the function, pack and range the other was running, and
    outputs came back wrong or unwritten (TA-45)."""
    backend = _fresh_backend(4, monkeypatch)
    n, threads, rounds = 400_000, 4, 25
    outs = [tack.zeros((n,), tack.i64) for _ in range(threads)]
    plans = [_compiled(backend, _number, [out]) for out in outs]
    errors = []

    def worker(k):
        compiled, prefix, _ = plans[k]
        try:
            for _ in range(rounds):
                backend._parallel_execute(compiled, prefix, 0, n, whole_range=True)
        except Exception as e:
            errors.append(e)

    ts = [threading.Thread(target=worker, args=(k,)) for k in range(threads)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errors
    for out in outs:
        np.testing.assert_array_equal(out.numpy(), rounds * np.arange(n, dtype=np.int64))


def test_a_failed_worker_start_stops_the_started_workers(tmp_path):
    """`pthread_create` failing after some workers exist: those workers
    wait in JIT'd code the constructor is about to free, so they must be
    told to leave and joined before the pool reports failure (TA-46: a
    SIGSEGV). A fresh process, with a `pthread_create` that fails on its
    second call; the thread count must come back to what it was, the
    backend must fall back to the executor, and the exit must be clean."""
    program = r"""
import ctypes, gc, os, re, sys
import numpy as np
import llvmlite.binding as llvm
import tack
from tack.runtime import cpu_pool
from tack.runtime.dispatch import get_backend

def threads():
    return int(re.search(r"Threads:\s+(\d+)", open("/proc/self/status").read()).group(1))

libc = ctypes.CDLL(None)
real = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                        ctypes.c_void_p)(ctypes.cast(libc.pthread_create, ctypes.c_void_p).value)
calls = []
@ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)
def failing_create(tid, attr, fn, arg):
    calls.append(1)
    return real(tid, attr, fn, arg) if len(calls) < 2 else 11      # EAGAIN
register = cpu_pool._register_libc_symbols
def patched():
    register()
    llvm.add_symbol("pthread_create", ctypes.cast(failing_create, ctypes.c_void_p).value)
cpu_pool._register_libc_symbols = patched

before = threads()
try:
    cpu_pool.CPool(4)
except RuntimeError as e:
    print("raised:", e)
else:
    sys.exit("the pool started")
gc.collect()
assert len(calls) == 2, calls
assert threads() == before, (before, threads())
tack.set_default_device("cpu", num_threads=4)
backend = get_backend()
assert backend._jit_pool() is None and backend.pool_kind == "python"
n = 300_000
@tack.kernel
def number(out):
    for i in range(out.shape[0]):
        out[i] = out[i] + i
out = tack.zeros((n,), tack.i64)
number(out)
np.testing.assert_array_equal(out.numpy(), np.arange(n, dtype=np.int64))
print("ok")
"""
    if platform.system() != "Linux":
        pytest.skip("counts threads through /proc")
    env = {k: v for k, v in os.environ.items() if k != "TACK_CPU_POOL"}
    # A file, not `-c`: the program defines a kernel, whose source Tack
    # reads back, and only Python 3.13 keeps the source of `-c` code.
    script = tmp_path / "failed_start.py"
    script.write_text(program)
    run = subprocess.run([sys.executable, str(script)], capture_output=True,
                         text=True, env=env, timeout=120)
    assert run.returncode == 0, run.stderr
    assert run.stdout.strip().endswith("ok"), run.stdout + run.stderr
    assert "raised: could not start a pool of 4 threads" in run.stdout


def test_the_python_path_still_works(monkeypatch):
    monkeypatch.setenv("TACK_CPU_POOL", "python")
    backend = CPUBackend(num_threads=4)
    n = 100_003
    out = tack.zeros((n,), tack.i64)
    compiled, prefix, _ = _compiled(backend, _number, [out])
    backend._parallel_execute(compiled, prefix, 0, n, whole_range=True)
    assert backend._cpool is None and backend._pool is not None
    np.testing.assert_array_equal(out.numpy(), np.arange(n, dtype=np.int64))


def test_the_pool_builds_in_a_fresh_process():
    """Building the pool must not abort the interpreter. It did on Apple
    silicon: the arm64 `yield` pause is inline asm and the asm parser was
    never initialized, and LLVM aborts rather than raising, so the Python
    fallback never ran. A fresh process, since an abort would take pytest
    with it."""
    program = (
        "import tack\n"
        "from tack.runtime.dispatch import get_backend\n"
        "tack.set_default_device('cpu', num_threads=2)\n"
        "pool = get_backend()._jit_pool()\n"
        "print('jit' if pool is not None else 'python')\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "TACK_CPU_POOL"}
    run = subprocess.run([sys.executable, "-c", program], capture_output=True,
                         text=True, env=env, timeout=120)
    assert run.returncode == 0, run.stderr
    assert run.stdout.strip() == "jit", run.stderr


def test_idle_workers_block_rather_than_poll(monkeypatch):
    """On Linux (futex) and macOS 14.4+ (os_sync_wait_on_address) a sleeping
    worker is woken by the launch, not by its own poll timer: polling cost
    up to 1 ms per launch on an M1 Max and made the policy under-thread."""
    backend = _fresh_backend(2, monkeypatch)
    pool = backend._jit_pool()
    assert pool is not None
    if platform.system() == "Linux":
        assert pool.waiter == "futex"
    elif platform.system() == "Darwin" and tuple(map(int, platform.mac_ver()[0].split(".")[:2])) >= (14, 4):
        assert pool.waiter == "wait_on_address"
    else:
        assert pool.waiter == "poll"
    # And it still runs every element once after the workers have slept.
    time.sleep(0.05)
    n = 100_000
    out = tack.zeros((n,), tack.i64)
    compiled, prefix, _ = _compiled(backend, _number, [out])
    backend._parallel_execute(compiled, prefix, 0, n, whole_range=True)
    np.testing.assert_array_equal(out.numpy(), np.arange(n, dtype=np.int64))


def test_a_pool_that_cannot_be_made_falls_back(monkeypatch):
    monkeypatch.delenv("TACK_CPU_POOL", raising=False)
    backend = CPUBackend(num_threads=4)
    monkeypatch.setattr(cpu_pool.CPool, "get", classmethod(lambda cls, threads: None))
    assert backend._jit_pool() is None
    assert backend.pool_kind == "python"
    n = 10_000
    out = tack.zeros((n,), tack.i64)
    compiled, prefix, _ = _compiled(backend, _number, [out])
    backend._parallel_execute(compiled, prefix, 0, n, whole_range=True)
    np.testing.assert_array_equal(out.numpy(), np.arange(n, dtype=np.int64))


def test_a_kernel_through_init_and_call(monkeypatch):
    """The ordinary path: tack.init, a kernel call big enough to thread."""
    monkeypatch.delenv("TACK_CPU_POOL", raising=False)
    tack.set_default_device("cpu", num_threads=4)
    n = 2_000_000
    out = tack.zeros((n,), tack.i64)
    for _ in range(3):
        _number(out)
    np.testing.assert_array_equal(out.numpy(), 3 * np.arange(n, dtype=np.int64))


def test_a_run_uses_only_the_first_k_workers(monkeypatch):
    """A run of k threads posts to workers 0 .. k - 2 and runs as worker
    k - 1 itself; the other workers are not woken and their slots read
    zero. Every element is still claimed exactly once."""
    # Six threads where the host has them; the pool is sized by the CPUs
    # the process may run on, and a CI runner has four or fewer.
    threads = min(6, max(2, cpu_pool._available_cpus()))
    backend = _fresh_backend(threads, monkeypatch)
    n = 200_000
    out = tack.zeros((n,), tack.i64)
    compiled, prefix, _ = _compiled(backend, _number, [out])
    backend._parallel_execute(compiled, prefix, 0, n, whole_range=True)   # makes the pool
    pool = backend._cpool
    assert pool is not None, backend.pool_kind
    wrapper = cpu_pool.chunk_wrapper(compiled)
    pack = cpu_pool.pack_args(compiled, prefix)
    out.fill(0)
    ks = [k for k in (1, 2, 3, 6) if k <= threads]
    for k in ks:
        spans, elems = pool.run(wrapper, pack, 0, n, k, measure=True)
        assert sum(elems) == n, (k, elems)
        assert all(e == 0 and s == 0 for e, s in zip(elems[k:], spans[k:])), (k, elems)
        # Which of the k claim is the scheduler's (a posted worker that is
        # descheduled, as under a loaded test run, may claim nothing).
    np.testing.assert_array_equal(out.numpy(), len(ks) * np.arange(n))
