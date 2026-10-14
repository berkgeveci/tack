"""NUMA interleave for CPU field allocations.

These tests need a machine with two or more memory nodes, so they skip almost
everywhere -- including CI, which is single-node.  They exist because the thing
they check cannot be checked any other way: the interleave failed silently for
its whole life, on a constant that reads correct, and every test that did not
look at where the pages physically landed passed throughout.

The one test that does not need a NUMA machine is the constant itself.
"""

import ctypes
import ctypes.util
import platform
from collections import Counter

import pytest

import tack
from tack.runtime import cpu
from tack.runtime.kernel_utils import resolve_variant

# x86-64 syscall number.  Only used on machines where the backend has already
# decided it knows the numbers, i.e. where _numa_available is True.
_SYS_MOVE_PAGES = 279

requires_numa = pytest.mark.skipif(
    not cpu._numa_available,
    reason="needs a multi-node NUMA machine with libnuma",
)


def pages_per_node(addr: int, nbytes: int) -> Counter:
    """Which NUMA node holds each 4K page of [addr, addr+nbytes)."""
    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    n = nbytes // 4096
    pages = (ctypes.c_void_p * n)(*[addr + i * 4096 for i in range(n)])
    status = (ctypes.c_int * n)()
    rc = libc.syscall(_SYS_MOVE_PAGES, 0, ctypes.c_ulong(n), pages, None,
                      status, 0)
    assert rc == 0, f"move_pages failed: errno {ctypes.get_errno()}"
    return Counter(status)


def test_interleave_constant_is_the_kernels():
    """MPOL_INTERLEAVE is 3.  5 is MPOL_PREFERRED_MANY, which does not bind.

    Runs anywhere -- it is a check on a number, not on a machine.  This is the
    regression: the constant was 5, so the policy installed cleanly and left
    every page on the faulting node.
    """
    assert cpu._MPOL_INTERLEAVE == 3
    assert cpu._MPOL_DEFAULT == 0


@pytest.mark.skipif(platform.machine() != "x86_64",
                    reason="syscall numbers are x86-64's")
def test_numa_declines_when_the_kernel_rejects_the_policy():
    """A policy the kernel refuses must leave NUMA support off, not enabled.

    This is the half the readback can catch: a mode the kernel rejects, which
    is what a wrong syscall number or an unsupported mode looks like.  It
    cannot catch a mode that exists and means something else -- the readback
    compares against the same constant that was sent, so a wrong-but-valid
    mode agrees with itself.  That is what the placement tests below are for,
    and it is exactly how MPOL_PREFERRED_MANY went unnoticed.
    """
    saved_mode = cpu._MPOL_INTERLEAVE
    saved_avail, saved_libc = cpu._numa_available, cpu._libc
    saved_mask, saved_max = cpu._numa_node_mask, cpu._numa_max_node
    try:
        cpu._MPOL_INTERLEAVE = 99  # no such mode; stands in for a bad syscall
        cpu._numa_available = False
        cpu._init_numa()
        assert not cpu._numa_available, (
            "enabled NUMA interleave for a policy the kernel rejected"
        )
    finally:
        cpu._MPOL_INTERLEAVE = saved_mode
        cpu._numa_available, cpu._libc = saved_avail, saved_libc
        cpu._numa_node_mask, cpu._numa_max_node = saved_mask, saved_max


@requires_numa
def test_policy_reads_back_as_interleave():
    assert cpu._mempolicy_interleaves()


@requires_numa
def test_interleave_context_spreads_pages():
    """The context manager must actually spread pages, not merely be entered.

    In a fresh process: a block this size that the process freed before
    would come back with its pages already placed (the CPU backend tells
    glibc to keep them), and the policy only places pages faulted in."""
    import os
    import subprocess
    import sys
    code = _INTERLEAVE_CHECK.format(tests=os.path.dirname(os.path.abspath(__file__)))
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         timeout=300)
    assert run.returncode == 0 and run.stdout.strip().endswith("ok"), \
        (run.stdout + run.stderr)[-1500:]


_INTERLEAVE_CHECK = """
import sys
sys.path.insert(0, {tests!r})
import numpy as np
from tack.runtime import cpu
from test_numa import pages_per_node

nbytes = 64 << 20
with cpu._NumaInterleave():
    arr = np.empty(nbytes // 4, dtype=np.float32)
    arr.fill(0)  # fault the pages under the policy
hist = pages_per_node(arr.ctypes.data, nbytes)
assert len(hist) > 1, f"all pages on one node: {{dict(hist)}}"
print("ok")
"""


@tack.kernel
def _number(out):
    for i in range(out.shape[0]):
        out[i] = i


def _fan_out(backend, kernel, field, n):
    """Run the whole range through the backend's workers, the way a
    dispatch the policy threads does."""
    variant, effective = resolve_variant(backend, kernel, (field,), {},
                                         build=backend._build_variant,
                                         specialize_disjoint=cpu._SPECIALIZE_DISJOINT)
    _, extents = cpu._get_launch(variant.ir, list(effective))
    prefix = variant.payload.bind(list(effective), extents)
    backend._parallel_execute(variant.payload, prefix, 0, n, whole_range=True)


_SPREAD_CHECK = """
import sys
sys.path.insert(0, {tests!r})
import numpy as np
import tack
from tack.lang.field import Field
from tack.runtime.cpu import CPUBackend
from test_numa import _fan_out, _number, pages_per_node

backend = CPUBackend(num_threads=4)
n = (64 << 20) // 4
buf = backend.allocate_field(tack.f32, (n,))
addr = buf._data.ctypes.data
untouched = pages_per_node(addr, 64 << 20)[-14]
assert untouched >= (64 << 20) // 4096 - 1, f"a new field's pages were touched: {{untouched}}"
_fan_out(backend, _number, Field(tack.f32, (n,), buf), n)
hist = pages_per_node(addr, 64 << 20)
assert len(hist) > 1, f"all pages on one node: {{dict(hist)}}"
# Interleave is round-robin, so no node should hold a lopsided share.
biggest = max(hist.values()) / sum(hist.values())
assert biggest < 0.75, f"lopsided placement: {{dict(hist)}}"
np.testing.assert_array_equal(buf._data[:1000], np.arange(1000, dtype=np.float32))
print("ok")
"""


@requires_numa
@pytest.mark.parametrize("pool", ["jit", "python"])
def test_a_field_the_workers_write_is_spread_across_the_nodes(pool):
    """The path a real kernel takes. A new field is not touched when it is
    made; the workers that first write it carry the interleave policy (the
    JIT'd pool's from `_jit_pool`, the executor's from `_get_pool`), so its
    pages are spread the way a buffer filled under the policy was.

    In a fresh process: the CPU backend tells glibc to keep freed pages,
    so in a process that already freed a block this size the "new" field
    would be that block, its pages present and placed by whoever touched
    them first. Only a process's first such block is a fresh mapping."""
    import os
    import subprocess
    import sys
    env = dict(os.environ, TACK_CPU_POOL=pool)
    code = _SPREAD_CHECK.format(tests=os.path.dirname(os.path.abspath(__file__)))
    run = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                         text=True, timeout=300)
    assert run.returncode == 0 and run.stdout.strip().endswith("ok"), \
        (run.stdout + run.stderr)[-1500:]


@requires_numa
def test_policy_is_restored_after_the_context():
    """Leaking MPOL_INTERLEAVE would change every later allocation in-process."""
    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    with cpu._NumaInterleave():
        pass
    mode = ctypes.c_int(-1)
    rc = libc.syscall(cpu._SYS_GET_MEMPOLICY, ctypes.byref(mode), None,
                      ctypes.c_ulong(0), None, 0)
    assert rc == 0
    assert mode.value == cpu._MPOL_DEFAULT, f"policy leaked: mode {mode.value}"


_MPOL_BIND = 2


def _set_policy(mode, nodes):
    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    if mode == cpu._MPOL_DEFAULT:
        rc = libc.syscall(cpu._SYS_SET_MEMPOLICY, mode, None, 0)
    else:
        mask = (ctypes.c_ulong * 1)(nodes)
        rc = libc.syscall(cpu._SYS_SET_MEMPOLICY, mode, mask, cpu._numa_max_node + 2)
    assert rc == 0, f"set_mempolicy failed: errno {ctypes.get_errno()}"


@requires_numa
def test_a_bound_thread_keeps_its_policy_and_its_node():
    """Under ``numactl --membind=0`` (here: the same policy set by hand) a
    field's pages stay on node 0, and the policy is still the binding
    afterwards -- not the default, which is what restoring the default
    unconditionally left behind.

    In a fresh process, as the spreading test is: a block this size that
    the process freed before keeps its pages where they were first
    placed, binding or not (the CPU backend tells glibc to keep them)."""
    import os
    import subprocess
    import sys
    code = _BOUND_CHECK.format(tests=os.path.dirname(os.path.abspath(__file__)))
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         timeout=300)
    assert run.returncode == 0 and run.stdout.strip().endswith("ok"), \
        (run.stdout + run.stderr)[-1500:]


_BOUND_CHECK = """
import sys
sys.path.insert(0, {tests!r})
import tack
from tack.runtime import cpu
from test_numa import _MPOL_BIND, _set_policy, pages_per_node

_set_policy(_MPOL_BIND, 1)
# One thread: the fill runs on this, bound, thread.
tack.init(arch=tack.cpu, num_threads=1)
n = (64 << 20) // 4
field = tack.zeros(tack.f32, (n,))
hist = pages_per_node(field._buffer._data.ctypes.data, 64 << 20)
assert set(hist) == {{0}}, f"pages left the bound node: {{dict(hist)}}"
assert cpu._read_mempolicy()[0] == _MPOL_BIND, "the binding was dropped"
with cpu._NumaInterleave():
    assert cpu._read_mempolicy()[0] == _MPOL_BIND
print("ok")
"""


@requires_numa
def test_the_init_probe_restores_the_policy():
    """`_mempolicy_interleaves` is the first thing `tack.init` runs that
    touches the policy; it must hand back what it found."""
    _set_policy(_MPOL_BIND, 1)
    try:
        assert cpu._mempolicy_interleaves()
        assert cpu._read_mempolicy() == (_MPOL_BIND, 1)
    finally:
        _set_policy(cpu._MPOL_DEFAULT, 0)
