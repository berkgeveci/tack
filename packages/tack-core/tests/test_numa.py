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

import numpy as np
import pytest

import tack
from tack.runtime import cpu

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
    """The context manager must actually spread pages, not merely be entered."""
    nbytes = 64 << 20
    with cpu._NumaInterleave():
        arr = np.empty(nbytes // 4, dtype=np.float32)
        arr.fill(0)  # fault the pages under the policy
    hist = pages_per_node(arr.ctypes.data, nbytes)
    assert len(hist) > 1, f"all pages on one node: {dict(hist)}"


@requires_numa
def test_allocate_field_spreads_pages():
    """The path a real kernel takes: tack.field() on the CPU backend."""
    tack.init(arch=tack.cpu)
    n = (64 << 20) // 4
    field = tack.field(tack.f32, (n,))
    hist = pages_per_node(field._buffer._data.ctypes.data, 64 << 20)
    assert len(hist) > 1, f"all pages on one node: {dict(hist)}"
    # Interleave is round-robin, so no node should hold a lopsided share.
    biggest = max(hist.values()) / sum(hist.values())
    assert biggest < 0.75, f"lopsided placement: {dict(hist)}"


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
