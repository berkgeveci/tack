"""A GPU launch must index every iteration once, or be refused up front.

CUDA and HIP computed the thread index as a 32-bit product, so a launch of
2^32 or more iterations wrapped silently. Each backend now checks the
iteration count against what one grid can index before dispatching.
"""

import numpy as np
import pytest

import tack
from tack.runtime.dispatch import get_backend
from tack.runtime.kernel_utils import check_launch_size

pytestmark = pytest.mark.quick      # the inner-loop set; see CLAUDE.md, Tests


@tack.kernel
def _fill_ones(out):
    for i in range(out.shape[0]):
        out[i] = 1.0


@tack.kernel
def _mark_past(out, n, base):
    for i in range(n):
        if i >= base:
            out[i - base] = 1


def _limit_launches(monkeypatch, items):
    """Make one launch of the active backend index at most `items` iterations.

    Every GPU backend's dispatch reads `_max_launch` (`GPUBackend.execute`);
    Metal's used to read its module's `_MAX_LAUNCH` directly."""
    monkeypatch.setattr(get_backend(), '_max_launch', items)


def test_launch_size_check_names_the_operation_and_limit():
    check_launch_size("Kernel 'k'", 1024, 1024, 'CUDA')
    with pytest.raises(ValueError, match=(
            r"^Kernel 'k': 1025 iterations exceed the 1024 that one CUDA "
            r"launch can index; split the work across several launches\.$")):
        check_launch_size("Kernel 'k'", 1025, 1024, 'CUDA')


def test_over_limit_launch_is_refused_before_dispatch(workgroup_backend, monkeypatch):
    _limit_launches(monkeypatch, 1024)
    at_limit = tack.empty((1024,))
    _fill_ones(at_limit)
    assert at_limit.numpy().min() == 1.0

    over = tack.empty((1025,))
    with pytest.raises(ValueError, match="Kernel '_fill_ones': 1025 iterations exceed the 1024"):
        _fill_ones(over)
    assert over.numpy().max() == 0.0


def test_over_limit_native_reduction(reduction_backend, monkeypatch):
    _limit_launches(monkeypatch, 1024)
    array = tack.empty((1025,))
    array.fill(1.0)
    if get_backend().name == 'metal':
        # Metal reduces an array its 32-bit kernel cannot index through NumPy.
        assert array.sum() == 1025.0
        return
    with pytest.raises(ValueError, match=r"Array sum\(\): 1025 iterations exceed the 1024"):
        array.sum()


@pytest.mark.parametrize('n', [1, 255, 257, 100_003])
def test_native_reduction_counts_every_element(reduction_backend, n):
    array = tack.empty((n,))
    array.fill(1.0)
    assert array.sum() == float(n)
    assert array.max() == 1.0


def test_launch_past_2_32_iterations_reaches_every_index(workgroup_backend):
    # Only the last 256 of 2^32 + 256 iterations store, so the array stays
    # tiny. A 32-bit index never reaches them, and they stayed zero.
    base = 2**32
    out = tack.empty((256,), tack.i32)
    if get_backend().name in ('hip', 'metal'):
        # One launch there indexes at most 2^32 threads: refuse, not wrap.
        with pytest.raises(ValueError, match="iterations exceed"):
            _mark_past(out, base + 256, base)
        return
    _mark_past(out, base + 256, base)
    np.testing.assert_array_equal(out.numpy(), np.ones(256, dtype=np.int32))
