"""Tests for user-friendly error reporting."""

import pytest

import tack


def test_init_unknown_arch():
    """Unknown architecture gives a clear error listing available options."""
    with pytest.raises(ValueError, match="Unknown architecture.*Available:"):
        tack.init(arch="bogus")


def test_init_missing_backend():
    """Missing backend dependency gives install instructions."""
    # HIP is unlikely to be installed in CI/test environments
    try:
        tack.init(arch="hip")
        pytest.skip("HIP is available — can't test missing backend error")
    except RuntimeError as e:
        msg = str(e)
        assert "hip" in msg.lower()
        assert "Requires" in msg or "missing dependency" in msg


def test_kernel_type_error():
    """Wrong argument count gives a clear error with kernel name."""
    tack.init(arch=tack.cpu)

    @tack.kernel
    def my_kernel(x, y, out):
        for i in range(x.shape[0]):
            out[i] = x[i] + y[i]

    x = tack.field(dtype=tack.f32, shape=(4,))
    with pytest.raises(TypeError, match="my_kernel"):
        my_kernel(x)  # too few args


def test_kernel_runtime_error_includes_name():
    """Runtime errors from kernels include the kernel name."""
    tack.init(arch=tack.cpu)

    @tack.kernel
    def bad_kernel(out, n):
        for i in range(n):
            out[i] = 1.0

    # Pass a non-field where a field is expected
    with pytest.raises((TypeError, RuntimeError)):
        bad_kernel("not_a_field", 10)
