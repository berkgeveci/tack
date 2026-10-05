"""Level Zero in a context another library owns.

A USM pointer means something only inside the context it was allocated
from, and neither DLPack nor a raw address carries one. So sharing device
memory with another Level Zero user means starting Tack inside *its*
context: `tack.init(arch=tack.level_zero, external_context=...)`.

These tests stand in for the other library with a second context created
directly through Level Zero, so they need a device but not VTK or SYCL.
"""

import ctypes

import numpy as np
import pytest

import tack
from tack.runtime.dispatch import get_backend


def test_an_option_the_backend_does_not_declare_is_refused():
    """Dropping it silently would start a private context while the caller
    believes memory is shared."""
    with pytest.raises(ValueError, match="does not accept.*external_context"):
        tack.init(arch=tack.cpu, external_context={"context": 1})
    with pytest.raises(ValueError, match="does not accept.*contxt"):
        tack.init(arch=tack.cpu, contxt=1)


@pytest.fixture
def foreign():
    """Driver, device and a context Tack did not create."""
    try:
        tack.init(arch=tack.level_zero)
    except RuntimeError as e:
        pytest.skip(f"no Level Zero backend: {e}")
    from tack.runtime import level_zero_backend as l0

    own = get_backend()
    ze = l0._get_ze()
    desc = l0.ze_context_desc_t(
        stype=l0.ZE_STRUCTURE_TYPE_CONTEXT_DESC, pNext=None, flags=0)
    context = l0.ze_context_handle_t()
    l0._check_ze(ze.zeContextCreate(own._driver, ctypes.byref(desc),
                                    ctypes.byref(context)), "zeContextCreate")
    yield {"driver": own._driver, "device": own._device,
           "context": context.value}
    tack.init(arch=tack.cpu)


def test_the_default_backend_owns_its_context(foreign):
    assert get_backend().shares_external_context is False


def test_an_external_context_is_adopted_not_replaced(foreign):
    tack.init(arch=tack.level_zero, external_context=foreign)
    backend = get_backend()

    assert backend.shares_external_context is True
    assert backend._context.value == foreign["context"]
    assert backend._device == foreign["device"]
    assert backend._driver == foreign["driver"]


def test_kernels_run_on_fields_allocated_in_the_adopted_context(foreign):
    tack.init(arch=tack.level_zero, external_context=foreign)

    @tack.kernel
    def ramp(out, n):
        for i in range(n):
            out[i] = float(i) * 3.0

    out = tack.field(dtype=tack.f32, shape=(64,))
    ramp(out, 64)

    np.testing.assert_array_equal(out.to_numpy(),
                                  np.arange(64, dtype=np.float32) * 3.0)


@pytest.mark.parametrize("missing", ["driver", "device", "context"])
def test_every_handle_is_required(foreign, missing):
    """A driver or device found separately need not be the one the context
    was created against, so none is filled in on the caller's behalf."""
    handles = dict(foreign)
    handles[missing] = 0
    with pytest.raises(ValueError, match=missing):
        tack.init(arch=tack.level_zero, external_context=handles)


def test_device_memory_is_told_apart_from_host_memory(foreign):
    tack.init(arch=tack.level_zero, external_context=foreign)
    backend = get_backend()
    field = tack.field(dtype=tack.f32, shape=(8,))
    host = np.zeros(8, dtype=np.float32)

    assert backend.memory_space(field._buffer._device_ptr.value) == "level_zero"
    assert backend.memory_space(host.ctypes.data) == "cpu"
    with pytest.raises(ValueError, match="'cpu' memory"):
        tack.field_from_ptr(host.ctypes.data, tack.f32, (8,))
