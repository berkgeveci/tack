"""Tests for tack.interop.vtk.

The module is now a thin layer over DLPack: VTK and Tack both speak it, so
the pointer plumbing that used to live here is gone. What remains is the
shape mapping between VTK's *tuples x components* and Tack's flat fields,
which is the one thing DLPack cannot know.

VTK is an optional dependency and is not installed in CI, so the exchange
itself is covered by tests that skip without it. The shape and validation
logic is tack's own and is tested with a stand-in for VTK's module, which
also keeps those tests honest about what they actually exercise.
"""

import sys
import types

import numpy as np
import pytest

import tack
from tack.interop import vtk as interop

try:
    from vtkmodules.util import dlpack_support as _real_dlpack_support
    HAVE_VTK_DLPACK = True
except ImportError:
    _real_dlpack_support = None
    HAVE_VTK_DLPACK = False

needs_vtk = pytest.mark.skipif(
    not HAVE_VTK_DLPACK,
    reason="needs VTK with vtkmodules.util.dlpack_support")


@pytest.fixture(autouse=True)
def cpu():
    tack.init(arch=tack.cpu)


# ── A stand-in for VTK's module ──────────────────────────────────────
#
# Records what tack hands it, so the shape mapping can be checked without
# a VTK build. It consumes the DLPack capsule for real, so a broken
# capsule still fails here.

class FakeVTKArray:
    def __init__(self, array, name):
        self.array = array
        self.name = name

    def GetNumberOfTuples(self):
        return self.array.shape[0]

    def GetNumberOfComponents(self):
        return self.array.shape[1] if self.array.ndim > 1 else 1


@pytest.fixture
def fake_vtk(monkeypatch):
    """Install a stand-in for vtkmodules.util.dlpack_support."""
    module = types.ModuleType("vtkmodules.util.dlpack_support")

    def dlpack_to_vtk(source, name=None):
        return FakeVTKArray(np.from_dlpack(source), name)

    def vtk_to_dlpack(array, device_id=0):
        return array.array.__dlpack__()

    module.dlpack_to_vtk = dlpack_to_vtk
    module.vtk_to_dlpack = vtk_to_dlpack
    monkeypatch.setattr(interop, "_dlpack_support", lambda: module)
    return module


def _field(values):
    f = tack.field(dtype=tack.f32, shape=(len(values),))
    f.from_numpy(np.asarray(values, dtype=np.float32))
    return f


# ── Shape mapping: Tack -> VTK ───────────────────────────────────────

def test_flat_field_becomes_tuples_of_one(fake_vtk):
    array = interop.field_to_vtk(_field(range(6)))
    assert array.GetNumberOfTuples() == 6
    assert array.GetNumberOfComponents() == 1


def test_components_group_the_values(fake_vtk):
    """3000 values with 3 components is 1000 points, not 3000."""
    array = interop.field_to_vtk(_field(range(12)), n_components=3)
    assert array.GetNumberOfTuples() == 4
    assert array.GetNumberOfComponents() == 3


def test_values_survive_the_grouping(fake_vtk):
    array = interop.field_to_vtk(_field(range(12)), n_components=3)
    np.testing.assert_array_equal(array.array.ravel(), np.arange(12))
    np.testing.assert_array_equal(array.array[1], [3, 4, 5])


def test_name_is_passed_through(fake_vtk):
    array = interop.field_to_vtk(_field(range(4)), name="velocity")
    assert array.name == "velocity"


def test_indivisible_size_is_rejected(fake_vtk):
    """7 values cannot be tuples of 3, and guessing would corrupt the data."""
    with pytest.raises(ValueError, match="does not divide"):
        interop.field_to_vtk(_field(range(7)), n_components=3)


def test_zero_components_is_rejected(fake_vtk):
    with pytest.raises(ValueError, match="at least 1"):
        interop.field_to_vtk(_field(range(6)), n_components=0)


def test_validation_does_not_need_vtk():
    """A shape mistake reports itself even where VTK is absent."""
    with pytest.raises(ValueError, match="does not divide"):
        interop.field_to_vtk(_field(range(7)), n_components=3)


# ── Shape mapping: VTK -> Tack ───────────────────────────────────────

def test_import_flattens_by_default(fake_vtk):
    """Tack kernels index interleaved data flat."""
    source = FakeVTKArray(np.arange(12, dtype=np.float32).reshape(4, 3), None)
    field = interop.vtk_to_field(source)
    assert field.shape == (12,)
    np.testing.assert_array_equal(field.to_numpy(), np.arange(12))


def test_import_can_keep_vtk_shape(fake_vtk):
    source = FakeVTKArray(np.arange(12, dtype=np.float32).reshape(4, 3), None)
    field = interop.vtk_to_field(source, flatten=False)
    assert field.shape == (4, 3)


def test_import_is_zero_copy(fake_vtk):
    data = np.arange(6, dtype=np.float32).reshape(3, 2)
    field = interop.vtk_to_field(FakeVTKArray(data, None))
    data[0, 0] = 99.0
    assert field.to_numpy()[0] == 99.0


def test_imported_field_outlives_the_source(fake_vtk):
    """The field holds the tensor, so the VTK array may be dropped."""
    import gc

    source = FakeVTKArray(np.arange(6, dtype=np.float32).reshape(3, 2), None)
    field = interop.vtk_to_field(source)
    del source
    for _ in range(3):
        gc.collect()
    np.testing.assert_array_equal(field.to_numpy(), np.arange(6))


def test_missing_vtk_reports_what_is_missing(monkeypatch):
    """The failure should name the module, not surface an ImportError."""
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) \
        else __builtins__.__import__

    def blocked(name, *args, **kwargs):
        if name.startswith("vtkmodules"):
            raise ImportError("no vtkmodules")
        return real_import(name, *args, **kwargs)

    monkeypatch.setitem(sys.modules, "vtkmodules", None)
    with pytest.raises(RuntimeError, match="dlpack_support"):
        interop._dlpack_support()


# ── Against a real VTK, when there is one ────────────────────────────

@needs_vtk
def test_round_trip_through_real_vtk():
    from vtkmodules.vtkCommonCore import vtkFloatArray

    array = vtkFloatArray()
    array.SetNumberOfComponents(3)
    array.SetNumberOfTuples(4)
    for i in range(12):
        array.SetValue(i, float(i))

    field = interop.vtk_to_field(array)
    assert field.shape == (12,)
    np.testing.assert_array_equal(field.to_numpy(), np.arange(12))

    back = interop.field_to_vtk(field, n_components=3, name="round")
    assert back.GetNumberOfTuples() == 4
    assert back.GetNumberOfComponents() == 3
    assert back.GetName() == "round"


@needs_vtk
def test_real_vtk_exchange_is_zero_copy():
    from vtkmodules.vtkCommonCore import vtkFloatArray

    array = vtkFloatArray()
    array.SetNumberOfComponents(1)
    array.SetNumberOfTuples(4)
    for i in range(4):
        array.SetValue(i, 0.0)

    field = interop.vtk_to_field(array)
    array.SetValue(2, 42.0)
    assert field.to_numpy()[2] == 42.0


@needs_vtk
def test_kernel_output_reaches_vtk_without_a_copy():
    """The point of the whole exercise."""

    @tack.kernel
    def ramp(out, n):
        for i in range(n):
            out[i] = float(i) * 2.0

    out = tack.field(dtype=tack.f32, shape=(9,))
    ramp(out, 9)

    array = interop.field_to_vtk(out, n_components=3, name="ramp")
    assert array.GetNumberOfTuples() == 3
    assert array.GetTuple3(1) == pytest.approx((6.0, 8.0, 10.0))
