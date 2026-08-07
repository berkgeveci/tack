"""Tests for tack.interop.vtk.

The module is a thin layer over DLPack: VTK and Tack both speak it, so the
pointer plumbing that used to live here is gone. What remains is the shape
mapping between VTK's *tuples x components* and Tack's fields -- which is
mostly a matter of not getting in the way, since a 2-D field already says
how many components it has.

VTK is an optional dependency and is not installed in CI, so the exchange
itself is covered by tests that skip without it. The shape and validation
logic is tack's own and is tested with a stand-in for VTK's module, which
also keeps those tests honest about what they actually exercise.
"""


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
# capsule still fails here, and it exports 2-D exactly as VTK does.

class FakeVTKArray:
    def __init__(self, array, name=None):
        if array.ndim == 1:
            array = array.reshape(len(array), 1)
        self.array = array
        self.name = name

    def GetNumberOfTuples(self):
        return self.array.shape[0]

    def GetNumberOfComponents(self):
        return self.array.shape[1]


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


def _field(values, shape=None):
    values = np.asarray(values, dtype=np.float32)
    f = tack.field(dtype=tack.f32, shape=shape or values.shape)
    f.from_numpy(values.reshape(shape) if shape else values)
    return f


# ── Tack -> VTK: the shape already says how many components ──────────

def test_two_d_field_needs_no_declaration(fake_vtk):
    """A (4, 3) field is 4 tuples of 3. Nothing to pass."""
    array = interop.field_to_vtk(_field(range(12), shape=(4, 3)))
    assert array.GetNumberOfTuples() == 4
    assert array.GetNumberOfComponents() == 3


def test_flat_field_is_one_component(fake_vtk):
    array = interop.field_to_vtk(_field(range(6)))
    assert array.GetNumberOfTuples() == 6
    assert array.GetNumberOfComponents() == 1


def test_components_group_a_flat_field(fake_vtk):
    """The interleaved form tack's own algorithms produce."""
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


def test_redundant_components_are_allowed(fake_vtk):
    """Saying what the shape already says is not an error."""
    array = interop.field_to_vtk(_field(range(12), shape=(4, 3)), n_components=3)
    assert array.GetNumberOfComponents() == 3


def test_components_contradicting_the_shape_are_rejected(fake_vtk):
    """A (4, 3) field is not tuples of 2, and one of the two is a mistake."""
    with pytest.raises(ValueError, match="contradicts"):
        interop.field_to_vtk(_field(range(12), shape=(4, 3)), n_components=2)


def test_indivisible_size_is_rejected(fake_vtk):
    """7 values cannot be tuples of 3, and guessing would corrupt the data."""
    with pytest.raises(ValueError, match="does not divide"):
        interop.field_to_vtk(_field(range(7)), n_components=3)


def test_zero_components_is_rejected(fake_vtk):
    with pytest.raises(ValueError, match="at least 1"):
        interop.field_to_vtk(_field(range(6)), n_components=0)


def test_three_d_field_is_rejected(fake_vtk):
    """VTK has no layout for it, so guessing one would be inventing data."""
    with pytest.raises(ValueError, match="no obvious VTK layout"):
        interop.field_to_vtk(_field(range(24), shape=(2, 3, 4)))


def test_three_d_field_can_be_flattened_explicitly(fake_vtk):
    array = interop.field_to_vtk(_field(range(24), shape=(2, 3, 4)), n_components=4)
    assert array.GetNumberOfTuples() == 6
    assert array.GetNumberOfComponents() == 4


def test_validation_does_not_need_vtk():
    """A shape mistake reports itself even where VTK is absent."""
    with pytest.raises(ValueError, match="does not divide"):
        interop.field_to_vtk(_field(range(7)), n_components=3)


# ── VTK -> Tack: the shape comes across intact ───────────────────────

def test_import_keeps_the_vtk_shape(fake_vtk):
    """4 tuples of 3 arrive as (4, 3), so field.shape[1] is the components."""
    source = FakeVTKArray(np.arange(12, dtype=np.float32).reshape(4, 3))
    field = interop.vtk_to_field(source)
    assert field.shape == (4, 3)
    np.testing.assert_array_equal(field.to_numpy(), np.arange(12).reshape(4, 3))


def test_single_component_arrives_flat(fake_vtk):
    """VTK exports (n, 1), but a scalar array is naturally 1-D."""
    source = FakeVTKArray(np.arange(5, dtype=np.float32))
    field = interop.vtk_to_field(source)
    assert field.shape == (5,)


def test_import_can_flatten_for_tacks_algorithms(fake_vtk):
    """flying_edges and compute_normals index points interleaved."""
    source = FakeVTKArray(np.arange(12, dtype=np.float32).reshape(4, 3))
    field = interop.vtk_to_field(source, flatten=True)
    assert field.shape == (12,)
    np.testing.assert_array_equal(field.to_numpy(), np.arange(12))


def test_import_is_zero_copy(fake_vtk):
    data = np.arange(6, dtype=np.float32).reshape(3, 2)
    field = interop.vtk_to_field(FakeVTKArray(data))
    data[0, 0] = 99.0
    assert field.to_numpy()[0, 0] == 99.0


def test_flattened_import_is_also_zero_copy(fake_vtk):
    """reshape must carry the DLPack hold, not just the buffer."""
    data = np.arange(6, dtype=np.float32).reshape(3, 2)
    field = interop.vtk_to_field(FakeVTKArray(data), flatten=True)
    data[2, 1] = 99.0
    assert field.to_numpy()[5] == 99.0


def test_imported_field_outlives_the_source(fake_vtk):
    """The field holds the tensor, so the VTK array may be dropped."""
    import gc

    source = FakeVTKArray(np.arange(6, dtype=np.float32).reshape(3, 2))
    field = interop.vtk_to_field(source)
    del source
    for _ in range(3):
        gc.collect()
    np.testing.assert_array_equal(field.to_numpy().ravel(), np.arange(6))


def test_round_trip_preserves_shape(fake_vtk):
    """(4, 3) out, (4, 3) back, with nothing declared either way."""
    field = _field(range(12), shape=(4, 3))
    back = interop.vtk_to_field(interop.field_to_vtk(field))
    assert back.shape == (4, 3)
    np.testing.assert_array_equal(back.to_numpy(), np.arange(12).reshape(4, 3))


def test_missing_vtk_reports_what_is_missing(monkeypatch):
    """The failure should name the module, not surface an ImportError.

    Blocking the import itself rather than clearing sys.modules, so this
    tests the same thing whether or not VTK is installed here.
    """
    import builtins
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.startswith("vtkmodules"):
            raise ImportError("no vtkmodules")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
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
    assert field.shape == (4, 3)
    np.testing.assert_array_equal(field.to_numpy(), np.arange(12).reshape(4, 3))

    back = interop.field_to_vtk(field, name="round")
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
@pytest.mark.parametrize("dtype,np_dtype", [
    (tack.f32, np.float32), (tack.f64, np.float64),
    (tack.i8, np.int8), (tack.u8, np.uint8),
    (tack.i16, np.int16), (tack.u16, np.uint16),
    (tack.i32, np.int32), (tack.u32, np.uint32),
    (tack.i64, np.int64), (tack.u64, np.uint64),
], ids=lambda v: str(v) if hasattr(v, "name") else "")
def test_every_dtype_survives_real_vtk(dtype, np_dtype):
    """Only f32 was ever exercised against a real VTK build."""
    values = np.arange(12, dtype=np_dtype).reshape(6, 2)
    field = tack.field(dtype=dtype, shape=(6, 2))
    field.from_numpy(values)

    back = interop.vtk_to_field(interop.field_to_vtk(field, name="t"))
    assert back.shape == (6, 2)
    np.testing.assert_array_equal(back.to_numpy(), values)


@needs_vtk
def test_a_struct_of_arrays_is_refused_not_flattened():
    """VTK can store a component per buffer; DLPack describes one.

    Flattening it would hand back a field over a temporary VTK generated:
    writes to it go nowhere and writes to the array are never seen, in
    silence. Refusing is the only honest answer.
    """
    from vtkmodules.vtkCommonCore import vtkSOADataArrayTemplate

    soa = vtkSOADataArrayTemplate["float32"]()
    soa.SetNumberOfComponents(2)
    soa.SetNumberOfTuples(3)
    for t in range(3):
        for c in range(2):
            soa.SetTypedComponent(t, c, float(t * 2 + c))

    with pytest.raises(ValueError, match="SoA|contiguous"):
        interop.vtk_to_field(soa)


@needs_vtk
def test_an_array_with_no_storage_is_refused():
    """An implicit array computes its values; there is nothing to share."""
    from vtkmodules.vtkCommonCore import vtkConstantArray

    array = vtkConstantArray["float32"]()
    array.ConstructBackend(3.0)
    array.SetNumberOfComponents(1)
    array.SetNumberOfTuples(4)

    with pytest.raises(ValueError, match="no memory"):
        interop.vtk_to_field(array)


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
