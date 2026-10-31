"""``tack.set_default_float(tack.f64)`` makes datasets f64 where nothing chooses
their type: the constructors, the sources and the readers, which then give what
``dtype=tack.f64`` gives."""

import numpy as np
import pytest

import tack
import tack.dataset as td

pytestmark = pytest.mark.quick


@pytest.fixture
def f64_default():
    tack.set_default_device("cpu")
    tack.set_default_float(tack.f64)
    yield
    tack.set_default_float(tack.f32)


def _float_type(data):
    return tack.arrays.dtype_of(data.geometry.values)


def test_datasets_follow_it(f64_default):
    assert _float_type(td.uniform_grid((3, 3, 3))) is tack.f64
    assert _float_type(td.rectilinear_grid([0.0, 1.0], [0.0, 2.0])) is tack.f64
    assert _float_type(td.point_cloud(np.random.default_rng(0).random((5, 3)))) is tack.f64
    tets = td.tetrahedralize(td.uniform_grid((3, 3, 3)))
    rebuilt = td.DataSet(tets.topology, tets.positions())
    assert _float_type(rebuilt) is tack.f64
    tangle = td.tangle((8, 8, 8))
    assert tack.arrays.dtype_of(tangle.fields["tangle"].values) is tack.f64
    assert _float_type(td.uniform_grid((3, 3, 3), dtype=tack.f32)) is tack.f32


def test_filters_give_what_an_explicit_f64_gives(f64_default):
    by_default = td.contour(td.tangle((12, 12, 12)), "tangle", [0.2])
    tack.set_default_float(tack.f32)
    explicit = td.contour(td.tangle((12, 12, 12), dtype=tack.f64), "tangle", [0.2])
    assert _float_type(by_default) is tack.f64
    np.testing.assert_array_equal(by_default.positions(), explicit.positions())


def test_the_vtk_reader_follows_it(f64_default):
    pytest.importorskip("vtkmodules.vtkCommonDataModel")
    from vtkmodules.vtkFiltersSources import vtkSphereSource

    from tack.interop.vtk import vtk_to_dataset

    sphere = vtkSphereSource()()
    assert _float_type(vtk_to_dataset(sphere)) is tack.f64
    assert _float_type(vtk_to_dataset(sphere, dtype=tack.f32)) is tack.f32
    assert _float_type(vtk_to_dataset(sphere, dtype=None)) is tack.f32   # VTK's own
