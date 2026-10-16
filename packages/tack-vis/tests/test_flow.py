"""Particle advection and streamlines, by Viskores' rules.

A linear velocity field is reproduced exactly by every linear shape's
interpolation, so a particle's RK4 path through a mesh is NumPy's RK4 path
through the field itself -- on hexahedra, tetrahedra, wedges, a mixed mesh and
a structured grid alike. Leaving the mesh follows Viskores' SmallStep: bisect
the step to the boundary, then push out by the step still left.
"""

import numpy as np
import pytest
from test_dataset_api import _two_hexes_and_a_pyramid

import tack
import tack.dataset as td

try:
    import vtkmodules.all as vtk

    from tack.interop.vtk import vtk_to_dataset
except ImportError:
    vtk = None


def _vectors(values):
    values = np.asarray(values, np.float32)
    field = tack.empty((len(values),), tack.vec(3, tack.f32))
    tack.copyto(field, values)
    return field


def _rotation(data, center=(0.0, 0.0)):
    x = data.positions()
    v = np.c_[-(x[:, 1] - center[1]), x[:, 0] - center[0], 0.25 * np.ones(len(x))]
    data.fields["v"] = td.Field(td.H1(data), _vectors(v))
    return data


def _numpy_rk4(x, h, steps, center=(0.0, 0.0)):
    def f(p):
        return np.array([-(p[1] - center[1]), p[0] - center[0], 0.25])

    x = np.asarray(x, float)
    for _ in range(steps):
        a = f(x)
        b = f(x + h / 2 * a)
        c = f(x + h / 2 * b)
        d = f(x + h * c)
        x = x + h / 6 * (a + 2 * b + 2 * c + d)
    return x


def _grid():
    axis = np.linspace(-2.0, 2.0, 9)
    return td.rectilinear_grid(axis, axis, np.linspace(-1.0, 2.0, 7))


def _vtk_blocks(cell_type):
    source = vtk.vtkCellTypeSource()
    source.SetCellType(cell_type)
    source.SetBlocksDimensions(4, 4, 4)
    source.Update()
    return vtk_to_dataset(source.GetOutput())


MESHES = {"structured": _grid}
if vtk is not None:
    MESHES.update({name: (lambda t=t: _vtk_blocks(t))
                   for name, t in (("hexahedra", 12), ("tetrahedra", 10), ("wedges", 13),
                                   ("pyramids", 14))})


@pytest.mark.parametrize("mesh", MESHES)
def test_rk4_through_a_linear_field_is_exact(backend, mesh):
    data = MESHES[mesh]()
    center = (0.0, 0.0) if mesh == "structured" else (2.0, 2.0)
    _rotation(data, center)
    seeds = np.array([[center[0] + 1.0, center[1], 0.1], [center[0] + 0.3, center[1] + 0.6, 0.2]])
    out = td.advect(data, "v", seeds, 0.02, 120)
    for i, seed in enumerate(seeds):
        np.testing.assert_allclose(out.positions()[i], _numpy_rk4(seed, 0.02, 120, center),
                                   atol=2e-4)
    np.testing.assert_array_equal(out.fields["steps"].values.numpy(), [120, 120])
    np.testing.assert_array_equal(out.fields["status"].values.numpy(),
                                  td.ParticleStatus.SUCCESS | td.ParticleStatus.TERMINATE | td.ParticleStatus.TOOK_ANY_STEPS)
    np.testing.assert_allclose(out.fields["time"].values.numpy(), 2.4, rtol=1e-5)


def test_a_mixed_mesh(backend):
    data = _rotation(_two_hexes_and_a_pyramid(), center=(1.0, 0.5))
    out = td.advect(data, "v", [[1.2, 0.5, 0.2]], 0.01, 50)
    np.testing.assert_allclose(out.positions()[0], _numpy_rk4([1.2, 0.5, 0.2], 0.01, 50,
                                                              (1.0, 0.5)), atol=1e-4)


def _uniform(direction=(1.0, 0.0, 0.0)):
    axis = np.linspace(0.0, 2.0, 11)
    data = td.rectilinear_grid(axis, axis, axis)
    data.fields["v"] = td.Field(td.H1(data), _vectors(np.tile(direction, (data.num_points, 1))))
    return data


def test_leaving_the_mesh_bisects_then_pushes_out(backend):
    """Four whole steps to x = 1.95; the fifth would leave, so it is bisected to the
    boundary (x = 2, within the locator's tolerance) and the particle pushed out
    by the step still left, as Viskores' SmallStep does."""
    out = td.advect(_uniform(), "v", [[1.55, 1.0, 1.0]], 0.1, 100)
    x = out.positions()[0]
    assert 2.0 < x[0] < 2.0 + 0.051 and x[1] == pytest.approx(1.0)
    assert out.fields["steps"].values.numpy()[0] == 5
    assert out.fields["status"].values.numpy()[0] == (
        td.ParticleStatus.SUCCESS | td.ParticleStatus.SPATIAL_BOUNDS | td.ParticleStatus.TOOK_ANY_STEPS)
    assert out.fields["time"].values.numpy()[0] == pytest.approx(0.45, abs=1e-3)


def test_seeds_outside_and_still_fields(backend):
    data = _uniform()
    out = td.advect(data, "v", [[5.0, 1.0, 1.0]], 0.1, 10)
    assert out.fields["steps"].values.numpy()[0] == 0
    assert out.fields["status"].values.numpy()[0] == td.ParticleStatus.SPATIAL_BOUNDS
    np.testing.assert_array_equal(out.positions()[0], [5.0, 1.0, 1.0])
    still = _uniform((0.0, 0.0, 0.0))
    out = td.advect(still, "v", [[1.0, 1.0, 1.0]], 0.1, 10)
    assert out.fields["steps"].values.numpy()[0] == 1
    assert out.fields["status"].values.numpy()[0] == (
        td.ParticleStatus.SUCCESS | td.ParticleStatus.TERMINATE | td.ParticleStatus.ZERO_VELOCITY | td.ParticleStatus.TOOK_ANY_STEPS)


def test_cell_velocity_and_euler(backend):
    """Cell data is constant in each cell; Euler steps by the velocity where the step
    starts, so it only learns it left at the next step -- as Viskores' does."""
    data = _uniform()
    data.fields["c"] = td.Field(td.Constant(data),
                                _vectors(np.tile([0.0, 0.0, 1.0], (data.num_cells, 1))))
    out = td.advect(data, "c", [[1.0, 1.0, 0.5]], 0.25, 100, integrator="euler")
    np.testing.assert_allclose(out.positions()[0], [1.0, 1.0, 2.25], atol=1e-5)
    assert out.fields["steps"].values.numpy()[0] == 7
    assert out.fields["status"].values.numpy()[0] == td.ParticleStatus.SPATIAL_BOUNDS | td.ParticleStatus.TOOK_ANY_STEPS


def test_streamlines_are_the_paths(backend):
    data = _rotation(_grid())
    seeds = np.array([[1.0, 0.0, 0.0], [0.0, 0.5, 0.0], [5.0, 5.0, 5.0]])
    lines = td.streamlines(data, "v", seeds, 0.05, 30)
    # One polyline per seed that moved; the third, outside, keeps its point.
    assert isinstance(lines.topology, td.PolylineTopology)
    assert lines.num_points == 31 + 31 + 1 and lines.num_cells == 2
    np.testing.assert_array_equal(lines.fields["seed"].values.numpy(), [0, 1])
    x = lines.positions()
    np.testing.assert_array_equal(x[0], seeds[0])
    np.testing.assert_allclose(x[30], _numpy_rk4(seeds[0], 0.05, 30), atol=2e-4)
    np.testing.assert_allclose(lines.fields["time"].values.numpy()[:31],
                               0.05 * np.arange(31), atol=1e-5)
    line_offsets, line_points = (a.numpy() for a in lines.topology.lines())
    np.testing.assert_array_equal(line_offsets, [0, 31, 62])
    np.testing.assert_array_equal(line_points, np.arange(62))
    # Each path's length is its steps', and its loose ends are its first and last.
    lengths = td.cell_geometry(lines).measure.values.numpy()
    np.testing.assert_allclose(lengths, [np.linalg.norm(np.diff(x[:31], axis=0), axis=1).sum(),
                                         np.linalg.norm(np.diff(x[31:62], axis=0), axis=1).sum()],
                               rtol=1e-5)
    ends = td.extract_surface(lines)
    np.testing.assert_array_equal(np.sort(ends.topology.connectivity.numpy()), [0, 30, 31, 61])
    advected = td.advect(data, "v", seeds, 0.05, 30)
    np.testing.assert_allclose(x[[30, 61]], advected.positions()[:2], atol=1e-6)


def test_particles_stay_on_the_device(backend, monkeypatch):
    """Seeds in an array of 3-vectors are copied on the device, the particles stay
    there, and the streamlines' paths are chosen and numbered there: nothing is
    read back but counts. The particles went to the host and back, the seeds
    too, and the paths' choice ran in NumPy. (Debug mode reads each launch's
    bounds check back, so it is off here.)"""
    from tack.lang.array import Array
    from tack.runtime import dispatch

    monkeypatch.setattr(dispatch, "_debug", False)

    data = _rotation(_grid())
    host = np.array([[1.0, 0.0, 0.0], [0.0, 0.5, 0.0], [5.0, 5.0, 5.0]], np.float32)
    seeds = tack.array(host, tack.vec3())
    locator = td.CellLocator(data)
    locator.cell_bins()
    expected = (td.advect(data, "v", host, 0.05, 30, locator=locator).positions(),
                td.streamlines(data, "v", host, 0.05, 30, locator=locator))
    read = []
    numpy = Array.numpy
    monkeypatch.setattr(Array, "numpy", lambda self, *a, **k: (read.append(self.shape),
                                                               numpy(self, *a, **k))[1])
    out = td.advect(data, "v", seeds, 0.05, 30, locator=locator)
    lines = td.streamlines(data, "v", seeds, 0.05, 30, locator=locator)
    assert read == []
    monkeypatch.undo()
    np.testing.assert_array_equal(out.positions(), expected[0])
    np.testing.assert_array_equal(lines.positions(), expected[1].positions())
    for a, b in zip(lines.topology.lines(), expected[1].topology.lines(), strict=True):
        np.testing.assert_array_equal(a.numpy(), b.numpy())
    np.testing.assert_array_equal(lines.fields["seed"].values.numpy(), [0, 1])
    still = td.streamlines(data, "v", tack.array(host[2:], tack.vec3()), 0.05, 30)
    assert (still.num_cells, still.num_points) == (0, 1)           # no step, no line


def test_refusals(backend):
    data = _two_hexes_and_a_pyramid()
    data.fields["s"] = td.Field(td.H1(data), tack.arrays.ConstantArray(1.0, data.num_points))
    with pytest.raises(TypeError, match="3-vectors"):
        td.advect(data, "s", [[0.5, 0.5, 0.5]], 0.1, 3)
    with pytest.raises(ValueError, match="steps"):
        td.advect(_uniform(), "v", [[1.0, 1.0, 1.0]], 0.1, 0)
    polyhedra = _rotation(td.as_polyhedra(_two_hexes_and_a_pyramid()), center=(1.0, 0.5))
    with pytest.raises(NotImplementedError, match="polyhedra"):
        td.advect(polyhedra, "v", [[1.0, 0.5, 0.5]], 0.1, 3)


def test_advection_keeps_the_input_id_type(backend):
    """A pipeline that starts in i64 stays there: advect's and streamlines' outputs
    chose their id type from their own sizes alone."""
    topology = td.StructuredTopology((2, 2, 2), id_dtype=tack.i64)
    data = td.DataSet(topology, tack.arrays.UniformCoordinates((2, 2, 2)))
    data.fields["v"] = td.Field(td.H1(data), tack.arrays.ConstantArray([1.0, 0.0, 0.0], data.num_points))
    for algorithm in (td.advect, td.streamlines):
        assert algorithm(data, "v", [[0.2, 0.5, 0.5]], 0.1, 1).id_dtype is tack.i64
    assert td.probe(data, [[0.5, 0.5, 0.5]]).id_dtype is tack.i64


def test_advection_preserves_f64_positions(f64_backend):
    data = td.uniform_grid((2, 2, 2), dtype=tack.f64)
    data.fields["v"] = td.Field(td.H1(data),
                                 tack.arrays.ConstantArray([0., 0., 0.], data.num_points, dtype=tack.f64))
    seed = np.array([[0.5 + 1e-9, 0.5, 0.5]], np.float64)
    out = td.advect(data, "v", seed, 0.1, 1)
    assert out.dtype is tack.f64
    np.testing.assert_array_equal(out.positions(), seed)


def test_advection_refuses_a_foreign_locator(backend):
    data = _uniform()
    locator = td.CellLocator(_uniform())
    with pytest.raises(ValueError, match="another dataset"):
        td.advect(data, "v", [[1., 1., 1.]], 0.1, 1, locator=locator)
