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
    return vtk_to_dataset(vtk.vtkCellTypeSource(cell_type=cell_type, blocks_dimensions=(4, 4, 4))())


MESHES = {"structured": _grid}
if vtk is not None:
    MESHES.update({name: (lambda t=t: _vtk_blocks(t))
                   for name, t in (("hexahedra", 12), ("tetrahedra", 10), ("wedges", 13),
                                   ("pyramids", 14))})
# The same meshes as polyhedra, located in sub-tetrahedra.
MESHES.update({f"{name} as polyhedra": (lambda make=make: td.as_polyhedra(make()))
               for name, make in list(MESHES.items())})


@pytest.mark.parametrize("mesh", MESHES)
def test_rk4_through_a_linear_field_is_exact(backend, mesh):
    data = MESHES[mesh]()
    center = (0.0, 0.0) if mesh.startswith("structured") else (2.0, 2.0)
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


def _graded():
    """Cells growing geometrically from z = 0, a boundary layer's, as tetrahedra."""
    axis = np.linspace(-2.0, 2.0, 9)
    z = np.concatenate(([0.0], 1e-4 * 1.6 ** np.arange(24)))
    return td.tetrahedralize(td.rectilinear_grid(axis, axis, z - 0.5))


@pytest.mark.parametrize("mesh", [*MESHES, "graded tetrahedra"])
def test_trying_the_last_cell_first_changes_nothing(backend, mesh, monkeypatch):
    """Advection tries the cell a particle was last found in before the bins
    (``_MeshSearch.find_from``); the traces are the search's, bit for bit, on
    every shape and on cells graded a millionfold, leaving the mesh included."""
    from tack.dataset.by_id import _MeshSearch

    data = _graded() if mesh == "graded tetrahedra" else MESHES[mesh]()
    center = (0.0, 0.0) if mesh.startswith(("structured", "graded")) else (2.0, 2.0)
    _rotation(data, center)
    seeds = np.array([[center[0] + 1.0, center[1], 0.1], [center[0] + 0.3, center[1] + 0.6, 0.2],
                      [center[0] + 1.9, center[1] + 0.5, -0.45]])
    traces = []
    for previous in (0, 1):
        monkeypatch.setattr(_MeshSearch, "PREVIOUS_CELL", previous)
        out = td.streamlines(data, "v", seeds, 0.03, 200)
        traces.append((out.geometry.values.numpy(), out.fields["time"].values.numpy()))
    np.testing.assert_array_equal(traces[0][0], traces[1][0])
    np.testing.assert_array_equal(traces[0][1], traces[1][1])


def test_a_step_after_a_push_out_starts_afresh(backend, monkeypatch):
    """Rotation inside a disk of cells, particles near its staircased rim: a
    step's stages leave the mesh, the step is bisected, and the particle pushed
    out often lands inside and goes on. Its next whole step must begin at its
    first stage and at the full size: they kept the bisection's last trial
    size and stage, so the next step sampled short of where it went or began
    mid-step on stale stage velocities, and some particles stopped as if
    seeded outside (``SUCCESS`` cleared). Launches of one step, each starting
    afresh, then disagreed with one launch on 20 of these 24 particles."""
    from tack.dataset.algorithms import _advection

    grid = td.uniform_grid((41, 41, 3), origin=(-2.0, -2.0, -0.1), spacing=(0.1, 0.1, 0.1))
    centers = td.cell_centers(grid).values.numpy()
    disk = td.extract_cells(grid, np.flatnonzero(np.hypot(centers[:, 0], centers[:, 1]) < 1.9))
    x = disk.positions()
    disk.fields["v"] = td.Field(td.H1(disk), _vectors(np.c_[-x[:, 1], x[:, 0], 0.0 * x[:, 2]]))
    angles = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    seeds = np.c_[1.84 * np.cos(angles), 1.84 * np.sin(angles), np.zeros(24)]
    monkeypatch.setattr(_advection, "_CHUNKED_ON_CPU", True)
    monkeypatch.setattr(_advection, "_GROWTH", 1)
    runs = []
    for first in (10**9, 1):
        monkeypatch.setattr(_advection, "_FIRST_STEPS", first)
        out = td.advect(disk, "v", seeds, 0.05, 300)
        runs.append([out.fields[f].values.numpy() for f in ("steps", "status", "time")]
                    + [out.positions()])
    for one, few in zip(*runs):
        np.testing.assert_array_equal(one, few)
    status = runs[0][1]
    moved = (status & td.ParticleStatus.TOOK_ANY_STEPS) != 0
    assert moved.sum() > 12 and (status[moved] & td.ParticleStatus.SUCCESS).all()


@pytest.mark.parametrize("budget", [1, 7])
@pytest.mark.parametrize("mesh", [*MESHES, "graded tetrahedra"])
def test_launches_of_a_few_steps_change_nothing(backend, mesh, budget, monkeypatch):
    """On a GPU, advection runs in launches of a bounded number of steps, each
    going on where the last paused (``_advection._launches``): a launch that
    runs too long loses the device (the ONERA M6's streamlines on headroom's
    Level Zero). Launches of 1 and 7 steps give one launch's paths, times,
    statuses and steps, bit for bit, on every shape, RK4 and Euler, and through
    a resumed call."""
    from tack.dataset.algorithms import _advection

    data = _graded() if mesh == "graded tetrahedra" else MESHES[mesh]()
    center = (0.0, 0.0) if mesh.startswith(("structured", "graded")) else (2.0, 2.0)
    _rotation(data, center)
    seeds = np.array([[center[0] + 1.0, center[1], 0.1], [center[0] + 0.3, center[1] + 0.6, 0.2],
                      [center[0] + 1.9, center[1] + 0.5, -0.45], [9.0, 9.0, 9.0]])
    monkeypatch.setattr(_advection, "_CHUNKED_ON_CPU", True)
    monkeypatch.setattr(_advection, "_GROWTH", 1)
    runs = []
    for first in (10**9, budget):                 # one launch, then launches of budget steps
        monkeypatch.setattr(_advection, "_FIRST_STEPS", first)
        out = td.streamlines(data, "v", seeds, 0.03, 200)
        moved = td.advect(data, "v", seeds, 0.03, 60, integrator="euler")
        on = td.advect(data, "v", moved, 0.03, 90)
        runs.append([out.geometry.values.numpy(), out.fields["time"].values.numpy()]
                    + [d.fields[f].values.numpy() for d in (moved, on)
                       for f in ("steps", "time", "status")]
                    + [moved.geometry.values.numpy(), on.geometry.values.numpy()])
    for one, few in zip(*runs):
        np.testing.assert_array_equal(one, few)


@pytest.mark.parametrize("polyhedra", [False, True])
def test_a_mixed_mesh(backend, polyhedra):
    data = _two_hexes_and_a_pyramid()
    data = _rotation(td.as_polyhedra(data) if polyhedra else data, center=(1.0, 0.5))
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


def test_streamlines_by_length_follow_the_direction(backend):
    """``step_unit="length"``: each step an arc length along the velocity's
    direction, as vtkStreamTracer steps by default, with the time it takes
    integrated beside it by the same weights -- NumPy's RK4 of v / |v| and
    1 / |v|. Over pieces, the whole's paths."""
    data = _rotation(_grid())
    seeds = np.array([[1.0, 0.0, 0.0], [0.0, 0.5, 0.0]])
    h, n = 0.05, 30

    def f(p):
        v = np.array([-p[1], p[0], 0.25])
        speed = np.linalg.norm(v)
        return v / speed, 1.0 / speed

    lines = td.streamlines(data, "v", seeds, h, n, step_unit="length")
    x = lines.positions()
    times = lines.fields["time"].values.numpy()
    for s, seed in enumerate(seeds):
        p, t = np.asarray(seed, float), 0.0
        for k in range(n):
            (a, ta), (b, tb) = f(p), f(p + h / 2 * f(p)[0])
            (c, tc) = f(p + h / 2 * b)
            (d, td_) = f(p + h * c)
            p = p + h / 6 * (a + 2 * b + 2 * c + d)
            t += h / 6 * (ta + 2 * tb + 2 * tc + td_)
        np.testing.assert_allclose(x[s * (n + 1) + n], p, atol=2e-4)
        np.testing.assert_allclose(times[s * (n + 1) + n], t, rtol=2e-4)
        chords = np.linalg.norm(np.diff(x[s * (n + 1):(s + 1) * (n + 1)], axis=0), axis=1)
        np.testing.assert_allclose(chords, h, rtol=2e-3)       # a chord of an arc of h
    pieces = td.streamlines(td.split(data, (2, 2, 1), ghost_layers=1), "v", seeds, h, n,
                            step_unit="length")
    traced = np.concatenate([q.positions() for q in pieces if q.num_points])
    for end in (x[n], x[2 * n + 1]):
        assert np.abs(traced - end).max(axis=1).min() < 1e-5
    with pytest.raises(ValueError, match="step_unit"):
        td.streamlines(data, "v", seeds, h, n, step_unit="cells")


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
    chain = td.DataSet(td.PolylineTopology([0, 3], [0, 1, 2]),
                       np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], np.float32))
    chain.fields["v"] = td.Field(td.H1(chain), _vectors(np.tile([1.0, 0.0, 0.0], (3, 1))))
    with pytest.raises(NotImplementedError, match="not polylines"):
        td.advect(chain, "v", [[0.5, 0.0, 0.0]], 0.1, 3)


def test_advection_over_polygons_is_over_their_quads(backend):
    """A quad grid as polygons: a particle in its plane moves as on the quads, the
    velocity linear on each polygon's triangles (exact for this linear field)."""
    quads = _rotation(td.uniform_grid((5, 5)), center=(2.0, 2.0))
    v = quads.fields["v"].values.numpy().copy()
    v[:, 2] = 0.0                                     # stay in the plane
    quads.fields["v"] = td.Field(td.H1(quads), _vectors(v))
    polygons = td.as_polygons(quads)
    seeds = [[3.0, 2.0, 0.0], [2.5, 2.7, 0.0]]
    ours, theirs = (td.advect(d, "v", seeds, 0.05, 40) for d in (polygons, quads))
    np.testing.assert_allclose(ours.positions(), theirs.positions(), atol=1e-5)
    np.testing.assert_array_equal(ours.fields["status"].values.numpy(),
                                  theirs.fields["status"].values.numpy())


# ── Polyhedra ───────────────────────────────────────────────────────

def test_polyhedra_leave_the_mesh_and_take_cell_velocity_as_shapes_do(backend):
    """A grid as polyhedra has the same boundary (its faces are planar), so leaving
    it -- the bisection to the boundary, the push out -- and Euler steps through
    cell data go as on the grid."""
    for direction, seed in (((1.0, 0.0, 0.0), [1.55, 1.0, 1.0]), ((0.6, 0.8, 0.0), [1.0, 1.0, 1.0])):
        grid = _uniform(direction)
        poly = td.as_polyhedra(grid)
        poly.fields["v"] = td.Field(td.H1(poly), grid.fields["v"].values)
        ours, theirs = (td.advect(d, "v", [seed], 0.1, 100) for d in (poly, grid))
        np.testing.assert_allclose(ours.positions(), theirs.positions(), atol=1e-5)
        for name in ("steps", "status"):
            np.testing.assert_array_equal(ours.fields[name].values.numpy(),
                                          theirs.fields[name].values.numpy())
        lines = [td.streamlines(d, "v", [seed], 0.1, 100) for d in (poly, grid)]
        np.testing.assert_allclose(lines[0].positions(), lines[1].positions(), atol=1e-5)
    grid = _uniform()
    poly = td.as_polyhedra(grid)
    poly.fields["c"] = td.Field(td.Constant(poly),
                                _vectors(np.tile([0.0, 0.0, 1.0], (poly.num_cells, 1))))
    out = td.advect(poly, "c", [[1.0, 1.0, 0.5]], 0.25, 100, integrator="euler")
    np.testing.assert_allclose(out.positions()[0], [1.0, 1.0, 2.25], atol=1e-5)
    assert out.fields["steps"].values.numpy()[0] == 7


def test_rk4_through_voronoi_polyhedra_is_exact(backend):
    """Cells of many faces and points, none of them a shape: a linear field is linear
    in every sub-tetrahedron, so RK4 is exact there too."""
    from test_polyhedra import _from_cell_faces, _voronoi_columns

    points, cells, _ = _voronoi_columns(thickness=0.4)
    data = _rotation(td.DataSet(_from_cell_faces(cells, len(points)), points),
                     center=(0.55, 0.5))
    seeds = np.array([[0.75, 0.5, 0.1], [0.6, 0.6, 0.2]])
    out = td.advect(data, "v", seeds, 0.02, 100)
    for i, seed in enumerate(seeds):
        np.testing.assert_allclose(out.positions()[i], _numpy_rk4(seed, 0.02, 100, (0.55, 0.5)),
                                   atol=1e-4)
    np.testing.assert_array_equal(out.fields["steps"].values.numpy(), [100, 100])


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
    with pytest.raises(ValueError, match="another topology or geometry"):
        td.advect(data, "v", [[1., 1., 1.]], 0.1, 1, locator=locator)
