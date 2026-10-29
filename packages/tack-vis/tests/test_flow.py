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


@pytest.mark.parametrize("mesh", [*MESHES, "graded tetrahedra"])
def test_walking_to_the_last_cells_neighbours_changes_nothing(backend, mesh):
    """``neighbors=True`` tries the last cell's face neighbours before the bins
    (``_MeshSearch.find_from``), taking one only where it holds the point well
    inside, as the last cell is taken: the traces, times and particles are the
    search's, bit for bit, on every shape (pyramids and polyhedra search the
    bins as before) and on cells graded a millionfold, leaving the mesh
    included."""
    data = _graded() if mesh == "graded tetrahedra" else MESHES[mesh]()
    center = (0.0, 0.0) if mesh.startswith(("structured", "graded")) else (2.0, 2.0)
    _rotation(data, center)
    seeds = np.array([[center[0] + 1.0, center[1], 0.1], [center[0] + 0.3, center[1] + 0.6, 0.2],
                      [center[0] + 1.9, center[1] + 0.5, -0.45]])
    lines = [td.streamlines(data, "v", seeds, 0.03, 200, neighbors=walk) for walk in (False, True)]
    for name in ("positions", "time"):
        a, b = (out.positions() if name == "positions" else out.fields["time"].values.numpy()
                for out in lines)
        np.testing.assert_array_equal(a, b)
    ends = [td.advect(data, "v", seeds, 0.03, 200, neighbors=walk) for walk in (False, True)]
    np.testing.assert_array_equal(ends[0].positions(), ends[1].positions())
    for name in ("steps", "status", "time"):
        np.testing.assert_array_equal(ends[0].fields[name].values.numpy(),
                                      ends[1].fields[name].values.numpy())


@pytest.mark.parametrize("mesh", ["structured", "tetrahedra", "hexahedra", "wedges", "pyramids"])
def test_the_neighbour_table_holds_each_cells_face_neighbours(backend, mesh):
    """``CellLocator.cell_neighbors``: cell c's neighbour across its local face k at
    ``6 c + k``, -1 on the boundary and past its faces; each pair of cells that
    share a face's points (and only those) are each other's neighbours."""
    if mesh not in MESHES:
        pytest.skip("needs VTK for this mesh")
    data = MESHES[mesh]()
    table = td.CellLocator(data).cell_neighbors().numpy().reshape(-1, 6)
    cells = td.extract_cells(data, np.arange(data.num_cells)).topology    # the same ids
    offsets, members = np.asarray(cells.offsets), np.asarray(cells.connectivity)
    points = [set(members[offsets[c]:offsets[c + 1]].tolist()) for c in range(data.num_cells)]
    faces = data.topology.faces()
    rows = faces.rows.numpy().reshape(-1, 4)
    expected = [set() for _ in range(data.num_cells)]
    for row in rows:
        face = {int(p) for p in row if p >= 0}
        sharing = [c for c in range(data.num_cells) if face <= points[c]]
        assert len(sharing) <= 2
        if len(sharing) == 2:
            expected[sharing[0]].add(sharing[1])
            expected[sharing[1]].add(sharing[0])
    for c in range(data.num_cells):
        assert {int(f) for f in table[c] if f >= 0} == expected[c]
        assert (table[c] >= 0).sum() == len(expected[c])


def test_no_neighbour_table_where_cells_are_not_solids(backend):
    """Surfaces and polyhedra walk nowhere: ``cell_neighbors`` is ``None``, and
    ``neighbors=True`` searches the bins, as before."""
    surface = td.extract_surface(_grid())
    assert td.CellLocator(surface).cell_neighbors() is None
    assert td.CellLocator(td.as_polyhedra(_grid())).cell_neighbors() is None


def test_pieces_walk_to_neighbours_as_the_whole_does(backend):
    """Over pieces, each piece walks its own cells: the paths are the pieces' paths
    without the walk, bit for bit."""
    data = _rotation(_grid())
    pieces = td.split(data, 2, ghost_layers=1)
    seeds = np.array([[1.0, 0.0, 0.1], [0.3, 0.6, 0.2]])
    out = [td.streamlines(pieces, "v", seeds, 0.03, 100, neighbors=walk) for walk in (False, True)]
    for a, b in zip(*out):
        np.testing.assert_array_equal(a.positions(), b.positions())


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


def _path_steps(out):
    """Each path's step lengths (its points' distances), by line."""
    p = out.positions()
    offsets, members = (a.numpy() for a in out.topology.lines())
    return [np.linalg.norm(np.diff(p[members[offsets[k]:offsets[k + 1]]], axis=0), axis=1)
            for k in range(len(offsets) - 1)]


def test_rk45_follows_a_rotation_to_its_tolerance(backend):
    """A rotation rising along z, which trilinear interpolation reproduces exactly:
    every point of an RK45 path (Cash-Karp, as VTK's vtkRungeKutta45) is where the
    exact solution is at its time, to about the tolerance, and the step adapts --
    the steps vary, within [min_step, max_step], and a tighter tolerance takes
    more, smaller steps."""
    data = _rotation(_grid())
    seeds = np.array([[1.0, 0.0, 0.1], [0.3, 0.6, 0.2]])
    counts = {}
    for tolerance in (1e-4, 1e-6):
        out = td.streamlines(data, "v", seeds, 0.1, 400, integrator="rk45",
                             tolerance=tolerance, min_step=0.01, max_step=0.5)
        p = out.positions().astype(float)
        t = out.fields["time"].values.numpy().astype(float)
        offsets, members = (a.numpy() for a in out.topology.lines())
        for k, seed in enumerate(seeds):
            rows = members[offsets[k]:offsets[k + 1] - 1]     # the push out at the top aside
            angle = np.arctan2(seed[1], seed[0]) + t[rows]
            radius = np.hypot(seed[0], seed[1])
            exact = np.c_[radius * np.cos(angle), radius * np.sin(angle),
                          seed[2] + 0.25 * t[rows]]
            np.testing.assert_allclose(p[rows], exact, atol=50 * tolerance + 2e-5)
        counts[tolerance] = [len(steps) for steps in _path_steps(out)]
        for steps, speed in zip(_path_steps(out), (np.hypot(1.0, 0.25), np.hypot(0.67, 0.25))):
            sizes = steps[:-1] / speed            # a step's time, from its distance
            assert sizes.min() >= 0.01 * 0.999 and sizes.max() <= 0.5 * 1.001
    assert all(a < b for a, b in zip(counts[1e-4], counts[1e-6]))



_WALL_GAP = tack.constant(0.0005)


@tack.func
def _back_from_the_wall(x):
    # Toward the wall at x = 1, turning back just inside it.
    return [(1.0 - _WALL_GAP - x[0]) * 50.0, 0.0 * x[1], 0.0 * x[2]]


def test_rk45_ends_at_a_wall_it_cannot_get_past(backend):
    """Where the flow turns back just inside a wall, RK45's stage points swing past it
    at every step size, or its stages' directions cancel: each step's bisection
    pushed the particle out by about an ulp, still inside, and it crawled there
    until its steps ran out (the aneurysm's wall, on every device, mustafar-linux
    2026-10-28: 1920 steps within 1.9e-4; GPUs paid for the whole launch). A push
    out that moves an RK45 particle less than min_step ends it, as
    vtkStreamTracer's RK45 ends such a particle out of its domain."""
    grid = td.uniform_grid((11, 11, 3), spacing=(0.1, 0.1, 0.1))
    grid.fields["v"] = td.interpolate(grid, _back_from_the_wall, td.H1(grid),
                                      dtype=tack.vec3())
    seeds = np.array([[0.5, 0.2, 0.1], [0.7, 0.3, 0.1], [0.9, 0.1, 0.1], [0.95, 0.5, 0.1]])
    for unit in ("length", "time"):
        out = td.streamlines(grid, "v", seeds, 0.02, 400, integrator="rk45", step_unit=unit)
        offsets = tack.arrays.to_host(out.topology.line_offsets)
        assert (np.diff(offsets) < 60).all(), np.diff(offsets)      # 401 each before
        last = out.positions()[offsets[1:] - 1].astype(float)
        assert (last[:, 0] > 0.95).all()                            # at the wall

def test_rk45_in_the_length_unit_keeps_its_step_bounds(backend):
    """In the length unit a step is an arc length: every step lies within
    [min_step, max_step], the path's last step short of it where max_length
    ends the path."""
    data = _rotation(_grid())
    out = td.streamlines(data, "v", np.array([[1.0, 0.0, 0.1]]), 0.05, 500, integrator="rk45",
                         tolerance=1e-6, min_step=0.005, max_step=0.2, step_unit="length",
                         max_length=3.0)
    steps, = _path_steps(out)
    assert steps[:-1].min() >= 0.005 * 0.99 and steps.max() <= 0.2 * 1.001
    assert steps.sum() == pytest.approx(3.0, rel=2e-3)        # chords of 3.0 of arc


@pytest.mark.parametrize("integrator", ["rk4", "euler", "rk45"])
def test_max_length_ends_a_path_where_it_says(backend, integrator):
    """``max_length`` ends a path that has gone that far: in the length unit its
    last step lands on it (no sliver of a step after), in the time unit it ends
    after the step that passes it; the particle is ``TERMINATE``d."""
    data = _rotation(_grid())
    seeds = np.array([[1.0, 0.0, 0.1], [0.3, 0.6, 0.2]])
    out = td.streamlines(data, "v", seeds, 0.05, 1000, integrator=integrator,
                         step_unit="length", max_length=2.0)
    for steps in _path_steps(out):
        assert steps.sum() == pytest.approx(2.0, rel=2e-3)
        assert steps[-1] > 1e-3
    ends = td.advect(data, "v", seeds, 0.05, 1000, integrator=integrator, max_length=2.0)
    assert (ends.fields["status"].values.numpy() & td.ParticleStatus.TERMINATE).all()
    assert (ends.fields["steps"].values.numpy() < 1000).all()


@pytest.mark.parametrize("budget", [1, 7])
def test_rk45_in_launches_of_a_few_steps_changes_nothing(backend, budget, monkeypatch):
    """A particle's step, which RK45 adapts, goes from launch to launch with it:
    launches of 1 and 7 steps give one launch's paths, times and steps, bit for
    bit, max_length and a resumed call included."""
    from tack.dataset.algorithms import _advection

    data = _rotation(_grid())
    seeds = np.array([[1.0, 0.0, 0.1], [0.3, 0.6, 0.2], [1.9, 0.5, -0.45], [9.0, 9.0, 9.0]])
    monkeypatch.setattr(_advection, "_CHUNKED_ON_CPU", True)
    monkeypatch.setattr(_advection, "_GROWTH", 1)
    runs = []
    for first in (10**9, budget):
        monkeypatch.setattr(_advection, "_FIRST_STEPS", first)
        out = td.streamlines(data, "v", seeds, 0.05, 300, integrator="rk45", tolerance=1e-6,
                             step_unit="length", max_length=4.0)
        moved = td.advect(data, "v", seeds, 0.05, 60, integrator="rk45", tolerance=1e-6)
        on = td.advect(data, "v", moved, 0.05, 90, integrator="rk45", tolerance=1e-6)
        runs.append([out.geometry.values.numpy(), out.fields["time"].values.numpy()]
                    + [d.fields[f].values.numpy() for d in (moved, on)
                       for f in ("steps", "time", "status")])
    for one, few in zip(*runs):
        np.testing.assert_array_equal(one, few)


def test_euler_and_rk4_are_unchanged_by_the_adaptive_state(backend):
    """The step and length a particle carries change nothing for Euler and RK4:
    with no max_length their paths are a fixed step's, as before RK45."""
    data = _rotation(_grid())
    seeds = np.array([[1.0, 0.0, 0.1]])
    out = td.advect(data, "v", seeds, 0.02, 120)
    np.testing.assert_allclose(out.positions()[0], _numpy_rk4(seeds[0], 0.02, 120), atol=2e-4)


def test_the_rk45_parameters_are_checked(backend):
    data = _rotation(_grid())
    seeds = np.array([[1.0, 0.0, 0.1]])
    with pytest.raises(ValueError, match="for integrator='rk45'"):
        td.advect(data, "v", seeds, 0.1, 10, tolerance=1e-6)
    with pytest.raises(ValueError, match="min_step <= step_size <= max_step"):
        td.advect(data, "v", seeds, 0.1, 10, integrator="rk45", min_step=0.2)
    with pytest.raises(ValueError, match="'rk45'"):
        td.advect(data, "v", seeds, 0.1, 10, integrator="rk5")
    with pytest.raises(ValueError, match="max_length must be positive"):
        td.advect(data, "v", seeds, 0.1, 10, max_length=0.0)


@pytest.mark.skipif(vtk is None, reason="needs VTK, the reference")
def test_rk45_is_as_close_to_the_path_as_vtks(backend):
    """A swirl whose speed grows with the radius, on an image: Tack's and VTK's
    vtkStreamTracer's RK45 at the same tolerance and step bounds (the length
    unit) take about as many steps and lie as close to the path (a fine RK4's),
    within a few 1e-5 over their first nine tenths."""
    from scipy.spatial import cKDTree

    image = vtk.vtkImageData(dimensions=(41, 41, 3), origin=(-2.0, -2.0, -0.1),
                             spacing=(0.1, 0.1, 0.1))
    x = np.asarray([image.GetPoint(i) for i in range(image.number_of_points)])
    r2 = x[:, 0] ** 2 + x[:, 1] ** 2
    image.point_data["v"] = np.c_[-x[:, 1] * (1 + r2), x[:, 0] * (1 + r2), np.zeros(len(x))]
    data = vtk_to_dataset(image, dtype=tack.f64 if tack.get_default_device()._backend.supports_f64
                          else tack.f32)
    seeds = np.array([[0.5, 0.0, 0.0], [1.2, 0.0, 0.0], [1.7, 0.1, 0.0]])

    def paths(out):
        p = out.positions()
        offsets, members = (a.numpy() for a in out.topology.lines())
        return [p[members[offsets[k]:offsets[k + 1]]] for k in range(len(offsets) - 1)]

    reference = paths(td.streamlines(data, "v", seeds, 0.0005, 20000, step_unit="length",
                                     max_length=4.0))
    ours = paths(td.streamlines(data, "v", seeds, 0.05, 2000, integrator="rk45", tolerance=1e-6,
                                min_step=0.001, max_step=0.1, step_unit="length",
                                max_length=4.0))
    tracer = vtk.vtkStreamTracer(
        integrator_type=vtk.vtkStreamTracer.RUNGE_KUTTA45,
        integration_direction=vtk.vtkStreamTracer.FORWARD, initial_integration_step=0.05,
        minimum_integration_step=0.001, maximum_integration_step=0.1, maximum_error=1e-6,
        maximum_number_of_steps=2000, maximum_propagation=4.0,
        integration_step_unit=vtk.vtkStreamTracer.LENGTH_UNIT)
    tracer.SetInputArrayToProcess(0, 0, 0, 0, "v")
    tracer.SetSourceData(vtk.vtkPolyData(points=vtk.vtkPoints(data=seeds)))
    lines = tracer(image)
    offsets, members = lines.lines.offsets_array, lines.lines.connectivity_array
    theirs = [np.asarray(lines.points.data)[members[offsets[k]:offsets[k + 1]]]
              for k in range(len(offsets) - 1)]
    for mine, vtks, path in zip(ours, theirs, reference):
        t = np.linspace(0.0, 1.0, 20)[:, None, None]
        tree = cKDTree((path[:-1][None] * (1 - t) + path[1:][None] * t).reshape(-1, 3))
        far_mine = tree.query(mine[:len(mine) * 9 // 10])[0].max()
        far_vtk = tree.query(vtks[:len(vtks) * 9 // 10])[0].max()
        assert far_mine < 5e-5 and far_vtk < 5e-5
        assert abs(len(mine) - len(vtks)) <= 0.2 * len(vtks)


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
