"""Temporal filters, phase T0 of the vault's proposal: each dataset is one step,
at ``time`` in ``time_unit``, kept by every filter as the origin is; pieces,
AMR boxes and octrees at two times are refused; VTK's ``DATA_TIME_STEP`` maps
both ways; and the step cache and time differences the temporal filters share."""

import itertools

import numpy as np
import pytest
from test_dataset_devices import CALLS, _hexes, _SecondCPU

import tack
import tack.dataset as td
from tack.dataset import temporal
from tack.runtime import dispatch
from tack.runtime.dispatch import env_flag

try:
    import vtkmodules.vtkCommonDataModel  # noqa: F401  (registers the pythonic API)
    from vtkmodules.vtkCommonDataModel import vtkDataObject

    from tack.interop.vtk import dataset_to_vtk, vtk_to_dataset
except ImportError:
    if env_flag("TACK_REQUIRE_VTK"):
        raise
    vtkDataObject = None

needs_vtk = pytest.mark.skipif(vtkDataObject is None, reason="needs VTK")

pytestmark = pytest.mark.quick


@pytest.fixture(autouse=True)
def cpu():
    tack.set_default_device("cpu")
    yield
    tack.set_default_device("cpu")


def _stepped(data, time=2.5, time_unit="s"):
    data.time, data.time_unit = time, time_unit
    return data


def test_a_dataset_is_at_a_time():
    data = td.uniform_grid((3, 3, 3), unit="m")
    assert (data.time, data.time_unit) == (None, None)
    data = td.DataSet(data.topology, data.geometry, time=3, time_unit="hr")
    assert (data.time, data.time_unit) == (3.0, "hr")
    assert data._frame() == {"origin": (0.0, 0.0, 0.0), "unit": "m", "time": 3.0,
                             "time_unit": "hr"}
    with pytest.raises(ValueError, match="not a unit of time"):
        data.time_unit = "m"
    with pytest.raises(ValueError, match="finite number"):
        data.time = float("nan")
    data.time = None
    assert data.time is None


#: Filters whose output is not at the input's step: probe and voxelize give
#: back their target, a dataset of no time here.
OTHER_TIME = {"probe", "voxelize"}


@pytest.mark.parametrize("name", sorted(CALLS))
def test_every_algorithm_keeps_the_step(name):
    data = _stepped(_hexes(tack.device("cpu")))
    out = CALLS[name](data)
    if isinstance(out, td.DataSet):
        expected = (None, None) if name in OTHER_TIME else (2.5, "s")
        assert (out.time, out.time_unit) == expected


def test_probe_points_take_the_steps_time_and_devices_keep_it():
    data = _stepped(_hexes(tack.device("cpu")))
    probed = td.probe(data, np.array([[0.5, 0.5, 0.5]]))
    assert (probed.time, probed.time_unit) == (2.5, "s")
    moved = data.to_device(dispatch.as_device(_SecondCPU()))
    assert (moved.time, moved.time_unit) == (2.5, "s")


def test_pieces_hierarchies_and_octrees_are_at_one_time():
    grid = _stepped(td.uniform_grid((5, 4, 3)), 1.0, "s")
    pieces = list(td.split(grid, (2, 1, 1)))
    assert all((p.time, p.time_unit) == (1.0, "s") for p in pieces)
    pieces[1].time, pieces[1].time_unit = 1000.0, "ms"           # the same time
    td.ghost_cells(td.PartitionedDataSet(pieces))
    pieces[1].time = 2000.0
    with pytest.raises(ValueError, match="pieces are at two times"):
        td.ghost_cells(td.PartitionedDataSet(pieces))
    import test_amr

    amr = test_amr.hierarchy()
    levels = [[td.DataSet(b.topology, b.geometry.values, fields=dict(b.fields), dtype=b.dtype,
                          time=float(level)) for b in boxes]
              for level, boxes in enumerate(amr.levels)]
    with pytest.raises(ValueError, match="boxes are at two times"):
        td.AMRDataSet(levels, amr.ratios)
    import test_octrees

    roots, level, centre, rho, temperature = test_octrees.ramses_like(2)
    roots = _stepped(td.uniform_grid(tuple(roots.topology.point_dims)), 7.0, "Myr")
    octree = td.OctreeDataSet.from_cells(roots, level, centre, fields={"T": temperature})
    assert (octree.time, octree.time_unit) == (7.0, "Myr")
    assert octree.leaves().time == 7.0
    assert td.probe(octree, np.array([[0.5, 0.5, 0.0]])).time_unit == "Myr"


@needs_vtk
def test_vtk_data_time_step_both_ways():
    key = vtkDataObject.DATA_TIME_STEP()
    data = _stepped(_hexes(tack.device("cpu")), 12.5, "s")
    grid = dataset_to_vtk(data)
    assert grid.GetInformation().Get(key) == 12.5
    back = vtk_to_dataset(grid)
    assert (back.time, back.time_unit) == (12.5, None)            # VTK has no unit
    pieces = td.PartitionedDataSet(td.split(_stepped(td.uniform_grid((5, 4, 3)), 3.0),
                                            (2, 1, 1)))
    assert all(p.time == 3.0 for p in vtk_to_dataset(dataset_to_vtk(pieces)))
    assert vtk_to_dataset(dataset_to_vtk(td.uniform_grid((2, 2, 2)))).time is None


def test_the_step_cache_copies_what_it_keeps():
    data = _stepped(td.uniform_grid((3, 3, 3), unit="m"), 1.0, "s")
    data.fields["v"] = td.Field(td.H1(data), tack.array(np.ones((27, 3), np.float32),
                                                        tack.vec3()), "m/s")
    kept = temporal._Step(data, ["v"], positions=True)
    shared = temporal._Step(data, ["v"], copy=False)
    data.fields["v"].values.fill(2.0)                 # the simulation's next step
    assert (kept.fields["v"].values.numpy() == 1.0).all()
    assert (shared.fields["v"].values.numpy() == 2.0).all()
    assert kept.fields["v"].unit == "m/s" and kept.geometry.unit == "m"
    assert kept.geometry.values is data.geometry.values           # implicit: kept as is
    assert kept.same_mesh(data) and not kept.same_mesh(td.uniform_grid((3, 3, 3)))
    with pytest.raises(KeyError, match="no field 'p'"):
        temporal._Step(data, ["p"])


def test_time_between_steps():
    grid = td.uniform_grid((2, 2, 2))
    a = td.DataSet(grid.topology, grid.geometry, time=1.0, time_unit="s")
    b = td.DataSet(grid.topology, grid.geometry, time=1500.0, time_unit="ms")
    assert temporal._elapsed(a, b) == pytest.approx(500.0)        # in the later's unit
    assert temporal._elapsed(a, b, "s") == pytest.approx(0.5)
    assert temporal._elapsed(b, a) == pytest.approx(-0.5)
    c = td.DataSet(grid.topology, grid.geometry, time=4.0)
    d = td.DataSet(grid.topology, grid.geometry, time=6.0)
    assert temporal._elapsed(c, d) == 2.0                         # no unit, no conversion
    with pytest.raises(ValueError, match="other's unknown"):
        temporal._elapsed(a, d)
    with pytest.raises(ValueError, match="has no time"):
        temporal._elapsed(grid, a)
    with pytest.raises(ValueError, match="no unit to convert from"):
        temporal._elapsed(c, d, "s")


# -- running statistics and time derivatives (T1) ----------------------------------------

def _steps(count=5, seed=0, vector=False, make=None):
    """A stream of steps of one mesh, a new dataset (and topology) each step, as a
    simulation hands them out, with ``T`` (degrees Celsius) and the values."""
    rng = np.random.default_rng(seed)
    out = []
    for k in range(count):
        data = make() if make else td.uniform_grid((4, 3, 3), unit="m")
        data.time, data.time_unit = 0.25 * k * k, "s"            # irregular steps
        shape = (data.num_points, 3) if vector else (data.num_points,)
        values = rng.random(shape).astype(np.float32)
        data.fields["T"] = td.Field(td.H1(data), tack.array(values, tack.vec3()) if vector
                                    else tack.array(values), "degC")
        out.append((data, values.astype(np.float64)))
    return out


@pytest.mark.parametrize("vector", [False, True])
def test_running_statistics_are_numpys(vector):
    stats = td.RunningStatistics("T")
    seen = []
    for data, values in _steps(vector=vector):
        out = stats.update(data)
        seen.append(values)
        stack = np.array(seen)
        for suffix, expected in (("mean", stack.mean(0)), ("variance", stack.var(0)),
                                 ("min", stack.min(0)), ("max", stack.max(0))):
            np.testing.assert_allclose(out.fields[f"T_{suffix}"].values.numpy(), expected,
                                       rtol=1e-6, atol=1e-7, err_msg=suffix)
    assert stats.count == 5
    assert out.fields["T_mean"].unit == "°C" and out.fields["T_variance"].unit == "K**2"
    assert out.time == data.time and out.fields["T"] is data.fields["T"]
    earlier = out.fields["T_mean"].values.numpy().copy()
    stats.update(_steps(1, seed=9)[0][0]) if not vector else None
    np.testing.assert_array_equal(out.fields["T_mean"].values.numpy(), earlier)  # a copy


def _hex_mesh():
    """A new unstructured topology over the same arrays' values each call."""
    grid = td.tetrahedralize(td.uniform_grid((3, 3, 3), unit="m"))
    t = grid.topology
    topology = td.UnstructuredTopology(tack.array(t.types.numpy() if hasattr(t.types, "numpy")
                                                  else tack.arrays.materialize(t.types).numpy()),
                                       tack.arrays.materialize(t.offsets),
                                       tack.arrays.materialize(t.connectivity),
                                       num_points=t.num_points)
    return td.DataSet(topology, grid.geometry.values, unit="m")


def test_statistics_take_one_mesh_and_refuse_another():
    stats = td.RunningStatistics("T")
    for data, _ in _steps(3, make=_hex_mesh):
        stats.update(data)                       # new topology objects, the same mesh
    other, _ = _steps(1, make=lambda: td.uniform_grid((4, 3, 3), unit="m"))[0]
    with pytest.raises(ValueError, match="remesh it onto the first step's mesh"):
        stats.update(other)
    changed = _hex_mesh()                         # another mesh of the same kind and size
    ids = changed.topology.connectivity.numpy()
    ids[[0, 1]] = ids[[1, 0]]
    reordered = td.DataSet(td.UnstructuredTopology(
        tack.arrays.materialize(changed.topology.types), changed.topology.offsets,
        tack.array(ids), num_points=changed.num_points), changed.geometry.values)
    reordered.fields["T"] = td.Field(td.H1(reordered), tack.zeros((changed.num_points,)))
    with pytest.raises(ValueError, match="remesh"):
        stats.update(reordered)
    with pytest.raises(KeyError, match="no field 'p'"):
        td.RunningStatistics("p").update(other)


def test_time_derivatives_are_backward_differences():
    rate = td.TimeDerivative("T")
    steps = _steps()
    assert rate.update(steps[0][0]) is None                     # no previous step
    for (before, b), (now, v) in itertools.pairwise(steps):
        out = rate.update(now)
        np.testing.assert_allclose(out.fields["T_dt"].values.numpy(),
                                   (v - b) / (now.time - before.time), rtol=1e-6)
    assert out.fields["T_dt"].unit == "K/s"
    late = steps[1][0]
    with pytest.raises(ValueError, match="do not increase"):
        rate.update(late)
    minutes = _steps(2)
    minutes[1][0].time, minutes[1][0].time_unit = 0.25 / 60, "min"   # 0.25 s, in minutes
    other = td.TimeDerivative("T")
    other.update(minutes[0][0])
    got = other.update(minutes[1][0])
    assert got.fields["T_dt"].unit == "K/min"
    np.testing.assert_allclose(got.fields["T_dt"].values.numpy(),
                               (minutes[1][1] - minutes[0][1]) / (0.25 / 60), rtol=1e-5)


def test_the_previous_step_is_a_copy_unless_asked():
    """A simulation overwriting its buffer in place: a copied previous step gives the
    true rate, references the overwritten values (no change)."""
    grid = td.uniform_grid((3, 3, 3))
    buffer = tack.array(np.zeros(27, np.float32))
    copied, shared = td.TimeDerivative("T"), td.TimeDerivative("T", copy=False)
    for k in range(2):
        tack.copyto(buffer, np.full(27, float(k), np.float32))
        step = td.DataSet(grid.topology, grid.geometry,
                          fields={"T": td.Field(td.H1(grid), buffer)}, time=float(k))
        a, b = copied.update(step), shared.update(step)
    np.testing.assert_array_equal(a.fields["T_dt"].values.numpy(), 1.0)
    np.testing.assert_array_equal(b.fields["T_dt"].values.numpy(), 0.0)


def test_pieces_stream_piece_by_piece():
    stats, rate = td.RunningStatistics("T"), td.TimeDerivative("T")
    seen = []
    for data, values in _steps(4):
        pieces = td.split(data, (2, 1, 1))
        out, d = stats.update(pieces), rate.update(pieces)
        seen.append(values)
    assert isinstance(out, td.PartitionedDataSet) and len(out) == 2
    whole = td.RunningStatistics("T")
    for data, _ in _steps(4):
        expected = whole.update(data)
    merged = td.merge_pieces(out)
    np.testing.assert_allclose(merged.fields["T_mean"].values.numpy(),
                               expected.fields["T_mean"].values.numpy(), rtol=1e-6)
    assert isinstance(d, td.PartitionedDataSet)
    with pytest.raises(ValueError, match="brings? .*pieces|2 pieces, then 3"):
        stats.update(td.split(_steps(1)[0][0], (3, 1, 1)))


def test_streaming_on_a_device_without_f64():
    from test_dataset_devices import _NoF64CPU

    device = dispatch.as_device(_NoF64CPU())
    stats, rate = td.RunningStatistics("T"), td.TimeDerivative("T")
    seen = []
    for data, values in _steps(3):
        there = data.to_device(device)
        out, d = stats.update(there), rate.update(there)
        seen.append(values)
    np.testing.assert_allclose(out.fields["T_mean"].values.numpy(), np.mean(seen, 0),
                               rtol=1e-5)
    assert d.fields["T_dt"].values.dtype == tack.f32


# -- advection, paths, pathlines and streaklines (T2) ------------------------------------

def _flow(t, velocity, unit="s", dims=(11, 11, 3), time_unit=None):
    """A step at ``t``: a grid with a velocity field ``velocity(positions, t)``."""
    grid = td.uniform_grid(dims, spacing=(1.0, 1.0, 1.0), unit="m")
    grid.time, grid.time_unit = t, time_unit or unit
    v = velocity(grid.positions().astype(np.float64), t).astype(np.float32)
    grid.fields["v"] = td.Field(td.H1(grid), tack.array(v, tack.vec3()), "m/s")
    return grid


def _along_x(speed):
    def velocity(x, t):
        v = np.zeros_like(x)
        v[:, 0] = speed(t)
        return v
    return velocity


def test_advection_is_exact_for_a_velocity_linear_in_time():
    """v(t) = t along x: a particle moves (t1^2 - t0^2) / 2, which RK4 over a
    velocity linear in time between steps gives exactly, an injected particle
    from its own release time."""
    advection = td.Advection("v", seeds=[[1.0, 5.0, 1.0]])
    advection.inject([[1.0, 4.0, 1.0]], time=0.5)
    for t in (0.0, 1.0, 2.0):
        particles = advection.update(_flow(t, _along_x(lambda t: t)))
    x = particles.world_positions()
    np.testing.assert_allclose(x[:, 0], [1.0 + 2.0, 1.0 + (4.0 - 0.25) / 2], rtol=1e-6)
    np.testing.assert_array_equal(particles.fields["seed"].values.numpy(), [0, 1])
    np.testing.assert_array_equal(particles.fields["release"].values.numpy(), [0.0, 0.5])
    assert particles.time == 2.0 and particles.fields["time"].unit == "s"
    assert (particles.fields["status"].values.numpy() == 17).all()   # SUCCESS, TOOK_ANY_STEPS


def test_advection_matches_advect_in_a_steady_flow():
    def rotation(x, t):
        v = np.zeros_like(x)
        v[:, 0], v[:, 1] = -(x[:, 1] - 5.0), x[:, 0] - 5.0
        return v

    seeds = np.array([[7.0, 5.0, 1.0], [5.0, 3.5, 1.0]])
    advection = td.Advection("v", seeds, step_size=0.05)
    for t in (0.0, 1.0):
        particles = advection.update(_flow(t, rotation))
    steady = td.advect(_flow(0.0, rotation), "v", seeds.astype(np.float32), 0.05, 20)
    np.testing.assert_allclose(particles.world_positions(), steady.world_positions(),
                               atol=2e-5)


def test_particles_leaving_the_mesh_stop_and_outside_seeds_are_marked():
    advection = td.Advection("v", seeds=[[9.0, 5.0, 1.0], [50.0, 5.0, 1.0]])
    for t in (0.0, 1.0, 2.0):
        particles = advection.update(_flow(t, _along_x(lambda t: 1.0)))
    status = particles.fields["status"].values.numpy()
    assert status[0] & td.ParticleStatus.SPATIAL_BOUNDS and status[0] & td.ParticleStatus.TERMINATE
    assert not status[1] & td.ParticleStatus.SUCCESS             # released outside
    assert particles.world_positions()[0, 0] <= 10.0


def test_advection_converts_times_and_refuses_what_it_cannot_integrate():
    """Steps in milliseconds with a velocity in metres per second."""
    advection = td.Advection("v", seeds=[[1.0, 5.0, 1.0]])
    for t in (0.0, 500.0, 1000.0):
        particles = advection.update(_flow(t, _along_x(lambda t: 2.0), time_unit="ms"))
    np.testing.assert_allclose(particles.world_positions()[0, 0], 3.0, rtol=1e-6)
    with pytest.raises(ValueError, match="do not increase"):
        advection.update(_flow(900.0, _along_x(lambda t: 2.0), time_unit="ms"))
    with pytest.raises(ValueError, match="before the last step's time"):
        advection.inject([[1.0, 1.0, 1.0]], time=10.0)
    late = td.Advection("v")
    late.inject([[1.0, 1.0, 1.0]], time=-1.0)
    with pytest.raises(ValueError, match="before the first step's time"):
        late.update(_flow(0.0, _along_x(lambda t: 1.0)))
    with pytest.raises(ValueError, match="'rk4' or 'euler'"):
        td.Advection("v", integrator="rk45")


def test_a_pending_release_waits_for_its_step():
    advection = td.Advection("v")
    advection.update(_flow(0.0, _along_x(lambda t: 1.0)))
    advection.inject([[1.0, 5.0, 1.0]], time=1.5)
    assert advection.update(_flow(1.0, _along_x(lambda t: 1.0))).num_points == 0
    particles = advection.update(_flow(2.0, _along_x(lambda t: 1.0)))
    np.testing.assert_allclose(particles.world_positions()[0, 0], 1.5, rtol=1e-6)


def test_the_mesh_may_change_between_steps():
    """Each step's velocity is probed in its own mesh: a finer grid, then a coarser
    one, of the same uniform flow."""
    advection = td.Advection("v", seeds=[[1.0, 5.0, 1.0]])
    advection.update(_flow(0.0, _along_x(lambda t: 1.0), dims=(21, 21, 3)))
    particles = advection.update(_flow(1.0, _along_x(lambda t: 1.0), dims=(11, 11, 3)))
    np.testing.assert_allclose(particles.world_positions()[0, 0], 2.0, rtol=1e-6)


def test_advection_keeps_a_copy_of_the_previous_step():
    """The simulation overwrites its velocity buffer in place: a copy blends the
    two steps' velocities, references only the last."""
    grid = td.uniform_grid((11, 11, 3), unit="m")
    buffer = tack.array(np.zeros((grid.num_points, 3), np.float32), tack.vec3())
    copied = td.Advection("v", seeds=[[1.0, 5.0, 1.0]])
    shared = td.Advection("v", seeds=[[1.0, 5.0, 1.0]], copy=False)
    for t, speed in ((0.0, 0.0), (1.0, 2.0)):
        values = np.zeros((grid.num_points, 3), np.float32)
        values[:, 0] = speed
        tack.copyto(buffer, values)
        step = td.DataSet(grid.topology, grid.geometry, unit="m", time=t, time_unit="s",
                          fields={"v": td.Field(td.H1(grid), buffer, "m/s")})
        a, b = copied.update(step), shared.update(step)
    np.testing.assert_allclose(a.world_positions()[0, 0], 1.0 + 1.0, rtol=1e-6)  # mean 1 m/s
    np.testing.assert_allclose(b.world_positions()[0, 0], 1.0 + 2.0, rtol=1e-6)


def test_extend_paths_joins_particles_written_step_by_step():
    advection = td.Advection("v", seeds=[[1.0, 5.0, 1.0], [1.0, 2.0, 1.0]])
    written = [advection.update(_flow(t, _along_x(lambda t: 1.0))) for t in (0.0, 1.0, 2.0)]
    paths = None
    for particles in written:
        paths = td.extend_paths(paths, particles)
    assert paths.num_cells == 2 and paths.num_points == 6
    np.testing.assert_array_equal(paths.fields["seed"].values.numpy(), [0, 0, 0, 1, 1, 1])
    np.testing.assert_allclose(paths.world_positions()[:3, 0], [1.0, 2.0, 3.0], rtol=1e-6)
    again = td.extend_paths(paths, written[-1])                    # nothing new
    assert again.num_points == paths.num_points


def test_pathlines_and_streaklines():
    pathlines = td.Pathlines("v", seeds=[[1.0, 5.0, 1.0]])
    first = pathlines.update(_flow(0.0, _along_x(lambda t: 1.0)))
    assert first.num_points == 1 and first.num_cells == 0          # one point, no line yet
    for t in (1.0, 2.0, 3.0):
        paths = pathlines.update(_flow(t, _along_x(lambda t: 1.0)))
    assert paths.num_cells == 1
    np.testing.assert_allclose(paths.world_positions()[:, 0], [1.0, 2.0, 3.0, 4.0], rtol=1e-6)
    streaks = td.Streaklines("v", seeds=[[1.0, 5.0, 1.0], [1.0, 2.0, 1.0]], releases=2)
    for t in (0.0, 1.0, 2.0):
        line = streaks.update(_flow(t, _along_x(lambda t: 1.0)))
    assert line.num_cells == 2 and line.num_points == 10
    first_line = line.fields["source"].values.numpy() == 0
    np.testing.assert_allclose(line.world_positions()[first_line, 0], [1.0, 1.5, 2.0, 2.5, 3.0],
                               rtol=1e-6)          # newest, at the seed, first
    np.testing.assert_allclose(line.fields["release"].values.numpy()[first_line],
                               [2.0, 1.5, 1.0, 0.5, 0.0])


def test_advection_over_pieces_is_the_wholes():
    def rotation(x, t):
        v = np.zeros_like(x)
        v[:, 0], v[:, 1] = -(x[:, 1] - 5.0), (x[:, 0] - 5.0) * (1.0 + 0.2 * t)
        return v

    seeds = np.array([[7.0, 5.0, 1.0], [5.0, 3.5, 1.0], [4.0, 6.0, 1.0]])
    whole, pieces = td.Advection("v", seeds, step_size=0.1), td.Advection("v", seeds,
                                                                           step_size=0.1)
    for t in (0.0, 0.5, 1.0):
        step = _flow(t, rotation)
        a = whole.update(step)
        b = pieces.update(td.split(step, (2, 2, 1)))
    assert isinstance(b, td.PartitionedDataSet)
    np.testing.assert_allclose(b[0].world_positions(), a.world_positions(), atol=1e-5)


def test_advection_on_a_device_without_f64():
    from test_dataset_devices import _NoF64CPU

    device = dispatch.as_device(_NoF64CPU())
    advection = td.Advection("v", seeds=[[1.0, 5.0, 1.0]])
    for t in (0.0, 1.0):
        particles = advection.update(_flow(t, _along_x(lambda t: 1.0)).to_device(device))
    assert particles.device == device
    np.testing.assert_allclose(particles.world_positions()[0, 0], 2.0, rtol=1e-6)


def _swirl(x, t):
    v = np.zeros_like(x)
    v[:, 0], v[:, 1] = -(x[:, 1] - 5.0), (x[:, 0] - 5.0) * (1.0 + 0.2 * t)
    return v


def advect_across(world):
    """This rank's particles (its own seeds) advected through steps whose pieces are
    shared out over ``world``'s ranks, and the same seeds through the whole steps."""
    seeds = np.array([[7.0, 5.0, 1.0], [5.0, 3.5, 1.0], [4.0, 6.0, 1.0],
                      [6.0, 6.5, 1.0]])[world.rank::world.size]
    shared, whole = td.Advection("v", seeds, step_size=0.1), td.Advection("v", seeds,
                                                                           step_size=0.1)
    for t in (0.0, 0.5, 1.0):
        step = _flow(t, _swirl)
        pieces = list(td.split(step, (2, 2, 1)))[world.rank::world.size]
        got = shared.update(td.PartitionedDataSet(pieces, world=world))
        want = whole.update(step)
    return got[0], want


def test_advection_across_mpi_ranks(mpiexec):
    import os

    script = os.path.join(os.path.dirname(__file__), "_temporal_mpi.py")
    for ranks in (2, 3):
        run = mpiexec(ranks, script)
        assert run.returncode == 0 and "ok" in run.stdout, run.stdout + run.stderr
