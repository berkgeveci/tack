"""Temporal filters, phase T0 of the vault's proposal: each dataset is one step,
at ``time`` in ``time_unit``, kept by every filter as the origin is; pieces,
AMR boxes and octrees at two times are refused; VTK's ``DATA_TIME_STEP`` maps
both ways; and the step cache and time differences the temporal filters share."""

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


#: Filters whose output is not at the input's step: probe gives back its
#: target, a dataset of no time here.
OTHER_TIME = {"probe"}


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
