"""``probe`` of a partitioned source.

Each probed point goes to the lowest piece one of whose own cells holds it
(``tack.communication.Routing``), is evaluated there, and its values come
back. No ghost cells are needed: a point is evaluated in one cell. The values
equal probing the whole grid, points on faces pieces share included (the
lower piece, as one dataset takes its lower cell), and points outside every
piece are ``valid`` 0. Across MPI ranks each rank probes its own points over
every rank's pieces (``_probe_pieces_mpi.py``).
"""

import os

import numpy as np
import pytest

import tack
import tack.dataset as td
from tack.dataset import Ghosts
from tack.runtime import dispatch
from tack.runtime.cpu import CPUBackend

pytestmark = pytest.mark.quick


class _SecondCPU(CPUBackend):
    """A CPU backend posing as another device's: its arrays are not the default's."""

    display_name = "second device"

    @property
    def memory_domain(self):
        return ("cpu", 1)


@pytest.fixture(autouse=True)
def cpu():
    tack.set_default_device("cpu")
    yield
    tack.set_default_device("cpu")


def source():
    whole = td.wavelet(5)
    whole.fields["c"] = td.project(whole, "RTData", td.Constant(whole))
    whole.fields["v"] = td.Field(td.H1(whole), tack.array(
        whole.positions().astype(np.float32), tack.vec3()))
    return whole


def points(whole):
    """Random points, grid points (on the faces pieces share), and two outside."""
    rng = np.random.default_rng(0)
    return np.concatenate([rng.uniform(-5.5, 5.5, (600, 3)), whole.positions()[::13],
                           [[0.0, 0.0, 0.0], [9.0, 9.0, 9.0], [-5.0, -5.0, -7.0]]]
                          ).astype(np.float32)


def _same(got, expected, names=("RTData", "c", "v", "valid")):
    for name in names:
        np.testing.assert_array_equal(got.fields[name].values.numpy(),
                                      expected.fields[name].values.numpy(), err_msg=name)


@pytest.mark.parametrize("layers", [0, 1])
def test_probe_of_pieces_is_the_wholes(layers):
    whole = source()
    pts = points(whole)
    expected = td.probe(whole, pts)
    got = td.probe(td.split(whole, (2, 2, 2), ghost_layers=layers), pts)
    _same(got, expected)
    valid = got.fields["valid"].values.numpy()
    assert valid[-2] == 0 and valid[-1] == 0 and valid.sum() > 0


def test_a_partitioned_target_gives_partitioned_results():
    whole = source()
    targets = td.split(td.uniform_grid((9, 7, 5), origin=(-4.0, -3.0, -2.0)), 3)
    got = td.probe(td.split(whole, 4, ghost_layers=1), targets, fields=["RTData"])
    assert isinstance(got, td.PartitionedDataSet) and len(got) == len(targets)
    for out, target in zip(got, targets):
        expected = td.probe(whole, target, fields=["RTData"])
        _same(out, expected, ("RTData", "valid"))
        assert out.ghosts is target.ghosts                      # the target's marks kept


def test_pieces_on_two_devices_and_hidden_cells():
    whole = source()
    pts = points(whole)
    second = dispatch.as_device(_SecondCPU())
    pieces = [p.to_device(second) if g % 2 else p for g, p in
              enumerate(td.split(whole, (2, 2, 2), ghost_layers=1))]
    _same(td.probe(td.PartitionedDataSet(pieces), pts), td.probe(whole, pts))
    # A hidden cell holds nothing: its points are probed as if it were not there.
    marks = np.zeros(whole.num_cells, np.uint8)
    marks[::7] = Ghosts.HIDDENCELL
    whole.ghosts = Ghosts(cells=marks)
    expected = td.probe(whole, pts)
    _same(td.probe(td.split(whole, (2, 2, 2), ghost_layers=1), pts), expected)


def test_probe_of_pieces_refuses_a_locator():
    whole = source()
    with pytest.raises(ValueError, match="own locator"):
        td.probe(td.split(whole, 2), points(whole), locator=td.CellLocator(whole))


def test_probe_of_pieces_across_mpi_ranks(mpiexec):
    script = os.path.join(os.path.dirname(__file__), "_probe_pieces_mpi.py")
    for ranks in (2, 3):
        run = mpiexec(ranks, script)
        assert run.returncode == 0 and "ok" in run.stdout, run.stdout + run.stderr


def probed_across(world):
    """This rank's share of the points probed over its share of eight pieces:
    ``(rows, {field: values})``, rows into ``points``."""
    whole = source()
    pts = points(whole)
    every = list(td.split(whole, (2, 2, 2), ghost_layers=1))
    mine = np.array_split(np.arange(len(every)), world.size)[world.rank]
    rows = np.array_split(np.arange(len(pts)), world.size)[world.rank]
    out = td.probe(td.PartitionedDataSet([every[i] for i in mine], world), pts[rows])
    return rows.tolist(), {n: out.fields[n].values.numpy().tolist()
                           for n in ("RTData", "c", "v", "valid")}
