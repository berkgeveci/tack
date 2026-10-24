"""Refreshing ghost values from their owners with a `tack.communication.Plan`.

A filter that needs a point's neighbours is right on a piece's owned points
and wrong on the ghost layer's outer points, whose neighbours the piece lacks.
A second such filter then reads those wrong values. Copying each ghost point's
value from the piece that owns it, between the two, makes the second right
too: here ``project`` of a cell field to points, then point gradients of the
result, over ``split``'s pieces against the whole grid.

The plan is built from positions gathered over the blocks, as ghost
generation (the partitioned proposal's phase 3) will build it from its
matching, so each rank needs only its own pieces: the same code runs in one
process and under MPI (``_ghost_update_mpi.py``).
"""

import os

import numpy as np
import pytest

import tack
import tack.dataset as td
from tack import communication as tc
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


@pytest.fixture
def cpu():
    tack.set_default_device("cpu")
    yield
    tack.set_default_device("cpu")


def _keys(positions):
    return [tuple(p) for p in np.round(positions, 6)]


def _duplicates(piece):
    return (piece.ghosts.points.numpy() & Ghosts.DUPLICATEPOINT) != 0


def ghost_plan(blocks, pieces):
    """A plan giving each of this rank's ``pieces`` its duplicate points' values from
    the pieces owning them, matched by position over every rank's blocks."""
    owned, wanted = {}, {}
    for g, piece in zip(blocks.local, pieces):
        keys, duplicate = _keys(piece.positions()), _duplicates(piece)
        owned[g] = {k: i for i, k in enumerate(keys) if not duplicate[i]}
        wanted[g] = [(i, k) for i, k in enumerate(keys) if duplicate[i]]
    owner = {}
    for a, table in enumerate(blocks.all_gather(owned)):
        for k, j in table.items():
            owner[k] = (a, j)
    requests = blocks.all_gather({g: [k for _, k in wanted[g]] for g in blocks.local})
    receives = {}
    for g in blocks.local:
        for i, k in wanted[g]:
            receives.setdefault(g, {}).setdefault(owner[k][0], []).append(i)
    sends = {}
    for g, keys in enumerate(requests):          # in each requester's order
        for k in keys:
            a, j = owner[k]
            if a in blocks.local:
                sends.setdefault(a, {}).setdefault(g, []).append(j)
    return tc.Plan(blocks, sends, receives)


def ghost_update_errors(world, pieces, devices=None):
    """Point gradients of projected values over this rank's share of ``pieces`` of
    a wavelet, against the whole's: the largest error on owned points before and
    after refreshing the ghosts, and the largest gradient, over every rank."""
    whole = td.wavelet(6)
    whole.fields["c"] = td.project(whole, "RTData", td.Constant(whole))
    whole.fields["p"] = td.project(whole, "c", td.H1(whole))
    expected = td.gradients(whole, "p", at="points").values.numpy()
    where = {key: i for i, key in enumerate(_keys(whole.positions()))}

    every = list(td.split(whole, pieces, ghost_layers=1))
    share = len(every) // world.size
    mine = every[world.rank * share:(world.rank + 1) * share]
    if devices is not None:
        mine = [p.to_device(devices[g % len(devices)]) for g, p in enumerate(mine)]
    blocks = tc.Blocks([p.device for p in mine], world)
    plan = ghost_plan(blocks, mine)
    for piece in mine:
        piece.fields["p"] = td.project(piece, "c", td.H1(piece))

    def error():
        worst = {}
        for g, piece in zip(blocks.local, mine):
            got = td.gradients(piece, "p", at="points").values.numpy()
            owned = ~_duplicates(piece)
            ids = [where[k] for k in _keys(piece.positions())]
            worst[g] = float(np.abs(got[owned] - expected[ids][owned]).max())
        return blocks.all_reduce(worst, "max")

    before = error()
    plan.run({g: p.fields["p"].values for g, p in zip(blocks.local, mine)})
    return before, error(), float(np.abs(expected).max())


def _check(before, after, scale):
    assert before > 1e-2 * scale          # the ghost layer's outer points are wrong
    assert after <= 1e-5 * scale          # 0 on the CPU


@pytest.mark.parametrize("pieces, two_devices", [(4, False), (6, True)])
def test_a_plan_refreshes_ghost_points_between_filters(cpu, pieces, two_devices):
    devices = [tack.device("cpu"), dispatch.as_device(_SecondCPU())] if two_devices else None
    _check(*ghost_update_errors(tc.World(), pieces, devices))


@pytest.mark.parametrize("ranks, pieces", [(2, 4), (3, 6)])
def test_a_plan_refreshes_ghost_points_across_mpi_ranks(mpiexec, ranks, pieces):
    run = mpiexec(ranks, os.path.join(os.path.dirname(__file__), "_ghost_update_mpi.py"),
                  pieces)
    assert run.returncode == 0 and "ok" in run.stdout, run.stdout + run.stderr
