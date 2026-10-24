"""Refreshing ghost values from their owners with a `tack.communication.Plan`.

A filter that needs a point's neighbours is right on a piece's owned points
and wrong on the ghost layer's outer points, whose neighbours the piece lacks.
A second such filter then reads those wrong values. Copying each ghost point's
value from the piece that owns it, between the two, makes the second right
too: here ``project`` of a cell field to points, then point gradients of the
result, over ``split``'s pieces against the whole grid. The plan is built from
positions, as ghost generation (the partitioned proposal's phase 3) will
build it from its matching.
"""

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


def _ghost_plan(blocks, pieces):
    """Each piece's duplicate points, received from the piece owning them."""
    owner = {}
    for g, piece in zip(blocks.local, pieces):
        marks = piece.ghosts.points.numpy()
        for i, key in enumerate(_keys(piece.positions())):
            if not marks[i] & Ghosts.DUPLICATEPOINT:
                owner[key] = (g, i)
    sends, receives = {}, {}
    for g, piece in zip(blocks.local, pieces):
        marks = piece.ghosts.points.numpy()
        for i, key in enumerate(_keys(piece.positions())):
            if marks[i] & Ghosts.DUPLICATEPOINT:
                a, j = owner[key]
                sends.setdefault(a, {}).setdefault(g, []).append(j)
                receives.setdefault(g, {}).setdefault(a, []).append(i)
    return tc.Plan(blocks, sends, receives)


def _owned_gradients(piece):
    return (piece.ghosts.points.numpy() & Ghosts.DUPLICATEPOINT) == 0


@pytest.mark.parametrize("pieces, devices", [(4, "one"), (6, "two")])
def test_a_plan_refreshes_ghost_points_between_filters(cpu, pieces, devices):
    whole = td.wavelet(6)
    whole.fields["c"] = td.project(whole, "RTData", td.Constant(whole))
    whole.fields["p"] = td.project(whole, "c", td.H1(whole))
    expected = td.gradients(whole, "p", at="points").values.numpy()
    where = {key: i for i, key in enumerate(_keys(whole.positions()))}

    parts = list(td.split(whole, pieces, ghost_layers=1))
    if devices == "two":
        second = dispatch.as_device(_SecondCPU())
        parts = [p.to_device(second) if g % 2 else p for g, p in enumerate(parts)]
    blocks = tc.Blocks([p.device for p in parts])
    plan = _ghost_plan(blocks, parts)
    for piece in parts:
        piece.fields["p"] = td.project(piece, "c", td.H1(piece))

    def errors():
        worst = 0.0
        for piece in parts:
            got = td.gradients(piece, "p", at="points").values.numpy()
            owned = _owned_gradients(piece)
            ids = [where[k] for k in _keys(piece.positions())]
            worst = max(worst, np.abs(got[owned] - expected[ids][owned]).max())
        return worst

    scale = np.abs(expected).max()
    assert errors() > 1e-2 * scale         # the ghost layer's outer points are wrong
    plan.run({g: p.fields["p"].values for g, p in zip(blocks.local, parts)})
    assert errors() <= 1e-5 * scale        # 0 on the CPU
