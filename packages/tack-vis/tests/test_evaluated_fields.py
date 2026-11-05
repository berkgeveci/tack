# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Fields evaluated where they are read, through a toy space: f = x y + z computed at
each point an algorithm asks for, with nothing stored. Each algorithm that reads such
a field is checked against the same function stored in H1 of order 2, which holds
it exactly on these hexahedra."""

import itertools

import numpy as np
import pytest

import tack
import tack.dataset as td
from tack.dataset.spaces import Space

_keys = itertools.count(1)


@tack.func
def _f(x):
    return x[0] * x[1] + x[2]


class _View:
    @tack.func
    def value(self, c, pc):
        return _f(self.dom.position(c, pc))

    @tack.func
    def corner_value(self, c, j):
        return self.value(c, self.dom.parametric_point(j))

    @tack.func
    def value_range(self, c):
        return -3.0e38, 3.0e38

    @tack.func
    def dof(self, c, k):
        return 0.0 * self.dom.parametric_center()[0]


class Analytic(Space):
    """``_f``, evaluated where it is read. ``carried`` says whether a filter's output
    can make it again (``rebind``)."""

    on = "cells"
    interpolated = True
    evaluated = True
    continuous = True
    default_subdivision = 2
    integration_order = 2

    @classmethod
    def _params(cls, key=0):
        return {"key": key}

    @property
    def size(self):
        return self.topology.num_cells

    def mixin_for(self, key, shape=None):
        return _View

    def attributes(self, group=None, dtype=None):
        return {} if group is None else {"dom": self.data.domain_view("cells", group)}

    def rebind(self, field, data):
        return analytic(data, self.carried) if self.carried else None

    def stored(self, field, data):
        return td.interpolate(data, _f, td.H1(data))


def analytic(data, carried=True):
    space = Analytic(data, key=next(_keys))
    space.data, space.carried = data, carried
    return td.Field(space, tack.zeros((data.num_cells,), tack.f32))


def _box():
    data = td.uniform_grid((5, 4, 4), spacing=(0.25, 1 / 3, 1 / 3))
    data.fields["f"] = analytic(data)
    data.fields["exact"] = td.interpolate(data, _f, td.H1(data, 2))
    return data


def test_space_declares_the_contract_with_its_defaults():
    from tack.dataset.spaces import H1, Constant
    data = td.uniform_grid((2, 2, 2))
    stored = td.Field(Constant(data), tack.zeros((data.num_cells,), tack.f32))
    for space in (H1(data), Constant(data)):
        assert not space.evaluated and space.default_subdivision == 1
        assert space.integration_order is None
        assert space.rebind(stored, data) is None
    assert H1(data).continuous and not Constant(data).continuous
    assert Constant(data).stored(stored, data) is stored


def test_contour_reads_it_through_its_view(backend):
    data = _box()
    mine = td.contour(data, "f", 0.6)
    exact = td.contour(data, "exact", 0.6)
    assert mine.num_cells == exact.num_cells
    # Continuous: crossings shared between cells, as H1's are.
    assert mine.num_points == exact.num_points
    assert td.integrate(mine).measure == pytest.approx(td.integrate(exact).measure, rel=1e-5)


def _cells(out):
    return sum(piece.num_cells for piece in out) if isinstance(
        out, td.PartitionedDataSet) else out.num_cells


def test_clip_and_threshold_subdivide_by_its_default(backend):
    data = _box()
    # Its value_range is unbounded, so every cell is cut where H1's bounds keep some
    # whole: the same region, in more pieces.
    mine, exact = (td.clip(data, f, 0.6, exact=False) for f in ("f", "exact"))
    assert _cells(mine) >= _cells(exact)
    assert td.integrate(mine).measure == pytest.approx(td.integrate(exact).measure, rel=1e-5)
    mine, exact = (td.threshold(data, f, 0.0, 0.6) for f in ("f", "exact"))
    assert isinstance(mine, td.PartitionedDataSet)
    assert td.integrate(mine).measure == pytest.approx(td.integrate(exact).measure, rel=1e-5)


def test_filters_make_it_again_on_their_output(backend):
    data = _box()
    data.fields["dropped"] = analytic(data, carried=False)
    x = td.cell_centers(data).values.numpy()[:, 0].astype(np.float32)
    data.fields["x"] = td.Field(td.Constant(data), tack.array(x))
    out = td.threshold(data, "x", 0.0, 0.6)
    assert out.fields["f"].space.evaluated
    assert out.fields["f"].space.data is out
    assert "dropped" not in out.fields


def test_integrate_to_its_order(backend):
    data = _box()
    # Over [0, 1]^3: the integral of x y + z is 1/4 + 1/2.
    assert td.integrate(data, "f").values["f"] == pytest.approx(0.75, rel=1e-6)
    data.fields["g"] = analytic(data)
    data.fields["g"].space.integration_order = None
    with pytest.raises(TypeError, match="integration_order"):
        td.integrate(data, "g")


def test_reductions_read_its_stored_values(backend):
    data = _box()
    points = td.interpolate(data, _f, td.H1(data))
    by_values = td.statistics(data, points)
    mine = td.statistics(data, "f")
    assert mine.n == data.num_points
    assert mine.mean == pytest.approx(by_values.mean)
    assert td.entropy(data, "f") == pytest.approx(td.entropy(data, points))
    np.testing.assert_allclose(td.owned_values(data, "f").numpy(), points.values.numpy())


def test_writers_take_its_stored_values():
    from tack.dataset.dataset import _with_stored_values

    data = _box()
    written = _with_stored_values(data)
    assert not written.fields["f"].space.evaluated
    assert written.fields["exact"] is data.fields["exact"]
    plain = td.uniform_grid((2, 2, 2))
    assert _with_stored_values(plain) is plain
