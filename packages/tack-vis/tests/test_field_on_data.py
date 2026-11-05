# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A field argument that is an object whose on(data) gives the Field: one argument
for a field in each piece of a PartitionedDataSet."""

import numpy as np
import pytest

import tack
import tack.dataset as td


class Scaled:
    """A stand-in for a field expression: ``factor`` times the field ``name``."""

    def __init__(self, name, factor):
        self.name, self.factor, self.calls = name, factor, 0

    def on(self, data):
        self.calls += 1
        field = data.fields[self.name]
        return td.Field(field.space, tack.array(field.values.numpy() * self.factor), field.unit)


def _piece(lo):
    data = td.uniform_grid((4, 3, 3), spacing=(0.5, 0.5, 0.5), origin=(lo, 0.0, 0.0))
    x = td.cell_centers(data).values.numpy()[:, 0].astype(np.float32)
    data.fields["x"] = td.Field(td.Constant(data), tack.array(x))
    return data


def test_an_object_is_read_on_the_dataset(backend):
    data = _piece(0.0)
    twice = Scaled("x", 2.0)
    by_name = td.statistics(data, "x")
    assert td.statistics(data, twice).max == pytest.approx(2 * by_name.max)
    assert td.integrate(data, {"x2": twice}).values["x2"] == pytest.approx(
        2 * td.integrate(data, "x").values["x"])
    assert td.threshold(data, twice, 0.0, 1.0).num_cells == td.threshold(
        data, "x", 0.0, 0.5).num_cells
    with pytest.raises(TypeError, match="on\\(data\\)"):
        td.statistics(data, object())


def test_an_object_reduces_over_pieces_once_a_piece(backend):
    pieces = td.PartitionedDataSet([_piece(0.0), _piece(1.5)])
    whole = td.statistics(pieces, "x")
    twice = Scaled("x", 2.0)
    assert td.statistics(pieces, twice).mean == pytest.approx(2 * whole.mean)
    assert twice.calls == len(pieces)
    twice.calls = 0
    assert td.entropy(pieces, twice) == pytest.approx(td.entropy(pieces, "x"))
    assert twice.calls == len(pieces)
    twice.calls = 0
    out = td.integrate(pieces, {"x2": twice})
    assert out.values["x2"] == pytest.approx(2 * td.integrate(pieces, "x").values["x"])
    assert twice.calls == len(pieces)
    with pytest.raises(KeyError):
        td.statistics(pieces, Scaled("nosuch", 1.0))
