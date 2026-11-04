# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Fields a dataset derives from itself where it stores none of the name: one name,
read at the location asked for."""

import numpy as np
import pytest

import tack
import tack.dataset as td
from tack.dataset.algorithms import _common


def _grid():
    return td.uniform_grid((4, 3, 3), spacing=(0.5, 0.5, 0.5))


def test_normals_at_each_location(backend):
    box = td.extract_surface(_grid())
    made = td.surface_normals(box)
    np.testing.assert_array_equal(td.named_field(box, "normals", "points").values.numpy(),
                                  made.fields["normals"].values.numpy())
    np.testing.assert_array_equal(td.named_field(box, "normals", "cells").values.numpy(),
                                  made.fields["cell_normals"].values.numpy())
    solid = _grid()
    np.testing.assert_array_equal(td.named_field(solid, "normals", "faces").values.numpy(),
                                  td.face_geometry(solid).normal.values.numpy())
    assert "normals" not in box.fields
    assert td.named_field(box, "normals", "cells") is td.named_field(box, "normals", "cells")
    np.testing.assert_allclose(td.named_field(box, "measures", "cells").values.numpy().sum(), 8.0,
                               rtol=1e-6)


def test_a_field_stored_at_the_location_comes_first(backend):
    box = td.extract_surface(_grid())
    mine = td.Field(td.Values(box, "cells"), td.surface_normals(box).fields["cell_normals"].values)
    box.fields["normals"] = mine
    assert td.named_field(box, "normals", "cells") is mine
    # Stored at the cells only: at the points, the derived ones.
    assert td.named_field(box, "normals", "points") is not mine
    assert td.named_field(box, "normals") is mine


def test_no_location_takes_the_first_the_dataset_has(backend):
    solid = _grid()
    # A solid has no surface normals at its points or cells: its faces'.
    assert td.named_field(solid, "normals").space.on == "faces"


def test_registered_names_and_misses(backend):
    calls = []

    def compute(data):
        calls.append(1)
        return td.Field(td.Values(data, "cells"), tack.array(np.arange(data.num_cells, dtype=np.float32)))

    td.register_derived_field("cell_index", compute, at="cells")
    try:
        box = td.extract_surface(_grid())
        assert "cell_index" in td.derived_names()
        assert td.statistics(box, "cell_index").max == box.num_cells - 1
        td.statistics(box, "cell_index")
        assert len(calls) == 1
        with pytest.raises(KeyError):
            td.named_field(box, "nosuch")
        with pytest.raises(ValueError):
            td.register_derived_field("x", compute, at="edges")
    finally:
        del _common._DERIVED["cell_index"]
