"""How filters carry fields: one set of rules (``tack.data.carry``) for every
filter that makes a new dataset, and ``fields=`` to choose which come."""

import numpy as np
import pytest
from test_dataset_api import _scalars, _two_hexes_and_a_pyramid

import tack
import tack.data as td


def _ints(values):
    field = tack.field(tack.i32, shape=(len(values),))
    field.from_numpy(np.asarray(values, np.int32))
    return field


def _every_kind():
    """Two hexahedra and a pyramid with a field of every kind."""
    data = _two_hexes_and_a_pyramid()
    x = data.positions()
    t = data.topology
    nf, ne = t.faces().num_faces, t.edges().num_edges
    data.fields["height"] = td.Field(td.H1(data), _scalars(x[:, 2] + 0.1 * x[:, 0]))
    data.fields["label"] = td.Field(td.Values(data, "points"), _ints(np.arange(data.num_points)))
    data.fields["cell"] = td.Field(td.Constant(data), _scalars([1.0, 2.0, 3.0]))
    data.fields["cell id"] = td.Field(td.Values(data, "cells"), _ints([0, 1, 2]))
    data.fields["dg"] = td.algorithms.discontinuous(data, data.fields["height"])
    data.fields["flux"] = td.Field(td.Values(data, "faces", oriented=True),
                                   _scalars(np.arange(nf, dtype=float)))
    data.fields["edge"] = td.Field(td.Values(data, "edges"), _scalars(np.arange(ne, dtype=float)))
    return data


def _kinds(out):
    return {name: (type(f.space).__name__, getattr(f.space, "on", None))
            for name, f in out.fields.items() if name != "shape"}


def _names(out):
    return set(out.fields) - {"shape"}          # the geometry is always there


def test_threshold_keeps_whole_cells_and_all_their_data(backend):
    data = _every_kind()
    out = td.threshold(data, "cell", 1.5, 3.5)
    assert _kinds(out) == {
        "height": ("H1", "points"), "label": ("Values", "points"),
        "cell": ("Constant", "cells"), "cell id": ("Values", "cells"),
        "dg": ("L2", "cells")}          # faces and edges are derived afresh
    np.testing.assert_array_equal(out.fields["cell id"].values.to_numpy(), [1, 2])
    assert out.fields["dg"].space.size == 8 + 5


def test_contour_interpolates_point_data_and_takes_its_cells(backend):
    data = _every_kind()
    out = td.contour(data, "height", 0.5)
    assert out.num_cells
    # Integer point data cannot be interpolated; L2 data belongs to whole cells.
    assert _kinds(out) == {"height": ("H1", "points"), "cell": ("Constant", "cells"),
                           "cell id": ("Values", "cells")}
    np.testing.assert_allclose(out.fields["height"].values.to_numpy(), 0.5, atol=1e-6)


def test_a_surface_takes_face_data_as_its_cells(backend):
    data = _every_kind()
    out = td.external_faces(data)
    kinds = _kinds(out)
    assert kinds["flux"] == ("Values", "cells") and kinds["cell"] == ("Constant", "cells")
    assert kinds["label"] == ("Values", "points") and "dg" not in kinds and "edge" not in kinds


def test_polyhedra_keep_every_entity(backend):
    data = _every_kind()
    out = td.as_polyhedra(data)
    assert _kinds(out)["flux"] == ("Values", "faces") and out.fields["flux"].space.oriented
    assert _kinds(out)["edge"] == ("Values", "edges") and "dg" not in out.fields


@pytest.mark.parametrize("filter_", ["threshold", "contour", "slice", "external faces",
                                     "polyhedra"])
def test_fields_selects_what_comes(backend, filter_):
    data = _every_kind()
    run = {"threshold": lambda f: td.threshold(data, "cell", 0.5, 3.5, fields=f),
           "contour": lambda f: td.contour(data, "height", 0.5, fields=f),
           "slice": lambda f: td.slice_plane(data, (0.5, 0.5, 0.5), (1, 0, 0), fields=f),
           "external faces": lambda f: td.external_faces(data, fields=f),
           "polyhedra": lambda f: td.as_polyhedra(data, fields=f)}[filter_]
    assert _names(run(["height", "cell"])) == {"height", "cell"}
    assert _names(run("cell")) == {"cell"}
    assert _names(run([])) == set()
    with pytest.raises(KeyError, match="no field 'hieght'"):
        run(["hieght"])
