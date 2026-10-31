"""Fields that share a name in different spaces (``data.fields``, ``Fields``): a point
field and a cell field called ``"p"``, as VTK keeps them, each found by its space,
carried by filters, chosen with ``fields=``, taken by algorithms as ``(name,
space)``, and sent between pieces under one name."""

import numpy as np
import pytest

import tack
import tack.dataset as td

pytestmark = pytest.mark.quick


def _data():
    """A grid with ``p`` at the points (``x``), at the cells (the cell id), and ``q``."""
    data = td.uniform_grid((4, 3, 3))
    x = data.positions()
    data.fields["p"] = td.Field(td.H1(data), tack.array(x[:, 0].astype(np.float32)))
    data.fields["p", td.Constant] = td.Field(
        td.Constant(data), tack.array(np.arange(data.num_cells, dtype=np.float32)))
    data.fields["q"] = td.Field(td.H1(data), tack.array(x[:, 1].astype(np.float32)))
    return data


def test_each_field_of_a_name_is_found_by_its_space():
    data = _data()
    point, cell = data.fields["p", td.H1], data.fields["p", td.Constant]
    assert isinstance(point.space, td.H1) and isinstance(cell.space, td.Constant)
    assert data.fields["p", td.H1(data)] is point             # the space itself
    assert data.fields["q"] is data.fields["q", td.H1]
    assert data.fields.named("p") == [point, cell]
    assert data.fields.entries() == [("p", point), ("p", cell), ("q", data.fields["q"])]
    assert list(data.fields) == [("p", td.H1(data)), ("p", td.Constant(data)), "q"]
    assert len(data.fields) == 3
    assert "p" in data.fields and ("p", td.Constant) in data.fields
    assert ("p", td.L2) not in data.fields and "r" not in data.fields


def test_a_name_two_fields_share_is_refused_not_guessed():
    data = _data()
    for read in (lambda: data.fields["p"], lambda: data.fields.get("p"),
                 lambda: data.fields.pop("p", None), lambda: data.fields.__delitem__("p")):
        with pytest.raises(KeyError, match="'p' names 2 fields"):
            read()
    with pytest.raises(KeyError, match="'p' names 2 fields"):
        td.gradients(data, "p")
    assert len(data.fields) == 3
    with pytest.raises(KeyError, match="no field"):
        data.fields["p", td.L2]
    assert data.fields.get(("p", td.L2)) is None


def test_putting_a_field_in():
    """By name alone it is the one field of that name, as a dict's item is; by
    ``(name, space)`` it goes beside the others, replacing the one in its space."""
    data = _data()
    again = td.Field(td.Constant(data), tack.zeros((data.num_cells,), tack.f32))
    data.fields["p", td.Constant] = again
    assert data.fields["p", td.Constant] is again and len(data.fields.named("p")) == 2
    with pytest.raises(ValueError, match="is in Constant"):
        data.fields["p", td.H1] = again
    data.fields["p"] = again
    assert data.fields.named("p") == [again] and data.fields["p"] is again
    del data.fields["p"]
    assert "p" not in data.fields
    with pytest.raises(ValueError, match="another topology"):
        data.fields["r"] = td.Field(td.H1(td.uniform_grid((2, 2))), tack.zeros((4,)))


def test_a_dataset_takes_its_fields_keys_back():
    """``dict(data.fields)`` keeps every field, and a dataset made from it has them all;
    two of a name in one space are refused."""
    data = _data()
    copy = td.DataSet(data.topology, data.geometry, fields=dict(data.fields))
    assert copy.fields.entries() == data.fields.entries()
    point = data.fields["p", td.H1]
    with pytest.raises(ValueError, match="two fields 'p' would be in H1"):
        td.DataSet(data.topology, data.geometry, fields={"p": point, ("p", td.H1): point})


def test_filters_carry_every_field_of_a_name():
    data = _data()
    kept = td.threshold(data, "q", 0.5, 2.0)
    assert [(n, type(f.space)) for n, f in kept.fields.entries()] == \
        [("p", td.H1), ("p", td.Constant), ("q", td.H1)]
    cells = td.threshold(data, "q", 0.5, 2.0, fields="p")       # a name: both
    assert [type(f.space) for f in cells.fields.named("p")] == [td.H1, td.Constant]
    assert "q" not in cells.fields
    one = td.threshold(data, "q", 0.5, 2.0, fields=[("p", td.Constant)])
    assert [type(f.space) for _, f in one.fields.entries()] == [td.Constant]
    # The cell field's values are the kept cells' ids.
    ids = np.asarray(td.cell_centers(data).values.numpy())
    assert one.fields["p"].values.numpy().size == kept.num_cells < ids.shape[0]


def test_fields_that_would_meet_in_one_space_are_chosen_not_guessed():
    """A probe puts every field at the target's points: a point and a cell field of
    one name would be two point fields there, so it asks which, as VTK (which keeps
    the point one) does not."""
    data = _data()
    points = np.array([[0.5, 0.5, 0.5], [2.25, 1.5, 1.75]])
    with pytest.raises(ValueError, match=r"choose one with fields=\[\('p', td.H1\)\]"):
        td.probe(data, points)
    probed = td.probe(data, points, fields=[("p", td.Constant), "q"])
    np.testing.assert_array_equal(probed.fields["p"].values.numpy(), [0, 11])
    np.testing.assert_allclose(probed.fields["q"].values.numpy(), points[:, 1])


def test_algorithms_take_a_name_and_a_space():
    data = _data()
    gradient = td.gradients(data, ("p", td.H1)).values.numpy()
    np.testing.assert_allclose(gradient, np.tile([1.0, 0.0, 0.0], (data.num_cells, 1)),
                               atol=1e-6)
    stats = td.statistics(data, ("p", td.Constant))
    assert stats.mean == pytest.approx((data.num_cells - 1) / 2)
    data.fields["p", td.H1] = td.Field(td.H1(data), data.fields["p", td.H1].values, "m")
    converted = td.convert(data, "cm", field=("p", td.H1))
    assert converted.fields["p", td.H1].values.numpy().max() == pytest.approx(300.0)
    assert converted.fields["p", td.Constant].values.numpy().max() == data.num_cells - 1


def test_pieces_exchange_every_field_of_a_name():
    """Ghost cells and statistics over pieces, with two fields of one name: each
    sent and reduced under its own space, structured pieces and unstructured."""
    data = _data()
    structured = td.split(data, 2)
    unstructured = td.PartitionedDataSet(
        [td.extract_cells(piece, np.arange(piece.num_cells)) for piece in structured])
    for pieces in (td.ghost_cells(structured), td.ghost_cells(unstructured)):
        for piece in pieces:
            x = piece.positions()[:, 0]
            np.testing.assert_allclose(piece.fields["p", td.H1].values.numpy(), x, atol=1e-6)
            centers = td.cell_centers(piece).values.numpy()
            # The cell field is the cell's id in the whole grid, from its center.
            i, j, k = (np.floor(centers[:, a]).astype(int) for a in range(3))
            np.testing.assert_array_equal(piece.fields["p", td.Constant].values.numpy(),
                                          i + 3 * (j + 2 * k))
    whole = td.statistics(data, ("p", td.Constant))
    split = td.statistics(structured, ("p", td.Constant))
    assert split.mean == pytest.approx(whole.mean) and split.n == whole.n == data.num_cells
