"""Empty and degenerate inputs: grids with sizes of 1 and 0, cells of no volume,
one-dimensional meshes, infinite values.

Survey and decisions: the "Degenerate inputs survey" note (2026-10-13). An
algorithm given one of these either returns a finite, meaningful result or
refuses with a ValueError, TypeError or NotImplementedError naming the reason.
"""

import numpy as np
import pytest

import tack
import tack.data as td

try:
    import vtkmodules.all as vtk

    from tack.interop.vtk import dataset_to_vtk, vtk_to_dataset
except ImportError:
    vtk = None


# ── Grids with sizes of 1 and 0 ────────────────────────────────────


@pytest.mark.parametrize("dims, points, cells, structured", [
    ((1,), 1, 0, False), ((1, 1), 1, 0, False), ((1, 1, 1), 1, 0, False),
    ((0,), 0, 0, False), ((2, 0), 0, 0, False), ((3, 0, 4), 0, 0, False),
    ((3, 1, 1), 3, 2, True), ((1, 3), 3, 2, True), ((1, 3, 1), 3, 2, True),
    ((3, 1, 3), 9, 4, True), ((1, 1, 4), 4, 3, True), ((2, 2, 2), 8, 1, True),
])
def test_a_size_of_one_adds_no_cells_in_any_position(backend, dims, points, cells, structured):
    data = td.uniform_grid(dims)
    assert (data.num_points, data.num_cells) == (points, cells)
    assert isinstance(data.topology, td.StructuredTopology) == structured
    assert data.positions().shape == (points, 3)


def test_grids_need_one_to_three_sizes_of_at_least_zero(backend):
    for dims in ((), (2, 2, 2, 2), (3, -1)):
        with pytest.raises(ValueError, match="1 to 3 sizes of at least 0"):
            td.uniform_grid(dims)


def test_a_line_along_y_has_its_cells_along_y(backend):
    data = td.uniform_grid((1, 3), origin=(1, 2, 3), spacing=(0.5, 2.0, 4.0))
    np.testing.assert_allclose(td.cell_centers(data).values.to_numpy(vectors=True),
                               [[1, 3, 3], [1, 5, 3]])
    lines = td.rectilinear_grid([0.0], [0.0, 1.0, 3.0])
    np.testing.assert_allclose(td.cell_centers(lines).values.to_numpy(vectors=True),
                               [[0, 0.5, 0], [0, 2, 0]])
    assert td.rectilinear_grid([]).num_points == 0
    assert td.sources.wavelet(extent=0).num_points == 1


def _with_points_stored(data):
    """The same grid with its points stored, so nothing reads them by (i, j, k)."""
    copy = td.DataSet(data.topology, data.positions().astype(np.float32))
    for name, field in data.fields.items():
        copy.fields[name] = td.Field(td.H1(copy), field.values)
    return copy


def _host(out):
    """Every array of a result, in its order: datasets' positions and fields."""
    if isinstance(out, td.DataSet):
        return [out.positions()] + [np.asarray(td.arrays.to_host(f.values), float)
                                    for f in out.fields.values()]
    if isinstance(out, tuple):
        return [a for part in out for a in _host(part)]
    return [np.asarray(td.arrays.to_host(out.values), float)]


@pytest.mark.parametrize("dims", [(1, 4, 3), (4, 1, 3), (3, 4, 1)])
def test_a_grid_squeezed_of_a_size_one_axis_is_the_same_grid(backend, dims):
    """Its topology's axes are renumbered; every algorithm agrees with the same
    grid read point by point."""
    data = td.uniform_grid(dims, origin=(0.3, -0.2, 0.1), spacing=(0.5, 0.7, 0.9))
    x = data.positions()
    data.fields["p"] = td.Field(td.H1(data), tack.field_from_numpy(
        (x @ [1.0, 2.0, 3.0]).astype(np.float32)))
    data.fields["v"] = td.Field(td.H1(data), td.ConstantArray([0.1, 0.2, 0.3], data.num_points))
    stored = _with_points_stored(data)
    seed, middle = x[len(x) // 2] + 0.01, (x.min(0) + x.max(0)) / 2
    for algorithm in (td.cell_geometry, td.cell_centers, td.face_geometry, td.external_faces,
                      lambda d: td.gradients(d, "p"),
                      lambda d: td.gradients(d, "p", at="points"),
                      lambda d: td.probe(d, [middle, seed]),
                      lambda d: td.advect(d, "v", [seed], 0.05, 3),
                      lambda d: td.contour(d, "p", float((x @ [1, 2, 3]).mean()))):
        for ours, theirs in zip(_host(algorithm(data)), _host(algorithm(stored)), strict=True):
            np.testing.assert_allclose(ours, theirs, atol=1e-5)


@pytest.mark.skipif(vtk is None, reason="needs VTK")
def test_a_vtk_image_with_a_size_of_one_first(backend):
    image = vtk.vtkImageData()
    image.SetDimensions(1, 4, 3)
    image.SetSpacing(1.0, 0.5, 0.25)
    data = vtk_to_dataset(image)
    assert (data.num_points, data.num_cells) == (12, 6)
    expected = np.array([image.GetPoint(p) for p in range(image.GetNumberOfPoints())])
    np.testing.assert_allclose(data.positions(), expected)
    back = dataset_to_vtk(data)
    assert back.IsA("vtkImageData") and back.GetDimensions() == (1, 4, 3)


# ── Meshes of lines ─────────────────────────────────────────────────


FACE_ALGORITHMS = {
    "face_geometry": td.face_geometry,
    "face_centers": td.face_centers,
    "boundary_faces": td.boundary_faces,
    "external_faces": td.external_faces,
    "traces": lambda d: td.traces(d, "p"),
    "jump": lambda d: td.jump(d, "c"),
    "upwind_flux": lambda d: td.upwind_flux(d, "c", (1.0, 0.0, 0.0)),
}


@pytest.mark.parametrize("name", FACE_ALGORITHMS)
def test_a_mesh_of_lines_has_no_faces_and_says_so(backend, name):
    """A line's faces would be its end points, which are not derived; asking
    gave no faces, so a line's boundary was empty, or an internal error."""
    data = td.uniform_grid((4,))
    data.fields["p"] = td.Field(td.H1(data), td.ConstantArray(1.0, 4))
    data.fields["c"] = td.Field(td.Constant(data), td.ConstantArray(1.0, 3))
    with pytest.raises(NotImplementedError, match="faces of a mesh of lines"):
        FACE_ALGORITHMS[name](data)


def test_a_mesh_of_lines_still_has_cells(backend):
    data = td.uniform_grid((4,), spacing=(0.5, 1.0, 1.0))
    data.fields["p"] = td.Field(td.H1(data), tack.field_from_numpy(np.array([0, 1, 4, 9], np.float32)))
    np.testing.assert_allclose(td.cell_centers(data).values.to_numpy(vectors=True)[:, 0],
                               [0.25, 0.75, 1.25])
    np.testing.assert_allclose(td.to_cells(data, "p").values.to_numpy(), [0.5, 2.5, 6.5])
    assert td.threshold(data, "p", 0.5, 5.0).num_cells == 1     # the line from 1 to 4


# ── Cells of no volume, faces of no area ────────────────────────────


def _collapsed(kind):
    """One hexahedron whose points coincide, or a grid of spacing 0."""
    if kind == "hexahedron":
        top = td.UnstructuredTopology([12], [0, 8], list(range(8)))
        return td.DataSet(top, np.full((8, 3), 2.0, np.float32))
    return td.uniform_grid((3, 3, 3), origin=(2, 2, 2), spacing=(0.0, 0.0, 0.0))


@pytest.mark.parametrize("kind", ["hexahedron", "grid"])
def test_geometry_of_cells_of_no_volume_is_finite(backend, kind):
    data = _collapsed(kind)
    data.fields["c"] = td.Field(td.Constant(data), td.ConstantArray(1.0, data.num_cells))
    measure, centroid = td.cell_geometry(data)
    np.testing.assert_array_equal(measure.values.to_numpy(), 0)
    np.testing.assert_allclose(centroid.values.to_numpy(vectors=True), 2.0)
    np.testing.assert_allclose(td.face_centers(data).values.to_numpy(vectors=True), 2.0)
    normals, areas = td.face_geometry(data)
    np.testing.assert_array_equal(normals.values.to_numpy(), 0)
    np.testing.assert_array_equal(areas.values.to_numpy(), 0)
    flux = td.upwind_flux(data, "c", (1.0, 0.0, 0.0))
    np.testing.assert_array_equal(flux.values.to_numpy(), 0)
    np.testing.assert_array_equal(td.divergence(data, flux).values.to_numpy(), 0)


@pytest.mark.parametrize("kind", ["hexahedron", "grid"])
def test_derivatives_on_cells_of_no_volume_are_nan(backend, kind):
    data = _collapsed(kind)
    data.fields["p"] = td.Field(td.H1(data), td.ConstantArray(1.0, data.num_points))
    data.fields["c"] = td.Field(td.Constant(data), td.ConstantArray(1.0, data.num_cells))
    assert np.isnan(td.gradients(data, "p").values.to_numpy()).all()
    assert np.isnan(td.perot(data, td.upwind_flux(data, "c", (1, 0, 0))).values.to_numpy()).all()


def test_polygons_and_polylines_of_no_extent(backend):
    flat = td.DataSet(td.PolygonalTopology([0, 3], [0, 1, 2]), np.ones((3, 3), np.float32))
    area, centroid = td.cell_geometry(flat)
    assert area.values.to_numpy()[0] == 0
    np.testing.assert_allclose(centroid.values.to_numpy(vectors=True), [[1, 1, 1]])
    line = td.DataSet(td.PolylineTopology([0, 3], [0, 1, 2]), np.ones((3, 3), np.float32))
    length, centroid = td.cell_geometry(line)
    assert length.values.to_numpy()[0] == 0
    np.testing.assert_allclose(centroid.values.to_numpy(vectors=True), [[1, 1, 1]])
    np.testing.assert_array_equal(td.face_geometry(line).normal.values.to_numpy(), 0)


# ── Values that are not finite ──────────────────────────────────────


def _ramp_with(value):
    data = td.uniform_grid((3, 3, 3))
    values = np.linspace(0, 1, 27).astype(np.float32)
    values[0] = value
    data.fields["p"] = td.Field(td.H1(data), tack.field_from_numpy(values))
    data.fields["v"] = td.Field(td.H1(data), td.ConstantArray([1.0, 2.0, 3.0], 27))
    return data


@pytest.mark.parametrize("value", [np.inf, -np.inf, np.nan])
def test_a_crossing_beside_a_value_that_is_not_finite_lies_at_the_other_end(backend, value):
    """The edge weight (iso - a) / (b - a) was NaN with an infinite end, and with
    it the crossing's position and every field carried to it."""
    data = _ramp_with(value)
    surface = td.contour(data, "p", 0.5)
    assert surface.num_cells and np.isfinite(surface.positions()).all()
    np.testing.assert_allclose(surface.fields["v"].values.to_numpy(vectors=True), [[1, 2, 3]] * surface.num_points)
    clipped = td.clip(data, "p", 0.03)
    assert np.isfinite(clipped.positions()).all()
    assert np.isfinite(clipped.fields["v"].values.to_numpy()).all()


def test_histograms_refuse_ranges_that_are_not_finite(backend):
    from tack.algorithms import histogram
    for value in (np.inf, np.nan):
        data = tack.field_from_numpy(np.array([0.0, 1.0, value], np.float32))
        with pytest.raises(ValueError, match="range of .* is not finite"):
            histogram(data)
        with pytest.raises(ValueError, match="is not finite"):
            td.entropy(_ramp_with(value), "p")
    finite = tack.field_from_numpy(np.array([0.0, 1.0], np.float32))
    with pytest.raises(ValueError, match="is not finite"):
        histogram(finite, range=(0.0, np.inf))
    with pytest.raises(ValueError, match="minimum above its maximum"):
        histogram(finite, range=(1.0, 0.0))
    with pytest.raises(ValueError, match="at least 1 bin"):
        histogram(finite, bins=0)


def test_entropy_of_one_value_and_of_none(backend):
    data = td.uniform_grid((3, 3))
    data.fields["one"] = td.Field(td.H1(data), td.ConstantArray(2.0, 9))
    assert str(td.entropy(data, "one")) == "0.0"
    with pytest.raises(ValueError, match="at least 1 bin"):
        td.entropy(data, "one", bins=0)
    empty = td.DataSet(td.UnstructuredTopology([], [0], [], num_points=0), np.empty((0, 3)))
    empty.fields["p"] = td.Field(td.H1(empty), td.ConstantArray(0.0, 0))
    with pytest.raises(ValueError, match="entropy of no values"):
        td.entropy(empty, "p")


# ── Sets ────────────────────────────────────────────────────────────


def test_extract_cells_takes_a_cell_set_empty_or_not(backend):
    data = td.uniform_grid((3, 3, 3))
    data.sets["none"] = td.Set(data, "cells", [])
    data.sets["two"] = td.Set(data, "cells", [1, 5])
    assert td.extract_cells(data, data.sets["none"]).num_cells == 0
    assert td.extract_cells(data, "none").num_cells == 0
    assert td.extract_cells(data, "two").num_cells == 2
    data.sets["corners"] = td.Set(data, "points", [0])
    with pytest.raises(ValueError, match="set of this dataset's cells, not a set of points"):
        td.extract_cells(data, "corners")
    other = td.uniform_grid((3, 3, 3))
    with pytest.raises(ValueError, match="of another topology"):
        td.extract_cells(data, td.Set(other, "cells", [0]))


@pytest.mark.skipif(vtk is None, reason="needs VTK")
@pytest.mark.parametrize("dims", [(1, 1, 1), (0, 0, 0), (1, 5, 4)])
def test_vtk_images_of_one_point_or_none_go_both_ways(backend, dims):
    image = vtk.vtkImageData()
    image.SetDimensions(*dims)
    data = vtk_to_dataset(image)
    assert data.num_points == int(np.prod(dims))
    back = dataset_to_vtk(data)
    assert back.IsA("vtkImageData") and back.GetDimensions() == dims
