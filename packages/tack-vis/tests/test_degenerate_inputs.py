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
    data.fields["p"] = td.Field(td.H1(data), tack.field_like(
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
    data.fields["p"] = td.Field(td.H1(data), tack.field_like(np.array([0, 1, 4, 9], np.float32)))
    np.testing.assert_allclose(td.cell_centers(data).values.to_numpy(vectors=True)[:, 0],
                               [0.25, 0.75, 1.25])
    np.testing.assert_allclose(td.to_cells(data, "p").values.to_numpy(), [0.5, 2.5, 6.5])
    assert td.threshold(data, "p", 0.5, 5.0).num_cells == 1     # the line from 1 to 4
