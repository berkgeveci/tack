"""Faces of a 2D mesh: codimension 1, so its edges -- on the shape path as on the
polygonal one. Their geometry (an edge's in-plane normal out of side 0, its
length as its area), their sides, and the finite-volume algorithms that work
on faces: divergence, Perot's reconstruction, the boundary as lines."""

import numpy as np
import pytest
from test_dataset_api import _scalars

import tack.data as td

try:
    import vtkmodules.all as vtk

    from tack.interop.vtk import vtk_to_dataset
except ImportError:
    vtk = None


def _quads():
    # 4 x 3 cells, unevenly spaced, in the xy plane.
    return td.rectilinear_grid([0.0, 1.0, 2.5, 3.0, 4.5], [0.0, 0.5, 2.0, 3.0])


def _vtk_blocks(cell_type):
    source = vtk.vtkCellTypeSource()
    source.SetCellType(cell_type)
    source.SetBlocksDimensions(3, 2, 0)
    source.Update()
    return vtk_to_dataset(source.GetOutput())


MESHES = {"structured quads": _quads}
if vtk is not None:
    MESHES.update({"triangles": lambda: _vtk_blocks(5), "pixels": lambda: _vtk_blocks(8),
                   "quads": lambda: _vtk_blocks(9),
                   "polygons": lambda: td.as_polygons(_vtk_blocks(9))})


def _edges_of(data):
    """Each face's two point ids, sorted, and its sides, on the host."""
    faces = data.topology.faces()
    if hasattr(faces, "rows"):
        rows = faces.rows.to_numpy(vectors=True)[:, :2]
    else:
        t = data.topology
        offsets, points = t.face_offsets.to_numpy(), t.face_points.to_numpy()
        rows = np.array([points[offsets[f]:offsets[f] + 2] for f in range(faces.num_faces)])
    return rows, faces.sides.to_numpy(vectors=True)


@pytest.mark.parametrize("mesh", MESHES)
def test_a_2d_meshs_faces_are_its_edges(backend, mesh):
    data = MESHES[mesh]()
    assert data.topology.dimension == 2
    rows, sides = _edges_of(data)
    edges = data.topology.edges()
    assert data.topology.faces().num_faces == edges.num_edges
    np.testing.assert_array_equal(sorted(map(tuple, np.sort(rows, axis=1))),
                                  sorted(map(tuple, edges.rows.to_numpy(vectors=True))))
    boundary = td.boundary_faces(data).to_numpy()
    assert ((sides[:, 2] < 0) == np.isin(np.arange(len(sides)), boundary)).all()
    # Every cell's edges: each interior one shared by two cells.
    assert (sides[:, 0] >= 0).all()


@pytest.mark.parametrize("mesh", MESHES)
def test_edge_geometry_and_closed_cells(backend, mesh):
    data = MESHES[mesh]()
    x = data.positions()
    rows, sides = _edges_of(data)
    normals, lengths = td.face_geometry(data)
    normals = normals.values.to_numpy(vectors=True)
    lengths = lengths.values.to_numpy()
    d = x[rows[:, 1]] - x[rows[:, 0]]
    np.testing.assert_allclose(lengths, np.linalg.norm(d, axis=1), rtol=1e-6)
    np.testing.assert_allclose(normals[:, 2], 0, atol=1e-6)            # in the plane
    np.testing.assert_allclose((normals * d).sum(1), 0, atol=1e-6)     # across the edge
    # Out of side 0: away from that cell's centroid.
    _, centroids = td.cell_geometry(data)
    c = centroids.values.to_numpy(vectors=True)[sides[:, 0]]
    mid = 0.5 * (x[rows[:, 0]] + x[rows[:, 1]])
    assert ((normals * (mid - c)).sum(1) > 0).all()
    # A uniform flow's flux sums to zero through every cell's edges.
    velocity = np.array([0.7, -0.4, 0.0])
    flux = td.Field(td.Values(data, "faces", oriented=True), _scalars(normals @ velocity * lengths))
    np.testing.assert_allclose(td.divergence(data, flux).values.to_numpy(), 0, atol=1e-5)
    # Perot's reconstruction gives the flow back in every cell.
    normal_component = td.Field(td.Values(data, "faces", oriented=True), _scalars(normals @ velocity))
    np.testing.assert_allclose(td.perot(data, normal_component).values.to_numpy(vectors=True),
                               np.tile(velocity, (data.num_cells, 1)), atol=1e-5)


@pytest.mark.parametrize("mesh", [m for m in MESHES if m != "polygons"])
def test_the_boundary_of_a_2d_mesh_is_lines(backend, mesh):
    data = MESHES[mesh]()
    data.fields["cell"] = td.Field(td.Constant(data), _scalars(np.arange(data.num_cells)))
    outside = td.external_faces(data)
    assert set(outside.topology.types.to_numpy()) == {3}               # VTK_LINE
    _, lengths = td.face_geometry(data)
    perimeter = lengths.values.to_numpy()[td.boundary_faces(data).to_numpy()].sum()
    x = data.positions()
    span = x.max(0) - x.min(0)
    assert perimeter == pytest.approx(2 * (span[0] + span[1]), rel=1e-6)
    assert outside.fields["cell"].space is td.Constant(outside)


def test_shape_and_polygonal_paths_agree(backend):
    if vtk is None:
        pytest.skip("needs VTK")
    shape = _vtk_blocks(9)
    polygons = td.as_polygons(shape)
    # The same edges, measured the same, matched by their sorted point ids.
    lengths = []
    for data in (shape, polygons):
        rows, _ = _edges_of(data)
        by_edge = dict(zip(map(tuple, np.sort(rows, axis=1).tolist()),
                           td.face_geometry(data).area.values.to_numpy().tolist()))
        lengths.append(by_edge)
    assert lengths[0].keys() == lengths[1].keys()
    np.testing.assert_allclose([lengths[0][k] for k in lengths[0]],
                               [lengths[1][k] for k in lengths[0]], rtol=1e-6)
    np.testing.assert_allclose(td.cell_geometry(shape).measure.values.to_numpy(),
                               td.cell_geometry(polygons).measure.values.to_numpy(), rtol=1e-6)
    with pytest.raises(ValueError, match="not 3D"):
        td.as_polyhedra(shape)


@pytest.mark.parametrize("mesh", sorted(MESHES))
def test_side_1_of_an_edge_is_reflected(backend, mesh):
    """As on a 3D face, a cell goes round its edge backwards exactly when it is the
    edge's side 1. An edge's two points read the same forwards and back from its
    second, so this is the bit that says so (``2 * r + reflected``)."""
    faces = MESHES[mesh]().topology.faces()
    if not hasattr(faces, "side_orientation"):
        pytest.skip("the polygonal path records sides per cell face, not orientations")
    slots = faces.side_slot.to_numpy()
    orientation = faces.side_orientation.to_numpy()
    np.testing.assert_array_equal(orientation & 1, slots)
    np.testing.assert_array_equal(orientation, 3 * slots)


def test_contour_of_l2_data_on_a_2d_mesh(backend):
    """An L2 field or geometry gives each cell its own lines' points: the same line
    as point data where they agree, and gaps kept where the geometry has them.
    (The 2D path took point values only, and merged crossings by face.)"""
    data = td.rectilinear_grid([0.0, 1.0], [0.0, 1.0, 2.0])
    x = td.Field(td.H1(data), _scalars(data.positions()[:, 0]))
    shared = td.contour(data, x, 0.5)
    own = td.contour(data, td.discontinuous(data, x), 0.5)
    assert (shared.num_cells, shared.num_points, own.num_cells, own.num_points) == (2, 3, 2, 4)
    lines = own.positions()[own.topology.connectivity.to_numpy()].reshape(-1, 2, 3)
    np.testing.assert_allclose(lines, [[[0.5, 0, 0], [0.5, 1, 0]], [[0.5, 1, 0], [0.5, 2, 0]]])
    # The second cell lifted to z = 10: its line stays on it.
    geometry = td.discontinuous(data, data.geometry)
    corners = geometry.values.to_numpy(vectors=True)
    corners[4:, 2] += 10.0
    geometry.values.from_numpy(corners)
    data.geometry = geometry
    cut = td.contour(data, x, 0.5)
    lines = cut.positions()[cut.topology.connectivity.to_numpy()].reshape(-1, 2, 3)
    np.testing.assert_allclose(lines[:, :, 2], [[0, 0], [10, 10]])
    sliced = td.slice_plane(data, (0.5, 0.0, 0.0), (1.0, 0.0, 0.0))
    assert sliced.num_cells == 2


def test_contour_of_no_cells(backend):
    """A dataset with no cells (a threshold that kept none) contours to nothing,
    its fields carried; it was refused as a mesh of lines."""
    data = td.rectilinear_grid([0.0, 1.0], [0.0, 1.0], [0.0, 1.0])
    data.fields["s"] = td.Field(td.H1(data), _scalars(data.positions()[:, 0]))
    empty = td.threshold(data, "s", 2.0, 3.0)
    out = td.contour(empty, "s", 0.5)
    assert (out.num_cells, out.num_points, list(out.fields)) == (0, 0, ["s"])


def test_contour_of_varying_l2_orders_on_mixed_2d_shapes(backend):
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0],
                          [2, 0, 0], [3, 0, 0], [3, 1, 0], [2, 1, 0],
                          [4, 0, 0], [5, 0, 0], [4, 1, 0], [5, 1, 0]], np.float32)
    top = td.UnstructuredTopology([5, 9, 8], [0, 3, 7, 11], np.arange(11))
    data = td.DataSet(top, positions)
    data.fields["cell"] = td.cell_ids(data)
    space = td.L2(data, order=np.array([1, 0, 1]))
    field = td.Field(space, _scalars([0., 1., 0., 1., 0., 1., 0., 1.]))
    out = td.contour(data, field, 0.5)
    segments = out.positions()[out.topology.connectivity.to_numpy()].reshape(-1, 2, 3)
    np.testing.assert_allclose(segments, [[[0.5, 0, 0], [0.5, 0.5, 0]],
                                        [[4.5, 0, 0], [4.5, 1, 0]]], atol=1e-6)
    np.testing.assert_array_equal(td.arrays.to_host(out.fields["cell"].values), [0, 2])
    assert np.all(segments[:, 1, 1] > segments[:, 0, 1])  # higher x on the right


def test_contour_cell_values_have_no_2d_level_set(backend):
    data = td.uniform_grid((2, 2))
    field = td.Field(td.Values(data, "cells"), td.ConstantArray(1., data.num_cells))
    assert td.contour(data, field, 0.5).num_cells == 0
    face_field = td.Field(td.Values(data, "faces"), td.ConstantArray(1., 4))
    with pytest.raises(TypeError, match="basis"):
        td.contour(data, face_field, 0.5)
