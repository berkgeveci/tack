"""``cell_quality``: Verdict's measures (VTK's ``vtkMeshQuality``) where VTK defines
them, the same definitions extended to every shape, and the cases VTK has no
measure for (order-2 cells, polyhedra, polygons)."""

import numpy as np
import pytest

import tack
import tack.dataset as td
from tack.runtime.dispatch import env_flag

try:
    import vtkmodules.vtkCommonDataModel  # noqa: F401  (registers the pythonic API)
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkUnstructuredGrid
    from vtkmodules.vtkFiltersVerdict import vtkMeshQuality
except ImportError:
    if env_flag("TACK_REQUIRE_VTK"):
        raise
    vtkMeshQuality = None

needs_vtk = pytest.mark.skipif(vtkMeshQuality is None, reason="needs VTK, the reference")

pytestmark = pytest.mark.quick

S2, S3 = np.sqrt(2.0), np.sqrt(3.0)
#: Each linear shape: its VTK type, ideal cell and edges.
SHAPES = {
    "triangle": (5, [[0, 0, 0], [1, 0, 0], [0.5, S3 / 2, 0]], [(0, 1), (1, 2), (2, 0)]),
    "quad": (9, [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
             [(0, 1), (1, 2), (2, 3), (3, 0)]),
    "tetra": (10, [[0, 0, 0], [1, 0, 0], [0.5, S3 / 2, 0], [0.5, S3 / 6, np.sqrt(2 / 3)]],
              [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]),
    "hexahedron": (12, [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1],
                        [1, 1, 1], [0, 1, 1]],
                   [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4),
                    (1, 5), (2, 6), (3, 7)]),
    "wedge": (13, [[0, 0, 0], [1, 0, 0], [0.5, S3 / 2, 0], [0, 0, 1], [1, 0, 1],
                   [0.5, S3 / 2, 1]],
              [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (0, 3), (1, 4), (2, 5)]),
    "pyramid": (14, [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, 1 / S2]],
                [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 4), (2, 4), (3, 4)]),
}
VTK_NAME = {"triangle": "Triangle", "quad": "Quad", "tetra": "Tet", "hexahedron": "Hex",
            "wedge": "Wedge", "pyramid": "Pyramid"}


def _distorted(shape, count=60, spread=0.12, seed=1):
    """``count`` cells of ``shape``, the first ideal, the rest with corners moved
    (surface cells kept flat in every other one, warped in the rest)."""
    rng = np.random.default_rng(seed)
    ideal = np.asarray(SHAPES[shape][1], np.float64)
    cells = [ideal]
    for i in range(1, count):
        noise = rng.normal(0.0, spread, ideal.shape)
        if shape in ("triangle", "quad") and i % 2:
            noise[:, 2] = 0.0
        cells.append(ideal + noise)
    return cells


def _dataset(shape, cells):
    vtk_type = SHAPES[shape][0]
    n = len(cells[0])
    points = np.concatenate(cells)
    topology = td.UnstructuredTopology(np.full(len(cells), vtk_type, np.uint8),
                                       np.arange(0, len(points) + 1, n),
                                       np.arange(len(points)), num_points=len(points))
    return td.DataSet(topology, points, dtype=tack.f64, unit="m")


def _vtk(shape, cells, measure):
    points = np.concatenate(cells)
    n = len(cells[0])
    grid = vtkUnstructuredGrid()
    grid.points = vtkPoints(data=points)
    grid.cells = (SHAPES[shape][0], vtkCellArray(offsets=np.arange(0, len(points) + 1, n),
                                                 connectivity=np.arange(len(points))))
    quality = vtkMeshQuality()
    getattr(quality, f"Set{VTK_NAME[shape]}QualityMeasureTo{measure}")()
    quality.SetInputData(grid)
    quality.Update()
    return np.asarray(quality.GetOutput().cell_data["Quality"])


#: Where VTK 9.7.1 defines the measure as Verdict documents it.
MATCHED = [(s, "scaled_jacobian", "ScaledJacobian") for s in SHAPES] + \
    [(s, "condition", "Condition") for s in ("triangle", "quad", "tetra", "hexahedron")] + \
    [(s, "edge_ratio", "EdgeRatio") for s in ("triangle", "quad", "tetra", "wedge")] + \
    [("hexahedron", "jacobian_ratio", "NodalJacobianRatio")]


@needs_vtk
@pytest.mark.parametrize("shape,measure,vtk_measure", MATCHED)
def test_verdicts_measures(shape, measure, vtk_measure):
    cells = _distorted(shape)
    got = td.cell_quality(_dataset(shape, cells), measure)[measure].values.numpy()
    want = _vtk(shape, cells, vtk_measure)
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-12)
    assert got[0] == pytest.approx(1.0, abs=1e-12)                 # the ideal cell


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_edges_are_the_cells_edges(shape):
    """Longest over shortest of the shape's edges, by brute force. VTK 9.7.1's
    hexahedron edge ratio is sometimes smaller than its longest over shortest edge
    (it leaves an edge out), so it is not the reference here."""
    cells = _distorted(shape)
    quality = td.cell_quality(_dataset(shape, cells))
    lengths = np.array([[np.linalg.norm(c[a] - c[b]) for a, b in SHAPES[shape][2]]
                        for c in cells])
    np.testing.assert_allclose(quality["min_edge"].values.numpy(), lengths.min(1), rtol=1e-12)
    np.testing.assert_allclose(quality["max_edge"].values.numpy(), lengths.max(1), rtol=1e-12)
    np.testing.assert_allclose(quality["edge_ratio"].values.numpy(),
                               lengths.max(1) / lengths.min(1), rtol=1e-12)
    assert quality["min_edge"].unit == "m" and quality["edge_ratio"].unit == "dimensionless"


def test_the_wedge_condition_is_its_corners_against_the_ideal_wedges():
    """The definition the other shapes share (VTK 9.7.1's wedge condition number
    takes another convention): at each corner, |T| |T^-1| / 3 for T = A W^-1, A the
    corner's edges and W the ideal wedge's, the worst corner."""
    cells = _distorted("wedge")
    got = td.cell_quality(_dataset("wedge", cells), "condition")["condition"].values.numpy()
    ideal = np.asarray(SHAPES["wedge"][1])
    neighbours = {0: (1, 2, 3), 1: (2, 0, 4), 2: (0, 1, 5), 3: (5, 4, 0), 4: (3, 5, 1),
                  5: (4, 3, 2)}
    for c, value in zip(cells, got):
        worst = 0.0
        for k, nb in neighbours.items():
            a = (c[list(nb)] - c[k]).T
            w = (ideal[list(nb)] - ideal[k]).T
            t = a @ np.linalg.inv(w)
            worst = max(worst, np.linalg.norm(t) * np.linalg.norm(np.linalg.inv(t)) / 3)
        assert value == pytest.approx(worst, rel=1e-9)


def test_ideal_cells_of_every_shape_score_one():
    grids = [td.uniform_grid((3, 3, 3), dtype=tack.f64),           # voxels
             td.uniform_grid((3, 3), dtype=tack.f64)]              # pixels
    for grid in grids:
        for name, field in td.cell_quality(grid).items():
            np.testing.assert_allclose(field.values.numpy(), 1.0, rtol=1e-12, err_msg=name)


def test_inverted_and_degenerate_cells():
    ideal = np.asarray(SHAPES["tetra"][1], np.float64)
    inverted = ideal[[0, 2, 1, 3]]
    flat = ideal.copy()
    flat[3, 2] = 0.0
    quality = td.cell_quality(_dataset("tetra", [ideal, inverted, flat]))
    sj = quality["scaled_jacobian"].values.numpy()
    assert sj[0] == pytest.approx(1.0) and sj[1] == pytest.approx(-1.0) and sj[2] == 0.0
    big = np.finfo(np.float64).max
    np.testing.assert_array_equal(quality["condition"].values.numpy()[1:], big)


def test_curved_cells_are_measured_along_their_edges():
    """An order-2 geometry: straight, it measures as the linear cell; with a corner
    pulled off the straight edge, the tangents at that corner turn and its scaled
    Jacobian falls, which the cell's corners alone would not show."""
    grid = td.tetrahedralize(td.uniform_grid((2, 2, 2), dtype=tack.f64))
    curved = td.DataSet(grid.topology, td.interpolate(grid, _identity, td.H1(grid, order=2),
                                                      dtype=tack.vec3(tack.f64)),
                        dtype=tack.f64)
    linear = td.cell_quality(grid, "scaled_jacobian")["scaled_jacobian"].values.numpy()
    straight = td.cell_quality(curved, "scaled_jacobian")["scaled_jacobian"].values.numpy()
    np.testing.assert_allclose(straight, linear, rtol=1e-12)
    values = curved.geometry.values.numpy().copy()
    values[grid.num_points:] += [0.15, -0.1, 0.05]                 # bend the edges
    bent = td.DataSet(grid.topology, td.Field(td.H1(grid, order=2),
                                              tack.array(values, tack.vec3(tack.f64))),
                      dtype=tack.f64)
    assert (td.cell_quality(bent, "scaled_jacobian")["scaled_jacobian"].values.numpy()
            < straight - 1e-3).any()


@tack.func
def _identity(x):
    return x


def test_polyhedra_polygons_and_lines():
    cube = td.uniform_grid((3, 3, 3), dtype=tack.f64)
    polyhedra = td.cell_quality(td.as_polyhedra(cube))
    sj = polyhedra["scaled_jacobian"].values.numpy()
    assert (sj > 0.0).all() and (sj < 1.0).all()                   # fan tets, not regular
    np.testing.assert_allclose(sj, sj[0], rtol=1e-9)               # every cube alike
    np.testing.assert_allclose(polyhedra["edge_ratio"].values.numpy(), 1.0, rtol=1e-12)
    flat = td.uniform_grid((3, 3), dtype=tack.f64)
    polygons = td.cell_quality(td.as_polygons(flat))
    assert (polygons["scaled_jacobian"].values.numpy() > 0.0).all()
    np.testing.assert_allclose(polygons["edge_ratio"].values.numpy(), 1.0, rtol=1e-12)
    lines = td.extract_edges(cube)
    quality = td.cell_quality(lines)
    assert np.isnan(quality["scaled_jacobian"].values.numpy()).all()
    np.testing.assert_allclose(quality["min_edge"].values.numpy(), 1.0)


def test_measures_by_name():
    grid = td.uniform_grid((3, 3, 3))
    assert set(td.cell_quality(grid, ["condition", "max_edge"])) == {"condition", "max_edge"}
    with pytest.raises(ValueError, match="no measure 'skew'"):
        td.cell_quality(grid, "skew")
