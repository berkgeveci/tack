"""``merge_points``: coincident points made one, against VTK's
``vtkStaticCleanUnstructuredGrid``, on meshes written without merging their points
(each cell's corners its own points, as the 512 ``eta`` pieces of berk's
collection store each point about 2.6 times)."""

import numpy as np
import pytest

import tack
import tack.dataset as td
from tack.runtime.dispatch import env_flag

try:
    import vtkmodules.vtkCommonDataModel  # noqa: F401  (registers the pythonic API)
    from vtkmodules.vtkFiltersCore import vtkStaticCleanUnstructuredGrid

    from tack.interop.vtk import dataset_to_vtk
except ImportError:
    if env_flag("TACK_REQUIRE_VTK"):
        raise
    vtkStaticCleanUnstructuredGrid = None

needs_vtk = pytest.mark.skipif(vtkStaticCleanUnstructuredGrid is None,
                               reason="needs VTK, the reference")

pytestmark = pytest.mark.quick


@pytest.fixture(autouse=True)
def on_each_f64_backend(f64_backend):
    """The meshes are in f64, so on every backend that has f64 -- the CPU, and
    CUDA, HIP or Level Zero where present -- whatever the run's default device.
    Metal has none; a device without f64 is checked by test_dataset_devices."""
    yield f64_backend
    tack.set_default_device("cpu")


def _soup(mesh, seed=0, jitter=0.0, unit=None):
    """``mesh``'s cells (of one shape, unstructured), each with its own copies of
    its corners, the points in a shuffled order, each copy moved by up to
    ``jitter`` along every axis."""
    t = mesh.topology
    types = tack.arrays.materialize(t.types).numpy()
    offsets = tack.arrays.materialize(t.offsets).numpy()
    corners = tack.arrays.materialize(t.connectivity).numpy()
    rng = np.random.default_rng(seed)
    positions = mesh.positions().astype(np.float64)[corners]
    positions += rng.uniform(-jitter, jitter, positions.shape)
    order = rng.permutation(len(positions))                 # point i is copy order[i]
    where = np.empty_like(order)
    where[order] = np.arange(len(order))
    topology = td.UnstructuredTopology(types, offsets, where, num_points=len(positions))
    return td.DataSet(topology, positions[order], dtype=tack.f64, unit=unit)


def _hexes(n=(4, 3, 3)):
    """An unstructured mesh of hexahedra on a lattice of ``n`` points, in VTK's
    corner order."""
    nx, ny, nz = n
    z, y, x = np.meshgrid(*(np.arange(m, dtype=np.float64) for m in (nz, ny, nx)),
                          indexing="ij")
    positions = np.stack([0.5 * x.ravel(), 0.25 * y.ravel(), 0.75 * z.ravel()], axis=1)
    k, j, i = (a.ravel() for a in np.meshgrid(np.arange(nz - 1), np.arange(ny - 1),
                                              np.arange(nx - 1), indexing="ij"))

    def point(di, dj, dk):
        return (i + di) + nx * ((j + dj) + ny * (k + dk))

    corners = np.stack([point(0, 0, 0), point(1, 0, 0), point(1, 1, 0), point(0, 1, 0),
                        point(0, 0, 1), point(1, 0, 1), point(1, 1, 1), point(0, 1, 1)],
                       axis=1).ravel()
    cells = len(i)
    topology = td.UnstructuredTopology(np.full(cells, 12, np.uint8), np.arange(cells + 1) * 8,
                                       corners, num_points=len(positions))
    return td.DataSet(topology, positions, dtype=tack.f64)


def _vtk_clean(data, tolerance=0.0):
    f = vtkStaticCleanUnstructuredGrid()
    if tolerance:
        f.SetToleranceIsAbsolute(True)
        f.SetAbsoluteTolerance(tolerance)
    else:
        f.SetTolerance(0.0)
    f.SetInputData(dataset_to_vtk(data))
    f.Update()
    out = f.GetOutput()
    points = np.asarray(out.GetPoints().GetData(), np.float64)
    connectivity = np.asarray(out.cells.connectivity_array)
    return points, connectivity


def _corner_positions(points, connectivity):
    return points[connectivity]


@needs_vtk
@pytest.mark.parametrize("tolerance, jitter", [(0.0, 0.0), (1e-6, 2e-7)])
def test_unmerged_cells_merge_as_vtks_static_clean(tolerance, jitter):
    """Each cell's own copies of its corners, shuffled (and moved by less than the
    tolerance): as many points as VTK's static clean leaves, every cell's
    corners at the same places, and the mesh's own point count."""
    mesh = _hexes()
    soup = _soup(mesh, jitter=jitter)
    ours = td.merge_points(soup, tolerance)
    points, connectivity = _vtk_clean(soup, tolerance)
    assert ours.num_points == len(points) == mesh.num_points
    ours_conn = tack.arrays.materialize(ours.topology.connectivity).numpy()
    np.testing.assert_array_equal(
        _corner_positions(ours.positions().astype(np.float64), ours_conn),
        _corner_positions(points, connectivity))


def test_a_point_keeps_its_smallest_copy_and_the_order_of_those():
    """A set becomes its smallest point, which keeps its position, point fields,
    ghost marks and global id; merged points are in the order of those ids, and
    cell fields are unchanged."""
    soup = _soup(_hexes((3, 2, 2)), seed=3)
    n = soup.num_points
    soup.fields["p"] = td.Field(td.H1(soup), tack.array(np.arange(n, dtype=np.float64)))
    soup.fields["c"] = td.Field(td.Constant(soup),
                                tack.array(10.0 * np.arange(soup.num_cells)))
    marks = np.zeros(n, np.uint8)
    marks[::5] = td.Ghosts.DUPLICATEPOINT
    soup.ghosts = td.Ghosts(np.zeros(soup.num_cells, np.uint8), marks)
    soup.global_ids = td.GlobalIds(np.arange(soup.num_cells, dtype=np.int64),
                                   np.arange(n, dtype=np.int64) + 100)
    out = td.merge_points(soup)
    x = soup.positions()
    _, first, label = np.unique(x, axis=0, return_index=True, return_inverse=True)
    kept = np.sort(first)                                  # each set's smallest point
    np.testing.assert_array_equal(out.positions(), x[kept])
    np.testing.assert_array_equal(out.fields["p"].values.numpy(), kept.astype(np.float64))
    np.testing.assert_array_equal(out.fields["c"].values.numpy(),
                                  soup.fields["c"].values.numpy())
    np.testing.assert_array_equal(out.ghosts.points.numpy(), marks[kept])
    np.testing.assert_array_equal(out.global_ids.points.numpy(), kept + 100)
    conn = tack.arrays.materialize(out.topology.connectivity).numpy()
    old = tack.arrays.materialize(soup.topology.connectivity).numpy()
    np.testing.assert_array_equal(x[kept][conn], x[old])


def test_chains_merge_and_zero_signs_are_one_point():
    """Points within the tolerance of each other join, and so do chains of them:
    0, 0.6 and 1.2 (times 1e-6) are one point at a tolerance of 1e-6, though the
    ends are 1.2e-6 apart. At zero tolerance only equal coordinates merge, -0
    and 0 alike."""
    x = np.array([[0.0, 0, 0], [0.6e-6, 0, 0], [1.2e-6, 0, 0], [1.0, 0, 0],
                  [-0.0, 1, 0], [0.0, 1, 0]])
    cloud = td.point_cloud(x, dtype=tack.f64)
    assert td.merge_points(cloud, 1e-6).num_points == 3
    assert td.merge_points(cloud, 0.5e-6).num_points == 5
    assert td.merge_points(cloud).num_points == 5


def test_quantities_hidden_cells_and_refusals():
    """The tolerance may be a quantity in the dataset's unit; hidden cells are
    dropped first; structured grids, polyhedra and a negative tolerance are
    refused."""
    soup = _soup(_hexes((3, 2, 2)), jitter=1e-7, unit="m")
    assert td.merge_points(soup, "1 um").num_points == 12
    assert td.merge_points(soup, "1 nm").num_points == soup.num_points
    hidden = np.zeros(soup.num_cells, np.uint8)
    hidden[0] = td.Ghosts.HIDDENCELL
    soup.ghosts = td.Ghosts(hidden, np.zeros(soup.num_points, np.uint8))
    out = td.merge_points(soup, 1e-6)
    assert out.num_cells == soup.num_cells - 1 and out.num_points == 8
    with pytest.raises(TypeError, match="unstructured"):
        td.merge_points(td.uniform_grid((3, 3, 3)))
    with pytest.raises(NotImplementedError, match="not supported yet"):
        td.merge_points(td.as_polyhedra(_hexes((3, 2, 2))))
    with pytest.raises(ValueError, match="at least zero"):
        td.merge_points(soup, -1.0)


def test_pieces_written_without_merging_number_as_the_whole():
    """The case it is for: pieces each storing a point once per cell, as the 512
    ``eta`` pieces do. Merged piece by piece, their points match exactly across
    the pieces, and global ids number the whole mesh's points."""
    mesh = _hexes((5, 4, 3))
    cells = np.arange(mesh.num_cells)
    pieces = []
    for part in np.array_split(cells, 3):
        piece = td.extract_cells(mesh, tack.array(part.astype(np.int64)))
        pieces.append(td.merge_points(_soup(piece, seed=len(pieces))))
    numbered = td.generate_global_ids(td.PartitionedDataSet(pieces))
    ids = max(int(p.global_ids.points.numpy().max()) for p in numbered) + 1
    assert ids == mesh.num_points
