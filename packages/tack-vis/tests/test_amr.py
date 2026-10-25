"""Block-structured AMR (``AMRDataSet``): the hierarchy, its marks, and what reads
them.

The marks are held to VTK's ``vtkAMRUtilities::BlankCells`` and its Gaussian
pulse source, cell for cell. The same hierarchy as images, as rectilinear
grids and as curvilinear grids gives the same marks, since they are computed
in index space. Reductions count each region of space and each location once,
checked against the hierarchy's geometry by brute force. The same checks run
with boxes on a simulated second device and over 2 and 3 MPI ranks
(``_amr_mpi.py``).
"""

import os

import numpy as np
import pytest

import tack
import tack.dataset as td
from tack.dataset import Ghosts
from tack.runtime import dispatch
from tack.runtime.cpu import CPUBackend
from tack.runtime.dispatch import env_flag

try:
    import vtkmodules.vtkCommonDataModel  # noqa: F401  (registers the pythonic API)
    from vtkmodules.vtkCommonDataModel import vtkAMRUtilities
    from vtkmodules.vtkFiltersAMR import vtkAMRGaussianPulseSource

    from tack.interop.vtk import dataset_to_vtk, vtk_to_dataset
except ImportError:
    if env_flag("TACK_REQUIRE_VTK"):
        raise
    vtkAMRGaussianPulseSource = None

needs_vtk = pytest.mark.skipif(vtkAMRGaussianPulseSource is None, reason="needs VTK")
pytestmark = pytest.mark.quick


class _SecondCPU(CPUBackend):
    """A CPU backend posing as another device's: its arrays are not the default's."""

    display_name = "second device"

    @property
    def memory_domain(self):
        return ("cpu", 1)


@pytest.fixture(autouse=True)
def cpu():
    tack.set_default_device("cpu")
    yield
    tack.set_default_device("cpu")


def image(lo, cells, spacing, origin=(0.0, 0.0, 0.0), values=None):
    """An image box of ``cells`` cells whose first cell is ``lo`` in its level, on the
    lattice of ``spacing`` from ``origin``; a cell field ``c`` of its centres' x
    and a point field ``p`` of its points' y."""
    d = len(cells)
    corner = np.asarray(origin[:d]) + np.asarray(lo) * spacing
    grid = td.uniform_grid(tuple(n + 1 for n in cells), origin=tuple(corner),
                           spacing=(spacing,) * d)
    topology = td.StructuredTopology(tuple(n + 1 for n in cells), offset=tuple(lo))
    box = td.DataSet(topology, grid.geometry.values)
    x = box.positions()
    box.fields["p"] = td.Field(td.H1(box), tack.array(x[:, 1].astype(np.float32)))
    centres = td.cell_centers(box).values.numpy().reshape(-1, 3)
    box.fields["c"] = td.Field(td.Constant(box), tack.array(centres[:, 0].astype(np.float32)))
    return box


def hierarchy(kind="image", d=2):
    """Three levels in 2D (or 3D): a root of two abutting boxes, three level-1 boxes
    (two abutting, one across the root boxes' seam), and a level-2 box; as
    images, or the same boxes as rectilinear or curvilinear grids."""
    if d == 2:
        levels = [[((0, 0), (4, 8)), ((4, 0), (4, 8))],
                  [((2, 2), (4, 4)), ((6, 2), (2, 4)), ((2, 10), (8, 4))],
                  [((6, 6), (4, 4))]]
    else:
        levels = [[((0, 0, 0), (4, 4, 4)), ((4, 0, 0), (4, 4, 4))],
                  [((2, 2, 2), (4, 4, 2)), ((6, 2, 2), (4, 4, 2))],
                  [((6, 6, 6), (4, 2, 2))]]
    spacings = [1.0, 0.5, 0.25]
    boxes = [[image(lo, cells, h) for lo, cells in level] for level, h in zip(levels, spacings)]
    if kind == "image":
        return td.AMRDataSet(boxes, ratios=[2, 2])
    return td.AMRDataSet([[_as(kind, box) for box in level] for level in boxes], ratios=[2, 2])


def _as(kind, box):
    """``box`` with its geometry as a rectilinear or curvilinear grid's, the same
    points, fields kept."""
    coordinates = box.geometry.values
    if kind == "rectilinear":
        dims, origin, spacing = coordinates.dims, coordinates._origin, coordinates._spacing
        axes = [origin[a] + spacing[a] * np.arange(dims[a]) for a in range(3)]
        geometry = tack.arrays.CartesianProduct(*axes)
    else:
        geometry = box.positions()
    out = td.DataSet(box.topology, geometry)
    for name, field in box.fields.items():
        space = td.H1(out) if isinstance(field.space, td.H1) else td.Constant(out)
        out.fields[name] = td.Field(space, field.values)
    return out


def marks(amr):
    return [(box.ghosts.cells.numpy(), box.ghosts.points.numpy()) for _, box in amr]


# -- the marks -------------------------------------------------------------------------

@needs_vtk
@pytest.mark.parametrize("dimension", [2, 3])
def test_the_gaussian_pulse_is_vtks(dimension):
    source = vtkAMRGaussianPulseSource(dimension=dimension)
    source.Update()
    theirs = vtk_to_dataset(source.GetOutput())
    ours = td.amr_gaussian_pulse(dimension=dimension)
    assert theirs.ratios == ours.ratios and theirs.num_levels == ours.num_levels == 2
    for level in range(2):
        np.testing.assert_array_equal(theirs.boxes(level), ours.boxes(level))
    for (_, a), (_, b) in zip(theirs, ours):
        np.testing.assert_array_equal(a.world_positions(), b.world_positions())
        np.testing.assert_array_equal(a.ghosts.cells.numpy(), b.ghosts.cells.numpy())
        for name in ("Gaussian-Pulse", "Centroid"):
            np.testing.assert_array_equal(a.fields[name].values.numpy(),
                                          b.fields[name].values.numpy())
    # VTK's own blanking of the source: the refined cells, by VTK's REFINEDCELL.
    vtk_marks = np.asarray(source.GetOutput().GetDataSetAsCartesianGrid(0, 0)
                           .cell_data["vtkGhostType"])
    np.testing.assert_array_equal(vtk_marks, ours.levels[0][0].ghosts.cells.numpy())


@needs_vtk
@pytest.mark.parametrize("d", [2, 3])
@pytest.mark.parametrize("kind", ["image", "rectilinear"])
def test_vtk_blanks_the_export_as_tack_does(kind, d):
    amr = hierarchy(kind, d)
    out = dataset_to_vtk(amr)
    assert out.class_name == "vtkOverlappingAMR"
    blocks = [(level, k) for level in range(out.GetNumberOfLevels())
              for k in range(out.GetNumberOfBlocks(level))]
    for level, k in blocks:
        out.GetDataSetAsCartesianGrid(level, k).cell_data.RemoveArray("vtkGhostType")
    vtkAMRUtilities.BlankCells(out)
    for (level, k), (_, box) in zip(blocks, amr):
        got = np.asarray(out.GetDataSetAsCartesianGrid(level, k).cell_data["vtkGhostType"])
        np.testing.assert_array_equal(got, box.ghosts.cells.numpy() & Ghosts.REFINEDCELL)
    back = vtk_to_dataset(dataset_to_vtk(amr))
    assert back.ratios == amr.ratios
    for (_, a), (_, b) in zip(back, amr):
        np.testing.assert_allclose(a.world_positions(), b.world_positions(), atol=1e-6)
        np.testing.assert_array_equal(a.ghosts.cells.numpy(), b.ghosts.cells.numpy())


@needs_vtk
def test_a_curvilinear_hierarchy_goes_to_vtk_as_levels():
    out = dataset_to_vtk(hierarchy("curvilinear"))
    assert out.class_name == "vtkPartitionedDataSetCollection"
    assert out.number_of_partitioned_data_sets == 3


@pytest.mark.parametrize("d", [2, 3])
def test_every_kind_of_box_is_marked_alike(d):
    expected = marks(hierarchy("image", d))
    for kind in ("rectilinear", "curvilinear"):
        for (a, b), (c, e) in zip(marks(hierarchy(kind, d)), expected):
            np.testing.assert_array_equal(a, c)
            np.testing.assert_array_equal(b, e)


def _world_points(amr):
    """Every box's points as world positions, with whether each is skipped."""
    out = []
    for _, box in amr:
        skipped = (box.ghosts.points.numpy() & Ghosts.SKIPPED_POINTS) != 0
        out.append((np.round(box.world_positions(), 9), skipped))
    return out


@pytest.mark.parametrize("d", [2, 3])
def test_each_location_and_region_counts_once(d):
    """Every location of every box's points is owned once: at the finest level
    holding it, by the lowest of the siblings sharing it. Every region is owned
    once: the owned cells' volumes add to the root's."""
    amr = hierarchy("image", d)
    owned = {}
    every = set()
    for positions, skipped in _world_points(amr):
        every |= set(map(tuple, positions))
        for x in map(tuple, positions[~skipped]):
            owned[x] = owned.get(x, 0) + 1
    assert set(owned) == every and set(owned.values()) == {1}
    volume = 0.0
    for level, box in amr:
        kept = (box.ghosts.cells.numpy() & Ghosts.SKIPPED_CELLS) == 0
        volume += kept.sum() * (0.5 ** level) ** d
    root = sum(np.prod(b[1] - b[0]) for b in amr.boxes(0))
    assert volume == root
    stats = td.statistics(amr, "c")
    assert stats.n == sum(((box.ghosts.cells.numpy() & Ghosts.SKIPPED_CELLS) == 0).sum()
                          for _, box in amr)
    assert td.statistics(amr, "p").n == len(every)


def test_marks_already_given_are_recomputed_and_others_kept():
    amr = hierarchy()
    first = amr.levels[0][0]
    cells = first.ghosts.cells.numpy().copy()
    cells[0] |= Ghosts.HIDDENCELL
    cells[1] |= Ghosts.REFINEDCELL          # stale: recomputed away
    first.ghosts = Ghosts(cells=cells, points=first.ghosts.points.numpy())
    again = td.AMRDataSet([list(amr.levels[0]), *map(list, amr.levels[1:])], amr.ratios)
    got = again.levels[0][0].ghosts.cells.numpy()
    assert got[0] & Ghosts.HIDDENCELL
    assert not got[1] & Ghosts.REFINEDCELL


def test_hierarchies_that_vtks_model_cannot_hold_are_refused():
    boxes = [[image((0, 0), (8, 8), 1.0)], [image((2, 2), (4, 4), 0.5)]]
    with pytest.raises(ValueError, match="overlap"):
        td.AMRDataSet([[image((0, 0), (4, 4), 1.0), image((2, 2), (4, 4), 1.0)]])
    with pytest.raises(ValueError, match="properly nested"):
        td.AMRDataSet([[image((0, 0), (4, 4), 1.0)], [image((6, 6), (4, 4), 0.5)]], [2])
    with pytest.raises(ValueError, match="one lattice"):
        td.AMRDataSet([[image((0, 0), (4, 4), 1.0), image((4, 0), (4, 4), 1.0, (0.1, 0, 0))]])
    with pytest.raises(ValueError, match="2 levels take 1"):
        td.AMRDataSet(boxes, [2, 2, 2])
    with pytest.raises(ValueError, match="whole number of at least 1"):
        td.AMRDataSet(boxes, [0])
    with pytest.raises(ValueError, match="one dimension"):
        td.AMRDataSet([[image((0, 0), (4, 4), 1.0)], [image((0, 0, 0), (2, 2, 2), 0.5)]], [2])
    with pytest.raises(ValueError, match="axis-aligned"):
        turned = td.uniform_grid((5, 5), direction=[[0.0, -1.0], [1.0, 0.0]])
        td.AMRDataSet([[turned]])
    with pytest.raises(ValueError, match="ghost cells"):
        td.AMRDataSet([list(td.split(td.uniform_grid((9, 9)), 2, ghost_layers=1))])
    with pytest.raises(TypeError, match="StructuredTopology"):
        td.AMRDataSet([[td.point_cloud(np.zeros((3, 3)))]])
    # Anisotropic ratios are Tack's: VTK's AMR takes one per level.
    flat = td.AMRDataSet([[image((0, 0), (4, 4), 1.0)],
                          [image((2, 0), (2, 4), 0.5).rebased((0.0, 0.0, 0.0))]], [(2, 1)])
    assert flat.ratios == ((2, 1, 1),)


# -- what reads them -------------------------------------------------------------------

def test_filters_leave_refined_cells_out():
    amr = td.amr_gaussian_pulse()
    root = amr.levels[0][0]
    refined = (root.ghosts.cells.numpy() & Ghosts.REFINEDCELL) != 0
    values = root.fields["Gaussian-Pulse"].values.numpy()
    kept = td.threshold(root, "Gaussian-Pulse", float(values.min()), float(values.max()))
    assert kept.num_cells == (~refined).sum()
    assert td.remove_ghosts(root).num_cells == (~refined).sum()
    # The same cells' results still cover every cell: refined data are valid.
    assert td.cell_centers(root).values.shape[0] == root.num_cells
    assert td.connected_components(root).fields["region"].values.numpy()[refined].max() == -1


def test_boxes_on_two_devices():
    second = dispatch.as_device(_SecondCPU())
    amr = hierarchy()
    moved = td.AMRDataSet([[box.to_device(second) if k % 2 else box
                            for k, box in enumerate(level)] for level in
                           (list(lv) for lv in amr.levels)], amr.ratios)
    for (_, a), (_, b) in zip(moved, amr):
        assert a.ghosts.device == a.device
        np.testing.assert_array_equal(a.ghosts.cells.numpy(), b.ghosts.cells.numpy())
    assert td.statistics(moved, "c").n == td.statistics(amr, "c").n


# -- ghost cells and averages --------------------------------------------------------

SPACING = [1.0, 0.5, 0.25]


def _expected_cell_x(amr, level, q):
    """The ghost cell value ``c`` (its centre's x) that ghost_cells gives cell ``q``:
    its own where a sibling holds it, else its coarse cell's, injected."""
    if any(np.all((q >= b[0, :2]) & (q < b[1, :2])) for b in amr.boxes(level)):
        return (q[0] + 0.5) * SPACING[level]
    return _expected_cell_x(amr, level - 1, q // 2)


@pytest.mark.parametrize("kind", ["image", "rectilinear", "curvilinear"])
def test_ghost_layers_come_from_siblings_and_the_coarser_level(kind):
    amr = hierarchy(kind)
    grown = td.ghost_cells(amr, layers=1)
    assert grown.layers == 1
    for (level, box), (_, own) in zip(grown, amr):
        x = box.world_positions()
        marks = box.ghosts
        # Grown by one cell within the level's domain, the extra marked.
        lo, hi = grown.level_domain(level)
        valid = amr.boxes(level)
        extent = np.array([box.topology.offset, np.asarray(box.topology.offset)
                           + np.asarray(box.topology.point_dims) - 1])
        assert any((extent[0] == np.maximum(b[0, :2] - 1, lo[:2])).all() and
                   (extent[1] == np.minimum(b[1, :2] + 1, hi[:2])).all() for b in valid)
        ghost_cells = (marks.cells.numpy() & Ghosts.DUPLICATECELL) != 0
        assert ghost_cells.sum() == box.num_cells - own.num_cells
        # Positions are the lattice everywhere, as an image's would be.
        ij = np.stack(np.meshgrid(*[np.arange(n) for n in box.topology.point_dims[::-1]],
                                  indexing="ij")[::-1], -1).reshape(-1, 2) + extent[0]
        np.testing.assert_allclose(x[:, :2], ij * SPACING[level], atol=1e-6)
        # A point field linear in position is exact at every ghost point,
        # copied or interpolated in the coarse cells.
        np.testing.assert_allclose(box.fields["p"].values.numpy(), x[:, 1], atol=1e-6)
        # A cell field: a sibling's value, else the coarse cell's.
        cells = np.stack(np.meshgrid(*[np.arange(n - 1) for n in
                                       box.topology.point_dims[::-1]],
                                     indexing="ij")[::-1], -1).reshape(-1, 2) + extent[0]
        got = box.fields["c"].values.numpy()
        expected = [_expected_cell_x(amr, level, q) for q in cells[ghost_cells]]
        np.testing.assert_allclose(got[ghost_cells], expected, atol=1e-6)
    # Ghosts are copies: the hierarchy's statistics do not change.
    assert td.statistics(grown, "c").n == td.statistics(amr, "c").n
    assert td.statistics(grown, "p").n == td.statistics(amr, "p").n


def test_every_kind_of_box_grows_alike():
    grown = [td.ghost_cells(hierarchy(kind)) for kind in ("image", "rectilinear",
                                                          "curvilinear")]
    for boxes in zip(*(list(g) for g in grown)):
        first = boxes[0][1]
        for _, box in boxes[1:]:
            np.testing.assert_array_equal(box.ghosts.cells.numpy(), first.ghosts.cells.numpy())
            np.testing.assert_array_equal(box.ghosts.points.numpy(),
                                          first.ghosts.points.numpy())
            np.testing.assert_allclose(box.world_positions(), first.world_positions(),
                                       atol=1e-6)
            for name in ("c", "p"):
                np.testing.assert_array_equal(box.fields[name].values.numpy(),
                                              first.fields[name].values.numpy())


@pytest.mark.parametrize("kind", ["image", "curvilinear"])
def test_ghosts_nothing_holds_are_hidden(kind):
    """An L of root boxes, and a fine box at the L's inner corner: ghost cells in the
    L's notch have no source, and are hidden, as are the points only they use;
    a curvilinear box's hidden points are extrapolated along the lattice."""
    root = [image((0, 0), (4, 4), 1.0), image((4, 0), (4, 4), 1.0),
            image((0, 4), (4, 4), 1.0)]
    fine = [image((6, 2), (2, 4), 0.5)]
    amr = td.AMRDataSet([root, fine], [2])
    if kind == "curvilinear":
        amr = td.AMRDataSet([[_as(kind, b) for b in level] for level in
                             ([*amr.levels[0]], [*amr.levels[1]])], [2])
    grown = td.ghost_cells(amr)
    second = grown.levels[0][1]                     # grows up into the notch at x 4..8
    hidden = (second.ghosts.cells.numpy() & Ghosts.HIDDENCELL) != 0
    assert hidden.sum() == 4                        # the notch's row above it, x 4..8
    np.testing.assert_allclose(second.world_positions()[:, :2] % 1.0, 0.0, atol=1e-6)
    kept = td.threshold(second, "c", -1e9, 1e9)
    assert kept.num_cells == second.num_cells - hidden.sum() - (
        (second.ghosts.cells.numpy() & Ghosts.REFINEDCELL) != 0).sum() + (
        hidden & ((second.ghosts.cells.numpy() & Ghosts.REFINEDCELL) != 0)).sum()


def test_update_ghosts_refreshes_both_kinds_of_ghost():
    grown = td.ghost_cells(hierarchy())
    for _, box in grown:
        owned = (box.ghosts.points.numpy() & Ghosts.DUPLICATEPOINT) == 0
        y = box.world_positions()[:, 1]
        values = np.where(owned, 2.0 * y, -1.0).astype(np.float32)
        box.fields["p"] = td.Field(td.H1(box), tack.array(values))
    td.update_ghosts(grown, "p")
    for _, box in grown:
        y = box.world_positions()[:, 1]
        hidden = (box.ghosts.points.numpy() & Ghosts.HIDDENPOINT) != 0
        np.testing.assert_allclose(box.fields["p"].values.numpy()[~hidden], 2.0 * y[~hidden],
                                   atol=1e-5)


@pytest.mark.parametrize("kind", ["image", "curvilinear"])
def test_average_down_puts_fine_data_into_coarse_cells_and_points(kind):
    amr = hierarchy(kind)
    for _, box in amr:                              # the superseded values, spoilt
        for name, bit in (("c", Ghosts.REFINEDCELL), ("p", Ghosts.REFINEDPOINT)):
            marks = box.ghosts.cells if name == "c" else box.ghosts.points
            values = box.fields[name].values.numpy().copy()
            values[(marks.numpy() & bit) != 0] = -1.0
            space = td.Constant(box) if name == "c" else td.H1(box)
            box.fields[name] = td.Field(space, tack.array(values))
    td.average_down(amr, "c")
    td.average_down(amr, "p")
    for _, box in amr:
        centres = td.cell_centers(box).values.numpy().reshape(-1, 3)[:, 0]
        np.testing.assert_allclose(box.fields["c"].values.numpy(), centres, atol=1e-6)
        np.testing.assert_allclose(box.fields["p"].values.numpy(),
                                   box.world_positions()[:, 1], atol=1e-6)


@pytest.mark.parametrize("kind", ["rectilinear", "curvilinear"])
def test_ghost_geometry_is_interpolated_in_the_coarse_cells(kind):
    """A root stretched along x (rectilinear) or sheared (curvilinear), and a fine
    box whose points subdivide its coarse cells linearly: the fine box's ghost
    layer, which no sibling holds, gets the same subdivision from the root."""
    def stretch(i):                     # the root's x, linear between its points
        i = np.asarray(i, np.float64)
        lo = np.floor(i)
        def at(k):
            return k + 0.05 * k * k
        return at(lo) + (i - lo) * (at(lo + 1) - at(lo))

    def map_(i, j):                     # world position at fractional root index (i, j)
        if kind == "rectilinear":
            return np.stack([stretch(i), j], -1)
        return np.stack([i + 0.3 * j, j + 0.1 * i], -1)

    def box(lo, cells, ratio):
        i = (lo[0] + np.arange(cells[0] + 1)) / ratio
        j = (lo[1] + np.arange(cells[1] + 1)) / ratio
        topology = td.StructuredTopology((cells[0] + 1, cells[1] + 1), offset=lo)
        if kind == "rectilinear":
            geometry = tack.arrays.CartesianProduct(stretch(i), j)
        else:
            jj, ii = np.meshgrid(j, i, indexing="ij")
            xy = map_(ii, jj).reshape(-1, 2)
            geometry = np.column_stack([xy, np.zeros(len(xy))])
        out = td.DataSet(topology, geometry)
        out.fields["p"] = td.Field(td.H1(out), tack.array(out.positions()[:, 1]))
        return out

    amr = td.AMRDataSet([[box((0, 0), (8, 8), 1)], [box((4, 4), (6, 6), 2)]], [2])
    fine = td.ghost_cells(amr).levels[1][0]
    lo = np.asarray(fine.topology.offset)
    dims = fine.topology.point_dims
    jj, ii = np.meshgrid(np.arange(dims[1]) + lo[1], np.arange(dims[0]) + lo[0], indexing="ij")
    expected = map_(ii / 2.0, jj / 2.0).reshape(-1, 2)
    np.testing.assert_allclose(fine.world_positions()[:, :2], expected, atol=1e-5)


def test_ghost_cells_refuses_what_it_cannot_do():
    grown = td.ghost_cells(hierarchy())
    with pytest.raises(ValueError, match="already has ghost cells"):
        td.ghost_cells(grown)
    with pytest.raises(ValueError, match="ghost_cells made"):
        td.update_ghosts(hierarchy(), "p")
    with pytest.raises(TypeError, match="AMRDataSet"):
        td.average_down(td.PartitionedDataSet([]), "p")


def test_ghost_cells_in_3d():
    grown = td.ghost_cells(hierarchy("curvilinear", 3))
    hidden_somewhere = False
    for level, box in grown:
        x = box.world_positions()
        filled = (box.ghosts.points.numpy() & Ghosts.HIDDENPOINT) == 0
        hidden_somewhere |= not filled.all()
        np.testing.assert_allclose(box.fields["p"].values.numpy()[filled], x[filled, 1],
                                   atol=1e-6)
        np.testing.assert_allclose(x[:, 2] % 0.25, 0.0, atol=1e-6)   # extrapolated too
    assert hidden_somewhere              # the level-2 box grows past its coarse box


def test_amr_across_mpi_ranks(mpiexec):
    script = os.path.join(os.path.dirname(__file__), "_amr_mpi.py")
    for ranks in (2, 3):
        run = mpiexec(ranks, script)
        assert run.returncode == 0 and "ok" in run.stdout, run.stdout + run.stderr


def across(world):
    """This rank's share of each level's boxes: their marks by (level, global index),
    and the statistics of the cell field over the hierarchy."""
    every = [list(level) for level in hierarchy().levels]
    mine, keys = [], []
    for level, boxes in enumerate(every):
        chosen = list(range(len(boxes)))[world.rank::world.size]
        mine.append([_unmarked(boxes[i]) for i in chosen])
        keys.append(chosen)
    # Boxes are numbered by rank, then in order: map ours back to the whole's.
    amr = td.AMRDataSet(mine, [2, 2], world=world)
    out = {}
    for level, boxes in enumerate(amr.levels):
        for box, whole in zip(boxes, keys[level]):
            out[(level, whole)] = (box.ghosts.cells.numpy().tolist(),
                                   (box.ghosts.points.numpy() & (0xFF ^ Ghosts.DUPLICATEPOINT))
                                   .tolist())
    grown = td.ghost_cells(amr)
    for level, boxes in enumerate(grown.levels):
        for box, whole in zip(boxes, keys[level]):
            out[(level, whole, "ghosts")] = (box.fields["c"].values.numpy().tolist(),
                                             box.fields["p"].values.numpy().tolist())
    return out, td.statistics(amr, "c").n


def _unmarked(box):
    return td.DataSet(box.topology, box.geometry.values, fields=dict(box.fields))
