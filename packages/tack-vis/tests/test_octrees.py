"""Octrees (``OctreeDataSet``): a grid of trees whose nodes are Morton-sorted, as a
linear octree.

The geometry the keys give is held to VTK's ``vtkHyperTreeGridGenerateFields``
(cell centres, sizes, which cells are valid, the total visible volume) node for
node, and conversions to and from ``vtkHyperTreeGrid`` keep the trees, values
and mask. Statistics, the focus of this pass, are held to NumPy over the
leaves, unweighted, volume-weighted and mass-weighted, and per level; volume
weights are checked on other kinds of data too. The trees can be spread over
ranks (``_octree_mpi.py``) or put on a second device.
"""

import os

import numpy as np
import pytest

import tack
import tack.dataset as td
from tack import arrays
from tack.dataset import Ghosts
from tack.runtime import dispatch
from tack.runtime.cpu import CPUBackend
from tack.runtime.dispatch import env_flag

try:
    import vtkmodules.vtkCommonDataModel  # noqa: F401  (registers the pythonic API)
    from vtkmodules.vtkFiltersHyperTree import vtkHyperTreeGridGenerateFields
    from vtkmodules.vtkFiltersSources import vtkHyperTreeGridSource, vtkRandomHyperTreeGridSource

    from tack.interop.vtk import dataset_to_vtk, vtk_to_dataset
except ImportError:
    if env_flag("TACK_REQUIRE_VTK"):
        raise
    vtkRandomHyperTreeGridSource = None

needs_vtk = pytest.mark.skipif(vtkRandomHyperTreeGridSource is None, reason="needs VTK")
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


def ramses_like(d=3, seed=3, levelmin=1, levelmax=4, trees=(2, 2, 2)):
    """Cells of a RAMSES-like run: a root grid of ``trees`` cells, each refined to
    ``levelmin`` everywhere, then further, oct by oct, where a blob is dense, to
    ``levelmax``; each cell's level, centre and a density (the blob, sampled at
    the centre) and a temperature. Whole octs, as RAMSES writes them."""
    rng = np.random.default_rng(seed)
    centre = np.array([0.8, 1.1, 0.9])[:d]
    roots_shape = trees[:d]
    levels, centres = [], []

    def refine(lo, size, level):
        mid = lo + 0.5 * size
        dense = np.exp(-8.0 * np.sum((mid - centre) ** 2)) > 0.3 / (level + 1)
        if level < levelmin or (level < levelmax and dense and rng.random() < 0.9):
            for child in range(2 ** d):
                offset = np.array([(child >> a) & 1 for a in range(d)]) * size / 2
                refine(lo + offset, size / 2, level + 1)
        else:
            levels.append(level)
            centres.append(mid)

    for index in np.ndindex(*roots_shape[::-1]):
        refine(np.array(index[::-1], float), np.ones(d), 0)
    centres = np.array(centres)
    rho = np.exp(-8.0 * np.sum((centres - centre) ** 2, axis=1))
    temperature = 1.0 + rng.random(len(centres))
    full = np.zeros((len(centres), 3))
    full[:, :d] = centres
    roots = td.uniform_grid(tuple(n + 1 for n in roots_shape), spacing=(1.0,) * d)
    return roots, np.array(levels), full, rho, temperature


# -- the tree --------------------------------------------------------------------------

@needs_vtk
@pytest.mark.parametrize("dims", [(4, 3, 3), (4, 3, 1)])
def test_the_geometry_is_vtks(dims):
    source = vtkRandomHyperTreeGridSource(dimensions=dims, max_depth=4, seed=7,
                                          split_fraction=0.5)
    fields = vtkHyperTreeGridGenerateFields()
    fields.SetInputConnection(source.GetOutputPort())
    fields.Update()
    octree = vtk_to_dataset(fields.GetOutput(), dtype=tack.f64)
    np.testing.assert_allclose(octree.centers(), octree.fields["CellCenter"].numpy(),
                               atol=1e-12)
    np.testing.assert_allclose(octree.measures(), octree.fields["CellSize"].numpy(),
                               rtol=1e-12)
    valid = octree.is_leaf & ((octree.marks.numpy() & Ghosts.HIDDENCELL) == 0)
    np.testing.assert_array_equal(valid, octree.fields["ValidCell"].numpy() != 0)
    total = float(np.asarray(fields.GetOutput().GetFieldData().GetArray("TotalVisibleVolume"))[0])
    assert np.isclose(octree.measures()[valid].sum(), total)
    assert np.isclose(td.statistics(octree, "Depth", weights="volume").weight, total)


@needs_vtk
@pytest.mark.parametrize("dims", [(4, 3, 3), (4, 3, 1)])
def test_vtk_round_trip_keeps_the_trees_values_and_mask(dims):
    source = vtkRandomHyperTreeGridSource(dimensions=dims, max_depth=4, seed=9,
                                          split_fraction=0.5)
    source.Update()
    first = vtk_to_dataset(source.GetOutput(), dtype=tack.f64)
    hidden = np.zeros(first.num_nodes, bool)
    hidden[::7] = True
    masked = td.OctreeDataSet(first.roots, first._tree, first._level, first._key,
                              fields={"Depth": first.fields["Depth"].numpy()}, mask=hidden)
    out = dataset_to_vtk(masked)
    assert out.class_name == "vtkHyperTreeGrid"
    back = vtk_to_dataset(out, dtype=tack.f64)
    for name in ("_tree", "_level", "_key"):
        np.testing.assert_array_equal(getattr(back, name), getattr(masked, name))
    np.testing.assert_array_equal(back.marks.numpy(), masked.marks.numpy())
    np.testing.assert_array_equal(back.fields["Depth"].numpy(), masked.fields["Depth"].numpy())


@needs_vtk
def test_a_branching_factor_of_three_is_refused():
    source = vtkHyperTreeGridSource(dimensions=(3, 3, 1), branch_factor=3, max_depth=1,
                                    descriptor="....")
    source.Update()
    with pytest.raises(ValueError, match="branching factor of 2"):
        vtk_to_dataset(source.GetOutput())


@pytest.mark.parametrize("d", [2, 3])
def test_ramses_cells_make_a_tree_of_whole_octs(d):
    roots, level, centre, rho, temperature = ramses_like(d)
    octree = td.OctreeDataSet.from_cells(roots, level, centre,
                                         fields={"rho": rho, "T": temperature})
    leaves = octree.is_leaf
    assert leaves.sum() == len(level)              # every cell given is a leaf
    np.testing.assert_allclose(octree.measures()[leaves].sum(), roots.num_cells)
    # Each leaf is a cell given, at its centre, with its values; the added
    # ancestors have NaN.
    order = np.lexsort(np.round(centre, 9).T)
    got = np.lexsort(np.round(octree.centers()[leaves], 9).T)
    np.testing.assert_allclose(octree.centers()[leaves][got], centre[order], atol=1e-12)
    np.testing.assert_array_equal(octree.fields["rho"].numpy()[leaves][got], rho[order])
    assert np.isnan(octree.fields["rho"].numpy()[~leaves]).all()
    # Pre-order: every node's parent comes before it.
    assert (np.diff(octree._tree) >= 0).all()


def test_trees_that_are_not_whole_are_refused():
    roots = td.uniform_grid((3, 2), spacing=(1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="fewer than 4 children"):
        td.OctreeDataSet.from_cells(roots, [1, 0], [[0.2, 0.2, 0.0], [1.5, 0.5, 0.0]])
    with pytest.raises(ValueError, match="no parent"):
        td.OctreeDataSet(roots, [0], [1], [0])
    with pytest.raises(ValueError, match="given twice"):
        td.OctreeDataSet(roots, [0, 0], [0, 0], [0, 0])
    with pytest.raises(ValueError, match="too large for its level"):
        td.OctreeDataSet(roots, [0], [0], [1])
    with pytest.raises(ValueError, match="trees are the roots' cells"):
        td.OctreeDataSet(roots, [5], [0], [0])
    with pytest.raises(TypeError, match="image or a rectilinear grid"):
        td.OctreeDataSet(td.point_cloud(np.zeros((2, 3))), [0], [0], [0])


# -- statistics ------------------------------------------------------------------------

def _weighted_numpy(x, w):
    mean = np.average(x, weights=w)
    d = x - mean
    total = w.sum()
    m2 = (w * d * d).sum()
    return {"mean": mean, "sum": (w * x).sum(), "weight": total,
            "population_variance": m2 / total,
            "sample_variance": m2 / (total - (w * w).sum() / total),
            "skewness": np.sqrt(total) * (w * d ** 3).sum() / m2 ** 1.5,
            "kurtosis": total * (w * d ** 4).sum() / m2 ** 2}


@pytest.mark.parametrize("d", [2, 3])
def test_statistics_of_the_leaves_weighted_and_not(d):
    roots, level, centre, rho, temperature = ramses_like(d)
    octree = td.OctreeDataSet.from_cells(roots, level, centre,
                                         fields={"rho": rho, "T": temperature})
    leaves = octree.is_leaf
    t = octree.fields["T"].numpy()[leaves].astype(np.float64)
    volume = octree.measures()[leaves]
    mass = octree.fields["rho"].numpy()[leaves].astype(np.float64) * volume
    plain = td.statistics(octree, "T")
    assert plain.n == leaves.sum() and plain.weight == leaves.sum()
    np.testing.assert_allclose(plain.mean, t.mean(), rtol=1e-6)
    np.testing.assert_allclose(plain.sample_variance, t.var(ddof=1), rtol=1e-5)
    for weights, w in (("volume", volume), ("rho", octree.fields["rho"].numpy()[leaves])):
        got = td.statistics(octree, "T", weights=weights)
        for name, value in _weighted_numpy(t, np.asarray(w, np.float64)).items():
            np.testing.assert_allclose(getattr(got, name), value, rtol=1e-5, err_msg=name)
    # Mass-weighted: with rho * volume, a field of ones gives the total mass.
    host = np.zeros(octree.num_nodes)
    host[leaves] = mass
    octree.fields["mass"] = tack.array(host.astype(np.float32))
    octree.fields["one"] = tack.array(np.ones(octree.num_nodes, np.float32))
    np.testing.assert_allclose(td.statistics(octree, "one", weights="mass").sum,
                               mass.sum(), rtol=1e-5)
    # One level's nodes, refined or not; volume weights make every node of a
    # level count alike.
    for lv in range(int(level.max()) + 1):
        chosen = octree._level == lv
        if not chosen.any():
            continue
        if np.isnan(octree.fields["T"].numpy()[chosen]).any():
            continue                       # an added ancestor: no values there
        np.testing.assert_allclose(td.statistics(octree, "T", level=lv).mean,
                                   octree.fields["T"].numpy()[chosen].mean(), rtol=1e-6)


def test_entropy_weighted_and_not():
    roots, level, centre, rho, temperature = ramses_like(3)
    octree = td.OctreeDataSet.from_cells(roots, level, centre, fields={"T": temperature})
    leaves = octree.is_leaf
    t = octree.fields["T"].numpy()[leaves].astype(np.float64)
    volume = octree.measures()[leaves]
    for weights, w in ((None, np.ones(len(t))), ("volume", volume)):
        counts, _ = np.histogram(t, bins=8, range=(t.min(), t.max()), weights=w)
        p = counts[counts > 0] / counts.sum()
        np.testing.assert_allclose(td.entropy(octree, "T", bins=8, weights=weights),
                                   -(p * np.log2(p)).sum(), rtol=1e-6)


def test_volume_weights_on_other_kinds_of_data():
    """An AMR hierarchy's cells differ in size by level: the volume-weighted mean of
    the cell centres' x over the hierarchy is the domain's mean x, exactly, where
    the unweighted one is pulled toward the fine boxes."""
    import test_amr

    amr = test_amr.hierarchy()
    weighted = td.statistics(amr, "c", weights="volume")
    np.testing.assert_allclose(weighted.mean, 4.0, rtol=1e-6)
    np.testing.assert_allclose(weighted.weight, 64.0, rtol=1e-6)
    assert abs(td.statistics(amr, "c").mean - 4.0) > 0.05
    # A uniform grid: equal weights, the plain statistics' moments.
    grid = td.wavelet(4)
    grid.fields["c"] = td.project(grid, "RTData", td.Constant(grid))
    a, b = td.statistics(grid, "c"), td.statistics(grid, "c", weights="volume")
    np.testing.assert_allclose([a.mean, a.population_variance, a.skewness],
                               [b.mean, b.population_variance, b.skewness], rtol=1e-5)
    with pytest.raises(TypeError, match="take a cell field"):
        td.statistics(grid, "RTData", weights="volume")
    with pytest.raises(TypeError, match="level= picks an octree"):
        td.statistics(grid, "c", level=1)


# -- leaves as cells, and probes ---------------------------------------------------------

def _corner_rows(data, positions, connectivity):
    """Each cell's corners as one row, rows sorted: cells compared as sets."""
    rows = np.round(positions[connectivity].reshape(len(connectivity), -1), 6)
    return rows[np.lexsort(rows.T[::-1])]


@needs_vtk
@pytest.mark.parametrize("dims", [(4, 3, 3), (4, 3, 1)])
def test_the_leaves_are_vtks_cells_with_shared_corners(dims):
    from vtkmodules.vtkFiltersHyperTree import vtkHyperTreeGridToUnstructuredGrid

    source = vtkRandomHyperTreeGridSource(dimensions=dims, max_depth=4, seed=7,
                                          split_fraction=0.5)
    source.Update()
    octree = vtk_to_dataset(source.GetOutput(), dtype=tack.f64)
    leaves = octree.leaves()
    convert = vtkHyperTreeGridToUnstructuredGrid()
    convert.SetInputData(source.GetOutput())
    convert.Update()
    vtk_cells = convert.GetOutput()
    # VTK gives every leaf its own corners; Tack's share them, and give the same
    # cells (VTK's points are f32).
    corners = 2 ** octree.dimension
    vtk_conn = np.asarray(vtk_cells.GetCells().GetConnectivityArray()).reshape(-1, corners)
    conn = leaves.topology.connectivity.numpy().reshape(-1, corners)
    np.testing.assert_allclose(
        _corner_rows(leaves, leaves.positions().astype(np.float64), conn),
        _corner_rows(None, np.asarray(vtk_cells.points.data, np.float64), vtk_conn),
        atol=1e-5)
    assert {vtk_cells.GetCellType(c) for c in range(vtk_cells.number_of_cells)} == \
        set(np.asarray(arrays.materialize(leaves.topology.types).numpy()).tolist())
    assert len(np.unique(leaves.positions(), axis=0)) == leaves.num_points
    np.testing.assert_array_equal(leaves.fields["Depth"].values.numpy(),
                                  octree.fields["Depth"].numpy()[octree.is_leaf])
    # Cell c is the c-th leaf: its centre is the node's.
    centres = leaves.positions().astype(np.float64)[conn].mean(axis=1)
    np.testing.assert_allclose(centres, octree.centers()[octree.is_leaf], atol=1e-12)


def test_leaves_share_corners_and_keep_hanging_ones():
    """One root refined into eight, one of them again: 27 corners, then the
    refined child's 3x3x3 lattice adds 19, hanging ones on its faces included."""
    roots = td.uniform_grid((2, 2, 2))
    tree = [0] * 17
    level = [0] + [1] * 8 + [2] * 8
    key = [0] + list(range(8)) + [(5 << 3) | c for c in range(8)]
    octree = td.OctreeDataSet(roots, tree, level, key)
    leaves = octree.leaves()
    assert (leaves.num_cells, leaves.num_points) == (15, 46)
    x = leaves.positions()
    lattice = np.round(x * 4).astype(int)
    np.testing.assert_allclose(x, lattice / 4)
    assert len(np.unique(lattice, axis=0)) == 46


def test_hidden_leaves_are_hidden_cells_and_reductions_agree():
    roots, level, centre, rho, temperature = ramses_like(3)
    whole = td.OctreeDataSet.from_cells(roots, level, centre, fields={"T": temperature})
    hidden = np.zeros(whole.num_nodes, bool)
    hidden[3::5] = True
    octree = td.OctreeDataSet(roots, whole._tree, whole._level, whole._key,
                              fields={"T": whole.fields["T"].numpy()}, mask=hidden)
    leaves = octree.leaves()
    marks = leaves.ghosts.cells.numpy()
    np.testing.assert_array_equal(marks != 0, hidden[octree.is_leaf])
    assert np.all(marks[marks != 0] == Ghosts.HIDDENCELL)
    a, b = td.statistics(octree, "T"), td.statistics(leaves, "T")
    assert a.n == b.n
    np.testing.assert_allclose([a.mean, a.population_variance],
                               [b.mean, b.population_variance], rtol=1e-12)


@needs_vtk
@pytest.mark.parametrize("dims", [(4, 3, 3), (4, 3, 1)])
def test_probe_is_vtks(dims):
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkPolyData
    from vtkmodules.vtkFiltersCore import vtkHyperTreeGridProbeFilter

    source = vtkRandomHyperTreeGridSource(dimensions=dims, max_depth=4, seed=7,
                                          split_fraction=0.5)
    source.Update()
    octree = vtk_to_dataset(source.GetOutput(), dtype=tack.f64)
    rng = np.random.default_rng(3)
    x = rng.uniform(-11.0, 11.0, (500, 3))
    if dims[2] == 1:
        x[:, 2] = 0.0
    points = vtkPolyData()
    points.points = vtkPoints(data=x)
    reference = vtkHyperTreeGridProbeFilter()
    reference.SetInputData(points)
    reference.SetSourceData(source.GetOutput())
    reference.Update()
    want_valid = np.asarray(reference.GetOutput().point_data["vtkValidPointMask"]).view(
        np.uint8)
    want = np.asarray(reference.GetOutput().point_data["Depth"])
    got = td.probe(octree, x)
    valid = got.fields["valid"].values.numpy()
    np.testing.assert_array_equal(valid, want_valid)
    depth = got.fields["Depth"].values.numpy()
    np.testing.assert_array_equal(depth[valid == 1], want[want_valid == 1])
    assert np.all(depth[valid == 0] == 0)


@pytest.mark.parametrize("d", [2, 3])
def test_probe_is_the_probe_of_the_leaves(d):
    roots, level, centre, rho, temperature = ramses_like(d)
    velocity = np.stack([rho, temperature, rho * temperature], axis=1)
    octree = td.OctreeDataSet.from_cells(roots, level, centre,
                                         fields={"rho": rho, "v": velocity})
    rng = np.random.default_rng(5)
    x = rng.uniform(-0.2, 2.2, (2000, 3))
    if d == 2:
        x[:, 2] = 0.0
    got, want = td.probe(octree, x), td.probe(octree.leaves(), x)
    valid = got.fields["valid"].values.numpy()
    np.testing.assert_array_equal(valid, want.fields["valid"].values.numpy())
    assert 0 < valid.sum() < len(x)
    inside = valid == 1
    # Away from faces, where both take the same leaf (on a face, the leaf whose
    # lower face it is, as one cell's search may take either).
    lattice = x[:, :d] * 2 ** octree.finest
    away = inside & np.all(np.abs(lattice - np.round(lattice)) > 1e-6, axis=1)
    for name in ("rho", "v"):
        np.testing.assert_array_equal(got.fields[name].values.numpy()[away],
                                      want.fields[name].values.numpy()[away])
    assert got.fields["v"].values.numpy().shape == (len(x), 3)


def test_probe_onto_a_dataset_keeps_it_and_chooses_fields():
    roots, level, centre, rho, temperature = ramses_like(3)
    octree = td.OctreeDataSet.from_cells(roots, level, centre,
                                         fields={"rho": rho, "T": temperature})
    image = td.uniform_grid((9, 9, 9), spacing=(0.25, 0.25, 0.25))
    image.fields["here"] = td.Field(td.H1(image), tack.zeros((image.num_points,), tack.f32))
    out = td.probe(octree, image, fields=["rho"])
    assert out.topology is image.topology
    assert set(out.fields) == {"here", "rho", "valid"}
    assert out.fields["valid"].values.numpy().all()
    # A point inside a leaf takes that leaf's value.
    x = image.positions().astype(np.float64)
    nodes = np.flatnonzero(octree.is_leaf)
    corner, sides = (a.numpy().reshape(-1, 3)[nodes] for a in octree.geometry())
    rho_leaves = octree.fields["rho"].numpy()[nodes]
    probed = out.fields["rho"].values.numpy()
    for q in range(0, len(x), 37):
        inside = np.all((x[q] >= corner) & (x[q] <= corner + sides), axis=1)
        assert probed[q] in rho_leaves[inside]
    with pytest.raises(KeyError, match="no field 'nothing'"):
        td.probe(octree, image, fields=["nothing"])


def test_probe_skips_hidden_leaves_and_points_off_a_2d_plane():
    roots = td.uniform_grid((3, 2), spacing=(1.0, 1.0))
    octree = td.OctreeDataSet(roots, [0, 1], [0, 0], [0, 0], fields={"f": [1.0, 2.0]},
                              mask=[0, 1])
    got = td.probe(octree, np.array([[0.5, 0.5, 0.0], [1.5, 0.5, 0.0], [0.5, 0.5, 0.3],
                                     [0.5, 0.5, 1e-5], [2.0, 1.0, 0.0], [2.5, 0.5, 0.0]]))
    np.testing.assert_array_equal(got.fields["valid"].values.numpy(), [1, 0, 0, 1, 0, 0])
    np.testing.assert_array_equal(got.fields["f"].values.numpy(), [1, 0, 0, 1, 0, 0])
    with pytest.raises(ValueError, match="takes no locator"):
        td.probe(octree, np.zeros((1, 3)), locator=object())


def test_an_octree_on_a_second_device():
    second = dispatch.as_device(_SecondCPU())
    roots, level, centre, rho, temperature = ramses_like(2)
    here = td.OctreeDataSet.from_cells(roots, level, centre, fields={"T": temperature})
    there = td.OctreeDataSet.from_cells(roots.to_device(second), level, centre,
                                        fields={"T": temperature})
    assert there.key.device == second and there.fields["T"].device == second
    a, b = td.statistics(here, "T", weights="volume"), td.statistics(there, "T",
                                                                     weights="volume")
    assert a == b
    leaves = there.leaves()
    assert leaves.device == second and leaves.fields["T"].values.device == second
    np.testing.assert_array_equal(leaves.positions(), here.leaves().positions())
    x = np.random.default_rng(2).uniform(0.0, 2.0, (300, 3))
    x[:, 2] = 0.0
    probed = td.probe(there, x)
    assert probed.fields["T"].values.device == tack.device("cpu")   # where x's points are
    np.testing.assert_array_equal(probed.fields["T"].values.numpy(),
                                  td.probe(here, x).fields["T"].values.numpy())


def test_octrees_across_mpi_ranks(mpiexec):
    script = os.path.join(os.path.dirname(__file__), "_octree_mpi.py")
    for ranks in (2, 3):
        run = mpiexec(ranks, script)
        assert run.returncode == 0 and "ok" in run.stdout, run.stdout + run.stderr


def across(world):
    """This rank's share of the trees, as an octree in ``world``: its statistics."""
    roots, level, centre, rho, temperature = ramses_like(3, trees=(3, 2, 2))
    whole = td.OctreeDataSet.from_cells(roots, level, centre, fields={"T": temperature})
    mine = np.isin(whole._tree, np.arange(roots.num_cells)[world.rank::world.size])
    octree = td.OctreeDataSet(roots, whole._tree[mine], whole._level[mine],
                              whole._key[mine], fields={"T": whole.fields["T"].numpy()[mine]},
                              world=world)
    return (td.statistics(octree, "T", weights="volume"), td.entropy(octree, "T", bins=6),
            td.statistics(whole, "T", weights="volume"), td.entropy(whole, "T", bins=6))


def probe_across(world):
    """This rank's points (a different number on each) probed over every rank's
    share of the trees, and over the whole octree."""
    roots, level, centre, rho, temperature = ramses_like(3, trees=(3, 2, 2))
    fields = {"T": temperature, "v": np.stack([rho, temperature, rho], axis=1)}
    whole = td.OctreeDataSet.from_cells(roots, level, centre, fields=fields)
    mine = np.isin(whole._tree, np.arange(roots.num_cells)[world.rank::world.size])
    octree = td.OctreeDataSet(roots, whole._tree[mine], whole._level[mine],
                              whole._key[mine],
                              fields={n: whole.fields[n].numpy()[mine] for n in fields},
                              world=world)
    x = np.random.default_rng(world.rank).uniform(-0.2, 3.2, (100 + 37 * world.rank, 3))
    return td.probe(octree, x), td.probe(whole, x)
