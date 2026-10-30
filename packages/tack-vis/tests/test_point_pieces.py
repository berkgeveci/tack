"""Point clouds in pieces: ``ghost_points`` and the point filters over a
``PartitionedDataSet``, each held to the whole's answer.

Pieces are cut from one cloud in index order (``_split``), so a piece's points
in order are the whole's: what is reduced in point order (clusters' numbers,
voxels' centroids, halos' sums) then has the whole's bits. A cut by position
(``by_x``) gives other pieces and the same answers up to numbering. In a
periodic box a piece's copies across a face are images, whose positions are
rounded once more than the whole's minimum-image offsets, so values measured
through them agree to a rounding. The same checks run on a simulated second
device and on 2 and 3 MPI ranks (``_point_pieces_mpi.py``).
"""

import itertools
import os

import numpy as np
import pytest

import tack
import tack.dataset as td
from tack.dataset import Ghosts
from tack.runtime import dispatch
from tack.runtime.cpu import CPUBackend

pytestmark = pytest.mark.quick

BOX = 10.0


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


def clumps(seed=7, count=30):
    """Clumps of points and a sprinkle of loners in a periodic box, shuffled, with a
    velocity, a class and a mass each."""
    rng = np.random.default_rng(seed)
    centres = rng.uniform(1.0, BOX - 1.0, (count, 3))
    x = np.concatenate([c + rng.normal(0.0, 0.3, (rng.integers(60, 250), 3)) for c in centres]
                       + [rng.uniform(0.0, BOX, (300, 3))]) % BOX
    x = x[rng.permutation(len(x))]
    return x, {"v": rng.normal(0.0, 1.0, (len(x), 3)),
               "cls": rng.integers(0, 5, len(x)).astype(np.uint8),
               "m": rng.uniform(1.0, 2.0, len(x))}


def cloud(x, fields, rows=None):
    rows = np.arange(len(x)) if rows is None else rows
    return td.point_cloud(x[rows].astype(np.float32),
                          fields={name: values[rows] for name, values in fields.items()},
                          origin=(0.0, 0.0, 0.0))


def _split(x, fields, count=4, order=None, world=None, mine=None):
    """``count`` pieces of the cloud: index ranges of ``order`` (index order by
    default), this rank's (``mine``) in ``world``; and each piece's rows."""
    order = np.arange(len(x)) if order is None else order
    rows = [order[r] for r in np.array_split(np.arange(len(x)), count)]
    mine = range(count) if mine is None else mine
    return td.PartitionedDataSet([cloud(x, fields, rows[i]) for i in mine], world), \
        [rows[i] for i in mine]


def by_x(x):
    return np.argsort(x[:, 0], kind="stable")


def joined(out, name):
    return np.concatenate([piece.fields[name].values.numpy() for piece in out])


def owned(piece):
    return (piece.ghosts.points.numpy() & Ghosts.DUPLICATEPOINT) == 0


# -- ghost_points ------------------------------------------------------------

@pytest.mark.parametrize("periodic", [None, BOX])
@pytest.mark.parametrize("how", [{"radius": 0.6}, {"k": 8}])
def test_ghost_points_give_owned_points_the_wholes_neighbours(how, periodic):
    x, fields = clumps()
    fields["i"] = np.arange(len(x))
    whole = cloud(x, fields)
    near = td.PointLocator(whole, periodic=periodic)
    for order in (None, by_x(x)):
        pieces, rows = _split(x, fields, order=order)
        out = td.ghost_points(pieces, periodic=periodic, **how)
        for piece, source, r in zip(out, pieces, rows):
            n = source.num_points
            mine = owned(piece)
            assert mine[:n].all() and not mine[n:].any()
            assert (piece.ghosts.cells.numpy()[n:] & Ghosts.DUPLICATECELL).all()
            np.testing.assert_array_equal(piece.positions()[:n], source.positions())
            ids = piece.fields["i"].values.numpy()
            queries = source.positions().astype(np.float64)
            here = td.PointLocator(piece)
            if "radius" in how:
                wo, wi = (a.numpy() for a in near.neighbors(how["radius"], queries))
                po, pi = (a.numpy() for a in here.neighbors(how["radius"], queries))
                for q in range(n):
                    expected = sorted(wi[wo[q]:wo[q + 1]].tolist())
                    assert sorted(ids[pi[po[q]:po[q + 1]]].tolist()) == expected
            else:
                _, wd = near.nearest(how["k"] + 1, queries)
                _, pd = here.nearest(how["k"] + 1, queries)
                np.testing.assert_allclose(pd.numpy(), wd.numpy(), rtol=1e-6, atol=1e-7)
            # Every copy is a point of the whole (moved by whole boxes) carrying
            # its own values.
            gap = piece.positions()[n:].astype(np.float64) - x[ids[n:]].astype(np.float32)
            np.testing.assert_allclose(gap - BOX * np.round(gap / BOX), 0.0, atol=1e-5)
            np.testing.assert_array_equal(piece.fields["cls"].values.numpy(), fields["cls"][ids])


def test_ghost_points_keep_global_ids_and_refresh_from_owners():
    x, fields = clumps(seed=3, count=12)
    pieces, rows = _split(x, fields, count=3)
    for piece, r in zip(pieces, rows):
        piece.global_ids = td.GlobalIds(cells=r, points=r)
    out = td.ghost_points(pieces, radius=0.5, fields=["m"])
    for piece in out:
        assert sorted(piece.fields) == ["m"]
        ids = piece.global_ids.points.numpy()
        np.testing.assert_array_equal(piece.global_ids.cells.numpy(), ids)
        np.testing.assert_array_equal(piece.fields["m"].values.numpy(),
                                      fields["m"][ids].astype(np.float32))
    # One piece in a periodic box is its own neighbour: its copies are images
    # of its own points, refreshed from itself.
    single = td.ghost_points(td.PartitionedDataSet([cloud(x, fields)]), radius=0.5,
                             periodic=BOX, fields=["m"])
    for data in (out, single):
        for piece in data:
            mine = owned(piece)
            assert (~mine).any()
            values = piece.fields["m"].values.numpy().copy()
            values[~mine] = -1.0
            piece.fields["k"] = td.Field(td.H1(piece), tack.array(values))
        td.update_ghosts(data, "k")
        for piece in data:
            np.testing.assert_array_equal(piece.fields["k"].values.numpy(),
                                          piece.fields["m"].values.numpy())


def test_ghost_points_refuses():
    x, fields = clumps(seed=1, count=4)
    pieces, _ = _split(x, fields, count=2)
    with pytest.raises(ValueError, match="radius .* or k"):
        td.ghost_points(pieces)
    with pytest.raises(ValueError, match="radius .* or k"):
        td.ghost_points(pieces, radius=1.0, k=4)
    with pytest.raises(ValueError, match="k is a whole number"):
        td.ghost_points(pieces, k=0)
    with pytest.raises(ValueError, match="k needs none"):
        td.ghost_points(pieces, k=4, around=pieces)
    with pytest.raises(TypeError, match="PartitionedDataSet"):
        td.ghost_points(pieces[0], radius=1.0)
    with pytest.raises(ValueError, match="half the periodic box"):
        td.ghost_points(pieces, radius=5.0, periodic=BOX)
    with pytest.raises(TypeError, match="point clouds"):
        td.ghost_points(td.PartitionedDataSet([td.wavelet(3)]), radius=1.0)
    with pytest.raises(ValueError, match="already has ghost points"):
        td.ghost_points(td.ghost_points(pieces, radius=0.5), radius=0.5)
    with pytest.raises(ValueError, match="one origin"):
        td.ghost_points(td.PartitionedDataSet([pieces[0], pieces[1].rebased((1.0, 0, 0))]),
                        radius=0.5)


# -- filters per piece, after ghost_points -----------------------------------

@pytest.mark.parametrize("how", [{"k": 10}, {"radius": 0.4}])
def test_normals_per_piece_after_ghost_points(how):
    x, fields = clumps(seed=5)
    expected = td.point_normals(cloud(x, fields), **how).fields["normals"].values.numpy()
    pieces, rows = _split(x, fields, order=by_x(x))
    got = np.empty_like(expected)
    for piece, source, r in zip(td.ghost_points(pieces, **how), pieces, rows):
        got[r] = td.point_normals(piece, **how).fields["normals"].values.numpy()[
            :source.num_points]
    # Covariances summed in another order of neighbours: equal to a rounding,
    # which a nearly flat neighbourhood's least axis magnifies.
    np.testing.assert_allclose(got, expected, atol=1e-3)


def test_sph_per_piece_after_ghost_points():
    """Densities per piece, their copies refreshed from the owners, then each
    piece's particles interpolated onto its own targets, which ``around`` widens
    the copies to."""
    rng = np.random.default_rng(11)
    x = np.sort(rng.uniform(0.0, 4.0, (3000, 3)), axis=0)
    fields = {"h": rng.uniform(0.15, 0.25, len(x)), "mass": rng.uniform(1.0, 2.0, len(x)),
              "T": rng.normal(size=len(x))}
    whole = td.sph_density(cloud(x, fields))
    targets = rng.uniform(0.0, 4.0, (2000, 3))
    expected = td.sph_interpolate(whole, targets, fields=["T"]).fields["T"].values.numpy()
    cuts = [-1.0, 1.0, 2.0, 3.0, 5.0]
    chosen = [(targets[:, 0] >= lo) & (targets[:, 0] < hi) for lo, hi in itertools.pairwise(cuts)]
    around = td.PartitionedDataSet([td.point_cloud(targets[c].astype(np.float32),
                                                   origin=(0.0, 0.0, 0.0)) for c in chosen])
    pieces, rows = _split(x, fields)
    out = td.ghost_points(pieces, radius=2.0 * fields["h"].max(), around=around)
    for piece in out:
        piece.fields["density"] = td.sph_density(piece).fields["density"]
    density = np.empty(len(x), np.float32)
    for piece, r in zip(out, rows):
        density[r] = piece.fields["density"].values.numpy()[:len(r)]
    np.testing.assert_allclose(density, whole.fields["density"].values.numpy(), rtol=1e-5)
    td.update_ghosts(out, "density")
    got = np.empty_like(expected)
    for piece, c in zip(out, chosen):
        got[c] = td.sph_interpolate(piece, targets[c], fields=["T"]).fields["T"].values.numpy()
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-6)


# -- filters over pieces -----------------------------------------------------

@pytest.mark.parametrize("periodic", [None, BOX])
@pytest.mark.parametrize("how", [{"radius": 0.3, "min_neighbors": 5}, {"k": 8}])
def test_outliers_over_pieces(how, periodic):
    x, fields = clumps()
    expected = td.outliers(cloud(x, fields), periodic=periodic, **how)
    out = td.outliers(_split(x, fields)[0], periodic=periodic, **how)
    assert not out[0].ghosts or not (out[0].ghosts.points.numpy() &
                                     Ghosts.DUPLICATEPOINT).any()
    name = "neighbors" if "radius" in how else "mean_distance"
    if periodic is None or "radius" in how:
        np.testing.assert_array_equal(joined(out, name), expected.fields[name].values.numpy())
    else:
        np.testing.assert_allclose(joined(out, name), expected.fields[name].values.numpy(),
                                   rtol=1e-6)
    marks = expected.fields["outlier"].values.numpy()
    np.testing.assert_array_equal(joined(out, "outlier"), marks)
    assert 0 < marks.sum() < len(marks)


def _labels(out, rows, count):
    labels = np.empty(count, np.int64)
    for piece, r in zip(out, rows):
        labels[r] = piece.fields["point_region"].values.numpy()
    return labels


@pytest.mark.parametrize("periodic", [None, BOX])
@pytest.mark.parametrize("least", [None, 20])
def test_clusters_over_pieces_number_regions_as_the_whole(least, periodic):
    x, fields = clumps(seed=5)
    expected = td.connected_components(cloud(x, fields), radius=0.3, min_size=least,
                                       periodic=periodic).fields["point_region"].values.numpy()
    pieces, rows = _split(x, fields, count=5)
    got = _labels(td.connected_components(pieces, radius=0.3, min_size=least,
                                          periodic=periodic), rows, len(x))
    np.testing.assert_array_equal(got, expected)
    # Cut by position: the same partition, numbered by (piece, point).
    pieces, rows = _split(x, fields, count=5, order=by_x(x))
    got = _labels(td.connected_components(pieces, radius=0.3, min_size=least,
                                          periodic=periodic), rows, len(x))
    pairs = set(zip(expected.tolist(), got.tolist()))
    assert len(pairs) == len(set(expected.tolist())) == len(set(got.tolist()))
    np.testing.assert_array_equal(got < 0, expected < 0)
    # With global ids, by the lowest id: the whole's numbers.
    for piece, r in zip(pieces, rows):
        piece.global_ids = td.GlobalIds(cells=r, points=r)
    got = _labels(td.connected_components(pieces, radius=0.3, min_size=least,
                                          periodic=periodic), rows, len(x))
    np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("keep", ["centroid", "first", "nearest"])
def test_voxel_downsample_over_pieces(keep):
    x, fields = clumps()
    expected = td.voxel_downsample(cloud(x, fields), 0.7, keep=keep)
    for pieces in (_split(x, fields)[0], td.ghost_points(_split(x, fields)[0], radius=0.3)):
        out = td.voxel_downsample(pieces, 0.7, keep=keep)
        want, got = expected.positions(), np.concatenate([p.positions() for p in out])
        a, b = np.lexsort(want.T), np.lexsort(got.T)
        np.testing.assert_array_equal(got[b], want[a])
        for name in ("v", "cls"):
            np.testing.assert_array_equal(joined(out, name)[b],
                                          expected.fields[name].values.numpy()[a])
        assert all(p.ghosts is None or not (p.ghosts.points.numpy() &
                                            Ghosts.DUPLICATEPOINT).any() for p in out)


@pytest.mark.parametrize("method", ["nearest", "cic"])
def test_deposit_over_pieces(method):
    x, fields = clumps()
    whole = cloud(x, fields)
    grid = td.uniform_grid((11, 11, 6), origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 2.0))
    pieces = _split(x, fields, count=3)[0]
    cases = [(None, None), ("sum", "m"), ("mean", "m"), ("count", "m")] + \
        ([("min", "m"), ("max", "m")] if method == "nearest" else [])
    for reduce, field in cases:
        for per_volume in (False, True):
            expected = td.deposit(whole, grid, field, reduce, method, per_volume).values.numpy()
            for data in (pieces, td.ghost_points(pieces, radius=0.5)):
                got = td.deposit(data, grid, field, reduce, method, per_volume)
                assert got.space.topology is grid.topology
                got = got.values.numpy()
                assert got.dtype == expected.dtype
                np.testing.assert_allclose(got, expected, rtol=2e-6, equal_nan=True)


@pytest.mark.parametrize("periodic", [None, BOX])
def test_halo_catalog_over_pieces_is_the_wholes(periodic):
    x, fields = clumps(seed=9)
    whole = td.connected_components(cloud(x, fields), radius=0.2, min_size=30,
                                    periodic=periodic)
    expected = td.halo_catalog(whole, mass="m", velocity="v", periodic=periodic)
    pieces, rows = _split(x, fields)
    found = td.connected_components(pieces, radius=0.2, min_size=30, periodic=periodic)
    out = td.halo_catalog(found, mass="m", velocity="v", periodic=periodic)
    assert len(out) == len(pieces) and sum(p.num_points for p in out) == expected.num_points
    for name in expected.fields:
        if name != "mbp":
            np.testing.assert_array_equal(joined(out, name), expected.fields[name].values.numpy())
    starts = np.concatenate(([0], np.cumsum([len(r) for r in rows])))
    np.testing.assert_array_equal(starts[joined(out, "mbp_piece")] + joined(out, "mbp"),
                                  expected.fields["mbp"].values.numpy())
    np.testing.assert_array_equal(np.concatenate([p.positions() for p in out]),
                                  expected.positions())
    with pytest.raises(TypeError, match="by name"):
        td.halo_catalog(found, mass=found[0].fields["m"])


def test_point_pieces_on_two_devices():
    x, fields = clumps(seed=2, count=12)
    second = dispatch.as_device(_SecondCPU())
    pieces, rows = _split(x, fields)
    moved = td.PartitionedDataSet([p.to_device(second) if g % 2 else p
                                   for g, p in enumerate(pieces)])
    ghosted = td.ghost_points(moved, k=6)
    assert [p.device for p in ghosted] == [p.device for p in moved]
    np.testing.assert_array_equal(
        _labels(td.connected_components(moved, radius=0.3, min_size=10), rows, len(x)),
        td.connected_components(cloud(x, fields), radius=0.3, min_size=10)
        .fields["point_region"].values.numpy())
    for out in (td.outliers(moved, k=6), td.voxel_downsample(moved, 1.0),
                td.halo_catalog(td.connected_components(moved, radius=0.3, min_size=10),
                                mass="m")):
        assert [p.device for p in out] == [p.device for p in moved]
    grid = td.uniform_grid((6, 6, 6), origin=(0.0, 0.0, 0.0), spacing=(2.0, 2.0, 2.0))
    np.testing.assert_array_equal(td.deposit(moved, grid).values.numpy(),
                                  td.deposit(cloud(x, fields), grid).values.numpy())


class _NoF64CPU(CPUBackend):
    """A CPU backend posing as a device without f64, as Metal is."""

    display_name = "device without f64"

    def __init__(self):
        super().__init__()
        self.supported_dtypes = frozenset(d for d in self.supported_dtypes
                                          if d is not tack.f64)

    @property
    def memory_domain(self):
        return ("cpu", 2)


def test_point_pieces_on_a_device_without_f64():
    """Metal's round found ghost_points(k=) sending its distances as an f64 device
    array, which Metal cannot hold: they travel as f64 bits in u64."""
    x, fields = clumps(seed=2, count=12)
    device = dispatch.as_device(_NoF64CPU())
    pieces, _ = _split(x, fields)
    there = td.PartitionedDataSet([p.to_device(device) for p in pieces])
    for how in ({"k": 6}, {"radius": 0.4}):
        got, want = td.ghost_points(there, **how), td.ghost_points(pieces, **how)
        assert [p.num_points for p in got] == [p.num_points for p in want]
        assert all(p.device == device for p in got)
    np.testing.assert_array_equal(
        np.concatenate([p.fields["outlier"].values.numpy() for p in td.outliers(there, k=6)]),
        np.concatenate([p.fields["outlier"].values.numpy() for p in td.outliers(pieces, k=6)]))


def test_point_pieces_across_mpi_ranks(mpiexec):
    script = os.path.join(os.path.dirname(__file__), "_point_pieces_mpi.py")
    for ranks in (2, 3):
        run = mpiexec(ranks, script)
        assert run.returncode == 0 and "ok" in run.stdout, run.stdout + run.stderr


def across(world):
    """What rank 0 compares under MPI: this rank's share of five pieces' clusters,
    outlier marks, halo catalogue rows and kept voxels, and the deposit."""
    x, fields = clumps(seed=5)
    mine = np.array_split(np.arange(5), world.size)[world.rank]
    pieces, rows = _split(x, fields, count=5, world=world, mine=mine)
    clusters = td.connected_components(pieces, radius=0.3, min_size=20, periodic=BOX)
    catalog = td.halo_catalog(clusters, mass="m", velocity="v", periodic=BOX)
    marks = td.outliers(pieces, k=8)
    kept = td.voxel_downsample(pieces, 0.7, keep="first")
    ghosted = td.ghost_points(pieces, radius=0.6, periodic=BOX, fields=[])
    grid = td.uniform_grid((6, 6, 6), origin=(0.0, 0.0, 0.0), spacing=(2.0, 2.0, 2.0))
    return {
        "labels": {int(i): int(v) for r, p in zip(rows, clusters)
                   for i, v in zip(r, p.fields["point_region"].values.numpy())},
        "outlier": {int(i): int(v) for r, p in zip(rows, marks)
                    for i, v in zip(r, p.fields["outlier"].values.numpy())},
        "halos": [(int(h), int(c)) for p in catalog for h, c in
                  zip(p.fields["halo"].values.numpy(), p.fields["count"].values.numpy())],
        "kept": sorted(map(tuple, np.concatenate([p.positions() for p in kept]).tolist())),
        "copies": sum(p.num_points - len(r) for p, r in zip(ghosted, rows)),
        "deposit": td.deposit(pieces, grid, "m", "sum").values.numpy().tolist(),
    }


def whole_answers():
    """``across``'s answers on the whole, for one rank to compare."""
    x, fields = clumps(seed=5)
    whole = cloud(x, fields)
    clusters = td.connected_components(whole, radius=0.3, min_size=20, periodic=BOX)
    catalog = td.halo_catalog(clusters, mass="m", velocity="v", periodic=BOX)
    grid = td.uniform_grid((6, 6, 6), origin=(0.0, 0.0, 0.0), spacing=(2.0, 2.0, 2.0))
    copies = sum(p.num_points - len(r) for p, r in zip(
        td.ghost_points(_split(x, fields, count=5)[0], radius=0.6, periodic=BOX, fields=[]),
        _split(x, fields, count=5)[1]))
    return {
        "labels": dict(enumerate(clusters.fields["point_region"].values.numpy().tolist())),
        "outlier": dict(enumerate(td.outliers(whole, k=8).fields["outlier"].values.numpy()
                                  .tolist())),
        "halos": list(zip(catalog.fields["halo"].values.numpy().tolist(),
                          catalog.fields["count"].values.numpy().tolist())),
        "kept": sorted(map(tuple, td.voxel_downsample(whole, 0.7, keep="first").positions()
                           .tolist())),
        "copies": copies,
        "deposit": td.deposit(whole, grid, "m", "sum").values.numpy(),
    }
