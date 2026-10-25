"""Cosmology halos: friends-of-friends by ``connected_components(radius=, periodic=)``
and ``halo_catalog``, ported from Viskores' cosmotools. Against Viskores' own
test cases (``UnitTestCosmoTools.cxx``) and brute force."""

import sys

import numpy as np
import pytest

import tack
import tack.dataset as td

pytestmark = pytest.mark.quick

#: The module, not the function of the same name ``tack.dataset.algorithms`` exports.
catalog_module = sys.modules["tack.dataset.algorithms.halo_catalog"]


@pytest.fixture(autouse=True)
def cpu():
    tack.set_default_device("cpu")
    yield
    tack.set_default_device("cpu")


def _halos(x, linking, size, periodic=None, **catalog):
    cloud = td.point_cloud(np.asarray(x, np.float32))
    fof = td.connected_components(cloud, radius=linking, min_size=size, periodic=periodic)
    return fof, td.halo_catalog(fof, periodic=periodic, **catalog)


def _partition(labels):
    """Each particle's group as the lowest index in it (Viskores' halo ids)."""
    out = np.full(len(labels), -1)
    for h in np.unique(labels[labels >= 0]):
        members = np.flatnonzero(labels == h)
        out[members] = members.min()
    return out


# ── Viskores' cases ─────────────────────────────────────────────────

def test_viskores_2d_halos():
    x = np.array([[1, 1], [1, 2], [2, 6], [1, 3], [3, 5], [1, 4], [1, 5], [3, 6], [2, 3],
                  [3, 3], [4, 3], [3, 7], [5, 2], [5, 3], [4, 6], [5, 4], [6, 3]], float)
    fof, catalog = _halos(np.column_stack([x, np.zeros(len(x))]), 1.0, 3)
    halo = fof.fields["point_region"].values.numpy()
    assert _partition(halo).tolist() == [0, 0, 2, 0, 2, 0, 0, 2, 0, 0, 0, 2, 0, 0, 2, 0, 0]
    mbp = dict(zip(catalog.fields["halo"].values.numpy().tolist(),
                   catalog.fields["mbp"].values.numpy().tolist()))
    assert [mbp[h] for h in halo] == [8, 8, 7, 8, 7, 8, 8, 7, 8, 8, 8, 7, 8, 8, 7, 8, 8]


def test_viskores_3d_halos():
    x = np.array([[20.8125, 10.8864, 0.309784], [29.6871, 15.4445, 1.55953],
                  [29.724, 15.4766, 1.51077], [29.6783, 15.4766, 1.5324],
                  [29.7051, 15.5052, 1.52008], [20.8172, 10.8534, 0.23461],
                  [20.8665, 10.8679, 0.254398], [20.8271, 10.8677, 0.234255],
                  [20.8592, 10.9505, 0.248716], [20.819, 10.8949, 0.304834],
                  [29.708, 15.4251, 1.53951], [20.8829, 10.9144, 0.261517],
                  [20.8379, 10.877, 0.27677], [29.7278, 15.5267, 1.50798]])
    fof, catalog = _halos(x, 0.2, 3)
    halo = fof.fields["point_region"].values.numpy()
    assert _partition(halo).tolist() == [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1]
    mbp = dict(zip(catalog.fields["halo"].values.numpy().tolist(),
                   catalog.fields["mbp"].values.numpy().tolist()))
    assert [mbp[h] for h in halo] == [9, 4, 4, 4, 4, 9, 9, 9, 9, 9, 4, 9, 9, 4]


# ── Against brute force ─────────────────────────────────────────────

BOX = 20.0


def _universe(seed=5):
    rng = np.random.default_rng(seed)
    centres = rng.random((15, 3)) * BOX
    centres[0] = [0.1, 0.1, 19.9]                       # across a corner of the box
    x = np.vstack([(c + rng.normal(0, 0.4, (rng.integers(30, 400), 3))) % BOX
                   for c in centres])
    x = np.vstack([x, rng.random((40, 3)) * BOX]).astype(np.float32)
    mass = (0.5 + rng.random(len(x))).astype(np.float32)
    velocity = rng.normal(0, 1, (len(x), 3)).astype(np.float32)
    return x, mass, velocity


def _expected(x, mass, velocity, labels, softening, center="mbp"):
    out = []
    for h in range(labels.max() + 1):
        idx = np.flatnonzero(labels == h)
        x0 = x[idx[0]].astype(np.float64)
        d = x[idx].astype(np.float64) - x0
        u = x0 + d - BOX * np.floor(d / BOX + 0.5)
        m = mass[idx].astype(np.float64)
        r = np.sqrt(((u[:, None] - u[None]) ** 2).sum(-1) + softening ** 2)
        np.fill_diagonal(r, np.inf)
        phi = -(m[None] / r).sum(axis=1)
        best = int(np.argmin(phi))
        com = (u * m[:, None]).sum(axis=0) / m.sum()
        vbar = (velocity[idx] * m[:, None]).sum(axis=0) / m.sum()
        spread = (m * ((velocity[idx] - vbar) ** 2).sum(axis=1)).sum()
        point = u[best] if center == "mbp" else com
        out.append({"count": len(idx), "mass": m.sum(), "com": com % BOX, "mbp": idx[best],
                    "potential": phi[best], "velocity": vbar,
                    "dispersion": np.sqrt(spread / (3 * m.sum())),
                    "radius": np.sqrt(((u - point) ** 2).sum(axis=1)).max(),
                    "point": point % BOX})
    return out


@pytest.mark.parametrize("exact", [2048, 16])        # every pair, and the pruned search
@pytest.mark.parametrize("center", ["mbp", "com"])
def test_catalog_is_brute_forces(monkeypatch, exact, center):
    monkeypatch.setattr(catalog_module, "_EXACT", exact)
    x, mass, velocity = _universe()
    cloud = td.point_cloud(x, fields={"m": mass, "v": velocity})
    fof = td.connected_components(cloud, radius=0.35, min_size=20, periodic=BOX)
    labels = fof.fields["point_region"].values.numpy()
    catalog = td.halo_catalog(fof, mass="m", velocity="v", softening=0.05, center=center,
                              periodic=BOX)
    expected = _expected(x, mass, velocity, labels, 0.05, center)
    got = {name: catalog.fields[name].values.numpy() for name in catalog.fields}
    assert catalog.num_points == len(expected) == labels.max() + 1
    assert got["halo"].tolist() == list(range(len(expected)))
    np.testing.assert_array_equal(got["count"], [e["count"] for e in expected])
    np.testing.assert_array_equal(got["mbp"], [e["mbp"] for e in expected])
    for name, tolerance in (("mass", 1e-5), ("potential", 1e-4), ("dispersion", 1e-4),
                            ("radius", 1e-4)):
        np.testing.assert_allclose(got[name], [e[name] for e in expected], rtol=tolerance)
    np.testing.assert_allclose(got["com"], [e["com"] for e in expected], atol=1e-3)
    np.testing.assert_allclose(got["velocity"], [e["velocity"] for e in expected], atol=1e-5)
    np.testing.assert_allclose(catalog.positions(), [e["point"] for e in expected], atol=1e-3)


def test_a_halo_across_a_corner_is_whole():
    x, _, _ = _universe()
    with_box, _ = _halos(x, 0.35, 20, periodic=BOX)
    without, _ = _halos(x, 0.35, 20)
    corner = np.all((x < 1.5) | (x > BOX - 1.5), axis=1)
    whole = with_box.fields["point_region"].values.numpy()
    split = without.fields["point_region"].values.numpy()
    assert len(np.unique(whole[corner & (whole >= 0)])) == 1    # a stray in no halo aside
    assert len(np.unique(split[corner & (split >= 0)])) > 1


def test_the_pruned_search_sums_few_particles(monkeypatch):
    """A concentrated halo of 40,000: the bounds leave a few percent to sum exactly,
    and the most bound particle is every pair's."""
    rng = np.random.default_rng(1)
    n = 40000
    r = rng.random(n) ** 2 * 3.0
    u = rng.normal(size=(n, 3))
    u /= np.linalg.norm(u, axis=1)[:, None]
    cloud = td.point_cloud((u * r[:, None] + 10).astype(np.float32),
                           fields={"halo": np.zeros(n, np.int32)})
    summed = []
    original = catalog_module._candidates

    def counted(u, m, first, count, lower, cutoff, g, eps2, phi):
        summed.append(int((lower.numpy() <= cutoff).sum()))
        return original(u, m, first, count, lower, cutoff, g, eps2, phi)

    pruned = td.halo_catalog(cloud, "halo", softening=0.01)
    monkeypatch.setattr(catalog_module, "_candidates", counted)
    td.halo_catalog(cloud, "halo", softening=0.01)
    monkeypatch.setattr(catalog_module, "_EXACT", n)
    exact = td.halo_catalog(cloud, "halo", softening=0.01)
    assert summed and summed[0] < 0.05 * n
    assert int(pruned.fields["mbp"].values.numpy()[0]) == int(exact.fields["mbp"].values
                                                              .numpy()[0])


def test_catalog_of_no_halos_and_of_unlabelled_particles():
    cloud = td.point_cloud(np.random.default_rng(0).random((50, 3)),
                           fields={"halo": np.full(50, -1, np.int32)})
    assert td.halo_catalog(cloud, "halo").num_points == 0
    labels = np.full(50, -1, np.int32)
    labels[10:20] = 7
    cloud.fields["halo"] = td.Field(td.H1(cloud), tack.array(labels))
    catalog = td.halo_catalog(cloud, "halo")
    assert catalog.fields["halo"].values.numpy().tolist() == [7]
    assert catalog.fields["count"].values.numpy().tolist() == [10]
    with pytest.raises(ValueError, match="center is one of"):
        td.halo_catalog(cloud, "halo", center="median")


# ── Periodic boxes in the locator ───────────────────────────────────

SIDES = np.array([10.0, 8.0, 6.0])


def _nearest_image(a, b):
    d = a.astype(np.float32) - b.astype(np.float32)
    d -= SIDES.astype(np.float32) * np.floor(d / SIDES.astype(np.float32) + 0.5)
    return (d * d).sum(axis=-1)


def test_periodic_neighbors_and_nearest_are_brute_forces():
    x = (np.random.default_rng(3).random((1500, 3)) * SIDES).astype(np.float32)
    locator = td.PointLocator(td.point_cloud(x), periodic=SIDES)
    offsets, ids = (a.numpy() for a in locator.neighbors(0.7))
    nearest, d2 = (a.numpy() for a in locator.nearest(6))
    for q in range(len(x)):
        d = _nearest_image(x, x[q])
        expected = np.flatnonzero(d <= 0.49)
        assert sorted(ids[offsets[q]:offsets[q + 1]].tolist()) == expected[expected != q].tolist()
        d[q] = np.inf
        np.testing.assert_allclose(np.sort(d2[q]), np.sort(d)[:6], rtol=1e-5, atol=1e-6)


def test_periodic_boxes_refuse_what_they_cannot_do():
    cloud = td.point_cloud(np.random.default_rng(0).random((20, 3)) * 6)
    locator = td.PointLocator(cloud, periodic=SIDES)
    with pytest.raises(ValueError, match="half the periodic box"):
        locator.neighbors(3.0)
    with pytest.raises(ValueError, match="half the periodic box"):
        td.connected_components(cloud, radius=3.5, periodic=SIDES)
    for bad in ((1, 2), ((0, 0, 0), (1, -1, 1)), (np.inf, 1, 1)):
        with pytest.raises(ValueError, match="periodic box"):
            td.PointLocator(cloud, periodic=bad)
    with pytest.raises(ValueError, match="periodic box"):
        td.outliers(cloud, radius=0.5, min_neighbors=1, locator=locator)   # no box given
    assert td.PointLocator(cloud, periodic=((1, 1, 1), (7, 9, 11))).periodic == (
        (1.0, 1.0, 1.0), (7.0, 9.0, 11.0))
