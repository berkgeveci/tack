"""``voxelize``: data onto an image's voxels, the fraction each voxel the data covers
and each field's mean over that part, by sampling. Checked where the answer is
exact (voxels aligned with the data, a linear field), against probe at the voxel
centres (one sample), and converging with more samples."""

import importlib

import numpy as np
import pytest

import tack
import tack.dataset as td

voxelize_module = importlib.import_module("tack.dataset.algorithms.voxelize")

pytestmark = pytest.mark.quick


@pytest.fixture(autouse=True)
def on_each_f64_backend(f64_backend):
    """Compared in f64, on every backend that has f64."""
    yield f64_backend
    tack.set_default_device("cpu")


def _box(dims=(5, 5, 5), spacing=0.25):
    """A unit cube of tetrahedra with a linear field and a cell field."""
    mesh = td.tetrahedralize(td.uniform_grid(dims, spacing=(spacing,) * 3, dtype=tack.f64,
                                             unit="m"))
    x = mesh.positions().astype(np.float64)
    mesh.fields["f"] = td.Field(td.H1(mesh), tack.array(x @ [1.0, 2.0, -0.5]), "K")
    mesh.fields["c"] = td.Field(td.Constant(mesh), tack.ones((mesh.num_cells,), tack.f64))
    return mesh


def _image(dims, origin, spacing, **kwargs):
    return td.uniform_grid(dims, origin=origin, spacing=spacing, dtype=tack.f64, unit="m",
                           **kwargs)


def _fraction(v):
    return v.fields["fraction"].values.numpy()


def test_aligned_voxels_are_wholly_in_or_out():
    v = td.voxelize(_box(), _image((7, 7, 7), (-0.25,) * 3, (0.25,) * 3))
    f = _fraction(v)
    assert set(np.unique(f)) == {0.0, 1.0}
    assert f.sum() * 0.25 ** 3 == pytest.approx(1.0, rel=1e-12)          # the cube
    hidden = (v.ghosts.cells.numpy() & td.Ghosts.HIDDENCELL) != 0
    np.testing.assert_array_equal(hidden, f == 0.0)
    # A linear field's mean over a voxel is its value at the voxel's centre.
    centres = td.cell_centers(v).values.numpy()
    inside = f == 1.0
    np.testing.assert_allclose(v.fields["f"].values.numpy()[inside],
                               centres[inside] @ [1.0, 2.0, -0.5], atol=1e-12)
    assert (v.fields["f"].values.numpy()[~inside] == 0.0).all()
    assert v.fields["f"].unit == "K" and v.fields["fraction"].unit == "dimensionless"


def test_voxels_across_the_boundary_are_partly_covered():
    """Voxels half outside the cube, an even number of samples: exactly half; a
    voxel at a corner, a quarter or an eighth."""
    v = td.voxelize(_box(), _image((6, 6, 6), (-0.125,) * 3, (0.25,) * 3), samples=4)
    f = _fraction(v).reshape(5, 5, 5)
    assert f[2, 2, 2] == 1.0
    assert f[0, 2, 2] == 0.5 and f[2, 0, 2] == 0.5
    assert f[0, 0, 2] == 0.25 and f[0, 0, 0] == 0.125
    assert f.sum() * 0.25 ** 3 == pytest.approx(1.0, rel=1e-12)


def test_more_samples_measure_a_curved_volume_better():
    mesh = _box((11, 11, 11), 0.1)
    x = mesh.positions().astype(np.float64)
    mesh.fields["r"] = td.Field(td.H1(mesh), tack.array(np.linalg.norm(x - 0.5, axis=1)))
    ball = td.clip(mesh, "r", 0.35, inside=True)
    exact = float(td.cell_geometry(ball).measure.values.numpy().sum())
    image = _image((7, 7, 7), (0.05, 0.05, 0.05), (0.15,) * 3)
    errors = [abs(_fraction(td.voxelize(ball, image, samples=k)).sum() * 0.15 ** 3 - exact)
              for k in (1, 4, 8)]
    assert errors[2] < errors[1] < errors[0] and errors[2] < 0.01 * exact


def test_one_sample_is_the_probe_at_the_voxel_centres():
    mesh = _box()
    image = _image((5, 4, 4), (0.1, 0.05, 0.0), (0.2, 0.3, 0.3))
    v = td.voxelize(mesh, image, samples=1)
    probed = td.probe(mesh, td.cell_centers(image).values.numpy())
    np.testing.assert_array_equal(_fraction(v), probed.fields["valid"].values.numpy())
    np.testing.assert_allclose(v.fields["f"].values.numpy(), probed.fields["f"].values.numpy(),
                               atol=1e-12)


def test_any_data_probe_takes():
    """Pieces as the whole; an AMR hierarchy and an octree as their probes at the
    voxel centres."""
    mesh = _box()
    image = _image((6, 6, 6), (-0.125,) * 3, (0.25,) * 3)
    whole = td.voxelize(mesh, image)
    pieces = td.voxelize(td.PartitionedDataSet([td.tetrahedralize(p) for p in td.split(
        td.uniform_grid((5, 5, 5), spacing=(0.25,) * 3, dtype=tack.f64, unit="m"),
        (2, 2, 1))]), image)
    np.testing.assert_allclose(_fraction(pieces), _fraction(whole), atol=1e-12)
    pulse = td.amr_gaussian_pulse(dimension=3, dtype=tack.f64)
    grid = td.uniform_grid((6, 6, 6), origin=(-1.9, -1.9, 0.1), spacing=(0.4,) * 3,
                           dtype=tack.f64)
    amr = td.voxelize(pulse, grid, samples=1)
    want = td.probe(pulse, td.cell_centers(grid).values.numpy())
    np.testing.assert_allclose(amr.fields["Gaussian-Pulse"].values.numpy(),
                               want.fields["Gaussian-Pulse"].values.numpy(), rtol=1e-12)
    import test_octrees

    roots, level, centre, rho, temperature = test_octrees.ramses_like(3)
    octree = td.OctreeDataSet.from_cells(roots, level, centre, fields={"rho": rho})
    voxels = td.uniform_grid((5, 5, 5), spacing=(0.5,) * 3, dtype=tack.f64)
    got = td.voxelize(octree, voxels, samples=1)
    want = td.probe(octree, td.cell_centers(voxels).values.numpy())
    np.testing.assert_array_equal(got.fields["rho"].values.numpy(),
                                  want.fields["rho"].values.numpy())
    assert (_fraction(got) == 1.0).all()


def test_2d_and_oriented_images():
    flat = td.uniform_grid((5, 5), spacing=(0.25, 0.25), dtype=tack.f64)
    flat.fields["f"] = td.Field(td.H1(flat), tack.array(flat.positions()[:, 0]))
    v = td.voxelize(flat, td.uniform_grid((3, 3), origin=(-0.25, -0.25), spacing=(0.5, 0.5),
                                          dtype=tack.f64), samples=4)
    np.testing.assert_allclose(_fraction(v), [0.25, 0.5, 0.5, 1.0])       # 2 x 2 voxels
    c, s = np.cos(0.4), np.sin(0.4)
    turned = _image((4, 4, 4), (0.5, 0.0, 0.1), (0.2,) * 3,
                    direction=[[c, -s, 0], [s, c, 0], [0, 0, 1]])
    mesh = _box()
    got = td.voxelize(mesh, turned, samples=1)
    want = td.probe(mesh, td.cell_centers(turned).values.numpy())
    np.testing.assert_array_equal(_fraction(got), want.fields["valid"].values.numpy())


def test_batches_give_the_same_voxels(monkeypatch):
    mesh = _box()
    image = _image((6, 6, 6), (-0.125,) * 3, (0.25,) * 3)
    whole = td.voxelize(mesh, image)
    monkeypatch.setattr(voxelize_module, "_BATCH", 64 * 7)             # 7 voxels a batch
    batched = td.voxelize(mesh, image)
    np.testing.assert_array_equal(_fraction(batched), _fraction(whole))
    np.testing.assert_array_equal(batched.fields["f"].values.numpy(),
                                  whole.fields["f"].values.numpy())


def test_refusals():
    mesh = _box()
    with pytest.raises(TypeError, match="image's voxels"):
        td.voxelize(mesh, td.tetrahedralize(td.uniform_grid((3, 3, 3))))
    with pytest.raises(ValueError, match="at least 1"):
        td.voxelize(mesh, _image((3, 3, 3), (0, 0, 0), (0.5,) * 3), samples=0)
    with pytest.raises(ValueError, match="'m' and 'ft'|'ft' and 'm'"):
        td.voxelize(mesh, td.uniform_grid((3, 3, 3), unit="ft"))
