"""Probing an AMR hierarchy (at the finest level holding each point; onto an image,
a resampling as VTK's ``vtkAMRResampleFilter``), the boxes on two devices, and the
hierarchy over 2 and 3 MPI ranks (``_amr_mpi.py``). Apart from ``test_amr.py``
since probe's kernels take a while to compile."""

import os

import numpy as np
import pytest
from test_amr import SPACING, _SecondCPU, hierarchy, needs_vtk

import tack
import tack.dataset as td
from tack.runtime import dispatch

try:
    import vtkmodules.vtkCommonDataModel  # noqa: F401  (registers the pythonic API)
    from vtkmodules.vtkFiltersAMR import vtkAMRGaussianPulseSource

    from tack.interop.vtk import vtk_to_dataset
except ImportError:
    vtkAMRGaussianPulseSource = None


@pytest.fixture(autouse=True)
def cpu():
    tack.set_default_device("cpu")
    yield
    tack.set_default_device("cpu")


def _finest_x(amr, x):
    """The cell field ``c`` (the centre's x) of the finest level's cell holding ``x``."""
    for level in range(amr.num_levels - 1, -1, -1):
        q = np.floor(x[:2] / SPACING[level] + 1e-9).astype(np.int64)
        for box in amr.boxes(level):
            if np.all(q >= box[0, :2]) and np.all(q < box[1, :2]):
                return (q[0] + 0.5) * SPACING[level]
    return None


@pytest.mark.parametrize("kind", ["image", "curvilinear"])
def test_probe_samples_the_finest_level(kind):
    amr = hierarchy(kind)
    rng = np.random.default_rng(0)
    # Random points, and points on a fine box's face with an unrefined coarse
    # cell beyond it, which go to the finer box (a point two cells of one level
    # share goes to either).
    points = np.column_stack([rng.uniform(0.0, 8.0, 500), rng.uniform(0.0, 8.0, 500),
                              np.zeros(500)])
    edges = np.array([[1.0, 1.3, 0.0], [1.7, 1.0, 0.0], [1.0, 6.2, 0.0], [1.5, 2.1, 0.0]])
    for where in (points, edges):
        out = td.probe(amr, where)
        assert (out.fields["valid"].values.numpy() == 1).all()
        np.testing.assert_allclose(out.fields["c"].values.numpy(),
                                   [_finest_x(amr, x) for x in where], atol=1e-6)
        np.testing.assert_allclose(out.fields["p"].values.numpy(), where[:, 1], atol=1e-5)


@needs_vtk
def test_a_probe_onto_an_image_resamples_as_vtk_does():
    """``vtkAMRResampleFilter`` samples the finest level's cells at an image's
    points; so does ``probe`` onto the image. VTK leaves points on the domain's
    upper faces unset (its values there are uninitialized), so those are left
    out of the comparison."""
    from vtkmodules.vtkFiltersAMR import vtkAMRResampleFilter

    source = vtkAMRGaussianPulseSource(dimension=3)
    source.Update()
    amr = vtk_to_dataset(source.GetOutput(), dtype=tack.f64)
    for lo, hi in (((-1.9, -1.9, 0.1), (0.4, 0.4, 2.4)), ((-2.0, -2.0, 0.0), (0.5, 0.5, 2.5))):
        resample = vtkAMRResampleFilter(number_of_samples=(10, 10, 10), min=lo, max=hi,
                                        demand_driven_mode=0, transfer_to_nodes=1)
        resample.SetInputConnection(source.GetOutputPort())
        resample.Update()
        block = resample.GetOutput().GetBlock(0)
        theirs = np.asarray(block.point_data["Gaussian-Pulse"])
        image = td.uniform_grid((10, 10, 10), origin=lo, spacing=tuple(block.spacing),
                                dtype=tack.f64)
        ours = td.probe(amr, image).fields["Gaussian-Pulse"].values.numpy()
        x = image.positions()
        kept = ~np.any(np.isclose(x, hi), axis=1)
        np.testing.assert_array_equal(ours[kept], theirs[kept])


def test_ghosts_and_probes_on_two_devices():
    second = dispatch.as_device(_SecondCPU())
    amr = hierarchy()
    moved = td.AMRDataSet([[box.to_device(second) if k % 2 else box
                            for k, box in enumerate(level)] for level in
                           (list(lv) for lv in amr.levels)], amr.ratios)
    for (_, a), (_, b) in zip(td.ghost_cells(moved), td.ghost_cells(amr)):
        assert a.device == a.fields["c"].values.device
        for name in ("c", "p"):
            np.testing.assert_array_equal(a.fields[name].values.numpy(),
                                          b.fields[name].values.numpy())
    points = np.column_stack([np.linspace(0.1, 7.9, 50), np.linspace(7.9, 0.1, 50),
                              np.zeros(50)])
    np.testing.assert_array_equal(td.probe(moved, points).fields["c"].values.numpy(),
                                  td.probe(amr, points).fields["c"].values.numpy())


def test_amr_across_mpi_ranks(mpiexec):
    script = os.path.join(os.path.dirname(__file__), "_amr_mpi.py")
    for ranks in (2, 3):
        run = mpiexec(ranks, script)
        assert run.returncode == 0 and "ok" in run.stdout, run.stdout + run.stderr
