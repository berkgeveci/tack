"""PyVista's example datasets (``_pyvista_datasets``): real meshes as VTK's readers
give them -- quadratic cells, hidden cells, boundary patches merged with the
volume, surfaces with lines, point clouds of 100k points, integer fields.
Each converts to a dataset or is refused for a stated reason, and every
algorithm of the device tests runs on each one, to an answer or a stated
refusal. Known failures are strict ``xfail``s naming what is wrong, so a fix
shows. Skipped without the files: ``python _pyvista_datasets.py`` fetches
them into ``TACK_PYVISTA_DATA``."""

import functools
import re
import warnings

import _pyvista_datasets as pv
import numpy as np
import pytest
import test_dataset_devices
import test_viskores_datasets

import tack
import tack.dataset as td

try:
    import vtkmodules.vtkCommonDataModel  # noqa: F401
except ImportError:
    have_vtk = False
else:
    have_vtk = True

pytestmark = pytest.mark.skipif(not have_vtk, reason="needs VTK, which reads the files")


@pytest.fixture(autouse=True)
def on_each_f64_backend(f64_backend):
    """On every backend that has f64: the scaled positions and ``s`` are f64."""
    yield f64_backend
    tack.set_default_device("cpu")


#: Refused when read, as they should be.
_REFUSED = {
    "unstructured_grid": "may hold only triangles, quads and polygons",   # polygons and more
    "prostar": "is not 3D: a polyhedral topology",                       # polyhedra and lines
}

_QUADRATIC = ("vtk_to_dataset keeps VTK's quadratic cells (types 21-29), which its "
              "docstring refuses and no algorithm takes ('are not linear shapes')")

#: Read wrongly: no algorithm runs on these.
_NOT_READ = {
    "aero_bracket": _QUADRATIC,
    "fea_bracket": _QUADRATIC,
    "pump_bracket": _QUADRATIC,
    "notch_displacement": _QUADRATIC,
    "disc_quads": _QUADRATIC,
    "quadratic_pyramid": _QUADRATIC,
    "tri_quadratic_hexahedron": _QUADRATIC,
}

#: Refusals of an algorithm beyond the Viskores datasets' (their ``_REFUSALS``).
_REFUSALS = re.compile("|".join([
    test_viskores_datasets._REFUSALS.pattern,
    "subdivision takes cells of one dimension",  # hexahedra and their boundary quads
]))

#: Refusals of one dataset's own: its voxels meet three at nine faces.
_REFUSALS_OF = {"blood_vessels": "faces are shared by more than two cells"}

_MIXED = ("a surface with lines: the face kernels resolve methods line cells lack "
          "('side_position', 'num_faces')")
_VERTICES = "vertices alone: the face kernels resolve 'num_faces', which vertices lack"
_HIDDEN = ("faces are derived over hidden cells, and the reservoir's inactive cells "
           "overlap ('969 faces are shared by more than two cells')")

#: Known failures, by (dataset, call).
_BUGS = {
    **{("cylinder_crossflow", call): _MIXED
       for call in ("surface_normals", "face_geometry", "upwind_flux", "net_flux",
                    "divergence", "perot")},
    **{(name, call): _VERTICES
       for name in ("particles_lethe", "stars_cloud_hyg")
       for call in ("net_flux", "divergence", "perot")},
    **{("reservoir", call): _HIDDEN
       for call in ("clip_polyhedra", "extract_surface", "boundary_faces", "feature_edges",
                    "as_polyhedra", "interpolate", "interpolate of order 4", "subdivide",
                    "contour of order 3", "contour by tolerance", "traces", "jump",
                    "face_centers", "face_geometry", "upwind_flux", "net_flux", "divergence",
                    "perot")},
}


def _box(d):
    return test_dataset_devices._box(d)


#: The calls whose cost grows with the points within a radius, with radii in
#: the dataset's point spacing ``h`` rather than the unit cube's: at the device
#: tests' radius of 1, 100k points in a box of 2 would be billions of pairs.
_BY_SPACING = {
    "outliers_radius": lambda d, h: td.outliers(d, radius=2 * h, min_neighbors=3),
    "point_normals_radius": lambda d, h: td.point_normals(d, radius=2 * h),
    "connected_components_radius": lambda d, h: td.connected_components(d, radius=h,
                                                                        min_size=2),
    "sph_interpolate": lambda d, h: td.sph_interpolate(d, d, ["s", "v"], smoothing_length=1.5 * h,
                                                       mass=1.0, density=1.0),
    "sph_density": lambda d, h: td.sph_density(d, smoothing_length=1.5 * h, mass=1.0),
    "connected_components_periodic": lambda d, h: td.connected_components(d, radius=h,
                                                                          periodic=_box(d)),
    "halo_catalog": lambda d, h: td.halo_catalog(td.connected_components(d, radius=0.6 * h),
                                                 velocity="v"),
    "halo_catalog_periodic": lambda d, h: td.halo_catalog(
        td.connected_components(d, radius=0.6 * h, periodic=_box(d)), center="com",
        periodic=_box(d)),
}


def _converted(name):
    if not pv.present(name):
        pytest.skip(f"{pv.path(name)} is missing: python _pyvista_datasets.py "
                    "(or set TACK_PYVISTA_DATA)")
    from tack.interop.vtk import vtk_to_dataset

    with warnings.catch_warnings():              # OpenFOAM's point and cell arrays share names
        warnings.filterwarnings("ignore", "VTK has a point array and a cell array named")
        return vtk_to_dataset(pv.read(name))


@functools.cache
def _prepared(name, device):
    """Dataset ``name`` on ``device``, scaled into [0, 2]^3 (the device tests'
    calls are placed for it), with ``s`` (the squared distance from the
    origin), ``v`` (the position) and a cell set ``corner``; and its point
    spacing."""
    data = _converted(name)
    x = data.positions().astype(np.float64) + data.origin
    low, high = x.min(0), x.max(0)
    scale = 2.0 / (high - low).max()
    matrix = np.eye(4)
    matrix[:3, :3] *= scale
    matrix[:3, 3] = -low * scale
    data = td.transform(data, matrix)
    x = data.positions().astype(np.float64) + data.origin
    extent = np.ptp(x, 0)
    extent = extent[extent > 1e-9]
    spacing = float((extent.prod() / data.num_points) ** (1 / len(extent)))
    data.fields["s"] = td.Field(td.H1(data), tack.array((x ** 2).sum(1)))
    data.fields["v"] = td.Field(td.H1(data), tack.array(x, tack.vec3(tack.f64)))
    data.sets["corner"] = td.Set(data, "cells", [0])
    return data, spacing


def _marked(names):
    return [pytest.param(name, marks=pytest.mark.xfail(reason=_NOT_READ[name], strict=True))
            if name in _NOT_READ else name for name in names]


@pytest.mark.parametrize("name", _marked(pv.DATASETS))
def test_each_dataset_is_read_or_refused(name):
    """Every file converts to a dataset of the cells Tack takes, or is refused
    saying why."""
    if name in _REFUSED:
        with pytest.raises(ValueError, match=_REFUSED[name]):
            _converted(name)
        return
    data = _converted(name)
    assert data.num_points > 0
    if isinstance(data.topology, td.UnstructuredTopology):
        types = np.unique(data.topology.types.numpy())
        assert types.max() <= 14, types          # linear shapes


def _cases():
    for name in pv.DATASETS:
        if name in _REFUSED or name in _NOT_READ:
            continue
        for call in test_dataset_devices.CALLS:
            reason = _BUGS.get((name, call))
            yield pytest.param(name, call, id=f"{name}-{call}", marks=[] if reason is None else
                               pytest.mark.xfail(reason=reason, strict=True))


@pytest.mark.parametrize("name, call", _cases())
def test_every_algorithm_answers_or_refuses(name, call, on_each_f64_backend):
    """Each algorithm of the device tests on each dataset: an answer, or one of
    the stated refusals."""
    data, spacing = _prepared(name, on_each_f64_backend)
    try:
        if call in _BY_SPACING:
            _BY_SPACING[call](data, spacing)
        else:
            test_dataset_devices.CALLS[call](data)
    except Exception as e:                       # a refusal, or a failure
        refused = _REFUSALS.search(str(e)) or (
            name in _REFUSALS_OF and re.search(_REFUSALS_OF[name], str(e)))
        if not refused:
            raise
