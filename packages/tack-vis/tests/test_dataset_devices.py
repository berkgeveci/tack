"""Datasets run where their arrays are.

A dataset is on the device its topology, geometry, fields and sets are on, and
every algorithm, derivation and iteration runs and allocates there
(`on_inputs_device`), whatever the default device. A second device is
simulated with a CPU backend of another memory domain, as tack-core's
``test_devices.py`` does, so these run anywhere; the default stays the CPU.
"""

import numpy as np
import pytest

import tack
import tack.dataset as td
from tack import arrays
from tack.runtime import dispatch
from tack.runtime.cpu import CPUBackend

pytestmark = pytest.mark.quick      # the inner-loop set; see CLAUDE.md, Tests


class _SecondCPU(CPUBackend):
    """A CPU backend posing as another device's: its arrays are not the default's."""

    display_name = "second device"

    @property
    def memory_domain(self):
        return ("cpu", 1)


@pytest.fixture
def second():
    tack.set_default_device("cpu")
    yield dispatch.as_device(_SecondCPU())
    tack.set_default_device("cpu")


class _NoF64CPU(CPUBackend):
    """A CPU backend posing as a device without f64, as Metal is: an f64 array
    reaching a kernel is refused, as it is there."""

    display_name = "device without f64"

    def __init__(self):
        super().__init__()
        self.supported_dtypes = frozenset(d for d in self.supported_dtypes
                                          if d is not tack.f64)

    @property
    def memory_domain(self):
        return ("cpu", 3)


@pytest.fixture
def no_f64():
    tack.set_default_device("cpu")
    yield dispatch.as_device(_NoF64CPU())
    tack.set_default_device("cpu")


def _on(device, values):
    """Whether ``values`` (an array, a dataset, a field or a set) is on ``device``."""
    dev = values._device
    return dev is not None and dev.memory_domain == device._backend.memory_domain


def _hexes(device, n=3):
    """An unstructured mesh of (n-1)^3 hexahedra, every array on ``device``, with a
    scalar point field ``s``, a vector point field ``v`` (the position) and a cell
    set ``corner``."""
    pts = np.array([[i, j, k] for k in range(n) for j in range(n) for i in range(n)],
                   np.float32)
    pid = lambda i, j, k: i + n * (j + n * k)
    conn = [pid(i + a, j + b, k + c) for k in range(n - 1) for j in range(n - 1)
            for i in range(n - 1) for a, b, c in ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                                                  (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))]
    cells = len(conn) // 8
    topology = td.UnstructuredTopology(
        tack.array(np.full(cells, 12, np.uint8), tack.u8, device=device),
        tack.array(np.arange(0, 8 * cells + 1, 8, dtype=np.int32), device=device),
        tack.array(np.array(conn, np.int32), device=device), num_points=len(pts))
    data = td.DataSet(topology, tack.array(pts, tack.vec3(), device=device))
    return _with_fields(data, device)


def _with_fields(data, device):
    x = data.positions()
    data.fields["s"] = td.Field(td.H1(data), tack.array((x ** 2).sum(1).astype(np.float32),
                                                        device=device))
    data.fields["v"] = td.Field(td.H1(data), tack.array(x.astype(np.float32), tack.vec3(),
                                                        device=device))
    data.sets["corner"] = td.Set(data, "cells", [0])
    return data


def _point(device):
    """One point, where ``probe`` samples, as a dataset on ``device``."""
    with device:
        return td.DataSet(td.UnstructuredTopology([1], [0, 1], [0], num_points=1),
                          np.array([[0.5, 0.5, 0.5]], np.float32))


def _grid(device):
    return _with_fields(td.uniform_grid((3, 3, 3), device=device), device)


@tack.func
def _norm2(x):
    return x.dot(x)


def _of_order_3(data):
    """``data`` with ``h``, its field ``s`` in ``H1`` of order 3."""
    data.fields["h"] = td.project(data, data.fields["s"], td.H1(data, order=3))
    return data


#: Every public algorithm, as a call on a dataset; each must run on the
#: dataset's device and leave its results there.
CALLS = {
    "contour": lambda d: td.contour(d, "s", 3.0),
    "contour_2d": lambda d: td.contour(td.extract_surface(d), "s", 3.0),
    "contour_1d": lambda d: td.contour(td.extract_edges(d), "s", 3.0),
    "slice": lambda d: td.slice(d, td.Plane((0.5, 0, 0), (1, 0, 0))),
    "clip": lambda d: td.clip(d, "s", 3.0),
    "clip_polyhedra": lambda d: td.clip(td.as_polyhedra(d), "s", 3.0),
    "clip_boolean": lambda d: td.clip(d, td.Difference(
        td.Sphere((1, 1, 1), 1.2), td.Planes([(1.5, 0, 0)], [(-1, 0.2, 0)]))),
    "threshold": lambda d: td.threshold(d, "s", 0.0, 3.0),
    "extract_geometry": lambda d: td.extract_geometry(d, td.Sphere((0, 0, 0), 1.5)),
    "extract_cells": lambda d: td.extract_cells(d, [0, 1]),
    "extract_extent": lambda d: (td.extract_extent(d, [(0, 1)] * len(d.topology.point_dims))
                                 if isinstance(d.topology, td.StructuredTopology) else None),
    "mask": lambda d: td.mask(d, 2),
    "mask_points": lambda d: td.mask_points(d, 2),
    "threshold_points": lambda d: td.threshold_points(d, "s", 0.0, 3.0),
    "extract_points": lambda d: td.extract_points(d, td.Sphere((0, 0, 0), 1.5)),
    "extract_surface": lambda d: td.extract_surface(d, td.boundary_faces(d)),
    "external_faces": td.extract_surface,
    "boundary_faces": td.boundary_faces,
    "extract_edges": td.extract_edges,
    "feature_edges": lambda d: td.feature_edges(td.extract_surface(d, td.boundary_faces(d))),
    "surface_normals": lambda d: td.surface_normals(td.extract_surface(d)),
    "tubes": lambda d: td.tubes(td.extract_edges(d), 0.05, radius_by="s"),
    "glyphs": lambda d: td.glyphs(d, "v", scale=0.1),
    "tetrahedralize": td.tetrahedralize,
    "merge_points": lambda d: (None if isinstance(d.topology, td.StructuredTopology)
                               else td.merge_points(d)),
    "shrink": td.shrink,
    "warp": lambda d: td.warp(d, (0, 0, 1), scale=0.5),
    "transform": lambda d: td.transform(d, np.eye(4)),
    "cylindrical": td.cylindrical,
    "spherical": td.spherical,
    "as_polyhedra": td.as_polyhedra,
    "cell_centers": td.cell_centers,
    "values_at_centers": lambda d: td.values_at_centers(d, "s"),
    "gradients": lambda d: td.gradients(d, "s"),
    "gradients_at_points": lambda d: td.gradients(d, "s", at="points"),
    "cell_ids": td.cell_ids,
    "point_ids": td.point_ids,
    "to_cells": lambda d: td.project(d, "s", td.Constant(d)),
    "to_points": lambda d: td.project(d, td.project(d, "s", td.Constant(d)), td.H1(d)),
    "discontinuous": lambda d: td.project(d, "s", td.L2(d)),
    "project": lambda d: td.project(d, "s", td.Constant(d)),
    "interpolate": lambda d: td.interpolate(d, _norm2, td.H1(d, order=2)),
    "interpolate of order 4": lambda d: td.interpolate(d, _norm2, td.H1(d, order=4)),
    "subdivide": lambda d: td.subdivide(d, 2),
    "contour of order 3": lambda d: td.contour(_of_order_3(d), "h", 3.0),
    "contour by tolerance": lambda d: td.contour(_of_order_3(d), "h", 3.0, tolerance=1e-2),
    "traces": lambda d: td.project(d, "s", td.SideTraces(d)),
    "jump": lambda d: td.jump(d, "s"),
    "face_centers": td.face_centers,
    "edge_lengths": td.edge_lengths,
    "magnitude": lambda d: td.magnitude(d, "v"),
    "log_values": lambda d: td.log_values(d, "s", min_value=1e-3),
    "dot": lambda d: td.dot(d, "v", "v"),
    "cross": lambda d: td.cross(d, "v", "v"),
    "composite": lambda d: td.composite(d, "s", "s", "s"),
    "implicit_values": lambda d: td.implicit_values(d, td.Sphere((0, 0, 0), 1.0)),
    "point_elevation": td.point_elevation,
    "cell_geometry": lambda d: td.cell_geometry(d).measure,
    "cell_quality": lambda d: td.cell_quality(d)["scaled_jacobian"],
    "face_geometry": lambda d: td.face_geometry(d).area,
    "flow_quantities": lambda d: td.flow_quantities(d, td.gradients(d, "v")).divergence,
    "upwind_flux": lambda d: td.upwind_flux(d, "s", (1.0, 0.5, 0.0)),
    "net_flux": lambda d: td.net_flux(d, td.face_geometry(d).normal),
    "divergence": lambda d: td.divergence(d, td.upwind_flux(d, "s", (1.0, 0.5, 0.0))),
    "perot": lambda d: td.perot(d, td.Field(td.Values(d, "faces"), tack.ones(
        (d.topology.faces().num_faces,), device=d.device))),
    "statistics": lambda d: td.statistics(d, "s"),
    "entropy": lambda d: td.entropy(d, "s"),
    "integrate": lambda d: td.integrate(d, ["s", "v"]),
    "probe": lambda d: td.probe(d, _point(d.device)),
    "advect": lambda d: td.advect(d, "v", np.array([[0.5, 0.5, 0.5]], np.float32), 0.01, 3),
    "streamlines": lambda d: td.streamlines(d, "v", np.array([[0.5, 0.5, 0.5]], np.float32),
                                            0.01, 3),
    "voxel_downsample": lambda d: td.voxel_downsample(d, 0.75),
    "voxel_downsample_nearest": lambda d: td.voxel_downsample(d, 0.75, keep="nearest"),
    "outliers_radius": lambda d: td.outliers(d, radius=1.0, min_neighbors=3),
    "outliers_statistical": lambda d: td.outliers(d, k=4),
    "point_normals": lambda d: td.point_normals(d, k=6),
    "point_normals_radius": lambda d: td.point_normals(d, radius=1.0),
    "connected_components_radius": lambda d: td.connected_components(d, radius=0.5, min_size=2),
    "deposit": lambda d: td.deposit(d, _image(d)),
    "voxelize": lambda d: td.voxelize(d, _image(d), samples=2),
    "deposit_cic": lambda d: td.deposit(d, _image(d), "s", "mean", method="cic"),
    "sph_interpolate": lambda d: td.sph_interpolate(d, d, ["s", "v"], smoothing_length=0.6,
                                                    mass=1.0, density=1.0),
    "sph_density": lambda d: td.sph_density(d, smoothing_length=0.6, mass=1.0),
    "connected_components_periodic": lambda d: td.connected_components(
        d, radius=0.6, periodic=_box(d)),
    "point_normals_periodic": lambda d: td.point_normals(d, k=6, periodic=_box(d)),
    "halo_catalog": lambda d: td.halo_catalog(td.connected_components(d, radius=1.1),
                                              velocity="v"),
    "halo_catalog_periodic": lambda d: td.halo_catalog(
        td.connected_components(d, radius=1.1, periodic=_box(d)), center="com",
        periodic=_box(d)),
}


def _box(data):
    """A periodic box about ``data``'s points, in world coordinates."""
    return (tuple(np.add((-0.5, -0.5, -0.5), data.origin)),
            tuple(np.add((2.5, 2.5, 2.5), data.origin)))


def _image(data):
    """An image over ``data``'s points, on its device and at its origin."""
    image = td.uniform_grid((4, 3, 3), spacing=(0.7, 1.0, 1.0), device=data.device)
    image.origin = data.origin
    return image


def _results(value):
    """The arrays, datasets, fields and sets in an algorithm's result."""
    if isinstance(value, (td.DataSet, td.Field, td.Set)):
        return [value]
    if isinstance(value, (tuple, list)):
        return [r for v in value for r in _results(v)]
    return []


@pytest.mark.parametrize("name", CALLS)
@pytest.mark.parametrize("make", ["hexes", "grid"])
def test_every_algorithm_runs_on_the_datasets_device(second, name, make):
    """The default is the CPU; the dataset is on the second device. Before, the
    first launch met a temporary on the default device ("arrays on two devices")."""
    data = (_hexes if make == "hexes" else _grid)(second)
    result = CALLS[name](data)
    for part in _results(result):
        assert _on(second, part), f"{name} left {part!r} on another device"
    assert _on(second, data)          # nothing derived it elsewhere


@pytest.mark.parametrize("name", CALLS)
@pytest.mark.parametrize("make", ["hexes", "grid"])
def test_every_algorithm_runs_on_a_device_without_f64(no_f64, name, make):
    """As on Metal, everywhere: an algorithm must not make an f64 array for its
    kernels on a device that has none (Metal's round found one in ghost_points,
    and octrees' fields)."""
    data = (_hexes if make == "hexes" else _grid)(no_f64)
    for part in _results(CALLS[name](data)):
        assert _on(no_f64, part), f"{name} left {part!r} on another device"


@pytest.mark.parametrize("make", ["hexes", "grid"])
def test_derived_state_and_iteration_are_on_the_datasets_device(second, make):
    data = (_hexes if make == "hexes" else _grid)(second)
    faces = data.topology.faces()
    assert _on(second, faces) and _on(second, faces.boundary())
    assert _on(second, data.topology.edges())
    offsets, entries = data.topology.point_links()
    assert _on(second, offsets) and _on(second, entries)
    assert _on(second, td.Set.from_sides(data, [0], [0]))
    data.sets["edge"] = td.Set(data, "edges", [0, 3])
    assert _on(second, data.sets["edge"]) and _on(second, td.extract_edges(data, "edge"))
    assert td.H1(data, order=2).size > data.num_points
    found = td.CellLocator(data).find(tack.array(np.array([[0.5, 0.5, 0.5]], np.float32),
                                                 tack.vec3(), device=second))
    assert _on(second, found.cells) and _on(second, found.pcs) and found.tets is None


@pytest.mark.parametrize("make", ["hexes", "grid"])
def test_the_public_building_blocks_run_on_the_datasets_device(second, make):
    """What filters of one's own are written with (`visible`, `owned_values`,
    `carry`), on marked data, so each allocates, and `orient`: called directly,
    not from inside an algorithm that already made the dataset's device the
    default, they once allocated on the default device ("arrays on two
    devices")."""
    data = (_hexes if make == "hexes" else _grid)(second)
    cells = np.zeros(data.num_cells, np.uint8)
    cells[0] = td.Ghosts.HIDDENCELL
    cells[1] = td.Ghosts.DUPLICATECELL
    points = np.zeros(data.num_points, np.uint8)
    points[0] = td.Ghosts.DUPLICATEPOINT
    data.ghosts = td.Ghosts(cells=tack.array(cells, device=second),
                            points=tack.array(points, device=second))
    kept = td.visible(data)
    assert kept.num_cells == data.num_cells - 1 and _on(second, kept)
    assert _on(second, td.owned_values(data, "s"))
    every = tack.array(np.arange(data.num_cells, dtype=np.int32), device=second)
    assert _on(second, td.carry(data, data.topology, points=td.Same(), cells=td.Take(every)))
    polyhedra = td.as_polyhedra(data).topology
    assert td.orient(polyhedra).device == polyhedra.device == second
    # And what the built-ins call on marked data, outside an algorithm's scope.
    from tack.dataset.ghosts import refuse_hidden, without_hidden

    flags = tack.array(np.ones(data.num_cells, np.int32), device=second)
    assert _on(second, without_hidden(data, flags))
    with pytest.raises(NotImplementedError, match="hidden"):
        refuse_hidden(data, "this")


def test_a_dataset_refuses_parts_on_two_devices(second):
    data = _hexes(second)
    with pytest.raises(ValueError, match="two devices"):
        td.Field(td.H1(data), tack.zeros((data.num_points,)))
    data.fields["elsewhere"] = td.Field.__new__(td.Field)
    data.fields["elsewhere"].space, data.fields["elsewhere"].values = (
        td.H1(data), tack.zeros((data.num_points,)))
    with pytest.raises(ValueError, match="two devices"):
        td.contour(data, "s", 3.0)


@pytest.mark.parametrize("make", ["hexes", "grid", "polyhedra"])
def test_to_device_moves_everything_and_computes_the_same(second, make):
    here = tack.device("cpu")
    data = {"hexes": lambda: _hexes(here), "grid": lambda: _grid(here),
            "polyhedra": lambda: td.as_polyhedra(_hexes(here))}[make]()
    moved = data.to_device(second)
    assert moved is not data and _on(second, moved) and moved.device == second
    assert all(_on(second, f) for f in moved.fields.values())
    assert all(_on(second, s) for s in moved.sets.values())
    assert set(moved.fields) == set(data.fields) and set(moved.sets) == set(data.sets)
    np.testing.assert_array_equal(moved.positions(), data.positions())
    a = td.cell_centers(data) if make != "polyhedra" else td.cell_geometry(data).centroid
    b = td.cell_centers(moved) if make != "polyhedra" else td.cell_geometry(moved).centroid
    np.testing.assert_allclose(arrays.to_host(b.values), arrays.to_host(a.values))
    assert moved.to_device(second) is moved and data.to_device(here) is data


def test_sources_take_a_device(second):
    for data in (td.uniform_grid((3, 3, 3), device=second),
                 td.rectilinear_grid([0, 1, 2], [0, 1], [0, 1], device=second),
                 td.wavelet(2, device=second), td.tangle((3, 3, 3), device=second)):
        assert _on(second, data) and data.device == second
        assert _on(second, td.contour(data, data.fields.get("RTData") or
                                      data.fields.get("tangle") or td.point_elevation(data),
                                      0.5))
    axes = tack.array(np.array([0.0, 1.0, 2.0], np.float32), device=second)
    assert _on(second, td.rectilinear_grid(axes, [0, 1], [0, 1]))   # the axes' device


def test_a_cell_locators_tables_are_on_its_datasets_device(second):
    """Found on cuda (2026-10-28): called directly, not from a search or an
    algorithm that made the dataset's device the default, ``cell_bins``,
    ``leaves`` and ``cell_neighbors`` built their tables on the default device
    and met the dataset's arrays on another ("arrays on two devices")."""
    data = _hexes(second)
    for name in ("cell_bins", "leaves", "cell_neighbors"):
        built = getattr(td.CellLocator(data), name)()
        assert all(_on(second, a) for a in (built if isinstance(built, tuple) else (built,)))


def test_a_point_locator_searches_on_its_datasets_device(second):
    """Found on every GPU (2026-10-25): with no Tack array among its arguments (its
    own points as the queries, or host positions), a search ran on the default
    device and met the locator's bins on another."""
    x = np.random.default_rng(0).random((500, 3)).astype(np.float32)
    here = td.point_cloud(x, origin=(0.0, 0.0, 0.0))
    want, got = td.PointLocator(here), td.PointLocator(here.to_device(second))
    for points in (None, x[::5].astype(np.float64)):
        for found, expected in ((got.neighbors(0.1, points), want.neighbors(0.1, points)),
                                (got.nearest(4, points), want.nearest(4, points))):
            for a, b in zip(found, expected):
                assert _on(second, a)
                np.testing.assert_array_equal(a.numpy(), b.numpy())
