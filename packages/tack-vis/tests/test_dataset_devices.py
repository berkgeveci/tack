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


#: Every public algorithm, as a call on a dataset; each must run on the
#: dataset's device and leave its results there.
CALLS = {
    "contour": lambda d: td.contour(d, "s", 3.0),
    "slice": lambda d: td.slice(d, td.Plane((0.5, 0, 0), (1, 0, 0))),
    "clip": lambda d: td.clip(d, "s", 3.0),
    "threshold": lambda d: td.threshold(d, "s", 0.0, 3.0),
    "extract_geometry": lambda d: td.extract_geometry(d, td.Sphere((0, 0, 0), 1.5)),
    "extract_cells": lambda d: td.extract_cells(d, [0, 1]),
    "mask": lambda d: td.mask(d, 2),
    "mask_points": lambda d: td.mask_points(d, 2),
    "threshold_points": lambda d: td.threshold_points(d, "s", 0.0, 3.0),
    "extract_points": lambda d: td.extract_points(d, td.Sphere((0, 0, 0), 1.5)),
    "extract_surface": lambda d: td.extract_surface(d, td.boundary_faces(d)),
    "external_faces": td.extract_surface,
    "boundary_faces": td.boundary_faces,
    "tetrahedralize": td.tetrahedralize,
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
    "face_geometry": lambda d: td.face_geometry(d).area,
    "flow_quantities": lambda d: td.flow_quantities(d, td.gradients(d, "v")).divergence,
    "upwind_flux": lambda d: td.upwind_flux(d, "s", (1.0, 0.5, 0.0)),
    "net_flux": lambda d: td.net_flux(d, td.face_geometry(d).normal),
    "divergence": lambda d: td.divergence(d, td.upwind_flux(d, "s", (1.0, 0.5, 0.0))),
    "perot": lambda d: td.perot(d, td.Field(td.Values(d, "faces"), tack.ones(
        (d.topology.faces().num_faces,), device=d.device))),
    "statistics": lambda d: td.statistics(d, "s"),
    "entropy": lambda d: td.entropy(d, "s"),
    "probe": lambda d: td.probe(d, _point(d.device)),
    "advect": lambda d: td.advect(d, "v", np.array([[0.5, 0.5, 0.5]], np.float32), 0.01, 3),
    "streamlines": lambda d: td.streamlines(d, "v", np.array([[0.5, 0.5, 0.5]], np.float32),
                                            0.01, 3),
}


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


@pytest.mark.parametrize("make", ["hexes", "grid"])
def test_derived_state_and_iteration_are_on_the_datasets_device(second, make):
    data = (_hexes if make == "hexes" else _grid)(second)
    faces = data.topology.faces()
    assert _on(second, faces) and _on(second, faces.boundary())
    assert _on(second, data.topology.edges())
    offsets, entries = data.topology.point_links()
    assert _on(second, offsets) and _on(second, entries)
    assert _on(second, td.Set.from_sides(data, [0], [0]))
    assert td.H1(data, order=2).size > data.num_points
    found = td.CellLocator(data).find(tack.array(np.array([[0.5, 0.5, 0.5]], np.float32),
                                                 tack.vec3(), device=second))
    assert all(_on(second, a) for a in found)


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
