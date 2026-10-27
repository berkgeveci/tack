"""``merge_points``: a dataset's coincident points made one, as VTK's
vtkStaticCleanUnstructuredGrid and Viskores' CleanGrid merge them."""

import tack
from tack import arrays
from tack.algorithms.scan import exclusive_scan
from tack.dataset import ids
from tack.dataset.algorithms._common import _length
from tack.dataset.algorithms.connected_components import _hook_near, _jump
from tack.dataset.carry import Same, Take, carry
from tack.dataset.ghosts import visible
from tack.dataset.point_locator import cached_point_locator
from tack.dataset.spaces import H1
from tack.dataset.topology import StructuredTopology, UnstructuredTopology
from tack.runtime.dispatch import get_backend, on_inputs_device

__all__ = ["merge_points"]


@tack.kernel
def _roots(labels, flags):
    for p in range(flags.shape[0]):
        flags[p] = 1 if labels[p] == p else 0


@tack.kernel
def _numbered(labels, index, new_id, kept):
    # Each point's merged point is its root's number; each root is kept there.
    for p in range(labels.shape[0]):
        r = labels[p]
        new_id[p] = index[r]
        if r == p:
            kept[index[p]] = p


@tack.kernel
def _renumbered(connectivity, new_id, out):
    for e in range(out.shape[0]):
        out[e] = new_id[connectivity[e]]


@on_inputs_device
def merge_points(data, tolerance=0.0, fields=None):
    """``data`` with each set of coincident points made one: points at most
    ``tolerance`` apart (inclusive; 0, the default, merges only points of equal
    coordinates) join, and so do chains of them, so a set may span more than
    ``tolerance``. A set becomes the point of its smallest id, which keeps that
    point's position, point fields, ghost marks and global id; the merged points
    keep the order of those ids, and the cells are the same cells on them, a
    cell's corners that merged kept as they are. Cell fields are unchanged.

    For meshes written without merging their points (each piece of a
    simulation's output storing a point once for every cell or block that
    uses it), which nothing can stitch by matching points across cells or
    pieces until each point is one. ``tolerance`` is in the dataset's spatial
    unit, or a quantity (``"1e-9 m"``); it should be far below the distance
    between points that are not copies, since every point compares itself with
    every point that close. Takes an unstructured topology; a structured grid's
    points are its lattice, and polyhedral, polygonal and polyline topologies
    are not handled yet. Hidden cells are dropped first, as every filter that
    makes cells drops them."""
    tolerance = _length(data, tolerance, "merge_points's tolerance")
    if not tolerance >= 0:
        raise ValueError(f"tolerance must be at least zero, not {tolerance!r}")
    if isinstance(data.topology, StructuredTopology):
        raise TypeError("merge_points takes an unstructured topology: a structured "
                        "grid's points are its lattice")
    if not isinstance(data.topology, UnstructuredTopology):
        raise NotImplementedError(f"merge_points of a {type(data.topology).__name__} is "
                                  f"not supported yet")
    if not isinstance(data.geometry.space, H1) or data.geometry.space.order != 1:
        raise TypeError("merge_points needs an order-1 H1 geometry: one position per point")
    data = visible(data)
    topology = data.topology
    n = data.num_points
    dtype = ids.at_least(tack.i32, n + 1)
    if dtype not in get_backend().supported_atomic_dtypes:
        raise NotImplementedError(f"merging {n} points needs {dtype} atomics, which the "
                                  f"{get_backend().name} backend does not have")
    # Union-find over the points within the tolerance, through the locator's
    # bins: each pair hooks the larger root under the smaller (an atomic
    # minimum), until nothing changes, so a set's root is its smallest point
    # whatever the order of the hooks.
    labels = tack.arange(n, dtype=dtype)
    if n:
        grid = cached_point_locator(data).grid()
        every = tack.empty((n,), tack.u8)
        every.fill(1)
        changed = tack.zeros((1,), tack.i32)
        while True:
            changed.fill(0)
            _hook_near(grid, 0, every, float(tolerance), labels, changed)
            _jump(labels)
            if not int(changed.numpy()[0]):
                break
    flags = tack.empty((max(n, 1),), dtype)
    index = tack.empty((max(n, 1),), dtype)
    count = 0
    if n:
        _roots(labels, flags)
        count = int(exclusive_scan(flags, index, n))
    new_id = tack.empty((max(n, 1),), dtype)
    kept = tack.empty((max(count, 1),), dtype)
    if n:
        _numbered(labels, index, new_id, kept)
    length = arrays.size_of(topology.connectivity)
    connectivity = tack.empty((length,), data.id_dtype)
    if length:
        _renumbered(topology.connectivity, new_id, connectivity)
    merged = UnstructuredTopology(topology.types, topology.offsets, connectivity,
                                  num_points=count)
    kept = kept if count else tack.empty((0,), dtype)
    return carry(data, merged, points=Take(kept), cells=Same(), fields=fields)
