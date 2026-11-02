"""``merge_points``: a dataset's coincident points made one, as VTK's
vtkStaticCleanUnstructuredGrid and Viskores' CleanGrid merge them."""

import numpy as np

import tack
from tack import arrays
from tack.algorithms.scan import exclusive_scan
from tack.dataset import ids
from tack.dataset.algorithms._common import _length
from tack.dataset.algorithms.connected_components import _hook_near, _jump
from tack.dataset.carry import Same, Take, carry
from tack.dataset.ghosts import visible
from tack.dataset.point_locator import cached_point_locator
from tack.dataset.polyhedra import (
    PolygonalTopology,
    PolyhedralTopology,
    PolylineTopology,
    _match_rings,
    _ranges,
    _side_counts,
)
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


def _renumber(point_ids, new_id, dtype):
    out = tack.empty((max(arrays.size_of(point_ids), 1),), dtype)
    if arrays.size_of(point_ids):
        _renumbered(point_ids, new_id, out)
    return out if arrays.size_of(point_ids) else tack.empty((0,), dtype)


@tack.kernel
def _distinct_points(face_offsets, face_points, out):
    # How many different points each face has once merged: under three, it has no area.
    for f in range(out.shape[0]):
        n = 0
        for i in range(face_offsets[f], face_offsets[f + 1]):
            new = 1
            for j in range(face_offsets[f], i):
                if face_points[j] == face_points[i]:
                    new = 0
            n += new
        out[f] = n


def _merged_polyhedra(topology, new_id, count):
    """``topology`` on the merged points, and the faces it keeps: faces renumbered, then
    matched by point set (``_match_rings``, as ``from_cell_faces`` matches its rings),
    so the copies of a face that pieces each stored once, with one side, become one
    face with two. The first copy is the face, in its winding; a cell on a later copy,
    which runs the other way, sees the face from its other side. A face left with
    fewer than three different points has no area and is matched with nothing: it
    stays as it is, as an unstructured cell's merged corners do."""
    renumbered = _renumber(topology.face_points, new_id, topology.id_dtype)
    nf = topology.num_faces
    distinct = tack.empty((max(nf, 1),), tack.i32)
    if nf:
        _distinct_points(topology.face_offsets, renumbered, distinct)
    face_points = renumbered.numpy()
    face_offsets = arrays.to_host(topology.face_offsets).astype(np.int64)
    # Match the faces with an area among themselves; each face's representative is its
    # first copy, or itself.
    flat = np.flatnonzero(distinct.numpy()[:nf] >= 3)
    if flat.size == nf:                                   # the usual case: every face
        ring_offsets, ring_points = face_offsets, face_points
    else:
        sizes = np.diff(face_offsets)[flat]
        ring_offsets = np.concatenate([[0], np.cumsum(sizes)])
        ring_points = face_points[_ranges(face_offsets[flat], sizes)]
    _, _, ring_face, ring_copy, same_way = _match_rings(ring_offsets, ring_points)
    if same_way:
        raise ValueError(f"{same_way} faces become one with another that runs the same way: "
                         "both their cells hold the merged face on the same side")
    first = flat[np.flatnonzero(ring_copy == 0)]          # each matched face's first copy
    representative = np.arange(nf)
    representative[flat] = first[ring_face]
    copy = (representative != np.arange(nf)).astype(np.uint8)
    kept = np.flatnonzero(copy == 0)
    number = np.full(nf, -1, np.int64)
    number[kept] = np.arange(kept.size)
    face_of = number[representative]
    kept_sizes = np.diff(face_offsets)[kept]
    offsets = np.concatenate([[0], np.cumsum(kept_sizes)])
    points = face_points[_ranges(face_offsets[kept], kept_sizes)]
    cell_faces = arrays.to_host(topology.cell_faces)
    sides = arrays.to_host(topology.cell_face_sides) ^ copy[cell_faces]
    merged = PolyhedralTopology(offsets, points, topology.cell_offsets, face_of[cell_faces],
                                sides, num_points=count, id_dtype=topology.id_dtype)
    counts, _ = _side_counts(merged)
    crowded = int(np.count_nonzero(counts.numpy() > 1))
    if crowded:
        raise ValueError(f"{crowded} merged faces have more than one cell on a side: more "
                         "than two cells meet at them")
    return merged, Take(tack.array(kept.astype(merged.id_dtype.numpy_dtype)))


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
    every point that close. Takes an unstructured, polyhedral, polygonal or
    polyline topology; a structured grid's points are its lattice. A polyhedral
    topology's faces are matched again on the merged points: the copies of a face
    that pieces each stored once, with one cell, become one face with both
    (OpenFOAM's processor boundaries, written without merging), its first copy
    kept with its values on faces; copies wound the same way, or a face with
    more than two cells, are refused. Polygonal and polyline topologies derive
    their facets from the merged loops and lines. Hidden cells are dropped
    first, as every filter that makes cells drops them."""
    tolerance = _length(data, tolerance, "merge_points's tolerance")
    if not tolerance >= 0:
        raise ValueError(f"tolerance must be at least zero, not {tolerance!r}")
    if isinstance(data.topology, StructuredTopology):
        raise TypeError("merge_points takes an unstructured topology: a structured "
                        "grid's points are its lattice")
    if not isinstance(data.topology, (UnstructuredTopology, PolyhedralTopology,
                                      PolylineTopology)):
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
    faces = None
    if isinstance(topology, PolygonalTopology):
        merged = PolygonalTopology(topology.loop_offsets,
                                   _renumber(topology.loop_points, new_id, data.id_dtype),
                                   num_points=count, id_dtype=data.id_dtype)
    elif isinstance(topology, PolyhedralTopology):
        merged, faces = _merged_polyhedra(topology, new_id, count)
    elif isinstance(topology, PolylineTopology):
        merged = PolylineTopology(topology.line_offsets,
                                  _renumber(topology.line_points, new_id, data.id_dtype),
                                  num_points=count, id_dtype=data.id_dtype)
    else:
        merged = UnstructuredTopology(topology.types, topology.offsets,
                                      _renumber(topology.connectivity, new_id, data.id_dtype),
                                      num_points=count)
    kept = kept if count else tack.empty((0,), dtype)
    return carry(data, merged, points=Take(kept), cells=Same(), faces=faces, fields=fields)
