"""Octrees: a grid of trees whose nodes are Morton-sorted, as a linear octree --
VTK's ``vtkHyperTreeGrid`` with a branching factor of 2, laid out for GPUs.

A grid of root cells (an image or a rectilinear grid, 2D or 3D) holds a tree in
each cell. Every node of every tree is kept, refined ones included, as
HyperTreeGrid and cell-based AMR codes (RAMSES) store values on them. A node is
three numbers: its tree (the root cell's index), its level (0 at the root) and
its Morton key at that level, its integer coordinates in its tree interleaved
bit by bit, x lowest -- so a child's key is its parent's shifted by the
dimension with the child's index (x fastest, HyperTreeGrid's order) below. The
nodes are sorted in pre-order: by tree, then by key carried to the finest
level, then by level, so a parent comes before its children and a subtree is
one run. Whether a node is a leaf, and its bounds, centre and measure, come
from the keys in kernels; nothing about position is stored.

Nodes a leaf does not cover are ``REFINEDCELL`` and masked ones ``HIDDENCELL``
(``marks``), so a reduction counts each region once, over the leaves, by the
rules an ``AMRDataSet`` follows. Cell data only: ``fields`` holds one value per
node.
"""

import numpy as np

import tack
from tack import arrays
from tack.arrays import CartesianProduct, UniformCoordinates
from tack.dataset.ghosts import Ghosts

__all__ = ["OctreeDataSet"]

#: Levels a key holds: 3 bits a level in 3D, and 5 bits for the level beside it.
MAX_LEVEL = {2: 29, 3: 19}


@tack.kernel
def _node_geometry(tree, level, key, root_lo, root_size, dimension, lo, size):
    # Each node's lower corner and sides: its tree's root cell, divided by 2 per
    # level, at the integer coordinates its key interleaves.
    for n in range(lo.shape[0]):
        t = tree[n]
        depth = level[n]
        k = key[n]
        i = tack.u64(0)
        j = tack.u64(0)
        m = tack.u64(0)
        for b in range(depth):
            i = i | (((k >> tack.u64(dimension * b)) & tack.u64(1)) << tack.u64(b))
            j = j | (((k >> tack.u64(dimension * b + 1)) & tack.u64(1)) << tack.u64(b))
            if dimension == 3:
                m = m | (((k >> tack.u64(dimension * b + 2)) & tack.u64(1)) << tack.u64(b))
        scale = 1.0 / tack.f64(tack.u64(1) << tack.u64(depth))
        r = root_size[t]
        s = [r[0] * scale, r[1] * scale, r[2] * scale]
        if dimension == 2:
            s = [s[0], s[1], r[2]]
        base = root_lo[t]
        lo[n] = [base[0] + tack.f64(i) * s[0], base[1] + tack.f64(j) * s[1],
                 base[2] + tack.f64(m) * s[2]]
        size[n] = s


class OctreeDataSet:
    """A grid of octrees, Morton-sorted: VTK's ``vtkHyperTreeGrid`` with branching
    factor 2, as a linear octree.

    ``roots`` is the grid of root cells, a dataset of a ``StructuredTopology``
    over an image's or a rectilinear grid's coordinates, 2D or 3D
    (``uniform_grid``, ``rectilinear_grid``); its cell ``t`` (x fastest) is
    tree ``t``. ``tree``, ``level`` and ``key`` give each node: its tree, its
    level (0 at the root) and its Morton key at that level (integer coordinates
    in the tree interleaved, x lowest). Every node's ancestors must be among
    the nodes. ``fields`` maps names to one value per node (cell data), in the
    order the nodes are given; ``mask`` (one per node, nonzero hides) becomes
    ``HIDDENCELL``. The nodes are sorted in pre-order and the fields with them.
    ``world`` is where other ranks' trees are, for reductions: each rank holds
    whole trees.

    ``OctreeDataSet.from_cells`` makes one from cells' levels and positions, as
    a RAMSES reader gives them; ``tack.interop.vtk`` converts
    ``vtkHyperTreeGrid``s."""

    def __init__(self, roots, tree, level, key, fields=None, mask=None, world=None):
        from tack.dataset.topology import StructuredTopology

        if not isinstance(roots.topology, StructuredTopology) or \
                not isinstance(roots.geometry.values, (UniformCoordinates, CartesianProduct)):
            raise TypeError("an octree's roots are an image or a rectilinear grid")
        self.roots = roots
        self.dimension = len(roots.topology.point_dims)
        if self.dimension not in (2, 3):
            raise ValueError("an octree is 2D or 3D")
        if world is None:
            from tack.communication import World
            world = World()
        self.world = world
        tree = np.asarray(arrays.to_host(tree) if isinstance(tree, tack.Array) else tree,
                          np.int64).reshape(-1)
        level = np.asarray(arrays.to_host(level) if isinstance(level, tack.Array) else level,
                           np.int64).reshape(-1)
        key = np.asarray(arrays.to_host(key) if isinstance(key, tack.Array) else key,
                         np.uint64).reshape(-1)
        if not len(tree) == len(level) == len(key):
            raise ValueError("tree, level and key give each node: one of each per node")
        d = self.dimension
        if len(level) and (level.min() < 0 or level.max() > MAX_LEVEL[d]):
            raise ValueError(f"levels run from 0 to {MAX_LEVEL[d]} in {d}D")
        if len(tree) and (tree.min() < 0 or tree.max() >= roots.num_cells):
            raise ValueError(f"trees are the roots' cells, 0 to {roots.num_cells - 1}")
        if len(key) and (key >> (d * level).astype(np.uint64)).any():
            raise ValueError("a key holds dimension bits a level: one is too large for its "
                             "level")
        order = _preorder(tree, level, key, d)
        self._tree, self._level, self._key = tree[order], level[order], key[order]
        _check_ancestors(self._tree, self._level, self._key, d)
        leaf = _leaves(self._tree, self._level, self._key, d)
        marks = np.where(leaf, 0, Ghosts.REFINEDCELL).astype(np.uint8)
        if mask is not None:
            mask = np.asarray(arrays.to_host(mask) if isinstance(mask, tack.Array) else mask
                              ).reshape(-1)
            marks[np.asarray(mask[order] != 0)] |= Ghosts.HIDDENCELL
        device = roots.device
        with device:
            self.tree = tack.array(self._tree, tack.i64)
            self.level = tack.array(self._level.astype(np.int32), tack.i32)
            self.key = tack.array(self._key, tack.u64)
            self.marks = tack.array(marks, tack.u8)
        self.fields = {}
        for name, values in (fields or {}).items():
            self.fields[name] = _reordered(values, order, device, len(order), name)
        self._geometry = None

    # -- what it holds ---------------------------------------------------

    @property
    def num_nodes(self):
        return len(self._key)

    @property
    def num_trees(self):
        """The trees this rank holds."""
        return len(np.unique(self._tree))

    @property
    def device(self):
        return self.roots.device

    @property
    def origin(self):
        return self.roots.origin

    def leaves(self):
        """Which nodes are leaves: a bool host array, in the nodes' order."""
        return (self.marks.numpy() & Ghosts.REFINEDCELL) == 0

    def __repr__(self):
        return (f"OctreeDataSet({self.num_trees} trees, {self.num_nodes} nodes, "
                f"{int(self.leaves().sum())} leaves, {self.dimension}D)")

    def geometry(self):
        """Each node's lower corner and sides, relative to ``origin``: two arrays of
        3-vectors (f64 where the device has it, else the roots' type), from the
        keys, made once."""
        if self._geometry is None:
            dtype = tack.f64 if tack.f64 in self.device._backend.supported_dtypes else \
                self.roots.dtype
            lo, size = _root_cells(self.roots)
            with self.device:
                root_lo = tack.array(lo.astype(dtype.numpy_dtype), tack.vec3(dtype))
                root_size = tack.array(size.astype(dtype.numpy_dtype), tack.vec3(dtype))
                corner = tack.empty((self.num_nodes,), tack.vec3(dtype))
                sides = tack.empty((self.num_nodes,), tack.vec3(dtype))
                if self.num_nodes:
                    _node_geometry(self.tree, self.level, self.key, root_lo, root_size,
                                   self.dimension, corner, sides)
            self._geometry = (corner, sides)
        return self._geometry

    def centers(self):
        """Each node's centre, relative to ``origin``: a host ``(n, 3)`` array."""
        corner, sides = self.geometry()
        return corner.numpy().reshape(-1, 3) + 0.5 * sides.numpy().reshape(-1, 3)

    def measures(self):
        """Each node's volume (area in 2D): a host array."""
        _, sides = self.geometry()
        s = sides.numpy().reshape(-1, 3)
        return s[:, 0] * s[:, 1] * (s[:, 2] if self.dimension == 3 else 1.0)

    # -- construction from cells ----------------------------------------

    @classmethod
    def from_cells(cls, roots, level, position, fields=None, world=None):
        """An octree of cells given by their ``level`` and ``position`` (an ``(n, 3)``
        world position inside each cell, its centre say), as a RAMSES reader or
        yt gives them -- whole octs: every refined cell's children. Each cell's
        tree is the root cell holding the position; its key, the integer
        coordinates there at its level. Ancestors no cell gives are added, their
        fields NaN (integer fields 0): reductions of the leaves never read
        them."""
        level = np.asarray(level, np.int64).reshape(-1)
        position = np.asarray(position, np.float64).reshape(-1, 3) - np.asarray(roots.origin)
        d = len(roots.topology.point_dims)
        axes = _root_axes(roots)
        tree = np.zeros(len(level), np.int64)
        stride = 1
        index = np.zeros((len(level), 3), np.int64)
        for a in range(d):
            edges = axes[a]
            cell = np.clip(np.searchsorted(edges, position[:, a], side="right") - 1, 0,
                           len(edges) - 2)
            tree += cell * stride
            stride *= len(edges) - 1
            fraction = (position[:, a] - edges[cell]) / (edges[cell + 1] - edges[cell])
            index[:, a] = np.clip(np.floor(fraction * (1 << level)), 0,
                                  (1 << level) - 1).astype(np.int64)
        key = _interleave(index, d)
        tree, level, key, fields = _with_ancestors(tree, level, key, fields or {}, d)
        return cls(roots, tree, level, key, fields=fields, world=world)

    # -- reductions ------------------------------------------------------

    def _selected(self, level):
        """Which nodes a reduction reads: the leaves, or the nodes of ``level``, less
        the hidden ones."""
        marks = self.marks.numpy()
        if level is None:
            chosen = (marks & Ghosts.REFINEDCELL) == 0
        else:
            chosen = self._level == int(level)
        return chosen & ((marks & Ghosts.HIDDENCELL) == 0)

    def _statistics(self, name, weights, level):
        from tack.dataset.algorithms.statistics import _array_moments, _merged, _summary

        values, chosen = self._field(name), self._selected(level)
        index = tack.array(np.flatnonzero(chosen).astype(np.int64), tack.i64,
                           device=self.device)
        from tack.communication._kernels import gather

        mine = None
        if len(index):
            w = None if weights is None else gather(self._weights(weights), index)
            mine = _array_moments(gather(values, index), w)
        moments = None
        for part in self.world._allgather(mine):
            if part is not None:
                moments = part if moments is None else _merged(moments, part)
        if moments is None:
            raise ValueError("statistics of no values")
        return _summary(*moments)

    def _entropy(self, name, bins, weights, level):
        from tack.dataset.algorithms.entropy import _bits, _weighted_counts

        chosen = self._selected(level)
        values = np.asarray(self._field(name).numpy())[chosen].astype(np.float64)
        w = np.ones(len(values)) if weights is None else \
            np.asarray(self._weights(weights).numpy(), np.float64)[chosen]
        extrema = [e for e in self.world._allgather(
            (float(values.min()), float(values.max())) if len(values) else None)
            if e is not None]
        if not extrema:
            raise ValueError("entropy of no values")
        value_range = (min(e[0] for e in extrema), max(e[1] for e in extrema))
        counts = _weighted_counts(values, w, bins, value_range) if len(values) else \
            np.zeros(bins)
        return _bits(sum(self.world._allgather(counts)))

    def _field(self, name):
        if name not in self.fields:
            raise KeyError(f"the octree has no field {name!r}: it has "
                           f"{', '.join(map(repr, self.fields)) or 'none'}")
        values = self.fields[name]
        if getattr(values, "_vector_n", None):
            raise TypeError("statistics takes a scalar field")
        return values

    def _weights(self, weights):
        if isinstance(weights, str) and weights == "volume":
            with self.device:
                return tack.array(self.measures())
        return self._field(weights)


# -- the keys --------------------------------------------------------------------------

def _preorder(tree, level, key, d):
    """The nodes' pre-order: by tree, then key carried to the finest level, then
    level. Two stable sorts: the finest-level key (and level), then the tree."""
    if not len(key):
        return np.zeros(0, np.int64)
    finest = int(level.max())
    carried = key << (d * (finest - level)).astype(np.uint64)
    secondary = (carried << np.uint64(5)) | level.astype(np.uint64)
    order = np.argsort(secondary, kind="stable")
    return order[np.argsort(tree[order], kind="stable")]


def _leaves(tree, level, key, d):
    """Which sorted nodes are leaves: those whose next node is not their child."""
    n = len(key)
    leaf = np.ones(n, bool)
    if n > 1:
        child = (tree[1:] == tree[:-1]) & (level[1:] == level[:-1] + 1) & \
            ((key[1:] >> np.uint64(d)) == key[:-1])
        leaf[:-1] = ~child
    return leaf


def _check_ancestors(tree, level, key, d):
    """Refuse a node whose parent is not among the nodes."""
    codes = _code(tree, level, key, d)
    if len(np.unique(codes)) != len(codes):
        raise ValueError("a node is given twice")
    deep = np.flatnonzero(level > 0)
    if not len(deep):
        return
    sorted_codes = np.sort(codes)
    parents = _code(tree[deep], level[deep] - 1, key[deep] >> np.uint64(d), d)
    at = np.minimum(np.searchsorted(sorted_codes, parents), len(sorted_codes) - 1)
    missing = sorted_codes[at] != parents
    if missing.any():
        raise ValueError(f"{int(missing.sum())} nodes have no parent among the nodes: give "
                         "every node's ancestors (from_cells adds them)")
    # A refined node has all 2^d children, as an oct and a HyperTreeGrid's
    # trees do: a missing one would be a hole no leaf covers.
    _, counts = np.unique(parents, return_counts=True)
    if (counts != 2 ** d).any():
        raise ValueError(f"{int((counts != 2 ** d).sum())} refined nodes have fewer than "
                         f"{2 ** d} children: give each refined node all its children")


def _code(tree, level, key, d):
    """One sortable code per node, (tree, level, key) as one Python integer: a
    host check, made once."""
    return (tree.astype(object) << 70) | (level.astype(object) << 64) | key.astype(object)


def _interleave(index, d):
    """Morton keys of integer coordinates ``(n, 3)``, x lowest."""
    key = np.zeros(len(index), np.uint64)
    if not len(index):
        return key
    bits = int(index[:, :d].max()).bit_length()
    for b in range(bits):
        for a in range(d):
            key |= ((index[:, a].astype(np.uint64) >> np.uint64(b)) & np.uint64(1)) \
                << np.uint64(d * b + a)
    return key


def _with_ancestors(tree, level, key, fields, d):
    """The nodes and every ancestor none of them is, fields filled with NaN (0 for
    integers) on the added ones."""
    have = set(zip(tree.tolist(), level.tolist(), key.tolist()))
    added = []
    for t, lv, k in zip(tree.tolist(), level.tolist(), key.tolist()):
        while lv > 0:
            lv, k = lv - 1, k >> d
            if (t, lv, k) in have:
                break
            have.add((t, lv, k))
            added.append((t, lv, k))
    if not added:
        return tree, level, key, fields
    extra = np.array(added, dtype=object)
    tree = np.concatenate([tree, extra[:, 0].astype(np.int64)])
    level = np.concatenate([level, extra[:, 1].astype(np.int64)])
    key = np.concatenate([key, extra[:, 2].astype(np.uint64)])
    filled = {}
    for name, values in fields.items():
        host = np.asarray(arrays.to_host(values) if isinstance(values, tack.Array) else values)
        pad = np.full((len(added), *host.shape[1:]), np.nan if host.dtype.kind == "f" else 0,
                      host.dtype)
        filled[name] = np.concatenate([host, pad])
    return tree, level, key, filled


def _reordered(values, order, device, n, name):
    """A field's values in the nodes' order, on ``device``."""
    host = np.asarray(arrays.to_host(values) if isinstance(values, tack.Array) else values)
    if host.shape[:1] != (n,):
        raise ValueError(f"field {name!r} has {host.shape[:1]} values for {n} nodes")
    host = host[order]
    with device:
        if host.ndim == 2:
            from tack.lang.types import from_numpy_dtype
            return tack.array(np.ascontiguousarray(host),
                              tack.vec(host.shape[1], from_numpy_dtype(host.dtype)))
        return tack.array(np.ascontiguousarray(host))


def _root_axes(roots):
    """The root grid's lines along each axis, relative to its origin (f64)."""
    coordinates = roots.geometry.values
    dims = roots.topology.point_dims
    if isinstance(coordinates, UniformCoordinates):
        return [coordinates._origin[a] + coordinates._spacing[a] * np.arange(dims[a])
                for a in range(len(dims))]
    return [np.asarray(arrays.to_host(axis), np.float64).reshape(-1)
            for axis in (coordinates.x, coordinates.y, coordinates.z)[:len(dims)]]


def _root_cells(roots):
    """Each root cell's lower corner and sides, ``(n, 3)`` f64 each, x fastest."""
    axes = _root_axes(roots)
    d = len(axes)
    # A 2D grid lies in its plane: the z of its points, no thickness.
    plane = np.zeros(1)
    coordinates = roots.geometry.values
    if d == 2:
        plane = np.array([coordinates._origin[2]]) if isinstance(coordinates,
                                                                  UniformCoordinates) \
            else np.asarray(arrays.to_host(coordinates.z), np.float64).reshape(-1)[:1]
    los = [a[:-1] for a in axes] + [plane] * (3 - d)
    sizes = [np.diff(a) for a in axes] + [np.zeros(1)] * (3 - d)
    mesh_lo = np.meshgrid(*los[::-1], indexing="ij")
    mesh_size = np.meshgrid(*sizes[::-1], indexing="ij")
    lo = np.stack([m.reshape(-1) for m in mesh_lo[::-1]], axis=1)
    size = np.stack([m.reshape(-1) for m in mesh_size[::-1]], axis=1)
    return lo, size
