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
from tack.dataset import units as _units
from tack.dataset.ghosts import Ghosts

__all__ = ["OctreeDataSet"]

#: Levels a key holds: 3 bits a level in 3D, and 5 bits for the level beside it.
MAX_LEVEL = {2: 29, 3: 19}


@tack.func
def _unkey(k, depth, dimension):
    # A key's integer coordinates in its tree at its level: every dimension-th
    # bit, x lowest.
    i = tack.u64(0)
    j = tack.u64(0)
    m = tack.u64(0)
    for b in range(depth):
        i = i | (((k >> tack.u64(dimension * b)) & tack.u64(1)) << tack.u64(b))
        j = j | (((k >> tack.u64(dimension * b + 1)) & tack.u64(1)) << tack.u64(b))
        if dimension == 3:
            m = m | (((k >> tack.u64(dimension * b + 2)) & tack.u64(1)) << tack.u64(b))
    return i, j, m


@tack.kernel
def _node_geometry(tree, level, key, root_lo, root_size, dimension, lo, size):
    # Each node's lower corner and sides: its tree's root cell, divided by 2 per
    # level, at the integer coordinates its key interleaves.
    for n in range(lo.shape[0]):
        t = tree[n]
        depth = level[n]
        i, j, m = _unkey(key[n], depth, dimension)
        scale = 1.0 / tack.f64(tack.u64(1) << tack.u64(depth))
        r = root_size[t]
        s = [r[0] * scale, r[1] * scale, r[2] * scale]
        if dimension == 2:
            s = [s[0], s[1], r[2]]
        base = root_lo[t]
        lo[n] = [base[0] + tack.f64(i) * s[0], base[1] + tack.f64(j) * s[1],
                 base[2] + tack.f64(m) * s[2]]
        size[n] = s


@tack.kernel
def _leaf_corners(leaf, tree, level, key, nx, ny, dimension, finest, ci, cj, ck):
    # Each leaf's 2^d corners on the lattice of the finest level over every root
    # cell (root t's lower corner at (tx, ty, tz) * 2^finest): corner v of leaf
    # c, x fastest (a voxel's order), at entry c * 2^d + v.
    for c in range(leaf.shape[0]):
        n = leaf[c]
        t = tree[n]
        depth = level[n]
        i, j, m = _unkey(key[n], depth, dimension)
        step = tack.i64(1) << tack.i64(finest - depth)
        bi = ((t % nx) << tack.i64(finest)) + tack.i64(i) * step
        bj = (((t // nx) % ny) << tack.i64(finest)) + tack.i64(j) * step
        bk = ((t // (nx * ny)) << tack.i64(finest)) + tack.i64(m) * step
        corners = 1 << dimension
        for v in range(corners):
            ci[c * corners + v] = bi + tack.i64(v & 1) * step
            cj[c * corners + v] = bj + tack.i64((v >> 1) & 1) * step
            ck[c * corners + v] = bk + tack.i64((v >> 2) & 1) * step


@tack.kernel
def _run_starts(si, sj, sk, flags):
    for s in range(flags.shape[0]):
        flags[s] = 1 if s == 0 or si[s] != si[s - 1] or sj[s] != sj[s - 1] or \
            sk[s] != sk[s - 1] else 0


@tack.kernel
def _numbered(order, ids, connectivity):
    # Each corner entry's point: its run's number.
    for s in range(order.shape[0]):
        connectivity[order[s]] = ids[s] - 1


@tack.func
def _along(axis, n, index, finest):
    # The position of lattice line ``index`` along a root axis of ``n`` cells.
    c = min(index >> tack.i64(finest), tack.i64(n - 1))
    f = tack.f64(index - (c << tack.i64(finest))) / tack.f64(tack.i64(1) << tack.i64(finest))
    return axis[c] + f * (axis[c + 1] - axis[c])


@tack.kernel
def _lattice_points(si, sj, sk, flags, ids, ax, ay, az, nx, ny, nz, finest, dimension,
                    plane, out):
    # Each distinct corner's position, from its lattice index alone, so every
    # cell that shares it shares the same bits.
    for s in range(flags.shape[0]):
        if flags[s] == 1:
            z = plane
            if dimension == 3:
                z = _along(az, nz, sk[s], finest)
            out[ids[s] - 1] = [_along(ax, nx, si[s], finest), _along(ay, ny, sj[s], finest), z]


@tack.func
def _root_cell(axis, n, x, tolerance):
    # The cell of axis lines axis[0..n] holding x (the last at its upper end), or
    # -1 outside by more than ``tolerance`` of the end cell's width.
    cell = -1
    if x >= axis[0] - tolerance * (axis[1] - axis[0]) and \
            x <= axis[n] + tolerance * (axis[n] - axis[n - 1]):
        lo = 0
        hi = n - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if axis[mid] <= x:
                lo = mid
            else:
                hi = mid - 1
        cell = lo
    return cell


@tack.func
def _lattice_index(axis, c, x, finest):
    # x's integer coordinate in root cell c at the finest level, clamped inside.
    cells = tack.i64(1) << tack.i64(finest)
    f = (x - axis[c]) / (axis[c + 1] - axis[c])
    index = tack.i64(floor(max(f, 0.0) * tack.f64(cells)))
    return tack.u64(min(index, cells - 1))


@tack.kernel
def _find_leaves(points, ax, ay, az, nx, ny, nz, dimension, plane, tolerance, tree, level,
                 carried, finest, marks, hidden, found):
    # The leaf holding each point: its root cell by binary searches of the axes,
    # its code at the finest level, then the last node whose (tree, key carried
    # to the finest level) is at most the point's -- in pre-order, the leaf whose
    # run of codes holds it.
    for q in range(found.shape[0]):
        p = points[q]
        found[q] = -1
        cx = _root_cell(ax, nx, p[0], tolerance)
        cy = _root_cell(ay, ny, p[1], tolerance)
        cz = 0
        if dimension == 3:
            cz = _root_cell(az, nz, p[2], tolerance)
        if cx >= 0 and cy >= 0 and cz >= 0:
            i = _lattice_index(ax, cx, p[0], finest)
            j = _lattice_index(ay, cy, p[1], finest)
            m = tack.u64(0)
            if dimension == 3:
                m = _lattice_index(az, cz, p[2], finest)
            code = tack.u64(0)
            for b in range(finest):
                code = code | (((i >> tack.u64(b)) & tack.u64(1)) << tack.u64(dimension * b))
                code = code | (((j >> tack.u64(b)) & tack.u64(1)) << tack.u64(dimension * b + 1))
                if dimension == 3:
                    code = code | (((m >> tack.u64(b)) & tack.u64(1))
                                   << tack.u64(dimension * b + 2))
            t = tack.i64(cx) + tack.i64(nx) * (tack.i64(cy) + tack.i64(ny) * tack.i64(cz))
            lo = tack.i64(0)
            hi = tack.i64(tree.shape[0])
            while lo < hi:
                mid = (lo + hi) // 2
                if tree[mid] < t or (tree[mid] == t and carried[mid] <= code):
                    lo = mid + 1
                else:
                    hi = mid
            n = lo - 1
            if n >= 0 and tree[n] == t and (marks[n] & hidden) == 0:
                # A point outside the roots (or off a 2D octree's plane) is held
                # within the share of the leaf's diagonal a dataset's locator
                # accepts outside a cell.
                scale = 1.0 / tack.f64(tack.i64(1) << tack.i64(level[n]))
                sx = (ax[cx + 1] - ax[cx]) * scale
                sy = (ay[cy + 1] - ay[cy]) * scale
                sz = 0.0
                if dimension == 3:
                    sz = (az[cz + 1] - az[cz]) * scale
                ox = max(ax[0] - p[0], p[0] - ax[nx], 0.0)
                oy = max(ay[0] - p[1], p[1] - ay[ny], 0.0)
                oz = abs(p[2] - plane)
                if dimension == 3:
                    oz = max(az[0] - p[2], p[2] - az[nz], 0.0)
                reach = tolerance * tolerance * (sx * sx + sy * sy + sz * sz)
                if ox * ox + oy * oy + oz * oz <= reach:
                    found[q] = n


@tack.kernel
def _take_found(found, values, out):
    for q in range(found.shape[0]):
        e = found[q]
        if e >= 0:
            out[q] = values[e]


@tack.kernel
def _found_valid(found, out):
    for q in range(found.shape[0]):
        out[q] = 1 if found[q] >= 0 else 0


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
    whole trees. ``units`` maps field names to their units
    (``tack.dataset.units``); the spatial unit is the roots' (``unit``).

    ``OctreeDataSet.from_cells`` makes one from cells' levels and positions, as
    a RAMSES reader gives them; ``tack.interop.vtk`` converts
    ``vtkHyperTreeGrid``s."""

    def __init__(self, roots, tree, level, key, fields=None, mask=None, world=None,
                 units=None):
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
        unknown = set(units or {}) - set(self.fields)
        if unknown:
            raise KeyError(f"units for fields not given: {', '.join(map(repr, sorted(unknown)))}")
        self.units = {name: _units.canonical(unit) for name, unit in (units or {}).items()}
        self._geometry = None
        self._carried_keys = None

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

    @property
    def unit(self):
        """The spatial unit, the roots' (``DataSet.unit``)."""
        return self.roots.unit

    @property
    def is_leaf(self):
        """Which nodes are leaves: a bool host array, in the nodes' order."""
        return (self.marks.numpy() & Ghosts.REFINEDCELL) == 0

    @property
    def finest(self):
        """The deepest level of this rank's nodes (0 with none)."""
        return int(self._level.max()) if len(self._level) else 0

    def __repr__(self):
        return (f"OctreeDataSet({self.num_trees} trees, {self.num_nodes} nodes, "
                f"{int(self.is_leaf.sum())} leaves, {self.dimension}D)")

    def geometry(self):
        """Each node's lower corner and sides, relative to ``origin``: two arrays of
        3-vectors (f64 where the device has it, else the roots' type), from the
        keys, made once."""
        if self._geometry is None:
            dtype = _compute_type(self.device, self.roots.dtype)
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

    # -- the leaves as cells --------------------------------------------

    def leaves(self):
        """This rank's leaves as a ``DataSet`` of voxels (pixels in 2D), relative to
        ``origin``: cell ``c`` is the ``c``-th leaf in the nodes' order
        (``np.flatnonzero(octree.is_leaf)[c]``), with every field's value there as
        cell data, and hidden leaves ``HIDDENCELL``. Corners that coincide are one
        point, found by sorting their indices on the finest level's lattice;
        hanging corners stay hanging, as the tree has them. Every filter that
        reads cell data works on it (``threshold``, ``cell_centers``, ``probe``).

        Each rank gives its own trees' leaves; a corner on a face between two
        ranks' trees is a point on both, unmarked."""
        from tack.algorithms import argsort, gather, inclusive_scan
        from tack.dataset import ids, shapes
        from tack.dataset.dataset import DataSet, Field
        from tack.dataset.spaces import Constant
        from tack.dataset.topology import UnstructuredTopology, uniform_cells

        d = self.dimension
        is_leaf = self.is_leaf
        count = int(is_leaf.sum())
        corners = count << d
        finest = self.finest
        nx, ny, nz = ([n - 1 for n in self.roots.topology.point_dims] + [1])[:3]
        idt = ids.choose(None, corners)
        dtype = _compute_type(self.device, self.roots.dtype)
        with self.device:
            leaf = tack.array(np.flatnonzero(is_leaf).astype(np.int64), tack.i64)
            columns = [tack.empty((corners,), tack.i64) for _ in range(3)]
            if count:
                _leaf_corners(leaf, self.tree, self.level, self.key, nx, ny, d, finest,
                              *columns)
            # Sorted by z, then y, then x (stable sorts from the fastest axis), so
            # points come x fastest; the runs of equal corners are the points.
            order = argsort(columns[0], index_dtype=tack.i64) if corners else \
                tack.empty((0,), tack.i64)
            for column in columns[1:d]:
                if corners:
                    order = gather(order, argsort(gather(column, order), index_dtype=tack.i64))
            si, sj, sk = (gather(c, order) for c in columns)
            flags = tack.empty((corners,), tack.i32)
            numbers = tack.empty((corners,), tack.i64)
            connectivity = tack.empty((corners,), idt)
            points = 0
            if corners:
                _run_starts(si, sj, sk, flags)
                points = int(inclusive_scan(flags, numbers, corners))
                _numbered(order, numbers, connectivity)
            axes, plane = self._axes(dtype)
            positions = tack.empty((points,), tack.vec3(self.roots.dtype))
            if points:
                _lattice_points(si, sj, sk, flags, numbers, *axes, nx, ny, nz, finest, d,
                                plane, positions)
            types, offsets = uniform_cells(shapes.VOXEL if d == 3 else shapes.PIXEL, count,
                                           idt)
            topology = UnstructuredTopology(types, offsets, connectivity, num_points=points)
            fields = {name: Field(Constant(topology), gather(values, leaf),
                                  self.units.get(name))
                      for name, values in self.fields.items()}
            marks = gather(self.marks, leaf)
            ghosts = None
            if count and (marks.numpy() & Ghosts.HIDDENCELL).any():
                hidden = marks.numpy() & Ghosts.HIDDENCELL
                ghosts = Ghosts(cells=tack.array(hidden.astype(np.uint8), tack.u8))
        return DataSet(topology, positions, fields=fields, dtype=self.roots.dtype,
                       ghosts=ghosts, origin=self.origin, unit=self.unit)

    # -- probing ---------------------------------------------------------

    def _probe(self, where, fields):
        """``probe(octree, where)``: each point's leaf value (``probe``'s
        docstring)."""
        from tack.communication._kernels import gather
        from tack.dataset.algorithms._extract import _vertices
        from tack.dataset.dataset import DataSet, Field, relative_positions
        from tack.dataset.ghosts import keep_marks
        from tack.dataset.partitioned import PartitionedDataSet
        from tack.dataset.spaces import H1, Values

        names = list(self.fields) if fields is None else list(fields)
        for name in names:
            self._field_values(name)
        many = isinstance(where, PartitionedDataSet)
        targets = list(where) if many else [where]
        for i, target in enumerate(targets):
            if not isinstance(target, DataSet):
                targets[i] = _vertices(relative_positions(target, self.origin, self.unit), self.roots)
            elif not isinstance(target.geometry.space, H1):
                raise TypeError("probe needs the probed points: an H1 geometry")
        points = [relative_positions(t, self.origin, self.unit) for t in targets]
        counts = [len(p) for p in points]
        x = np.concatenate(points) if points else np.zeros((0, 3))
        if self.world.size == 1:
            values = self._probed(x, names)
        else:
            values = self._probed_over_ranks(x, names)
        out = []
        start = 0
        for target, n in zip(targets, counts):
            kept = dict(target.fields)
            rows = tack.arange(start, start + n, dtype=tack.i64, device=self.device)
            for name in [*names, "valid"]:
                array = gather(values[name], rows).to_device(target.device)
                kept[name] = Field(Values(target, "points") if name == "valid" else
                                   H1(target), array, self.units.get(name))
            start += n
            out.append(keep_marks(target, DataSet(target.topology, target.geometry,
                                                  fields=kept, sets=dict(target.sets))))
        return PartitionedDataSet(out, world=where.world) if many else out[0]

    def _found(self, x):
        """The leaf holding each point of ``x`` (``(n, 3)`` f64 host, relative to
        ``origin``), or -1: a device array."""
        from tack.dataset.locator import _TOLERANCE

        dtype = _compute_type(self.device, self.roots.dtype)
        nx, ny, nz = ([n - 1 for n in self.roots.topology.point_dims] + [1])[:3]
        with self.device:
            points = tack.array(np.ascontiguousarray(x, dtype.numpy_dtype).reshape(-1, 3),
                                tack.vec3(dtype))
            found = tack.empty((len(x),), tack.i64)
            if len(x):
                axes, plane = self._axes(dtype)
                _find_leaves(points, *axes, nx, ny, nz, self.dimension, plane, _TOLERANCE,
                             self.tree, self.level, self._carried(), self.finest, self.marks,
                             int(Ghosts.HIDDENCELL), found)
        return found

    def _probed(self, x, names):
        """Each field's leaf value at the points of ``x``, 0 where none holds one,
        and ``valid``: arrays on the octree's device."""
        found = self._found(x)
        out = {}
        with self.device:
            for name in names:
                values = self._field_values(name)
                width = arrays.width_of(values)
                dtype = arrays.dtype_of(values)
                taken = tack.empty((len(x),), tack.vec(width, dtype) if width else dtype)
                if len(x):
                    taken.fill(0)
                    _take_found(found, values, taken)
                out[name] = taken
            valid = tack.empty((len(x),), tack.i32)
            if len(x):
                _found_valid(found, valid)
            out["valid"] = valid
        return out

    def _probed_over_ranks(self, x, names):
        """``_probed`` with every rank's trees: each point to the rank whose leaf
        holds it (the lowest on a tie), probed there and the values sent back."""
        from tack import communication as tc

        blocks = tc.Blocks([self.device], self.world)
        g = blocks.local.start

        def holds(gid, y):
            return (arrays.to_host(self._found(y)) >= 0).astype(np.int64)

        routing = tc.Routing(blocks, {g: x}, {g: self._box()}, holds)
        y, _ = routing.send()[g]
        like = {n: v.dtype for n, v in self._probed(np.zeros((0, 3)), names).items()}
        answers = {g: self._probed(np.asarray(y, np.float64), names)} if len(y) else {}
        return routing.reply(answers, like)[g]

    def _box(self):
        """The box of this rank's trees, relative to ``origin``, grown by what the
        search accepts outside (a share of a root cell, so of any leaf), or None.
        A 2D octree's box is its plane's, grown the same way along z."""
        from tack.dataset.locator import _TOLERANCE

        if not len(self._tree):
            return None
        lo, size = _root_cells(self.roots)
        held = np.unique(self._tree)
        grow = _TOLERANCE * float(np.linalg.norm(size[held], axis=1).max())
        return lo[held].min(axis=0) - grow, (lo[held] + size[held]).max(axis=0) + grow

    def _axes(self, dtype):
        """The root grid's lines along x, y and z as device arrays of ``dtype`` (a 2D
        grid's z axis is its x, unread), and its plane's z."""
        lines = _root_axes(self.roots)
        plane = float(_root_cells(self.roots)[0][0, 2]) if self.dimension == 2 else 0.0
        with self.device:
            axes = [tack.array(np.asarray(a, dtype.numpy_dtype)) for a in lines]
        if self.dimension == 2:
            axes.append(axes[0])
        return axes, plane

    def _carried(self):
        """Each node's key carried to the finest level, on the device, made once."""
        if self._carried_keys is None:
            shift = (self.dimension * (self.finest - self._level)).astype(np.uint64)
            with self.device:
                self._carried_keys = tack.array(self._key << shift, tack.u64)
        return self._carried_keys

    # -- construction from cells ----------------------------------------

    @classmethod
    def from_cells(cls, roots, level, position, fields=None, world=None, units=None):
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
        return cls(roots, tree, level, key, fields=fields, world=world, units=units)

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

    def _field_values(self, name):
        if name not in self.fields:
            raise KeyError(f"the octree has no field {name!r}: it has "
                           f"{', '.join(map(repr, self.fields)) or 'none'}")
        return self.fields[name]

    def _field(self, name):
        values = self._field_values(name)
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


def _compute_type(device, fallback):
    """The type positions are computed in: f64 where the device has it."""
    return tack.f64 if tack.f64 in device._backend.supported_dtypes else fallback


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
