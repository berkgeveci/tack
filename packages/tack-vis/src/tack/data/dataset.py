"""Datasets: a topology, a geometry, fields on spaces, and sets.

``docs/design/dataset-api.md`` §3 in prototype form:

- *spaces* (``tack.data.spaces``) say where a field's values live, on one
  topology, and own the layout that follows: ``H1(data)`` one per point,
  interpolated linearly (point data); ``Constant(data)`` one per cell (cell
  data); ``L2(data)`` each cell's own corner values (a linear discontinuous
  Galerkin field); ``Values(data, on)`` one value per point, edge, face or
  cell, with no basis -- data *about* the entity;
- a *field* is a space and its values, which are any array: a device
  field, or an implicit array from ``tack.data.arrays`` (a
  ``CartesianProduct`` of three axes, a ``ConstantArray``, a
  ``CountingArray``);
- the *geometry* is a field too, kept apart from the named fields
  (``data.geometry``): ``H1`` positions, one per point (a rectilinear grid's
  are a ``CartesianProduct``), or ``L2`` positions, each cell's own corners;
- *sets* (``tack.data.sets``) are named selections of points, cells or faces,
  such as the boundary faces.

``for_each(kernel, data, domain, *args)`` runs ``kernel`` once per shape in
the domain -- ``"cells"``, ``"faces"``, ``"edges"``, or a cell or face set --
passing the domain's view and, for every ``Field`` among ``args``, the
field's view for the same group. When a field's space varies from cell to
cell (an order per cell), each shape group is split by the keys of every
such space among ``args``, and the kernel runs once per subgroup, each
launch specialized to its keys.
"""

import numpy as np

import tack
from tack.algorithms.sort import _run_offsets, argsort, gather
from tack.data import arrays, shapes, views
from tack.data.arrays import CartesianProduct, UniformCoordinates
from tack.data.sets import Sets
from tack.data.spaces import H1, L2, Space

__all__ = ["DataSet", "Field", "for_each", "rectilinear_grid"]


class Field:
    """A space and its values: any array, explicit or implicit, of scalars or vectors,
    with one value per degree of freedom of the space. Everything else a kernel needs
    to find them belongs to the space."""

    def __init__(self, space, values):
        if not isinstance(space, Space):
            raise TypeError(f"a field's space is a Space on a topology, not {space!r}")
        arrays.storage(values)                       # refuses what is not an array
        if arrays.size_of(values) != space.size:
            raise ValueError(f"{space!r} holds {space.size} values, not "
                             f"{arrays.size_of(values)}")
        self.space = space
        self.values = values

    def view(self, group):
        """This field's view for ``group``: the group's kind and shape, the space, and
        the storage its values are read through. A space that varies reads its key
        for the group from ``group.keys``: launch it through ``for_each``."""
        mixins, attributes = _space_parts(self, group)
        return group.view(*mixins, **attributes)

    def __repr__(self):
        values = self.values
        stored = repr(values) if isinstance(values, arrays._Implicit) else "stored"
        return f"Field({self.space!r}, {arrays.size_of(values)} values, {stored})"


def _is_cells(group):
    return issubclass(group.kind, (views._Cells, views._StructuredCells))


def _space_parts(field, group, space_mixin=None, corners_only=False):
    """The mixins and attributes that read ``field`` for ``group``: its space's (or
    ``space_mixin``, for the geometry), an addressing mixin for values shared by
    points, and its storage's."""
    storage, attributes = arrays.storage(field.values)
    space = field.space
    if space.varies and space not in group.keys:
        raise ValueError(f"{space!r} varies from cell to cell: its views come from the "
                         "subgroups for_each makes")
    if (isinstance(space, H1) and space.order == 2 and not corners_only
            and not _is_cells(group)):
        raise TypeError(f"{space!r} is read cell by cell: on faces, take its traces "
                        "(tack.data.traces)")
    mixins = [space_mixin or space.mixin_for(group.keys.get(space))]
    if isinstance(space, H1):
        mixins.append(views.point_address(group.kind, storage))
    if not space_mixin:
        # A geometry's incidence is the domain view's own.
        mixins.extend(space.extra_mixins())
    mixins.append(storage)
    # Reading only the points' values -- an order-2 geometry on faces and edges
    # -- needs none of the space's own layout.
    layout = {} if corners_only else space.attributes(group)
    for name, value in layout.items():
        # The geometry's L2 offsets sit beside the domain view's own names.
        attributes["point_offsets" if space_mixin and name == "offsets" else name] = value
    return mixins, attributes


# ── Geometry ────────────────────────────────────────────────────────

def _as_geometry(geometry, topology, dtype):
    """``geometry`` as a ``Field``: positions per point (a host array, or any array of
    3-vectors, such as a ``CartesianProduct``) become an ``H1`` field on ``topology``;
    a ``Field`` is checked."""
    if not isinstance(geometry, Field):
        try:
            arrays.storage(geometry)
        except TypeError:
            positions = np.ascontiguousarray(geometry)
            geometry = tack.Vector.field(3, dtype, shape=(positions.shape[0],))
            geometry.from_numpy(positions.astype(dtype.numpy_dtype))
        geometry = Field(H1(topology), geometry)
    if geometry.space.topology is not topology:
        raise ValueError("the geometry's space is on another topology")
    if not isinstance(geometry.space, (H1, L2)):
        raise TypeError(f"geometry lives in H1 or L2, not {geometry.space!r}")
    if arrays.width_of(geometry.values) != 3:
        raise TypeError("geometry values must be 3-vectors")
    return geometry


def _geometry_parts(data, group, kind):
    """The mixins and attributes that give ``group``'s view, an entity of ``kind``,
    ``data``'s geometry field."""
    geometry = data.geometry
    if isinstance(geometry.space, L2):
        if kind == "faces":
            # Each side's cell has its own corners: a face is where its side 0
            # puts it.
            return [views._SideZeroGeometry], {"face_points": data._side_zero_points()}
        if kind != "cells":
            raise NotImplementedError(
                f"the {kind} of an L2 geometry have no positions of their own: each cell "
                "around an edge has its own, and edges have no sides to choose from")
        return _space_parts(geometry, group, views._L2Geometry)
    if group.shape in (shapes.Polyhedron, shapes.Polygon, shapes.PolyLine):
        # No reference element: the points' positions, nothing to map.
        return _space_parts(geometry, group, views._PointGeometry)
    if geometry.space.order == 2 and kind == "cells":
        # Curved: the domain view's incidence gives the quadratic nodes' ids.
        data.topology.edges()
        data.topology.faces()
        return _space_parts(geometry, group, views._H1Order2Geometry)
    # Faces and edges of a curved geometry see its corners.
    return _space_parts(geometry, group, views._H1Geometry, corners_only=True)


# ── Datasets ────────────────────────────────────────────────────────

class DataSet:
    """A topology, a geometry, named fields and named sets. The geometry is a
    ``Field`` in ``H1`` or ``L2`` of 3-vectors -- or positions per point (a host
    array, or any array of 3-vectors, such as a device field or a
    ``CartesianProduct``), which become an ``H1`` field -- kept apart from
    ``fields`` as ``data.geometry``: algorithms move it with the cells, never as
    one of the fields they carry or select."""

    def __init__(self, topology, geometry, fields=None, sets=None, dtype=tack.f32):
        self.topology = topology
        self._geometry = _as_geometry(geometry, topology, dtype)
        self.fields = dict(fields or {})
        for name, field in self.fields.items():
            if field.space.topology is not topology:
                raise ValueError(f"field {name!r} is on another topology")
        self.sets = sets

    @property
    def sets(self):
        """Named ``Set``s of this dataset's points, cells or faces, checked as they are
        put in: a set of another topology, or named for a domain, is refused."""
        return self._sets

    @sets.setter
    def sets(self, sets):
        self._sets = Sets(self.topology, sets)

    def _kind(self, domain):
        """The entities ``domain`` -- a domain or a set's name -- iterates over."""
        if domain in ("cells", "faces", "edges"):
            return domain
        if domain in self.sets:
            kind = self.sets[domain].on
            if kind == "points":
                raise ValueError(f"set {domain!r} selects points, which are not an "
                                 "iteration domain: loop over its ids")
            return kind
        raise ValueError(f"no domain {domain!r}: cells, faces, edges, or a cell or face set "
                         f"({', '.join(self.sets) or 'none'})")

    @property
    def geometry(self):
        """The geometry, a field of positions in ``H1`` or ``L2``."""
        return self._geometry

    @geometry.setter
    def geometry(self, geometry):
        self._geometry = _as_geometry(geometry, self.topology, self.dtype)

    @property
    def dtype(self):
        """The geometry's floating-point type."""
        return self.geometry.values.dtype

    @property
    def id_dtype(self):
        """The topology's integer type for ids: ``tack.i32`` or ``tack.i64``."""
        return self.topology.id_dtype

    @property
    def num_points(self):
        return self.topology.num_points

    @property
    def num_cells(self):
        return self.topology.num_cells

    def positions(self):
        """Every point's position as a host array, ``(num_points, 3)``: an ``H1``
        geometry's values. An ``L2`` geometry has none per point."""
        if isinstance(self.geometry.space, L2):
            raise ValueError("an L2 geometry has positions per cell corner, not per point")
        # An order-2 geometry's first values are the points'.
        return arrays.to_host(self.geometry.values)[:self.num_points]

    def domain_groups(self, domain):
        """The ``DomainGroup``s of ``domain``: ``"cells"``, ``"faces"``, ``"edges"``, or
        the name of a cell or face set in ``sets`` -- a cell set's as the cell
        groups' subgroups of its members."""
        kind = self._kind(domain)
        if domain == "cells":
            return self.topology.groups()
        if domain == "faces":
            return self.topology.faces().groups()
        if domain == "edges":
            return self.topology.edges().groups()
        if kind == "faces":
            return self.topology.faces().groups(self.sets[domain].ids)
        return self.launch_groups(domain)

    def domain_view(self, domain, group):
        """``group``'s view: kind, shape, geometry, and for cells their incidence when
        faces and edges have been derived."""
        mixins = []
        kind = self._kind(domain)
        geometry, attributes = _geometry_parts(self, group, kind)
        mixins.extend(geometry)
        if kind == "cells" and issubclass(group.kind, views._CellsById):
            mixins.append(views._PolyhedralWalk)
        if (kind == "cells" and issubclass(group.kind, views._PolylineCells)
                and self.topology._faces is not None):
            mixins.append(views._PolylineIncidence)
            attributes.update(cell_faces=self.topology.cell_faces,
                              cell_face_sides=self.topology.cell_face_sides)
        if kind == "cells":
            # A subgroup's index(c) is its position in the topology's group, so
            # it shares that group's incidence.
            faces = self.topology._faces
            # The cells the faces were derived from: those of the top dimension.
            if faces is not None and id(group.root) in getattr(faces, "group_starts", {}):
                mixins.extend((views._FaceIncidence, views._FaceWalk))
                attributes.update(side_face=faces.side_face, side_slot=faces.side_slot,
                                  side_orientation=faces.side_orientation,
                                  face_start=faces.group_starts[id(group.root)])
            edges = self.topology._edges
            if edges is not None and group.shape.NUM_EDGES:
                mixins.append(views._EdgeIncidence)
                attributes.update(side_edge=edges.side_edge, side_sign=edges.side_sign,
                                  edge_start=edges.group_starts[id(group.root)])
        # A key that shapes the kernel -- a size bucket's cap -- brings its mixin.
        for key, value in group.keys.items():
            if hasattr(key, "domain_mixin"):
                mixins.append(key.domain_mixin(value))
        return group.view(*mixins, **attributes)

    def _side_zero_points(self):
        """An L2 geometry's corners per (face, side, point): its traces, kept while the
        geometry is the same field."""
        from tack.data.algorithms.traces import traces

        kept = self.__dict__.get("_side_zero")
        if kept is None or kept[0] is not self.geometry:
            kept = (self.geometry, traces(self, self.geometry).values)
            self._side_zero = kept
        return kept[1]

    def launch_groups(self, domain, fields=(), keys=()):
        """The groups a kernel over ``domain`` launches once each, given the ``Field``s
        it reads: the domain's groups, each split by the keys of the spaces among
        ``fields`` that vary from cell to cell, and by any other ``keys`` -- objects
        that give each cell a key, as ``polyhedra.SizeBuckets`` does."""
        kind = self._kind(domain)
        members = self.sets[domain] if kind == "cells" and domain != "cells" else None
        groups = self.topology.groups() if members is not None else self.domain_groups(domain)
        spaces = [members] if members is not None else []
        for field in fields:
            if field.space.varies and field.space not in spaces:
                spaces.append(field.space)
        spaces.extend(k for k in keys if k not in spaces)
        if kind != "cells" or not spaces:
            return groups
        # A cell set splits the groups by membership, and keeps its members.
        return [sub for group in groups if group.count
                for sub in _subgroups(group, tuple(spaces))
                if members is None or sub.keys[members] == 1]


def for_each(kernel, data, domain, *args):
    """Run ``kernel(view, *args)`` once per shape in ``domain`` of ``data``.

    ``view`` is the domain's view for that shape (``for i in view``); every
    ``Field`` among ``args`` becomes its view for the same shape, so
    ``u.value(i, pc)`` and ``u.at(i)`` read the entity ``i`` the loop is at.
    Other arguments pass through.
    """
    kind = data._kind(domain)
    for arg in args:
        if isinstance(arg, Field):
            if arg.space.topology is not data.topology:
                raise ValueError(f"a field on {arg.space!r} of another topology")
            _check_domain(arg.space, kind)
    for group in data.launch_groups(domain, [a for a in args if isinstance(a, Field)]):
        if not group.count:
            continue
        view = data.domain_view(domain, group)
        kernel(view, *(a.view(group) if isinstance(a, Field) else a for a in args))


# ── Per-side traces ─────────────────────────────────────────────────


# ── Subgroups ───────────────────────────────────────────────────────

_KEY_BASE = 16           # keys (orders) below 16; up to 7 varying spaces in an i32


@tack.kernel
def _clear_keys(keys):
    for i in range(keys.shape[0]):
        keys[i] = 0


@tack.kernel
def _add_key(cells, cell_keys, keys, bad):
    for c in cells:
        k = cell_keys[cells.entity_id(c)]
        if k < 0 or k >= 16:
            tack.atomic_add(bad, 0, 1)
        keys[cells.index(c)] = keys[cells.index(c)] * 16 + k


@tack.kernel
def _invert(perm, rank):
    for i in range(perm.shape[0]):
        rank[perm[i]] = i


@tack.kernel
def _gather_selected(cells, rank, rows, ids, positions):
    for c in cells:
        p = cells.index(c)
        i = rank[p]
        for j in range(cells.NUM_POINTS):
            rows[i, j] = cells.point_id(c, j)
        ids[i] = cells.entity_id(c)
        positions[i] = p


def _subgroups(group, spaces):
    """``group`` split by the keys ``spaces`` give its cells: one subgroup per
    combination present, made on the device (key per cell, a stable sort, runs)
    and kept on the topology, since the spaces' keys are fixed when they are made.
    A subgroup's cells are gathered into rows, ids and positions sorted by key,
    shared by all the subgroups; each is a slice of them."""
    cache = spaces[0].topology.__dict__.setdefault("_subgroups", {})
    cache_key = (id(group), spaces)
    if cache_key in cache:
        return cache[cache_key]
    if len(spaces) > 7:
        raise NotImplementedError("at most 7 spaces that vary per cell in one launch")
    n = group.count
    cells = group.view()
    keys = tack.field(tack.i32, shape=(n,))
    bad = tack.zeros(tack.i32, (1,))
    _clear_keys(keys)
    for space in spaces:
        _add_key(cells, space.cell_keys(), keys, bad)
    if bad[0]:
        raise ValueError(f"{bad[0]} cells have a key outside [0, {_KEY_BASE})")
    idt = spaces[0].topology.id_dtype
    perm = argsort(keys, index_dtype=idt)
    sorted_keys = gather(keys, perm)
    offsets, runs = _run_offsets(sorted_keys, n, idt)
    bounds = offsets.to_numpy()
    if issubclass(group.kind, views._CellsById):
        # A polyhedral or polyline cell is its id: subgroups select ids, sorted by
        # key (the group's arguments, the count last, then the selection).
        subgroups = []
        for r in range(runs):
            first, count = int(bounds[r]), int(bounds[r + 1] - bounds[r])
            subgroups.append(views.DomainGroup(
                group.kind._selected, group.shape, (*group.args[:-1], count, perm, first),
                count, group.start, parent=group,
                keys=_decode(int(sorted_keys[first]), spaces)))
        cache[cache_key] = subgroups
        return subgroups
    rank = tack.field(idt, shape=(n,))
    _invert(perm, rank)
    rows = tack.field(idt, shape=(n, group.shape.NUM_POINTS))
    cell_ids = tack.field(idt, shape=(n,))
    positions = tack.field(idt, shape=(n,))
    _gather_selected(cells, rank, rows, cell_ids, positions)
    subgroups = []
    for r in range(runs):
        first, count = int(bounds[r]), int(bounds[r + 1] - bounds[r])
        subgroups.append(views.DomainGroup(views._SelectedCells, group.shape,
                                           (rows, cell_ids, count, positions, first), count,
                                           group.start, parent=group,
                                           keys=_decode(int(sorted_keys[first]), spaces)))
    cache[cache_key] = subgroups
    return subgroups


def _decode(combined, spaces):
    """A combined key back into each space's key, in ``spaces`` order."""
    digits = []
    for _ in spaces:
        digits.append(combined % _KEY_BASE)
        combined //= _KEY_BASE
    return dict(zip(spaces, reversed(digits)))


def _check_domain(space, kind):
    """A field's view must make sense where the loop is."""
    if isinstance(space, H1):
        return                                   # points belong to every entity
    if space.on == kind:
        return
    raise TypeError(f"a field on {space!r} cannot be viewed while iterating {kind}")


def rectilinear_grid(x, y=(0.0,), z=(0.0,), dtype=tack.f32):
    """A dataset of the grid whose lines run through ``x``, ``y`` and ``z``."""
    from tack.data.topology import StructuredTopology

    coordinates = CartesianProduct(x, y, z, dtype=dtype)
    return DataSet(StructuredTopology(coordinates.point_dims), coordinates)


def uniform_grid(dims, origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0), direction=None,
                 dtype=tack.f32):
    """A dataset of ``dims`` points (one to three sizes) at ``origin + direction @
    (ijk * spacing)``: VTK's image data, Viskores' uniform data set. Nothing per
    point is stored (``UniformCoordinates``)."""
    from tack.data.topology import StructuredTopology

    coordinates = UniformCoordinates(dims, origin, spacing, direction, dtype=dtype)
    return DataSet(StructuredTopology(coordinates.point_dims), coordinates)
