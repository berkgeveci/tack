"""Datasets: a topology, a geometry, fields on spaces, and sets.

As ``docs/vis-guide/design/dataset-api.md`` §3 describes them:

- *spaces* (``tack.dataset.spaces``) say where a field's values live, on one
  topology, and own the layout that follows: ``H1(data)`` one per point,
  interpolated linearly (point data); ``Constant(data)`` one per cell (cell
  data); ``L2(data)`` each cell's own corner values (a linear discontinuous
  Galerkin field); ``Values(data, on)`` one value per point, edge, face or
  cell, with no basis -- data *about* the entity;
- a *field* is a space and its values, which are any array: a device
  field, or an implicit array from ``tack.arrays`` (a
  ``CartesianProduct`` of three axes, a ``ConstantArray``, a
  ``CountingArray``);
- the *geometry* is a field too, kept apart from the named fields
  (``data.geometry``): ``H1`` positions, one per point (a rectilinear grid's
  are a ``CartesianProduct``), or ``L2`` positions, each cell's own corners;
- *sets* (``tack.dataset.sets``) are named selections of points, cells, faces or
  edges, such as the boundary faces.

``for_each(kernel, data, domain, *args)`` runs ``kernel`` once per shape in
the domain -- ``"cells"``, ``"faces"``, ``"edges"``, or a cell, face or edge set --
passing the domain's view and, for every ``Field`` among ``args``, the
field's view for the same group. When a field's space varies from cell to
cell (an order per cell), each shape group is split by the keys of every
such space among ``args``, and the kernel runs once per subgroup, each
launch specialized to its keys.
"""

import numpy as np

import tack
from tack import arrays
from tack.algorithms.sort import _run_offsets, argsort, gather
from tack.arrays import CartesianProduct, UniformCoordinates
from tack.dataset import bases, shapes, units, views
from tack.dataset.sets import Sets
from tack.dataset.spaces import H1, L2, Space
from tack.runtime.dispatch import as_device, backend_of, get_backend, on_inputs_device, same_device

__all__ = ["DataSet", "Field", "for_each", "rectilinear_grid"]


class Field:
    """A space and its values: any array, explicit or implicit, of scalars or vectors,
    with one value per degree of freedom of the space. Everything else a kernel needs
    to find them belongs to the space.

    ``unit`` is the values' physical unit, a string such as ``"m/s"`` kept in
    unyt's canonical spelling (``tack.dataset.units``), or ``None`` for unknown:
    meaning, which kernels never see. Filters that keep a quantity keep its
    unit; anything else gives ``None`` rather than a guess."""

    def __init__(self, space, values, unit=None):
        if not isinstance(space, Space):
            raise TypeError(f"a field's space is a Space on a topology, not {space!r}")
        views.storage(values)                       # refuses what is not an array
        if arrays.size_of(values) != space.size:
            raise ValueError(f"{space!r} holds {space.size} values, not "
                             f"{arrays.size_of(values)}")
        self.space = space
        self.values = values
        self._unit = units.canonical(unit)
        _ = self._device                    # refuses values on another device

    @property
    def unit(self):
        """The values' unit, a string, or ``None`` if unknown."""
        return self._unit

    @property
    def _device(self):
        """The device of its values and its space's topology, which must agree."""
        space = self.__dict__.get("space")
        return same_device("Field", (space.topology if space is not None else None,
                                     self.__dict__.get("values")))

    @property
    def device(self):
        """The `tack.Device` its values are on."""
        return as_device(self._device or get_backend())

    def view(self, group):
        """This field's view for ``group``: the group's kind and shape, the space, and
        the storage its values are read through. A space that varies reads its key
        for the group from ``group.keys``: launch it through ``for_each``."""
        mixins, attributes = _space_parts(self, group)
        return group.view(*mixins, **attributes)

    def __repr__(self):
        values = self.values
        stored = repr(values) if isinstance(values, arrays._Implicit) else "stored"
        unit = "" if self._unit is None else f", in {self._unit}"
        return f"Field({self.space!r}, {arrays.size_of(values)} values, {stored}{unit})"


def _is_cells(group):
    return issubclass(group.kind, (views._Cells, views._StructuredCells))


def _space_parts(field, group, space_mixin=None, corners_only=False):
    """The mixins and attributes that read ``field`` for ``group``: its space's (or
    ``space_mixin``, for the geometry), an addressing mixin for values shared by
    points, and its storage's."""
    space = field.space
    values = field.values
    if isinstance(space, H1) and space.basis == bases.SERENDIPITY:
        from tack.dataset._serendipity import full_values

        values = full_values(space, values)       # read as the full order-2 space
    storage, attributes = views.storage(values)
    if space.varies and space not in group.keys:
        raise ValueError(f"{space!r} varies from cell to cell: its views come from the "
                         "subgroups for_each makes")
    if (isinstance(space, H1) and space.order >= 2 and not corners_only
            and not _is_cells(group)):
        raise TypeError(f"{space!r} is read cell by cell: on faces, take its traces "
                        "(tack.dataset.traces)")
    mixins = [space_mixin or space.mixin_for(group.keys.get(space), group.shape)]
    if isinstance(space, H1):
        # A grid's coordinates read a corner by (i, j, k) when their axes are
        # the topology's; a grid with a size of 1 before another axis has its
        # topology's axes renumbered (grid_topology), and reads flat indices.
        aligned = (getattr(field.values, "point_dims", None)
                   == getattr(space.topology, "point_dims", ()))
        mixins.append(views.point_address(group.kind, storage, aligned))
    if not space_mixin:
        # A geometry's incidence is the domain view's own.
        mixins.extend(space.extra_mixins(group))
    mixins.append(storage)
    # Reading only the points' values -- an order-2 geometry on faces and edges
    # -- needs none of the space's own layout.
    layout = {} if corners_only else space.attributes(group, dtype=_basis_type(field))
    for name, value in layout.items():
        # The geometry's L2 offsets sit beside the domain view's own names.
        attributes["point_offsets" if space_mixin and name == "offsets" else name] = value
    return mixins, attributes


def _basis_type(field):
    """The float type ``field``'s basis is evaluated in: its own (``spaces._float_type``)."""
    from tack.dataset.spaces import _float_type

    return _float_type(arrays.dtype_of(field.values), backend_of(field.device))


# ── Geometry ────────────────────────────────────────────────────────

def as_origin(origin):
    """``origin`` as three Python floats: ``(0.0, 0.0, 0.0)`` for ``None``; refuses
    anything else that is not three finite numbers."""
    if origin is None:
        return (0.0, 0.0, 0.0)
    try:
        values = np.asarray(origin, np.float64).reshape(-1)
    except (TypeError, ValueError):
        values = None
    if values is None or values.shape != (3,) or not np.all(np.isfinite(values)):
        raise ValueError(f"an origin is three finite numbers, not {origin!r}")
    return tuple(float(v) for v in values)


def same_origin(what, datasets):
    """The origin every one of ``datasets`` has, or ``ValueError`` naming ``what``
    if two differ: positions compared or moved between datasets must be relative
    to one origin."""
    datasets = list(datasets)
    units.common(f"{what}'s geometry", [d.unit for d in datasets])
    origins = {d.origin for d in datasets}
    if len(origins) > 1:
        raise ValueError(f"{what} needs its datasets on one origin, not "
                         f"{sorted(origins)}: rebase them to one first (DataSet.rebased)")
    return origins.pop() if origins else (0.0, 0.0, 0.0)


def relative_positions(where, origin, unit=None):
    """``where``'s points relative to ``origin``, as a float64 host array ``(n, 3)``:
    a dataset's positions plus the difference of the two origins (none when they
    are equal), or an array of world positions less ``origin``. A dataset in a
    spatial unit other than ``unit`` is refused (positions are not converted)."""
    origin = np.asarray(as_origin(origin))
    if isinstance(where, DataSet):
        units.check("the points' spatial unit and the dataset's", where.unit, unit)
        x = where.positions().astype(np.float64)
        return x if where.origin == tuple(origin) else x + (np.asarray(where.origin) - origin)
    return np.asarray(where, np.float64).reshape(-1, 3) - origin


def _as_geometry(geometry, topology, dtype, unit=None):
    """``geometry`` as a ``Field``: positions per point (a host array, or any array of
    3-vectors, such as a ``CartesianProduct``) become an ``H1`` field on ``topology``
    in ``unit``; a ``Field`` is checked, and given ``unit`` if it has none (two
    units are refused)."""
    if isinstance(geometry, Field) and unit is not None:
        unit = units.canonical(unit)
        if geometry.unit is None:
            geometry = Field(geometry.space, geometry.values, unit)
        elif not units.same(geometry.unit, unit):
            raise ValueError(f"the geometry is in {geometry.unit!r}, not {unit!r}")
    if not isinstance(geometry, Field):
        try:
            views.storage(geometry)
        except TypeError:
            positions = np.ascontiguousarray(geometry)
            geometry = tack.empty((positions.shape[0],), tack.vec(3, dtype))
            tack.copyto(geometry, positions.astype(dtype.numpy_dtype))
        geometry = Field(H1(topology), geometry, unit)
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
        if geometry.space.tensor:
            # Curved, each cell its own nodes (a mesh displaced by a DG field, a
            # periodic mesh of order 2 or more).
            space = geometry.space
            bernstein = space.basis == bases.BERNSTEIN
            if bases.simplex_shape(group.shape):
                mixins, attributes = _space_parts(geometry, group, views.of_simplex_order(
                    views._L2SimplexGeometry, space.order, group.shape.DIMENSION, bernstein))
                return [*mixins, views._SimplexNewton], attributes
            if bases.wedge_shape(group.shape):
                mixins, attributes = _space_parts(geometry, group, views.of_wedge_order(
                    views._L2WedgeGeometry, space.order, bernstein))
                return [*mixins, views._WedgeNewton], attributes
            if not bases.tensor_shape(group.shape):
                raise NotImplementedError(
                    f"an L2 geometry of order {space.order} on {group.shape.__name__} "
                    "cells: curved L2 geometries are on segments, quads, hexahedra, "
                    "triangles, tetrahedra and wedges")
            mixin = (views._L2CellsTensorGeometry if space.layout == "cells"
                     else views._L2TensorGeometry)
            mixins, attributes = _space_parts(
                geometry, group, views.of_order(mixin, space.order,
                                                space.basis == bases.BERNSTEIN))
            return [*mixins, views._TensorNewton], attributes
        mixins, attributes = _space_parts(geometry, group, views._L2Geometry)
        if group.shape not in (shapes.Polyhedron, shapes.Polygon, shapes.PolyLine):
            mixins = [*mixins, views._CornerBox, views._CurvedNewton]
        return mixins, attributes
    if group.shape in (shapes.Polyhedron, shapes.Polygon, shapes.PolyLine):
        # No reference element: the points' positions, nothing to map.
        return _space_parts(geometry, group, views._PointGeometry)
    bernstein = geometry.space.basis == bases.BERNSTEIN
    high = geometry.space.order >= 3 or (geometry.space.order == 2 and (
        bernstein or getattr(geometry.space, "dofs", None) is not None))
    if geometry.space.order == 2 and kind == "cells" and not bases.tensor_shape(group.shape) \
            and not high:
        # Curved: the domain view's incidence gives the quadratic nodes' ids.
        data.topology.edges()
        if group.shape.QUADRATIC_FACES:
            data.topology.faces()
        mixins, attributes = _space_parts(geometry, group, views._H1Order2Geometry)
        return [*mixins, views.quadratic_faces(group.shape), views._QuadraticNewton], attributes
    if high and kind == "cells" and bases.wedge_shape(group.shape):
        # Curved wedges of order 3 and up (2 too in a Bernstein basis).
        space = geometry.space
        data.topology.edges()
        data.topology.faces()
        mixins, attributes = _space_parts(
            geometry, group, space._tabled(views.of_wedge_order(views._H1WedgeGeometry,
                                                                space.order, bernstein)))
        entities = () if space.dofs is not None else views.wedge_mixins(group.shape)
        return [*mixins, *entities, views._WedgeNewton], attributes
    if high and kind == "cells" and bases.simplex_shape(group.shape):
        # Curved simplices of order 3 and up (2 too in a Bernstein basis): the simplex
        # basis of the geometry's order.
        space = geometry.space
        data.topology.edges()
        if group.shape.DIMENSION == 3:
            data.topology.faces()
        mixins, attributes = _space_parts(
            geometry, group, space._tabled(views.of_simplex_order(
                views._H1SimplexGeometry, space.order, group.shape.DIMENSION, bernstein)))
        entities = () if space.dofs is not None else views.simplex_mixins(group.shape)
        return [*mixins, *entities, views._SimplexNewton], attributes
    if geometry.space.order >= 2 and kind == "cells":
        # Curved, on tensor shapes (order 2 too: the same layout, read as a tensor
        # basis, which bounds the cell by its Bernstein control points).
        space = geometry.space
        if space._has_edges(group.shape):
            data.topology.edges()
        if space._has_faces(group.shape):
            data.topology.faces()
        mixins, attributes = _space_parts(
            geometry, group, space._tabled(views.of_order(views._H1TensorGeometry, space.order,
                                                          space.basis == bases.BERNSTEIN)))
        # A table numbers the nodes itself: no edges or faces to read.
        entities = () if space.dofs is not None else views.tensor_mixins(group.shape)
        return [*mixins, *entities, views._TensorNewton], attributes
    # Faces and edges of a curved geometry see its corners.
    mixins, attributes = _space_parts(geometry, group, views._H1Geometry, corners_only=True)
    if kind == "cells":
        mixins = [*mixins, views._CornerBox, views._CurvedNewton]
    return mixins, attributes


# ── Datasets ────────────────────────────────────────────────────────

@tack.kernel
def _transposed(values, rows, cols, out):
    # values as rows x cols, row-major, into out as cols x rows.
    for i in range(values.shape[0]):
        out[(i % cols) * rows + i // cols] = values[i]


def _relaid(values, source, space):
    """``values`` of ``source`` laid out as ``space`` lays them out: blocks of one size
    to cell-fastest, or back -- a transpose."""
    n = space.topology.num_cells
    per_cell = space.size // max(n, 1)
    out = tack.empty_like(values)
    if arrays.size_of(values):
        rows, cols = (n, per_cell) if source.layout == "blocks" else (per_cell, n)
        _transposed(arrays.materialize(values), rows, cols, out)
    return out


@tack.kernel
def _copy_edge_traces(cells, edge_nodes, view_of, q, edges):
    # Each of the cells' edges' geometry values, from the edge's first corner by
    # position: every cell holding an edge copies the same values to the same places.
    # view_of: a lexicographic node's number in the geometry view's dof_index.
    for c in cells:
        for e in range(cells.NUM_EDGES):
            flip = views._before(cells.corner_position(c, cells.edge_corner(e, 1)),
                           cells.corner_position(c, cells.edge_corner(e, 0)))
            base = cells.edge_id(c, e) * q
            for k in range(q):
                j = q - 1 - k if flip else k
                edges[base + k] = cells.get(cells.dof_index(c, view_of[edge_nodes[e * q + j]]))


@tack.kernel
def _copy_face_traces(cells, face_nodes, q, faces):
    # Each of the cells' quad faces' geometry values, in the face's frame
    # (views._square_frame's), as _copy_edge_traces copies edges'.
    for c in cells:
        for f in range(cells.NUM_FACES):
            if cells.face_num_points(f) == 4:
                a = 0
                for k in range(1, 4):
                    if views._before(cells.corner_position(c, cells.face_corner(f, k)),
                               cells.corner_position(c, cells.face_corner(f, a))):
                        a = k
                b = (a + 1) % 4
                d = (a + 3) % 4
                if views._before(cells.corner_position(c, cells.face_corner(f, d)),
                           cells.corner_position(c, cells.face_corner(f, b))):
                    b, d = d, b
                ax, ay = views._square_corner(a)
                bx, by = views._square_corner(b)
                dx, dy = views._square_corner(d)
                p = q - 1
                base = cells.face_id(c, f) * q * q
                for k in range(q * q):
                    fa = ax * p + (k % q) * (bx - ax) + (k // q) * (dx - ax)
                    fb = ay * p + (k % q) * (by - ay) + (k // q) * (dy - ay)
                    faces[base + k] = cells.get(cells.dof_index(c, face_nodes[f * q * q + fa
                                                                               + q * fb]))


def _entity_traces(data, faces=True):
    """A curved geometry's values on every edge and (``faces``, tensor cells only) quad
    face, ``q`` a side (``q - 1`` its order), for ``views._CurvedNodePosition`` and
    ``extract_edges``' subdivided edges: ``(edges, faces, attributes)``, the
    attributes its 1D basis."""
    from tack.dataset import bases

    geometry = data.geometry
    space = geometry.space
    q = space.order + 1
    topology = data.topology
    vec = arrays.element_type(geometry.values)
    edges = tack.empty((max(topology.edges().num_edges * q, 1),), vec)
    solid = faces and any(g.count and g.shape.DIMENSION == 3 for g in topology.groups())
    faces = tack.empty((max(topology.faces().num_faces * q * q, 1) if solid else 1,), vec)
    for group in topology.groups():
        if group.count and group.shape.DIMENSION > 1:
            shape = group.shape
            edge_nodes, face_nodes = (
                tack.array(x.reshape(-1) if x.size else np.zeros(1, np.int32))
                for x in bases.entity_nodes(shape, space.order))
            tensor = not (bases.simplex_shape(shape) or bases.wedge_shape(shape))
            view_of = tack.array(
                np.arange(q ** shape.DIMENSION, dtype=np.int32) if tensor else
                np.asarray(bases.view_index(shape, space.order, space._any_order(shape)),
                           np.int32))
            cells = data.domain_view("cells", group)
            _copy_edge_traces(cells, edge_nodes, view_of, q, edges)
            if solid and shape.DIMENSION == 3 and tensor:
                _copy_face_traces(cells, face_nodes, q, faces)
    family = space.basis or "gauss_lobatto"
    dtype = _basis_type(geometry)
    tables = []
    for values in (bases.nodes(family, space.order), bases.weights(family, space.order)):
        out = tack.empty((len(values),), dtype)
        tack.copyto(out, np.asarray(values).astype(dtype.numpy_dtype))
        tables.append(out)
    return edges, faces, {"trace_q": q, "trace_nodes": tables[0], "trace_weights": tables[1],
                          "trace_bernstein": int(family == bases.BERNSTEIN)}


def _space_nodes(space, geometry):
    """``space``'s 1D nodes as a device array in the geometry's float type, where
    node views place its nodes."""
    from tack.dataset import bases

    dtype = _basis_type(geometry)
    key = ("_node_points", space.basis or "gauss_lobatto", space.order, dtype)
    cache = space.__dict__.setdefault("_tables", {})
    if key not in cache:
        values = bases.nodes(key[1], space.order)
        out = tack.empty((len(values),), dtype)
        tack.copyto(out, values.astype(dtype.numpy_dtype))
        cache[key] = out
    return cache[key]


class DataSet:
    """A topology, a geometry, named fields and named sets. The geometry is a
    ``Field`` in ``H1`` or ``L2`` of 3-vectors -- or positions per point (a host
    array, or any array of 3-vectors, such as a device array or a
    ``CartesianProduct``), which become an ``H1`` field -- kept apart from
    ``fields`` as ``data.geometry``: algorithms move it with the cells, never as
    one of the fields they carry or select. ``ghosts``, a
    ``tack.dataset.Ghosts`` or ``None``, marks the cells and points
    another piece of a partitioned dataset owns; ``global_ids``, a
    ``tack.dataset.GlobalIds`` or ``None``, numbers them across the pieces.

    ``origin``, three floats kept in double precision on the host, is where the
    geometry's coordinates start: a position is ``origin + value``, the value
    in the geometry's type (``tack.f32``, or ``tack.f64`` on a device that has
    it). Kernels see only the values, so coordinates far from zero -- a LiDAR
    survey's eastings and northings -- keep their precision in single
    precision. Filters keep it; what takes or gives absolute positions
    (implicit functions, probe points, seeds, ``world_positions``, the VTK
    interop) converts on the host in double precision.

    ``unit`` is the spatial unit, the geometry field's (``DataSet.unit``), in
    which ``origin`` is too; a geometry ``Field`` brings its own.

    ``time`` is when the data are, a double on the host or ``None``, in
    ``time_unit`` (a unit of time, or ``None`` for unknown): one step of a
    simulation as it streams through filters, which keep both, as they keep
    the origin. Tack holds no series of steps; filters that need earlier steps
    keep what they need (``tack.dataset.temporal``)."""

    @on_inputs_device
    def __init__(self, topology, geometry, fields=None, sets=None, dtype=tack.f32,
                 ghosts=None, global_ids=None, origin=None, unit=None, time=None,
                 time_unit=None):
        self.topology = topology
        self._geometry = _as_geometry(geometry, topology, dtype, unit)
        self.origin = origin
        self.time = time
        self.time_unit = time_unit
        self.fields = dict(fields or {})
        for name, field in self.fields.items():
            if field.space.topology is not topology:
                raise ValueError(f"field {name!r} is on another topology")
        self.sets = sets
        # Set by a filter that dropped its input's ghost marks (ghosts.drop_marks).
        self._ghosts_lost = False
        self.ghosts = ghosts
        self.global_ids = global_ids
        _ = self._device                    # refuses parts on two devices

    @property
    def _device(self):
        """The device its topology, geometry, fields and sets are on; parts on two
        devices are refused (the check every algorithm makes on entry, through
        `on_inputs_device`)."""
        parts = self.__dict__
        held = [parts.get("topology")]
        if parts.get("_geometry") is not None:
            held.append(parts["_geometry"].values)
        held.extend(field.values for field in parts.get("fields", {}).values())
        held.extend(s.ids for s in (parts.get("_sets") or {}).values())
        held.append(parts.get("_ghosts"))
        held.append(parts.get("_global_ids"))
        return same_device("DataSet", held)

    @property
    def device(self):
        """The `tack.Device` it is on: where its algorithms run and allocate."""
        return as_device(self._device or get_backend())

    def to_device(self, device):
        """This dataset on ``device``: itself if it is already there, else a copy --
        the topology made from its arrays copied there, each field in the same
        space on it, the sets by the same ids. What the topology derives (faces,
        groups, layouts) is derived there afresh, not copied."""
        from tack.dataset.sets import Set

        target = as_device(backend_of(device))
        here = self._device
        if here is not None and target._backend.memory_domain == here.memory_domain:
            return self
        topology = self.topology.to_device(target)

        def moved(field):
            space = field.space._rebuilt_on(topology)
            values = field.values.to_device(target)
            if isinstance(space, L2) and space.layout != field.space.layout:
                values = _relaid(values, field.space, space)      # "auto" chose another
            return Field(space, values, field.unit)

        sets = {name: Set._from_sorted_ids(topology, s.on, s.ids.to_device(target))
                for name, s in self.sets.items()}
        ghosts = None if self.ghosts is None else self.ghosts.to_device(target)
        global_ids = None if self.global_ids is None else self.global_ids.to_device(target)
        out = DataSet(topology, moved(self.geometry),
                      {name: moved(field) for name, field in self.fields.items()},
                      sets, dtype=self.dtype, ghosts=ghosts, global_ids=global_ids,
                      **self._frame())
        out._ghosts_lost = self._ghosts_lost
        return out

    @property
    def sets(self):
        """Named ``Set``s of this dataset's points, cells, faces or edges, checked as
        they are put in: a set of another topology, or named for a domain, is
        refused."""
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
        raise ValueError(f"no domain {domain!r}: cells, faces, edges, or a cell, face or "
                         f"edge set ({', '.join(self.sets) or 'none'})")

    @property
    def ghosts(self):
        """The ghost marks of its cells and points (``tack.dataset.Ghosts``),
        or ``None``: which cells and points another piece owns, in VTK's
        ``vtkGhostType`` bits.

        Filters that keep the cells and points (``warp``, ``transform``,
        ``shrink``, ``as_polyhedra``, ``probe``'s target, ...) keep the marks;
        others do not carry them yet, and their output remembers that it lost
        them, so ``statistics`` and ``entropy`` refuse it rather than count
        shared cells and points once per piece. Setting marks again clears
        that."""
        return self._ghosts

    @ghosts.setter
    def ghosts(self, ghosts):
        from tack.dataset.ghosts import Ghosts

        if ghosts is not None:
            if not isinstance(ghosts, Ghosts):
                raise TypeError(f"ghost marks are a tack.dataset.ghosts.Ghosts, not "
                                f"{type(ghosts).__name__}")
            ghosts._check(self.topology)
            self._ghosts_lost = False
        self._ghosts = ghosts

    @property
    def global_ids(self):
        """The global ids of its points and cells (``tack.dataset.GlobalIds``), or
        ``None``: each one's number across every piece of a partitioned dataset.
        Filters carry them where an output entity is an input entity (kept points,
        kept or extracted cells); what a filter makes has none."""
        return self._global_ids

    @global_ids.setter
    def global_ids(self, global_ids):
        from tack.dataset.global_ids import GlobalIds

        if global_ids is not None:
            if not isinstance(global_ids, GlobalIds):
                raise TypeError(f"global ids are a tack.dataset.GlobalIds, not "
                                f"{type(global_ids).__name__}")
            global_ids._check(self.topology)
        self._global_ids = global_ids

    @property
    def origin(self):
        """Where the geometry's coordinates start, ``(x, y, z)`` in double precision:
        a position is ``origin`` plus the geometry's value. ``(0.0, 0.0, 0.0)``
        unless set."""
        return self._origin

    @origin.setter
    def origin(self, origin):
        self._origin = as_origin(origin)

    @property
    def time(self):
        """When the data are, a float, or ``None``: one step's time, in
        ``time_unit``."""
        return self._time

    @time.setter
    def time(self, time):
        if time is not None:
            value = float(time)
            if not np.isfinite(value):
                raise ValueError(f"a time is a finite number or None, not {time!r}")
            time = value
        self._time = time

    @property
    def time_unit(self):
        """The unit of ``time``, a unit of time, or ``None`` if unknown."""
        return self._time_unit

    @time_unit.setter
    def time_unit(self, unit):
        self._time_unit = units.time_unit(unit)

    def _frame(self):
        """Where and when its positions are, as keywords for a ``DataSet`` made from
        it: ``origin``, ``unit``, ``time``, ``time_unit``."""
        return {"origin": self.origin, "unit": self.unit, "time": self.time,
                "time_unit": self.time_unit}

    @property
    def geometry(self):
        """The geometry, a field of positions in ``H1`` or ``L2``, relative to
        ``origin``."""
        return self._geometry

    @geometry.setter
    def geometry(self, geometry):
        # Positions given as values stay in the spatial unit; a Field brings its own.
        unit = None if isinstance(geometry, Field) else self.unit
        self._geometry = _as_geometry(geometry, self.topology, self.dtype, unit)

    @property
    def unit(self):
        """The spatial unit, the geometry's (``origin`` is in it too), a string, or
        ``None`` if unknown."""
        return self._geometry.unit

    @property
    def dtype(self):
        """The geometry's floating-point type."""
        return arrays.dtype_of(self.geometry.values)

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
        geometry's values, relative to ``origin`` (``world_positions`` adds it).
        An ``L2`` geometry has none per point."""
        if isinstance(self.geometry.space, L2):
            raise ValueError("an L2 geometry has positions per cell corner, not per point")
        # An order-2 geometry's first values are the points'.
        return arrays.to_host(self.geometry.values)[:self.num_points]

    def rebased(self, origin):
        """This dataset with its geometry relative to ``origin`` instead: the same
        world positions, each value moved by the difference of the two origins
        (taken in double precision, added in the geometry's type). An image
        stays an image, its corner moved in double precision, so an image at
        survey coordinates rebased to a nearby origin holds its points exactly;
        a rectilinear grid moves its axes. Other implicit geometry becomes
        positions per point."""
        from tack.arrays import CartesianProduct, UniformCoordinates
        from tack.dataset.algorithms._moving import _moved, _shifted

        origin = as_origin(origin)
        if origin == self.origin:
            return self
        offset = np.asarray(self.origin) - np.asarray(origin)
        values = self.geometry.values
        if isinstance(values, UniformCoordinates):
            moved = UniformCoordinates(values.dims, np.asarray(values._origin) + offset,
                                       values._spacing,
                                       values._matrix if values.ORIENTED else None,
                                       dtype=values.dtype, device=self.device)
        elif isinstance(values, CartesianProduct):
            axes = [arrays.to_host(axis).astype(np.float64).reshape(-1) + offset[a]
                    for a, axis in enumerate((values.xs, values.ys, values.zs))]
            moved = CartesianProduct(*axes, dtype=values.dtype, device=self.device)
        else:
            moved = _shifted(self, offset)
        out = _moved(self, moved)
        out.origin = origin
        return out

    def world_positions(self):
        """Every point's position as a float64 host array, ``(num_points, 3)``:
        ``origin`` plus ``positions()``, added in double precision."""
        return self.positions().astype(np.float64) + np.asarray(self.origin)

    @on_inputs_device
    def domain_groups(self, domain):
        """The ``DomainGroup``s of ``domain``: ``"cells"``, ``"faces"``, ``"edges"``, or
        the name of a cell, face or edge set in ``sets`` -- a cell set's as the
        cell groups' subgroups of its members."""
        kind = self._kind(domain)
        if domain == "cells":
            return self.topology.groups()
        if domain == "faces":
            return self.topology.faces().groups()
        if domain == "edges":
            return self.topology.edges().groups()
        if kind == "faces":
            return self.topology.faces().groups(self.sets[domain].ids)
        if kind == "edges":
            return self.topology.edges().groups(self.sets[domain].ids)
        return self.launch_groups(domain)

    @on_inputs_device
    def domain_view(self, domain, group, extra=(), **extra_attributes):
        """``group``'s view: kind, shape, geometry, and for cells their incidence when
        faces and edges have been derived; ``extra`` mixins and attributes last
        (a space's node mixins, ``node_views``)."""
        mixins = []
        kind = self._kind(domain)
        geometry, attributes = _geometry_parts(self, group, kind)
        mixins.extend(geometry)
        if kind == "cells" and issubclass(group.kind, views._CellsById):
            mixins.append(views._PolyhedralWalk)
            if group.shape is shapes.Polyhedron:
                mixins.append(views._PolyhedralLocation)
            elif group.shape is shapes.Polygon:
                mixins.append(views._PolygonLocation)
            elif group.shape is shapes.PolyLine:
                mixins.append(views._PolylineLocation)
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
            if faces is not None and group.root in getattr(faces, "group_starts", {}):
                mixins.extend((views._FaceIncidence, views._FaceWalk))
                attributes.update(side_face=faces.side_face, side_slot=faces.side_slot,
                                  side_orientation=faces.side_orientation,
                                  face_start=faces.group_starts[group.root])
            elif group.shape.NUM_FACES and not issubclass(group.kind, (views._CellsById,
                                                                       views._PolylineCells)):
                # A shape's faces walk from its tables alone, with no derived
                # faces: a fresh mesh's volumes need not sort them.
                mixins.append(views._FaceWalk)
            edges = self.topology._edges
            if edges is not None and group.shape.NUM_EDGES:
                mixins.append(views._EdgeIncidence)
                attributes.update(side_edge=edges.side_edge, side_sign=edges.side_sign,
                                  edge_start=edges.group_starts[group.root])
        # A key that shapes the kernel -- a size bucket's cap -- brings its mixin.
        for key, value in group.keys.items():
            if hasattr(key, "domain_mixin"):
                mixins.append(key.domain_mixin(value))
        mixins.extend(extra)
        attributes.update(extra_attributes)
        return group.view(*mixins, **attributes)

    def _tensor_nodes(self, space):
        """``node_views``' edges, quad faces and interiors of an ``H1`` space of order 3
        or more. An edge's and a face's nodes are placed on its straight geometry, are
        the geometry's own values when it has the space's layout, or the geometry's
        trace there when it is curved and of another order (``_entity_traces``; on
        hexahedra, quads and their kin, refused on curved simplices and wedges for
        now)."""
        geometry = self.geometry
        face_dofs, cell_dofs, _ = space._layout()
        points = _space_nodes(space, geometry)
        same = geometry.space is space
        where, edge_trace, face_trace = views._StraightNodePosition, {}, {}
        if same:
            where = views._GeometryAtDof
        elif geometry.space.order > 1:
            if any(g.count and (bases.simplex_shape(g.shape) or bases.wedge_shape(g.shape)
                                or not bases.has_bases(g.shape))
                   for g in self.topology.groups()):
                raise NotImplementedError(
                    f"the nodes of {space!r} on a curved geometry of another order "
                    f"({geometry.space!r}) on cells other than tensor ones: interpolate "
                    "into the geometry's own space, or make the geometry straight")
            edges, faces, basis = _entity_traces(self)
            where = views._CurvedNodePosition
            edge_trace, face_trace = dict(basis, trace=edges), dict(basis, trace=faces)
        out = []
        topology = self.topology
        dims = {g.shape.DIMENSION for g in topology.groups() if g.count}
        if dims - {1}:
            for group in topology.edges().groups():
                if group.count:
                    out.append(self.domain_view(
                        "edges", group, (views.of_node_order(views._TensorEdgeNodes, space.order),
                                         where),
                        node_edge_base=self.num_points, node_points=points, **edge_trace))
        if 3 in dims:
            for group in topology.faces().groups():
                if group.count and group.shape is shapes.Quad:
                    out.append(self.domain_view(
                        "faces", group, (views.of_node_order(views._TensorFaceNodes, space.order),
                                         where),
                        node_face_dofs=face_dofs, node_points=points, **face_trace))
                elif group.count and group.shape is shapes.Triangle and space.order > 2:
                    out.append(self.domain_view(
                        "faces", group, (views.of_node_order(views._SimplexFaceNodes, space.order),
                                         where),
                        node_face_dofs=face_dofs, node_points=points))
        for group in topology.groups():
            if not group.count:
                continue
            if bases.wedge_shape(group.shape):
                kind = bases.dof_tables(group.shape, space.order)[0]
                lattice = bases.wedge_lattice(space.order)[kind == bases.INTERIOR]
                table = tack.empty((max(lattice.size, 3),), tack.i32)
                tack.copyto(table, np.r_[lattice.reshape(-1), np.zeros(max(0, 3 - lattice.size),
                                                                        np.int32)])
                if not lattice.size:
                    continue                          # order 3: no interior nodes
                out.append(self.domain_view(
                    "cells", group,
                    (views.of_node_order(views._TableInteriorNodes, space.order),),
                    node_cell_dofs=cell_dofs, node_points=points, node_interior=table))
                continue
            interior = (views._SimplexInteriorNodes if bases.simplex_shape(group.shape)
                        else views._TensorInteriorNodes)
            out.append(self.domain_view(
                "cells", group, (views.of_node_order(interior, space.order),),
                node_cell_dofs=cell_dofs, node_points=points))
        return out

    @on_inputs_device
    def node_views(self, space):
        """Views over ``space``'s nodes, each visited once: entity views -- points,
        edges, quad faces, cells -- whose ``node_count(i)`` nodes have
        ``node_dof(i, j)``, where the node's value is in ``space``'s values, and
        ``node_position(i, j)``, where the geometry puts it. ``for_each(kernel,
        data, space, ...)`` runs a kernel over them:

            for i in nodes:
                for j in range(nodes.node_count(i)):
                    values[nodes.node_dof(i, j)] = f(nodes.node_position(i, j))

        ``H1``'s nodes are its points and, at order 2, its edges' middles, its
        quad faces' centers and its hexahedra's and quads' interiors; a shared
        node's position is the mean of its entity's points on a straight
        geometry, the geometry's own value there on one of the space's layout,
        and on a curved geometry of another order (tensor cells) its trace on
        the edge or face at the node. ``L2``'s
        are its cells' corners (order 1) or parametric centers (order 0),
        ``Constant``'s and values on cells' the parametric centers, values on
        points' the points. Shared nodes need positions per point: an ``L2``
        geometry has none, and is refused for them. On polyhedra, polygons and
        polylines only point nodes exist."""
        from tack.dataset.spaces import Values, reference_cells

        if space.topology is not self.topology:
            raise ValueError("a space of another topology")
        geometry = self.geometry
        point_nodes = ((isinstance(space, H1) and space.order == 1)
                       or (isinstance(space, Values) and space.on == "points"))
        shared = point_nodes or isinstance(space, H1)
        if shared and not isinstance(geometry.space, H1):
            raise NotImplementedError(
                f"the nodes of {space!r} are shared by cells, and an L2 geometry gives a "
                "shared node no single position")
        out = []
        if shared and self.num_points:
            group = views.DomainGroup(views._Points, shapes.Vertex, (self.num_points,),
                                self.num_points, 0)
            mixins, attributes = _geometry_parts(self, group, "points")
            out.append(group.view(*mixins, views._PointDof, views._MeanPosition, **attributes))
        if point_nodes:
            return out
        if isinstance(space, Values) and space.on not in ("points", "cells"):
            raise TypeError(f"values on {space.on} have no nodes to visit as such; "
                            f"for_each over {space.on!r} visits them")
        reference_cells(self.topology, f"the nodes of {space!r}")
        # Order 2 on a geometry curved to a higher order needs its traces too.
        if isinstance(space, H1) and (space.order > 2 or (space.order == 2 and (all(
                space._any_order(g.shape) for g in self.topology.groups() if g.count)
                or geometry.space.order > 2))):
            return out + self._tensor_nodes(space)
        if isinstance(space, H1):
            face_dofs, cell_dofs, _ = space._layout()
            curved = geometry.space.order == 2
            where = views._GeometryAtDof if curved else views._MeanPosition
            for group in self.topology.edges().groups():
                if group.count:
                    out.append(self.domain_view("edges", group, (views._EdgeDof, where),
                                                node_edge_base=self.num_points))
            if space.basis == bases.SERENDIPITY:
                return out                       # no face centres or insides stored
            if any(g.count and g.shape.QUADRATIC_FACES for g in self.topology.groups()):
                for group in self.topology.faces().groups():
                    if group.count and group.shape is shapes.Quad:
                        out.append(self.domain_view("faces", group, (views._FaceDof, where),
                                                    node_face_dofs=face_dofs))
            for group in self.topology.groups():
                if group.count and group.shape.QUADRATIC_INTERIOR:
                    out.append(self.domain_view("cells", group, (views._InteriorDof, where),
                                                node_cell_dofs=cell_dofs))
            return out
        groups = (self.launch_groups("cells", keys=(space,)) if space.varies
                  else self.topology.groups())
        out += [view for _, view in self._cell_node_views(space, groups)]
        return out

    def _cell_node_views(self, space, groups):
        """``(group, view)`` for each of ``groups``' cell views over ``space``'s nodes in
        the cells (``node_views``' last part): ``node_count``, ``node_dof``,
        ``node_position`` and ``node_parametric_point``."""
        from tack.dataset.spaces import Constant, Values

        out = []
        for group in groups:
            if not group.count:
                continue
            if isinstance(space, L2) and space.tensor and (bases.simplex_shape(group.shape)
                                                           or bases.wedge_shape(group.shape)):
                lattice = bases.lattice_ijk(group.shape, space.order).reshape(-1)
                ijk = tack.empty((len(lattice),), tack.i32)
                tack.copyto(ijk, lattice)
                nodes_of = views.of_node_order(views._SimplexCellNodes, space.order,
                                               len(lattice) // group.shape.DIMENSION)
                view = self.domain_view("cells", group, (nodes_of,),
                                        node_offsets=space.offsets, node_ijk=ijk,
                                        node_points=_space_nodes(space, self.geometry))
            elif isinstance(space, L2) and space.tensor:
                cells = space.layout == "cells"
                mixin = views._TensorCellNodesCells if cells else views._TensorCellNodes
                view = self.domain_view("cells", group, (views.of_node_order(mixin, space.order),),
                                        node_offsets=space.offsets, node_cells=self.num_cells,
                                        node_points=_space_nodes(space, self.geometry))
            elif isinstance(space, L2):
                order = group.keys[space] if space.varies else space.order
                dof = views._CornerDof if order == 1 else views._OffsetDof
                place = views._CornerPosition if order == 1 else views._CenterPosition
                view = self.domain_view("cells", group, (dof, place), node_offsets=space.offsets)
            elif isinstance(space, (Constant, Values)):
                view = self.domain_view("cells", group, (views._CellDof, views._CenterPosition))
            else:
                raise TypeError(f"no nodes for {space!r}")
            out.append((group, view))
        return out

    @on_inputs_device
    def _side_zero_points(self):
        """An L2 geometry's current corners per (face, side, point).

        Geometry values are mutable, so field identity cannot cache positions.
        """
        from tack.dataset.algorithms.traces import traces

        return traces(self, self.geometry).values

    @on_inputs_device
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


@on_inputs_device
def for_each(kernel, data, domain, *args):
    """Run ``kernel(view, *args)`` once per shape in ``domain`` of ``data``.

    ``view`` is the domain's view for that shape (``for i in view``); every
    ``Field`` among ``args`` becomes its view for the same shape, so
    ``u.value(i, pc)`` and ``u.at(i)`` read the entity ``i`` the loop is at.
    Other arguments pass through.

    ``domain`` may also be a ``Space``: the kernel then runs over the space's
    nodes (``DataSet.node_views``), taking arrays, not fields.
    """
    if isinstance(domain, Space):
        if any(isinstance(arg, Field) for arg in args):
            raise TypeError("a kernel over a space's nodes takes arrays, not fields: "
                            "pass field.values")
        for view in data.node_views(domain):
            kernel(view, *args)
        return
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
    cache_key = (group, spaces)
    if cache_key in cache:
        return cache[cache_key]
    if len(spaces) > 7:
        raise NotImplementedError("at most 7 spaces that vary per cell in one launch")
    n = group.count
    cells = group.view()
    keys = tack.empty((n,), tack.i32)
    bad = tack.zeros((1,), tack.i32)
    _clear_keys(keys)
    for space in spaces:
        _add_key(cells, space.cell_keys(), keys, bad)
    if bad[0]:
        raise ValueError(f"{bad[0]} cells have a key outside [0, {_KEY_BASE})")
    idt = spaces[0].topology.id_dtype
    perm = argsort(keys, index_dtype=idt)
    sorted_keys = gather(keys, perm)
    offsets, runs = _run_offsets(sorted_keys, n, idt)
    bounds = offsets.numpy()
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
    rank = tack.empty((n,), idt)
    _invert(perm, rank)
    rows = tack.empty((n, group.shape.NUM_POINTS), idt)
    cell_ids = tack.empty((n,), idt)
    positions = tack.empty((n,), idt)
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


def grid_topology(dims, id_dtype=None, device=None, offset=None):
    """The topology of a grid of ``dims`` points (one to three sizes, x fastest).

    A size of 1 adds no cells, in any position: ``(1, 3)`` is a line of three
    points (along y; the coordinates say where), as ``(3, 1)`` is. With no axis
    of two or more points the grid is points without cells: one point, or
    none when a size is 0. The flat point index is the same either way, so
    the coordinates of all ``dims`` serve. It is on ``device`` (by default the
    default device).
    """
    from tack.dataset.topology import StructuredTopology, UnstructuredTopology

    dims = tuple(int(d) for d in dims)
    if not 1 <= len(dims) <= 3 or min(dims) < 0:
        raise ValueError(f"a grid has 1 to 3 sizes of at least 0 points, not {dims}")
    axes = tuple(d for d in dims if d != 1)
    if 0 in dims or not axes:
        with as_device(backend_of(device)):
            return UnstructuredTopology([], [0], [], num_points=int(np.prod(dims)),
                                        id_dtype=id_dtype)
    if offset is not None:                  # the axes of more than one point
        offset = [o for o, d in zip(offset, dims) if d != 1]
    return StructuredTopology(axes, id_dtype=id_dtype, device=device, offset=offset)


def rectilinear_grid(x, y=(0.0,), z=(0.0,), dtype=tack.f32, device=None, unit=None):
    """A dataset of the grid whose lines run through ``x``, ``y`` and ``z``; see
    ``grid_topology`` for axes of one point or none. It is on ``device``, by
    default that of axes given as device arrays, else the default device.
    ``unit`` is the spatial unit (``DataSet.unit``)."""
    coordinates = CartesianProduct(x, y, z, dtype=dtype, device=device)
    return DataSet(grid_topology(coordinates.dims, device=coordinates.device), coordinates,
                   unit=unit)


def uniform_grid(dims, origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0), direction=None,
                 dtype=tack.f32, device=None, unit=None):
    """A dataset of ``dims`` points (one to three sizes) at ``origin + direction @
    (ijk * spacing)``: VTK's image data, Viskores' uniform data set. Nothing per
    point is stored (``UniformCoordinates``). A size of 1 adds no cells, in any
    position, and a grid with no axis of two points has none (``grid_topology``).
    ``origin`` and ``spacing`` take a value per size given (or three), and
    ``direction`` is 3 x 3 (2 x 2 for a 2D image). It is on ``device`` (by
    default the default device).

    The image's coordinates are in the geometry's type, so one far from zero
    -- a raster at survey coordinates -- loses digits in single precision:
    ``rebased(origin)`` keeps it an image with its corner relative to a nearby
    dataset origin, moved in double precision, as a point cloud's origin
    keeps its points. ``unit`` is the spatial unit (``DataSet.unit``), which
    ``origin`` and ``spacing`` are in."""
    topology = grid_topology(dims, device=device)   # the sizes as given, before padding
    return DataSet(topology, UniformCoordinates(dims, origin, spacing, direction, dtype=dtype,
                                                device=device), unit=unit)


def point_cloud(positions, fields=None, origin=None, dtype=tack.f32, device=None, unit=None,
                units=None):
    """A point cloud: a dataset of ``positions``, one vertex cell per point (types,
    offsets and connectivity implicit, stored nowhere), with ``fields`` (a dict of
    names to one value per point, scalars or rows of components) as point data.

    ``positions`` is an ``(n, 3)`` host array of world positions, or a Tack array
    of 3-vectors already relative to ``origin``. Host positions are taken
    relative to ``origin`` in double precision before they become ``dtype``;
    without one, float64 positions get the lower corner of their box, rounded
    down to whole units, so a LiDAR survey's eastings and northings keep their
    precision in single precision, and other positions get zero. ``dtype`` is
    ``tack.f32``, or ``tack.f64`` on a device that has it. Floating-point fields
    become ``dtype`` too; integer fields (a LAS classification, intensities) keep
    their type. ``unit`` is the spatial unit (``DataSet.unit``), which ``origin``
    is in, and ``units`` maps field names to their units."""
    from tack.dataset import ids
    from tack.dataset.algorithms._extract import _vertex_cells
    from tack.dataset.topology import UnstructuredTopology

    target = as_device(backend_of(device))
    if dtype not in target._backend.supported_dtypes:
        raise TypeError(f"the {target} device has no {dtype} arrays; a point cloud there "
                        f"takes positions of {tack.f32} relative to an origin")
    if isinstance(positions, tack.Array):
        n = int(positions.shape[0])
        local = positions
    else:
        world = np.asarray(positions, np.float64 if np.asarray(positions).dtype == np.float64
                           else None).reshape(-1, 3)
        n = len(world)
        if origin is None and world.dtype == np.float64 and n:
            origin = np.floor(world.min(axis=0))
        local = world.astype(np.float64) - np.asarray(as_origin(origin))
    with target:
        topology = UnstructuredTopology(*_vertex_cells(n, ids.choose(None, n)), num_points=n)
        data = DataSet(topology, local, dtype=dtype, origin=origin, unit=unit)
        units = dict(units or {})
        unknown = set(units) - set(fields or {})
        if unknown:
            raise KeyError(f"units for fields not given: {', '.join(map(repr, sorted(unknown)))}")
        for name, values in (fields or {}).items():
            if not isinstance(values, tack.Array):
                host = np.asarray(values)
                if host.shape[:1] != (n,):
                    raise ValueError(f"field {name!r} has {host.shape[:1]} values for "
                                     f"{n} points")
                if host.dtype.kind == "f":
                    host = host.astype(dtype.numpy_dtype)
                width = host.shape[1] if host.ndim == 2 else None
                if width:
                    from tack.lang.types import from_numpy_dtype

                    values = tack.array(np.ascontiguousarray(host),
                                        tack.vec(width, from_numpy_dtype(host.dtype)))
                else:
                    values = tack.array(np.ascontiguousarray(host))
            data.fields[name] = Field(H1(data), values, units.get(name))
    return data
