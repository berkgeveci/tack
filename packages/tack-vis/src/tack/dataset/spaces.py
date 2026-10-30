"""Spaces: where a field's values live, on one topology, and the layout that follows.

A space is constructed on a topology (or a dataset, for its topology) and
parameters, as MFEM's ``FiniteElementSpace`` is on a mesh::

    td.H1(data)                 # one value per point, interpolated linearly
    td.H1(data, order=2)        # quadratic: values on points, edges, quad faces, cells
    td.L2(data)                 # each cell's own corner values: linear DG
    td.L2(data, order=orders)   # an order per cell (0 or 1 here): p-adaptive DG
    td.Constant(data)           # one value per cell
    td.Values(data, "faces")    # one value per face, no basis
    td.SideTraces(data)         # values on each side of each face, at its points

Spaces are interned on their topology: the same kind and parameters on the
same topology is the same object, so fields that share a space share its
layout -- and equality is identity. The topology holds them weakly: a space
holds its topology, and a cycle would keep a dropped dataset's arrays (on a
GPU, its device memory) until Python's cycle collector ran. The layout is what a kernel needs
beyond a field's values to find them:

- ``H1``, order 1: nothing of its own. Its values are the points', found
  through the topology's connectivity.
- ``H1``, order 2: its numbering of values -- the points', then one per
  edge (``edge_base + edge``), one per quad face (``face_dofs``) and one per
  hexahedron (``cell_dofs``) -- built from the derived edges and faces. Its
  views read a cell's through the cell's edge and face ids.
- ``L2``, order 1: ``offsets``, where each cell's values start. An
  unstructured topology's are its connectivity offsets, so they are shared,
  not copied. With an order per cell, the offsets come from a scan of each
  cell's count of values (1 at order 0, its corners at order 1).
- ``Constant`` and ``Values``: nothing; value ``k`` belongs to entity ``k``.

Every space knows ``size``, the number of values a field on it holds, and
``on``, the entities a loop must run over to read it through ``view``.

A space whose cells differ -- an order per cell -- ``varies``: it gives
each cell a key (``cell_keys``, here the order), and ``tack.dataset.for_each``
launches each shape group once per key, with the view ``mixin_for(key)``,
so every launch is specialized as a uniform space's is.
"""

import os
import weakref

import numpy as np

import tack
from tack import arrays
from tack.algorithms.scan import exclusive_scan
from tack.arrays import CountingArray, size_of
from tack.dataset import bases, ids, shapes, views
from tack.runtime.dispatch import on_inputs_device

__all__ = ["H1", "L2", "Constant", "SideTraces", "Space", "Values"]


class Space:
    """What every space has: ``topology``, ``size``, ``on`` and a field-view ``mixin``,
    and the layout ``attributes`` that mixin reads."""

    on = "cells"
    mixin = None
    #: Spaces with a basis evaluate at parametric coordinates (``value``,
    #: ``parametric_gradient``); the others only give each entity its value.
    interpolated = False

    #: Whether cells differ in how their values are read (``cell_keys``).
    varies = False

    def __new__(cls, where, *args, **kwargs):
        topology = getattr(where, "topology", where)
        params = cls._params(*args, **kwargs)
        # An array parameter (an order per cell) is not hashable: the space is
        # one per array object, which it keeps alive, so its id stays unique.
        key = (cls, tuple(sorted((k, ("array", id(v)) if _is_array(v) else v)
                                 for k, v in params.items())))
        interned = topology.__dict__.setdefault("_spaces", weakref.WeakValueDictionary())
        space = interned.get(key)
        if space is None:
            space = super().__new__(cls)
            space.topology = topology
            space.params = params
            for name, value in params.items():
                # A property of the same name resolves the parameter itself (L2's
                # layout, whose "auto" depends on the device).
                if not isinstance(getattr(cls, name, None), property):
                    setattr(space, name, value)
            interned[key] = space
        return space

    @classmethod
    def _params(cls):
        return {}

    @property
    def _device(self):
        """Its topology's device, where its layout is made (`on_inputs_device`)."""
        return self.topology._device

    def _rebuilt_on(self, topology):
        """This space on ``topology`` (another dataset's, or this one's moved to
        another device): the same kind and parameters, array parameters copied
        to that topology's device."""
        target = topology.device
        return type(self)(topology, **{k: v.to_device(target) if hasattr(v, "to_device") else v
                                       for k, v in self.params.items()})

    def attributes(self, group=None, dtype=None):
        """The layout a view of this space reads for ``group``, beside the values;
        ``dtype`` the float type a basis is evaluated in, for the spaces that have
        tables of nodes."""
        return {}

    def extra_mixins(self, group):
        """Mixins a field view of this space needs for ``group`` beyond its own and
        the storage."""
        return ()

    def mixin_for(self, key, shape=None):
        """The field-view mixin for cells of ``key`` (``None`` for a uniform space) and
        ``shape``."""
        return self.mixin

    def cell_keys(self):
        """For a space that ``varies``: an i32 array of each cell's key, by cell id."""
        return

    @property
    def size(self):
        raise NotImplementedError

    def __repr__(self):
        args = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
        return f"{type(self).__name__}({args})"


def _needs_reference_cells(space, family):
    reference_cells(space.topology, family)


def reference_cells(topology, what):
    """Refuse ``what`` on a topology whose cells have no reference element --
    polyhedra, polygons and polylines -- rather than let it compile against a
    shape that has no shape functions, or launch over cells it would skip."""
    if not getattr(topology, "reference_cells", True):
        raise NotImplementedError(f"{what} needs a reference element, which polyhedral, "
                                  "polygonal and polyline cells do not have")


def _is_array(value):
    return isinstance(value, np.ndarray) or hasattr(value, "numpy")


@tack.kernel
def _table_corners(cells, table, per_cell, corner_node, bad):
    # 1 for a cell whose table row does not put its corners at its points.
    for c in cells:
        e = cells.entity_id(c)
        wrong = 0
        for j in range(cells.NUM_POINTS):
            if table[e * per_cell + corner_node[j]] != cells.point_id(c, j):
                wrong = 1
        bad[e] = wrong


def _checked_table(space):
    """``space``'s element-to-DOF table as ``(table, nodes a cell, size)``: flat, of
    ids on the topology's device, refused unless every cell has the same number of
    nodes, a row of that many, and its corners at its points."""
    from math import comb

    topology = space.topology
    _needs_reference_cells(space, "an element-to-DOF table")
    p = space.order
    counts = set()
    for group in topology.groups():
        if group.count:
            shape = group.shape
            _refuse_bernstein(space, shape)
            if not bases.has_bases(shape):
                raise NotImplementedError(f"an element-to-DOF table is on segments, quads, "
                                          f"pixels, hexahedra, voxels, triangles, tetrahedra "
                                          f"and wedges, not {shape.__name__} cells")
            _refuse_simplex_nodes(space, shape)
            counts.add(comb(p + shape.DIMENSION, shape.DIMENSION) if bases.simplex_shape(shape)
                       else comb(p + 2, 2) * (p + 1) if bases.wedge_shape(shape)
                       else (p + 1) ** shape.DIMENSION)
    if len(counts) > 1:
        raise NotImplementedError("an element-to-DOF table takes cells of one node count")
    per_cell = counts.pop() if counts else 1
    n = topology.num_cells
    dofs = space.dofs
    host = np.asarray(dofs if isinstance(dofs, np.ndarray) else arrays.to_host(dofs)).reshape(-1)
    if host.size != n * per_cell or not np.issubdtype(host.dtype, np.integer):
        raise ValueError(f"dofs is an integer array of {n} rows of {per_cell} nodes, not "
                         f"{host.dtype} of {host.size} values")
    if host.size and host.min() < 0:
        raise ValueError("dofs holds value indices, not negative numbers")
    size = max(int(host.max()) + 1 if host.size else 0, topology.num_points)
    table = tack.empty((max(host.size, 1),), ids.at_least(topology.id_dtype, size),
                       device=topology.device)
    if host.size:
        tack.copyto(table, host.astype(table.dtype.numpy_dtype))
        bad = tack.empty((n,), tack.i32, device=topology.device)
        for group in topology.groups():
            if group.count:
                corners = bases.dof_tables(group.shape, p)[4]
                _table_corners(group.view(), table, per_cell,
                               tack.array(np.asarray(corners, np.int32)), bad)
        if int(bad.numpy().sum()):
            raise ValueError("an element-to-DOF table puts each cell's corners at its points; "
                             f"{int(bad.numpy().sum())} of {n} cells' rows do not")
    return table, per_cell, size


def _only_order_one(family, order):
    if order != 1:
        raise NotImplementedError(
            f"{family} order {order} is not supported yet: its values on edges, faces "
            "and interiors need a cell -> DOF map built from the derived edges and faces")
    return {"order": order}


@tack.kernel
def _face_counts(kinds, per_quad, per_triangle, counts):
    for f in range(kinds.shape[0]):
        counts[f] = per_quad if kinds[f] == shapes.QUAD else (
            per_triangle if kinds[f] == shapes.TRIANGLE else 0)


@tack.kernel
def _place(counts, slots, base, out):
    for i in range(counts.shape[0]):
        out[i] = base + slots[i] if counts[i] > 0 else -1


@tack.kernel
def _interior_counts(cells, order, counts):
    # Order 2 keeps the quadratic elements' interiors (a hexahedron's and a
    # quad's); order 3 and up, and 2 on a segment (no quadratic element): (p - 1)^d
    # a tensor cell, the nodes strictly inside a simplex ((p - 1)(p - 2) / 2 a
    # triangle, (p - 1)(p - 2)(p - 3) / 6 a tetrahedron). The two agree at order 2
    # wherever both are defined.
    for c in cells:
        n = cells.QUADRATIC_INTERIOR
        if order != 2 or cells.NUM_QUADRATIC == 0:
            n = 1
            if cells.NUM_POINTS == 6:
                # A wedge: the triangle's interior times the segment's.
                n = (order - 1) * (order - 1) * (order - 2) // 2
            elif cells.NUM_POINTS == cells.DIMENSION + 1 and cells.DIMENSION >= 2:
                for k in range(cells.DIMENSION):
                    n = n * (order - 1 - k) // (k + 1)
            else:
                for _ in range(cells.DIMENSION):
                    n *= order - 1
        counts[cells.entity_id(c)] = n


def _basis_param(family, order, families, space):
    """``basis`` as a space keeps it: ``None`` where the order makes the families
    agree (at most 2 for the closed node sets, at most 1 for Bernstein's), so equal
    spaces are one."""
    if family not in families:
        raise ValueError(f"{space}'s basis is one of {', '.join(families)}, not {family!r}")
    if family in bases.CLOSED and order <= (1 if family == bases.BERNSTEIN else 2):
        return None
    return family


def _basis_tables(space, group, dtype):
    """A tensor basis' arrays for ``group``'s shape: its 1D nodes and weights in
    ``dtype`` (the field's float type) and, for ``H1``, where each node's value
    lives (``bases.dof_tables``)."""
    family = space.basis or "gauss_lobatto"
    key = (group.shape, family, dtype)
    cache = space.__dict__.setdefault("_tables", {})
    if key not in cache:
        def device(values, kind):
            out = tack.empty((len(values),), kind)
            tack.copyto(out, np.asarray(values, kind.numpy_dtype))
            return out

        kind, entity, a, b, corner = bases.dof_tables(group.shape, space.order)
        cache[key] = {"basis_nodes": device(bases.nodes(family, space.order), dtype),
                      "basis_weights": device(bases.weights(family, space.order), dtype),
                      "basis_bernstein": device(bases.bernstein(family, space.order).reshape(-1),
                                                dtype),
                      "node_kind": device(kind, tack.i32), "node_entity": device(entity, tack.i32),
                      "node_a": device(a, tack.i32), "node_b": device(b, tack.i32),
                      "corner_node": device(corner, tack.i32)}
    return cache[key]


def _simplex_tables(space, group, dtype):
    """A simplex basis' arrays for ``group``'s shape: its nodes' integer coordinates
    and Bernstein matrix (``bases.simplex_lattice``, ``simplex_bernstein``) in
    ``dtype``, the 1D nodes along its edges, and where each node's value lives
    (``bases.dof_tables``)."""
    key = (group.shape, "simplex", space.basis, dtype)
    cache = space.__dict__.setdefault("_tables", {})
    if key not in cache:
        def device(values, kind):
            values = np.asarray(values).reshape(-1)
            out = tack.empty((len(values),), kind)
            tack.copyto(out, values.astype(kind.numpy_dtype))
            return out

        kind, entity, a, b, corner = bases.dof_tables(group.shape, space.order)
        cache[key] = {"simplex_ijk": device(bases.simplex_lattice(group.shape, space.order),
                                            tack.i32),
                      "simplex_bernstein": device(
                          np.eye(len(bases.simplex_lattice(group.shape, space.order)))
                          if space.basis == bases.BERNSTEIN else
                          bases.simplex_bernstein(group.shape, space.order), dtype),
                      "node_kind": device(kind, tack.i32), "node_entity": device(entity, tack.i32),
                      "node_a": device(a, tack.i32), "node_b": device(b, tack.i32),
                      "corner_node": device(corner, tack.i32)}
    return cache[key]


def _wedge_tables(space, group, dtype):
    """A wedge basis' arrays: its nodes' integer coordinates and Bernstein matrix
    (``bases.wedge_lattice``, ``wedge_bernstein``) in ``dtype``, and where each node's
    value lives (``bases.dof_tables``)."""
    key = (group.shape, "wedge", space.basis, dtype)
    cache = space.__dict__.setdefault("_tables", {})
    if key not in cache:
        def device(values, kind):
            values = np.asarray(values).reshape(-1)
            out = tack.empty((len(values),), kind)
            tack.copyto(out, values.astype(kind.numpy_dtype))
            return out

        kind, entity, a, b, corner = bases.dof_tables(group.shape, space.order)
        cache[key] = {"wedge_ijk": device(bases.wedge_lattice(space.order), tack.i32),
                      "wedge_bernstein": device(
                          np.eye(len(bases.wedge_lattice(space.order)))
                          if space.basis == bases.BERNSTEIN else
                          bases.wedge_bernstein(space.order), dtype),
                      "node_kind": device(kind, tack.i32), "node_entity": device(entity, tack.i32),
                      "node_a": device(a, tack.i32), "node_b": device(b, tack.i32),
                      "corner_node": device(corner, tack.i32)}
    return cache[key]


def _refuse_simplex_nodes(space, shape):
    """Simplices and wedges above order 2 have equispaced nodes only (round 2): any
    other family is refused, with the remedy."""
    _refuse_bernstein(space, shape)
    if (bases.simplex_shape(shape) or bases.wedge_shape(shape)) and space.order > 2 \
            and space.basis not in ("equispaced", bases.BERNSTEIN):
        raise NotImplementedError(
            f"{type(space).__name__} of order {space.order} on {shape.__name__} cells has "
            f"the equispaced basis only (VTK's and gmsh's), not {space.basis!r}: pass "
            "basis=\"equispaced\"")


def _refuse_bernstein(space, shape):
    """Bernstein bases are on the shapes with bases of any order: pyramids and cells
    without a reference element are refused."""
    if space.basis == bases.BERNSTEIN and shape is not None and not bases.has_bases(shape):
        raise NotImplementedError(
            f"the Bernstein basis is on segments, quads, pixels, hexahedra, voxels, "
            f"triangles, tetrahedra and wedges, not {shape.__name__} cells")


def _on_gpu(topology):
    """Whether ``topology`` lives on a GPU, where ``L2``'s ``"auto"`` layout is
    cell-fastest above order 2."""
    return topology.device.type != "cpu"


def _float_type(dtype, device):
    """The float type a basis is evaluated in for values of ``dtype``: theirs if
    floating, else f32 (f64 where the device has it, for 64-bit integers)."""
    if dtype in (tack.f32, tack.f64):
        return dtype
    return tack.f64 if dtype.bits == 64 and device.supports_f64 else tack.f32


class H1(Space):
    """Continuous: values shared by the cells around them, interpolated by the shape's
    functions. Order 1 holds one value per point -- today's point data; order 2
    adds one per edge, per quad face and per hexahedron or quad, for the shape's
    quadratic Lagrange functions (tetrahedra, hexahedra, voxels, wedges,
    triangles, quads and pixels). Order 3 and up are on segments, quads, pixels,
    hexahedra and voxels (``tack.dataset.bases``): ``order - 1`` values per edge,
    ``(order - 1)^2`` per quad face and ``(order - 1)^d`` per cell, in the
    ``basis`` ``"gauss_lobatto"`` (the default: Lagrange polynomials on
    Gauss-Lobatto nodes) or ``"equispaced"`` (on equispaced nodes), which agree
    up to order 2; or ``"bernstein"``: the Bernstein polynomials' coefficients,
    not values at nodes (VTK's Bezier cells; from order 2, on every shape here).
    On triangles, tetrahedra and wedges with ``basis="equispaced"`` (VTK's and
    gmsh's) or ``"bernstein"``: ``(order - 1)(order - 2) / 2`` per
    triangle face or triangle, ``(order - 1)(order - 2)(order - 3) / 6`` per
    tetrahedron, ``(order - 1)^2 (order - 2) / 2`` per wedge.

    ``dofs``, from order 2, numbers the values by an element-to-DOF table instead
    of by entity, as MFEM and VTK's cell grids store them: an integer array of a
    row per cell (``(num_cells, nodes)``, or flat), each node's value index in
    Tack's node order for the shape (``bases.lattice_ijk``: the tensor lattice,
    first axis fastest; the simplex's and the wedge's lattices). Each cell lists
    its shared values in its own local order, so the field is continuous when the
    table shares them; a cell's corners must be its points (so a corner's value
    is its point's), and the space holds as many values as the largest index and
    the points need. A filter's output numbers its values by entity again."""

    on = "points"
    interpolated = True
    basis = None
    dofs = None

    @classmethod
    def _params(cls, order=1, basis="gauss_lobatto", dofs=None):
        order = int(order)
        if order < 1:
            raise ValueError(f"H1's order is at least 1, not {order}")
        family = _basis_param(basis, order, bases.CLOSED, "H1")
        out = {"order": order} if family is None else {"order": order, "basis": family}
        if dofs is not None:
            if order < 2:
                raise ValueError("an element-to-DOF table numbers values of order 2 and "
                                 "up; order 1's are the points'")
            if not _is_array(dofs):
                raise TypeError(f"dofs is an integer array of a row per cell, not {dofs!r}")
            out["dofs"] = dofs
        return out

    def _table(self):
        """The element-to-DOF table, flat on the topology's device, checked: ``(table,
        nodes a cell, size)``."""
        if "_dof_table" not in self.__dict__:
            self.__dict__["_dof_table"] = _checked_table(self)
        return self.__dict__["_dof_table"]

    @property
    def mixin(self):
        if self.order == 1:
            return views._H1Field
        if self.order == 2 and self.basis != bases.BERNSTEIN and self.dofs is None:
            return views._H1Order2Field
        return self._tabled(views.of_order(views._H1TensorField, self.order,
                                           self.basis == bases.BERNSTEIN))

    def _tabled(self, mixin):
        """``mixin``, read through the element-to-DOF table if this space has one."""
        return mixin if self.dofs is None else views.with_table(mixin)

    @on_inputs_device
    def _layout(self):
        """Order 2 and up: ``(face_dofs, cell_dofs, size)``, made on first use and kept."""
        if self.dofs is not None:
            raise NotImplementedError(f"{self!r} numbers its values by its table, not by "
                                      "entity: this needs the space numbered by entity")
        if "_order2" not in self.__dict__:
            _needs_reference_cells(self, f"H1 of order {self.order}")
            topology = self.topology
            p = self.order
            for group in topology.groups():
                if not group.count:
                    continue
                _refuse_bernstein(self, group.shape)
                if p == 2 and not self._any_order(group.shape) and not group.shape.NUM_QUADRATIC:
                    raise NotImplementedError(
                        f"H1 order 2 on {group.shape.__name__} cells is not supported yet; "
                        "it is on tetrahedra, hexahedra, voxels, wedges, triangles, quads "
                        "and pixels")
                if p > 2 and not bases.has_bases(group.shape):
                    raise NotImplementedError(
                        f"H1 order {p} on {group.shape.__name__} cells is not supported "
                        "yet; orders above 2 are on segments, quads, pixels, hexahedra, "
                        "voxels, triangles, tetrahedra and wedges")
                _refuse_simplex_nodes(self, group.shape)
            dims = {g.shape.DIMENSION for g in topology.groups() if g.count}
            edges = topology.edges() if dims - {1} else None
            ne = edges.num_edges if edges is not None else 0
            base = topology.num_points + ne * (p - 1)
            # Only quad faces of solids hold values: a mesh of tetrahedra, triangles
            # or quads needs no faces derived.
            solid = (any(g.count and g.shape.QUADRATIC_FACES for g in topology.groups())
                     if p == 2 else 3 in dims)
            faces = topology.faces() if solid else None
            nf = faces.num_faces if faces is not None else 0
            n = topology.num_cells
            idt = ids.at_least(topology.id_dtype, base + nf * (p - 1) ** 2 + n * (p - 1) ** 3)
            counts = tack.empty((nf,), tack.i32)
            slots = tack.empty((nf,), idt)
            face_dofs = tack.empty((nf,), idt)
            on_faces = 0
            if nf:
                _face_counts(faces.kinds, (p - 1) ** 2,
                             (p - 1) * (p - 2) // 2 if p > 2 else 0, counts)
                on_faces = exclusive_scan(counts, slots, nf)
                _place(counts, slots, base, face_dofs)
            counts = tack.zeros((n,), tack.i32)
            for group in topology.groups():
                if group.count:
                    _interior_counts(group.view(), p, counts)
            slots = tack.empty((n,), idt)
            cell_dofs = tack.empty((n,), idt)
            inside = exclusive_scan(counts, slots, n) if n else 0
            if n:
                _place(counts, slots, base + on_faces, cell_dofs)
            self._order2 = (face_dofs, cell_dofs, base + on_faces + inside)
        return self._order2

    @property
    def size(self):
        if self.order == 1:
            return self.topology.num_points
        if self.dofs is not None:
            return self._table()[2]
        return self._layout()[2]

    def __repr__(self):
        args = ", ".join(f"{k}={'<table>' if k == 'dofs' else repr(v)}"
                         for k, v in self.params.items())
        return f"H1({args})"

    def attributes(self, group=None, dtype=None):
        if self.order == 1:
            return {}
        if self.dofs is not None:
            table, per_cell, _ = self._table()
            attributes = {"dof_table": table, "dof_count": per_cell}
            if group is not None:
                dtype = dtype or tack.f32
                shape = group.shape
                attributes.update(_simplex_tables(self, group, dtype)
                                  if bases.simplex_shape(shape) else
                                  _wedge_tables(self, group, dtype)
                                  if bases.wedge_shape(shape) else
                                  _basis_tables(self, group, dtype))
            return attributes
        face_dofs, cell_dofs, _ = self._layout()
        topology = self.topology
        attributes = {"edge_base": topology.num_points, "face_dofs": face_dofs,
                      "cell_dofs": cell_dofs}
        if group is not None:
            root = group.root
            if self._has_edges(group.shape):
                edges = topology.edges()
                attributes.update(side_edge=edges.side_edge, side_sign=edges.side_sign,
                                  edge_start=edges.group_starts[root])
            if self._has_faces(group.shape):
                faces = topology.faces()
                attributes.update(side_face=faces.side_face, side_slot=faces.side_slot,
                                  side_orientation=faces.side_orientation,
                                  face_start=faces.group_starts[root])
            high = self.order > 2 or self.basis == bases.BERNSTEIN
            if high and bases.simplex_shape(group.shape):
                attributes.update(_simplex_tables(self, group, dtype or tack.f32))
            elif high and bases.wedge_shape(group.shape):
                attributes.update(_wedge_tables(self, group, dtype or tack.f32))
            elif self.order > 2 or (self.order == 2 and bases.tensor_shape(group.shape)):
                # Order 2 on tensor shapes: for a curved geometry's tensor views.
                attributes.update(_basis_tables(self, group, dtype or tack.f32))
        return attributes

    def _has_edges(self, shape):
        return shape.DIMENSION >= 2 if self._any_order(shape) else True

    def _has_faces(self, shape):
        return shape.DIMENSION == 3 if self._any_order(shape) else bool(shape.QUADRATIC_FACES)

    def _any_order(self, shape):
        """Whether views read this space on ``shape`` cells as a basis of any order --
        above order 2, or at 2 in a Bernstein basis or on a shape with such bases
        but no quadratic functions (the segment) -- not as the quadratic shapes'
        functions."""
        return self.order > 2 or (self.order == 2 and bases.has_bases(shape) and (
            self.basis == bases.BERNSTEIN or self.dofs is not None
            or not shape.NUM_QUADRATIC))

    def mixin_for(self, key, shape=None):
        _refuse_bernstein(self, shape)
        # Polyhedra evaluate in their sub-tetrahedra.
        if shape in (shapes.Polyhedron, shapes.Polygon) and self.order == 1:
            return views._PolyhedralPointField
        if shape is shapes.PolyLine and self.order == 1:
            return views._PolylinePointField
        bernstein = self.basis == bases.BERNSTEIN
        high = self.order > 2 or bernstein or self.dofs is not None
        if high and bases.simplex_shape(shape):
            return self._tabled(views.of_simplex_order(views._H1SimplexField, self.order,
                                                       shape.DIMENSION, bernstein))
        if high and bases.wedge_shape(shape):
            return self._tabled(views.of_wedge_order(views._H1WedgeField, self.order,
                                                     bernstein))
        if self.order == 2 and shape is not None and self._any_order(shape):
            return self._tabled(views.of_order(views._H1TensorField, 2, bernstein))
        return self.mixin

    def extra_mixins(self, group):
        if self.order == 1 or self.dofs is not None:
            # A table numbers the values itself: no edges or faces to read.
            return ()
        shape = group.shape
        incidence = ((views._EdgeIncidence,) if self._has_edges(shape) else ()) + \
            ((views._FaceIncidence,) if self._has_faces(shape) else ())
        if self.order == 2 and not self._any_order(shape):
            return (*incidence, views.quadratic_faces(shape))
        if bases.simplex_shape(shape):
            return (*incidence, *views.simplex_mixins(shape))
        if bases.wedge_shape(shape):
            return (*incidence, *views.wedge_mixins(shape))
        return (*incidence, *views.tensor_mixins(shape))


@tack.kernel
def _l2_counts(cells, orders, counts):
    for c in cells:
        e = cells.entity_id(c)
        counts[e] = 1 if orders[e] == 0 else cells.NUM_POINTS


@tack.kernel
def _tensor_counts(cells, order, counts):
    # (p + 1)^d a tensor cell's nodes, C(p + d, d) a simplex's.
    for c in cells:
        n = 1
        if cells.NUM_POINTS == 6:
            n = (order + 1) * (order + 2) // 2 * (order + 1)       # a wedge
        elif cells.NUM_POINTS == cells.DIMENSION + 1 and cells.DIMENSION >= 2:
            for k in range(cells.DIMENSION):
                n = n * (order + 1 + k) // (k + 1)
        else:
            for _ in range(cells.DIMENSION):
                n *= order + 1
        counts[cells.entity_id(c)] = n


@tack.kernel
def _set_last(offsets, n, total):
    for i in range(1):
        offsets[n] = total


class L2(Space):
    """Discontinuous: each cell's own values, at ``offsets[cell]``. Order 1 holds a
    value at each of the cell's corners -- a linear DG field -- and order 0 one
    value per cell. ``order`` is one order for every cell, or an array of one per
    cell (p-adaptive, orders 0 and 1), read when the space is made. Order 2 and
    up, or ``basis="gauss_legendre"`` at any order from 1, are on segments, quads,
    pixels, hexahedra and voxels (``tack.dataset.bases``): ``(order + 1)^d`` values
    a cell in node order, the first axis fastest, in the ``basis``
    ``"gauss_lobatto"`` (the default), ``"equispaced"`` or ``"gauss_legendre"``
    (Lagrange polynomials on those nodes) or ``"bernstein"`` (the Bernstein
    polynomials' coefficients); and from order 2 on triangles, tetrahedra and
    wedges with ``basis="equispaced"`` or ``"bernstein"``: each cell's values at its lattice's nodes
    (``bases.lattice_ijk``), always in blocks.

    ``layout`` says how a tensor basis' values are stored: ``"blocks"``, each
    cell's together (value ``n`` of cell ``c`` at ``offsets[c] + n``: MFEM's and
    VTK cell grids' layout), or ``"cells"``, cell-fastest (``n * num_cells + c``,
    so neighbouring GPU threads read neighbouring values). ``"auto"`` (the
    default) is ``"cells"`` for a tensor basis of order 3 or more on a GPU and
    ``"blocks"`` otherwise -- each faster where it is chosen (the vault's
    coefficient-layout proposal) -- read from the topology's device, so moving
    a field to another device may change it (``to_device`` transposes). The
    environment variable ``TACK_L2_LAYOUT`` (``cells`` or ``blocks``) overrides
    ``"auto"`` for tensor bases, to run everything one way. Corner values (order
    0 and 1) and an order per cell are always blocks."""

    interpolated = True
    basis = None
    LAYOUTS = ("auto", "blocks", "cells")

    @classmethod
    def _params(cls, order=1, basis="gauss_lobatto", layout="auto"):
        if layout not in cls.LAYOUTS:
            raise ValueError(f"L2's layout is one of {', '.join(cls.LAYOUTS)}, not {layout!r}")
        if _is_array(order):
            if basis != "gauss_lobatto":
                raise NotImplementedError("an order per cell is on corner values, the "
                                          "default basis")
            if layout == "cells":
                raise ValueError("an order per cell has no common stride: its layout is "
                                 "blocks")
            return {"order": order}
        order = int(order)
        if order < 0:
            raise ValueError(f"L2's order is at least 0, not {order}")
        family = _basis_param(basis, order, bases.FAMILIES, "L2")
        if order == 0:
            family = None                   # one value at the center, whatever the basis
        tensor = order >= 2 or family == "gauss_legendre"
        if layout == "cells" and not tensor:
            raise ValueError("corner values (order 0 and 1) are stored in blocks; a "
                             "cell-fastest layout is for tensor bases")
        out = {"order": order} if family is None else {"order": order, "basis": family}
        if layout != "auto" and tensor:
            out["layout"] = layout
        return out

    @property
    def layout(self):
        """``"blocks"`` or ``"cells"``: how this space stores a cell's values."""
        asked = self.params.get("layout", "auto")
        if not self.tensor:
            return "blocks"
        if asked == "cells" and any(g.count and (bases.simplex_shape(g.shape)
                                                 or bases.wedge_shape(g.shape))
                                    for g in self.topology.groups()):
            raise ValueError("a cell-fastest layout is for tensor bases; L2 on triangles, "
                             "tetrahedra and wedges is stored in blocks")
        if asked != "auto":
            return asked
        forced = os.environ.get("TACK_L2_LAYOUT", "")
        if forced:
            if forced not in ("blocks", "cells"):
                raise ValueError(f"TACK_L2_LAYOUT is blocks or cells, not {forced!r}")
            return forced if self._uniform() else "blocks"
        return "cells" if _on_gpu(self.topology) and self.order >= 3 and self._uniform() \
            else "blocks"

    def _uniform(self):
        """Whether every cell holds as many values of a tensor basis: cells of one
        dimension, none a simplex (cell-fastest is for tensor bases)."""
        groups = [g for g in self.topology.groups() if g.count]
        return (len({g.shape.DIMENSION for g in groups}) <= 1
                and not any(bases.simplex_shape(g.shape) or bases.wedge_shape(g.shape)
                            for g in groups))

    def indices(self):
        """A host ``(num_cells, values a cell)`` array: where each cell's value ``n`` is
        in this space's values (blocks of one size only)."""
        n = self.topology.num_cells
        per_cell = self.size // max(n, 1)
        if self.layout == "cells":
            return np.arange(per_cell)[None, :] * n + np.arange(n)[:, None]
        offsets = np.asarray(arrays.to_host(self.offsets)).reshape(-1)
        return offsets[:n, None] + np.arange(per_cell)[None, :]

    def blocks(self, values):
        """``values`` (a host array of a field of this space) as ``(num_cells, values a
        cell, ...)``, each cell's in node order, whatever the layout. Blocks of one
        size only (a tensor basis, or corner values of one shape)."""
        values = np.asarray(values)
        n = self.topology.num_cells
        per_cell = self.size // max(n, 1)
        if self.layout == "cells":
            return np.moveaxis(values.reshape((per_cell, n, *values.shape[1:])), 0, 1)
        return values.reshape((n, per_cell, *values.shape[1:]))

    @property
    def varies(self):
        return _is_array(self.order)

    @property
    def tensor(self):
        """Whether the cells' values are a tensor basis' nodes, not their corners."""
        return not self.varies and (self.order >= 2 or self.basis == "gauss_legendre")

    @property
    def mixin(self):
        if self.tensor:
            cells = self.layout == "cells"
            return views.of_order(views._L2CellsField if cells else views._L2TensorField,
                                  self.order, self.basis == bases.BERNSTEIN)
        return views._L2Field

    def mixin_for(self, key, shape=None):
        _refuse_bernstein(self, shape)
        bernstein = self.basis == bases.BERNSTEIN
        if self.tensor and bases.simplex_shape(shape):
            return views.of_simplex_order(views._L2SimplexField, self.order, shape.DIMENSION,
                                          bernstein)
        if self.tensor and bases.wedge_shape(shape):
            return views.of_wedge_order(views._L2WedgeField, self.order, bernstein)
        if self.tensor:
            return self.mixin
        order = key if self.varies else self.order
        return views._L2Order0Field if order == 0 else views._L2Field

    @on_inputs_device
    def cell_keys(self):
        """Each cell's order, an i32 array (for a space with an order per cell)."""
        if not self.varies:
            return None
        if "_orders" not in self.__dict__:
            host = self.order.numpy() if hasattr(self.order, "numpy") else self.order
            host = np.asarray(host).reshape(-1)
            if host.shape[0] != self.topology.num_cells:
                raise ValueError(f"an order per cell: {self.topology.num_cells} orders, "
                                 f"not {host.shape[0]}")
            if not np.isin(host, (0, 1)).all():
                bad = sorted(set(host.tolist()) - {0, 1})
                raise NotImplementedError(f"orders {bad} are not supported yet: L2 "
                                          "has orders 0 and 1")
            orders = tack.empty(host.shape, tack.i32)
            if host.size:
                tack.copyto(orders, host.astype(np.int32))
            self._orders = orders
        return self._orders

    @on_inputs_device
    def _layout(self):
        """``(offsets, size)``, made on first use and kept."""
        if "_offsets" not in self.__dict__:
            _needs_reference_cells(self, "L2")
            topology = self.topology
            n = topology.num_cells
            if self.varies:
                orders = self.cell_keys()
                counts = tack.empty((n,), tack.i32)
                for group in topology.groups():
                    if group.count:
                        _l2_counts(group.view(), orders, counts)
                offsets = tack.empty((n + 1,), ids.at_least(topology.id_dtype, 8 * n))
                size = exclusive_scan(counts, offsets, n) if n else 0
                _set_last(offsets, n, size)
                self._offsets, self._size = offsets, int(size)
            elif self.tensor:
                for group in topology.groups():
                    if group.count and not (bases.tensor_shape(group.shape) or (
                            (bases.simplex_shape(group.shape) or bases.wedge_shape(group.shape))
                            and self.order >= 2)):
                        raise NotImplementedError(
                            f"{self!r} on {group.shape.__name__} cells is not supported "
                            "yet; it is on segments, quads, pixels, hexahedra and voxels, "
                            "and from order 2 on triangles, tetrahedra and wedges")
                    if group.count:
                        _refuse_simplex_nodes(self, group.shape)
                counts = tack.empty((n,), tack.i32)
                for group in topology.groups():
                    if group.count:
                        _tensor_counts(group.view(), self.order, counts)
                offsets = tack.empty((n + 1,), ids.at_least(topology.id_dtype,
                                                            (self.order + 1) ** 3 * n))
                size = exclusive_scan(counts, offsets, n) if n else 0
                _set_last(offsets, n, size)
                self._offsets, self._size = offsets, int(size)
            elif self.order == 0:
                # A ramp, computed where it is read rather than stored.
                self._offsets = CountingArray(n + 1, dtype=topology.id_dtype)
                self._size = n
            elif hasattr(topology, "offsets"):
                self._offsets = topology.offsets          # the connectivity's: shared
                self._size = int(size_of(topology.connectivity))
            else:
                k = topology.groups()[0].shape.NUM_POINTS
                dtype = ids.at_least(topology.id_dtype, n * k)
                self._offsets = CountingArray(n + 1, 0, k, dtype=dtype)    # k values a cell
                self._size = n * k
        return self._offsets, self._size

    @property
    def offsets(self):
        """Where each cell's values start: value ``j`` of cell ``c`` is value
        ``offsets[c] + j``, and ``offsets[num_cells]`` is ``size``."""
        return self._layout()[0]

    @property
    def size(self):
        return self._layout()[1]

    def attributes(self, group=None, dtype=None):
        attributes = {"offsets": self.offsets}
        if self.tensor and group is not None and bases.simplex_shape(group.shape):
            attributes.update(_simplex_tables(self, group, dtype or tack.f32))
        elif self.tensor and group is not None and bases.wedge_shape(group.shape):
            attributes.update(_wedge_tables(self, group, dtype or tack.f32))
        elif self.tensor and group is not None:
            attributes.update(_basis_tables(self, group, dtype or tack.f32))
        if self.layout == "cells":
            attributes["layout_cells"] = self.topology.num_cells
        return attributes

    def __repr__(self):
        if self.varies:
            return "L2(order=per cell)"
        return super().__repr__()


class Constant(Space):
    """One value per cell (L2, order 0). Today's cell data."""

    mixin = views._ConstantField
    interpolated = True

    def mixin_for(self, key, shape=None):
        no_reference = (shapes.Polyhedron, shapes.Polygon, shapes.PolyLine)
        return views._PolyhedralCellField if shape in no_reference else self.mixin

    @property
    def size(self):
        return self.topology.num_cells


class Values(Space):
    """One value per entity of kind ``on`` -- points, edges, faces or cells -- with no
    basis: data about the entity, not a function to interpolate.

    On faces, ``oriented=True`` says the value is measured along the face's stored
    normal (out of its side 0): a normal flux or velocity component, as MPAS's
    ``normalVelocity`` or ``upwind_flux``'s output. Turning a face around -- as
    ``threshold`` must when it keeps only a face's side-1 cell -- negates it. A
    plain face value (an area, an id, a quality) is left alone.
    """

    mixin = views._ValuesField

    @classmethod
    def _params(cls, on, oriented=False):
        if on not in ("points", "edges", "faces", "cells"):
            raise ValueError(f"values live on points, edges, faces or cells, not {on!r}")
        if oriented and on != "faces":
            raise ValueError("only values on faces are oriented, along the face's normal")
        return {"on": on, "oriented": bool(oriented)}

    @property
    def size(self):
        topology = self.topology
        return {"points": lambda: topology.num_points,
                "cells": lambda: topology.num_cells,
                "faces": lambda: topology.faces().num_faces,
                "edges": lambda: topology.edges().num_edges}[self.on]()


class SideTraces(Space):
    """Per side: each face holds, for each of its sides, a value at each of its
    points (in the face's own row) -- a field's trace from each cell onto the
    face, MFEM's double-valued face E-vector. Laid out ``(face * 2 + side) * 4 +
    point``, triangles leaving the fourth slot unused and boundary faces side 1.
    ``tack.dataset.algorithms.traces`` fills one from any field with a basis."""

    on = "faces"
    mixin = views._SideTracesField
    interpolated = True
    #: Values per face: two sides of at most four points.
    PER_FACE = 8

    @property
    def size(self):
        return self.topology.faces().num_faces * self.PER_FACE
