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

import weakref

import numpy as np

import tack
from tack.algorithms.scan import exclusive_scan
from tack.arrays import CountingArray, size_of
from tack.dataset import ids, shapes, views
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
        return type(self)(topology, **{k: v.to_device(target) if _is_array(v) else v
                                       for k, v in self.params.items()})

    def attributes(self, group=None):
        """The layout a view of this space reads for ``group``, beside the values."""
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


def _only_order_one(family, order):
    if order != 1:
        raise NotImplementedError(
            f"{family} order {order} is not supported yet: its values on edges, faces "
            "and interiors need a cell -> DOF map built from the derived edges and faces")
    return {"order": order}


@tack.kernel
def _quad_face_dofs(kinds, flags):
    for f in range(kinds.shape[0]):
        flags[f] = 1 if kinds[f] == shapes.QUAD else 0


@tack.kernel
def _place(flags, slots, base, out):
    for i in range(flags.shape[0]):
        out[i] = base + slots[i] if flags[i] == 1 else -1


@tack.kernel
def _interiors(cells, flags):
    for c in cells:
        flags[cells.entity_id(c)] = cells.QUADRATIC_INTERIOR


class H1(Space):
    """Continuous: values shared by the cells around them, interpolated by the shape's
    functions. Order 1 holds one value per point -- today's point data; order 2
    adds one per edge, per quad face and per hexahedron or quad, for the shape's
    quadratic Lagrange functions (tetrahedra, hexahedra, voxels, wedges,
    triangles, quads and pixels)."""

    on = "points"
    interpolated = True

    @classmethod
    def _params(cls, order=1):
        if order not in (1, 2):
            _only_order_one("H1", order)
        return {"order": int(order)}

    @property
    def mixin(self):
        return views._H1Field if self.order == 1 else views._H1Order2Field

    @on_inputs_device
    def _layout(self):
        """Order 2: ``(face_dofs, cell_dofs, size)``, made on first use and kept."""
        if "_order2" not in self.__dict__:
            _needs_reference_cells(self, "H1 of order 2")
            topology = self.topology
            for group in topology.groups():
                if group.count and not group.shape.NUM_QUADRATIC:
                    raise NotImplementedError(
                        f"H1 order 2 on {group.shape.__name__} cells is not supported yet; "
                        "it is on tetrahedra, hexahedra, voxels, wedges, triangles, quads "
                        "and pixels")
            edges = topology.edges()
            base = topology.num_points + edges.num_edges
            # Only quad faces of solids hold values: a mesh of tetrahedra, triangles
            # or quads needs no faces derived.
            faces = (topology.faces() if any(g.shape.QUADRATIC_FACES for g in topology.groups())
                     else None)
            nf = faces.num_faces if faces is not None else 0
            n = topology.num_cells
            idt = ids.at_least(topology.id_dtype, base + nf + n)
            flags = tack.empty((nf,), tack.i32)
            slots = tack.empty((nf,), idt)
            face_dofs = tack.empty((nf,), idt)
            quads = 0
            if nf:
                _quad_face_dofs(faces.kinds, flags)
                quads = exclusive_scan(flags, slots, nf)
                _place(flags, slots, base, face_dofs)
            flags = tack.zeros((n,), tack.i32)
            for group in topology.groups():
                if group.count:
                    _interiors(group.view(), flags)
            slots = tack.empty((n,), idt)
            cell_dofs = tack.empty((n,), idt)
            inside = exclusive_scan(flags, slots, n) if n else 0
            if n:
                _place(flags, slots, base + quads, cell_dofs)
            self._order2 = (face_dofs, cell_dofs, base + quads + inside)
        return self._order2

    @property
    def size(self):
        if self.order == 1:
            return self.topology.num_points
        return self._layout()[2]

    def attributes(self, group=None):
        if self.order == 1:
            return {}
        face_dofs, cell_dofs, _ = self._layout()
        topology = self.topology
        attributes = {"edge_base": topology.num_points, "face_dofs": face_dofs,
                      "cell_dofs": cell_dofs}
        if group is not None:
            root = group.root
            edges = topology.edges()
            attributes.update(side_edge=edges.side_edge, side_sign=edges.side_sign,
                              edge_start=edges.group_starts[root])
            if group.shape.QUADRATIC_FACES:
                faces = topology.faces()
                attributes.update(side_face=faces.side_face, side_slot=faces.side_slot,
                                  side_orientation=faces.side_orientation,
                                  face_start=faces.group_starts[root])
        return attributes

    def mixin_for(self, key, shape=None):
        # Polyhedra evaluate in their sub-tetrahedra.
        if shape in (shapes.Polyhedron, shapes.Polygon) and self.order == 1:
            return views._PolyhedralPointField
        if shape is shapes.PolyLine and self.order == 1:
            return views._PolylinePointField
        return self.mixin

    def extra_mixins(self, group):
        if self.order == 1:
            return ()
        faces = (views._FaceIncidence,) if group.shape.QUADRATIC_FACES else ()
        return (views._EdgeIncidence, *faces, views.quadratic_faces(group.shape))


@tack.kernel
def _l2_counts(cells, orders, counts):
    for c in cells:
        e = cells.entity_id(c)
        counts[e] = 1 if orders[e] == 0 else cells.NUM_POINTS


@tack.kernel
def _set_last(offsets, n, total):
    for i in range(1):
        offsets[n] = total


class L2(Space):
    """Discontinuous: each cell's own values, at ``offsets[cell]``. Order 1 holds a
    value at each of the cell's corners -- a linear DG field -- and order 0 one
    value per cell. ``order`` is one order for every cell, or an array of one per
    cell (p-adaptive), read when the space is made."""

    mixin = views._L2Field
    interpolated = True

    @classmethod
    def _params(cls, order=1):
        if _is_array(order):
            return {"order": order}
        if order not in (0, 1):
            _only_order_one("L2", order)
        return {"order": int(order)}

    @property
    def varies(self):
        return _is_array(self.order)

    def mixin_for(self, key, shape=None):
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

    def attributes(self, group=None):
        return {"offsets": self.offsets}

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
