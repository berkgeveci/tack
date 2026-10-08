"""Datasets: a topology, a geometry, fields on spaces, and sets.

``docs/design/dataset-api.md`` §3 in prototype form:

- *spaces* say where a field's values live: ``H1()`` one per point,
  interpolated linearly (point data); ``Constant()`` one per cell (cell
  data); ``L2()`` each cell's own corner values (a linear discontinuous
  Galerkin field); ``Values(on)`` one value per point, edge, face or cell,
  with no basis -- data *about* the entity;
- a *field* is a space and its values, which are any array: a device
  field, or an implicit array from ``tack.data.arrays`` (a
  ``CartesianProduct`` of three axes, a ``ConstantArray``, a
  ``CountingArray``);
- the *geometry* is a field too, named ``"shape"``: ``H1`` positions, one
  per point (a rectilinear grid's are a ``CartesianProduct``), or ``L2``
  positions, each cell's own corners;
- *sets* are named arrays of entity ids, such as the boundary faces.

``for_each(kernel, data, domain, *args)`` runs ``kernel`` once per shape in
the domain -- ``"cells"``, ``"faces"``, ``"edges"``, or a face set --
passing the domain's view and, for every ``Field`` among ``args``, the
field's view for the same group.
"""

import numpy as np

import tack
from tack.data import arrays, views
from tack.data.arrays import CartesianProduct

__all__ = ["H1", "L2", "Constant", "DataSet", "Field", "Values", "for_each",
           "rectilinear_grid"]


# ── Spaces ──────────────────────────────────────────────────────────

class Space:
    """Where a field's values live. ``on`` names the entity kind they belong to."""

    on = "cells"
    mixin = None

    def __eq__(self, other):
        return type(self) is type(other) and vars(self) == vars(other)

    def __hash__(self):
        return hash((type(self), tuple(sorted(vars(self).items()))))

    def __repr__(self):
        args = ", ".join(f"{k}={v!r}" for k, v in vars(self).items())
        return f"{type(self).__name__}({args})"


class H1(Space):
    """Continuous, linear: one value per point (``Shared`` layout), interpolated by the
    cell's shape functions. Today's point data."""

    on = "points"
    mixin = views._H1Field


class Constant(Space):
    """One value per cell (L2, order 0). Today's cell data."""

    mixin = views._ConstantField


class L2(Space):
    """Discontinuous, linear: each cell's own value at each of its corners (``PerCell``
    layout), at ``offsets[cell]`` in the field's values. A linear DG field."""

    mixin = views._L2Field


class Values(Space):
    """One value per entity of kind ``on`` (points, edges, faces or cells), no basis."""

    mixin = views._ValuesField

    def __init__(self, on):
        if on not in ("points", "edges", "faces", "cells"):
            raise ValueError(f"values live on points, edges, faces or cells, not {on!r}")
        self.on = on


class Field:
    """A space and its values -- any array, explicit or implicit, of scalars or
    vectors -- and for ``L2`` the per-cell ``offsets`` into them."""

    def __init__(self, space, values, offsets=None):
        arrays.storage(values)                       # refuses what is not an array
        self.space = space
        self.values = values
        self.offsets = offsets
        if isinstance(space, L2) and offsets is None:
            raise ValueError("an L2 field needs the offset of each cell's values")

    def view(self, group):
        """This field's view for ``group``: the group's kind and shape, the space, and
        the storage its values are read through."""
        mixins, attributes = _space_parts(self, group)
        return group.view(*mixins, **attributes)

    def __repr__(self):
        values = self.values
        stored = repr(values) if isinstance(values, arrays._Implicit) else "stored"
        return f"Field({self.space!r}, {arrays.size_of(values)} values, {stored})"


def _space_parts(field, group, space_mixin=None):
    """The mixins and attributes that read ``field`` for ``group``: its space's (or
    ``space_mixin``, for the geometry), an addressing mixin for values shared by
    points, and its storage's."""
    storage, attributes = arrays.storage(field.values)
    mixins = [space_mixin or field.space.mixin]
    if isinstance(field.space, H1):
        mixins.append(views.point_address(group.kind, storage))
    mixins.append(storage)
    if field.offsets is not None:
        attributes["point_offsets" if space_mixin else "offsets"] = field.offsets
    return mixins, attributes


# ── Geometry ────────────────────────────────────────────────────────

def _as_geometry(geometry, dtype):
    """``geometry`` as a ``Field``: positions per point (a host array, or any array of
    3-vectors, such as a ``CartesianProduct``) become an ``H1`` field; a ``Field``
    is checked."""
    if not isinstance(geometry, Field):
        try:
            arrays.storage(geometry)
        except TypeError:
            positions = np.ascontiguousarray(geometry)
            geometry = tack.Vector.field(3, dtype, shape=(positions.shape[0],))
            geometry.from_numpy(positions.astype(dtype.numpy_dtype))
        geometry = Field(H1(), geometry)
    if not isinstance(geometry.space, (H1, L2)):
        raise TypeError(f"geometry lives in H1 or L2, not {geometry.space!r}")
    if arrays.width_of(geometry.values) != 3:
        raise TypeError("geometry values must be 3-vectors")
    return geometry


def _geometry_parts(geometry, group, kind):
    """The mixins and attributes that give ``group``'s view, an entity of ``kind``, the
    geometry field ``geometry``."""
    if isinstance(geometry.space, L2):
        if kind != "cells":
            raise NotImplementedError(
                f"the {kind} of an L2 geometry have no positions of their own: each side's "
                "cell has its own corners there, which needs per-side traces "
                "(docs/design/dataset-api.md, section 9)")
        return _space_parts(geometry, group, views._L2Geometry)
    return _space_parts(geometry, group, views._H1Geometry)


# ── Datasets ────────────────────────────────────────────────────────

class DataSet:
    """A topology, named fields and named sets. The geometry is the field named
    ``"shape"``, given as ``geometry``: a ``Field`` in ``H1`` or ``L2``, or positions
    per point (a host array, or any array of 3-vectors, such as a device field or
    a ``CartesianProduct``), which become an ``H1`` field."""

    def __init__(self, topology, geometry, fields=None, sets=None, dtype=tack.f32):
        self.topology = topology
        self.fields = dict(fields or {})
        if "shape" in self.fields:
            raise ValueError('"shape" is the geometry; pass it as geometry')
        self.fields["shape"] = _as_geometry(geometry, dtype)
        self.sets = dict(sets or {})

    @property
    def geometry(self):
        """The geometry: the field named ``"shape"``."""
        return self.fields["shape"]

    @property
    def dtype(self):
        """The geometry's floating-point type."""
        return self.geometry.values.dtype

    @property
    def num_points(self):
        if isinstance(self.geometry.space, L2):
            return self.topology.num_points
        return arrays.size_of(self.geometry.values)

    @property
    def num_cells(self):
        return self.topology.num_cells

    def positions(self):
        """Every point's position as a host array, ``(num_points, 3)``: an ``H1``
        geometry's values. An ``L2`` geometry has none per point."""
        if isinstance(self.geometry.space, L2):
            raise ValueError("an L2 geometry has positions per cell corner, not per point")
        return arrays.to_host(self.geometry.values)

    def l2_offsets(self):
        """Where each cell's values start in an ``L2`` field: cell ``c``'s corner ``j``
        is at ``offsets[c] + j``, and ``offsets[num_cells]`` is the field's size.
        An unstructured topology's are its connectivity offsets."""
        topology = self.topology
        if hasattr(topology, "offsets"):
            return topology.offsets
        k = topology.groups()[0].shape.NUM_POINTS
        offsets = tack.field(tack.i32, shape=(topology.num_cells + 1,))
        offsets.from_numpy(np.arange(topology.num_cells + 1, dtype=np.int32) * k)
        return offsets

    def domain_groups(self, domain):
        """The ``DomainGroup``s of ``domain``: ``"cells"``, ``"faces"``, ``"edges"``, or
        the name of a face set in ``sets``."""
        if domain == "cells":
            return self.topology.groups()
        if domain == "faces":
            return self.topology.faces().groups()
        if domain == "edges":
            return self.topology.edges().groups()
        if domain in self.sets:
            return self.topology.faces().groups(self.sets[domain])
        raise ValueError(f"no domain {domain!r}: cells, faces, edges, or a set "
                         f"({', '.join(self.sets) or 'none'})")

    def domain_view(self, domain, group):
        """``group``'s view: kind, shape, geometry, and for cells their incidence when
        faces and edges have been derived."""
        mixins = []
        kind = domain if domain in ("cells", "faces", "edges") else "faces"
        geometry, attributes = _geometry_parts(self.geometry, group, kind)
        mixins.extend(geometry)
        if domain == "cells":
            faces = self.topology._faces
            if faces is not None and group.shape.NUM_FACES:
                mixins.append(views._FaceIncidence)
                attributes.update(side_face=faces.side_face, side_slot=faces.side_slot,
                                  face_start=faces.group_starts[id(group)])
            edges = self.topology._edges
            if edges is not None and group.shape.NUM_EDGES:
                mixins.append(views._EdgeIncidence)
                attributes.update(side_edge=edges.side_edge, side_sign=edges.side_sign,
                                  edge_start=edges.group_starts[id(group)])
        return group.view(*mixins, **attributes)


def for_each(kernel, data, domain, *args):
    """Run ``kernel(view, *args)`` once per shape in ``domain`` of ``data``.

    ``view`` is the domain's view for that shape (``for i in view``); every
    ``Field`` among ``args`` becomes its view for the same shape, so
    ``u.value(i, pc)`` and ``u.at(i)`` read the entity ``i`` the loop is at.
    Other arguments pass through.
    """
    kind = domain if domain in ("cells", "faces", "edges") else "faces"
    for arg in args:
        if isinstance(arg, Field):
            _check_domain(arg.space, kind)
    for group in data.domain_groups(domain):
        if not group.count:
            continue
        view = data.domain_view(domain, group)
        kernel(view, *(a.view(group) if isinstance(a, Field) else a for a in args))


def _check_domain(space, kind):
    """A field's view must make sense where the loop is."""
    if isinstance(space, H1):
        return                                   # points belong to every entity
    if isinstance(space, (Constant, L2)) and kind == "cells":
        return
    if isinstance(space, Values) and space.on == kind:
        return
    raise TypeError(f"a field on {space!r} cannot be viewed while iterating {kind}")


def rectilinear_grid(x, y=(0.0,), z=(0.0,), dtype=tack.f32):
    """A dataset of the grid whose lines run through ``x``, ``y`` and ``z``."""
    from tack.data.topology import StructuredTopology

    coordinates = CartesianProduct(x, y, z, dtype=dtype)
    return DataSet(StructuredTopology(coordinates.point_dims), coordinates)
