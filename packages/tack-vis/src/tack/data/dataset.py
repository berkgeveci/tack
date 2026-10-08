"""Datasets: a topology, a geometry, fields on spaces, and sets.

``docs/design/dataset-api.md`` §3 in prototype form:

- *spaces* say where a field's values live: ``H1()`` one per point,
  interpolated linearly (point data); ``Constant()`` one per cell (cell
  data); ``L2()`` each cell's own corner values (a linear discontinuous
  Galerkin field); ``Values(on)`` one value per point, edge, face or cell,
  with no basis -- data *about* the entity;
- a *field* is a space and its values;
- the *geometry* is a field too: positions per point (an explicit H1
  vector field), or ``RectilinearCoordinates``, which store only the axes;
- *sets* are named arrays of entity ids, such as the boundary faces.

``for_each(kernel, data, domain, *args)`` runs ``kernel`` once per shape in
the domain -- ``"cells"``, ``"faces"``, ``"edges"``, or a face set --
passing the domain's view and, for every ``Field`` among ``args``, the
field's view for the same group.
"""

import numpy as np

import tack
from tack.data import views
from tack.lang.field import Field as _DeviceField

__all__ = ["H1", "L2", "Constant", "DataSet", "Field", "RectilinearCoordinates", "Values",
           "for_each", "rectilinear_grid"]


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
    """A space and its values: a field of scalars or vectors, and for ``L2`` the
    per-cell ``offsets`` into them."""

    def __init__(self, space, values, offsets=None):
        self.space = space
        self.values = values
        self.offsets = offsets
        if isinstance(space, L2) and offsets is None:
            raise ValueError("an L2 field needs the offset of each cell's values")

    def view(self, group):
        """This field's view for ``group``: the group's kind and shape, and the space."""
        attributes = {"values": self.values}
        if self.offsets is not None:
            attributes["offsets"] = self.offsets
        return group.view(self.space.mixin, **attributes)

    def __repr__(self):
        return f"Field({self.space!r}, {self.values.shape[0]} values)"


# ── Geometry ────────────────────────────────────────────────────────

class RectilinearCoordinates:
    """The points of a grid whose lines run along the axes: (x[i], y[j], z[k]),
    numbered x fastest. Only the three axes are stored."""

    def __init__(self, x, y=(0.0,), z=(0.0,), dtype=tack.f32):
        self.dtype = dtype
        self.x, self.y, self.z = (self._axis(a) for a in (x, y, z))
        self.dims = (self.x.shape[0], self.y.shape[0], self.z.shape[0])

    def _axis(self, values):
        values = np.ascontiguousarray(values, dtype=self.dtype.numpy_dtype).reshape(-1)
        field = tack.field(self.dtype, shape=values.shape)
        field.from_numpy(values)
        return field

    @property
    def num_points(self):
        return int(np.prod(self.dims))

    @property
    def point_dims(self):
        dims = list(self.dims)
        while len(dims) > 1 and dims[-1] == 1:
            dims.pop()
        return tuple(dims)

    def to_numpy(self):
        z, y, x = np.meshgrid(self.z.to_numpy(), self.y.to_numpy(), self.x.to_numpy(),
                              indexing="ij")
        return np.stack([x, y, z], axis=-1).reshape(-1, 3)


def _geometry_mixin(geometry, group):
    """The mixin and attributes that give ``group``'s view its positions."""
    if isinstance(geometry, RectilinearCoordinates):
        structured = issubclass(group.kind, views._StructuredCells)
        mixin = views._RectilinearStructured if structured else views._RectilinearPoints
        return mixin, {"xs": geometry.x, "ys": geometry.y, "zs": geometry.z,
                       "px": geometry.dims[0], "py": geometry.dims[1]}
    return views._ExplicitPoints, {"points": geometry}


# ── Datasets ────────────────────────────────────────────────────────

class DataSet:
    """A topology, a geometry (positions per point, or ``RectilinearCoordinates``),
    named fields and named sets."""

    def __init__(self, topology, geometry, fields=None, sets=None, dtype=tack.f32):
        self.topology = topology
        if isinstance(geometry, (RectilinearCoordinates, _DeviceField)):
            self.geometry = geometry
        else:
            geometry = np.ascontiguousarray(geometry)
            field = tack.Vector.field(3, dtype, shape=(geometry.shape[0],))
            field.from_numpy(geometry.astype(dtype.numpy_dtype))
            self.geometry = field
        self.fields = dict(fields or {})
        self.sets = dict(sets or {})

    @property
    def num_points(self):
        if isinstance(self.geometry, RectilinearCoordinates):
            return self.geometry.num_points
        return self.geometry.shape[0] // 3

    @property
    def num_cells(self):
        return self.topology.num_cells

    def positions(self):
        """Every point's position as a host array, ``(num_points, 3)``."""
        if isinstance(self.geometry, RectilinearCoordinates):
            return self.geometry.to_numpy()
        return self.geometry.to_numpy(vectors=True)

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
        geometry, attributes = _geometry_mixin(self.geometry, group)
        mixins.append(geometry)
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

    coordinates = RectilinearCoordinates(x, y, z, dtype=dtype)
    return DataSet(StructuredTopology(coordinates.point_dims), coordinates)
