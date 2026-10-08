"""Spaces: where a field's values live, on one topology, and the layout that follows.

A space is constructed on a topology (or a dataset, for its topology) and
parameters, as MFEM's ``FiniteElementSpace`` is on a mesh::

    td.H1(data)                 # one value per point, interpolated linearly
    td.H1(data, order=2)        # (not in the prototype: needs a DOF map)
    td.L2(data)                 # each cell's own corner values: linear DG
    td.Constant(data)           # one value per cell
    td.Values(data, "faces")    # one value per face, no basis

Spaces are interned on their topology: the same kind and parameters on the
same topology is the same object, so fields that share a space share its
layout -- and equality is identity. The layout is what a kernel needs
beyond a field's values to find them:

- ``H1``, order 1: nothing of its own. Its values are the points', found
  through the topology's connectivity. (Order ``p`` would own a cell -> DOF
  map built from the derived edges and faces, one per shape group.)
- ``L2``, order 1: ``offsets``, where each cell's values start. An
  unstructured topology's are its connectivity offsets, so they are shared,
  not copied.
- ``Constant`` and ``Values``: nothing; value ``k`` belongs to entity ``k``.

Every space knows ``size``, the number of values a field on it holds, and
``on``, the entities a loop must run over to read it through ``view``.
"""

import numpy as np

import tack
from tack.data import views

__all__ = ["H1", "L2", "Constant", "Space", "Values"]


class Space:
    """What every space has: ``topology``, ``size``, ``on`` and a field-view ``mixin``,
    and the layout ``attributes`` that mixin reads."""

    on = "cells"
    mixin = None
    #: Spaces with a basis evaluate at parametric coordinates (``value``,
    #: ``parametric_gradient``); the others only give each entity its value.
    interpolated = False

    def __new__(cls, where, *args, **kwargs):
        topology = getattr(where, "topology", where)
        params = cls._params(*args, **kwargs)
        key = (cls, tuple(sorted(params.items())))
        interned = topology.__dict__.setdefault("_spaces", {})
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

    def attributes(self):
        """The layout a view of this space reads, beside the field's values."""
        return {}

    @property
    def size(self):
        raise NotImplementedError

    def __repr__(self):
        args = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
        return f"{type(self).__name__}({args})"


def _only_order_one(family, order):
    if order != 1:
        raise NotImplementedError(
            f"{family} order {order} is not in the prototype: its values on edges, faces "
            "and interiors need a cell -> DOF map built from the derived edges and faces")
    return {"order": order}


class H1(Space):
    """Continuous: one value per point, shared by the cells around it, interpolated by
    the shape's functions. Order 1 is today's point data."""

    on = "points"
    mixin = views._H1Field
    interpolated = True

    @classmethod
    def _params(cls, order=1):
        return _only_order_one("H1", order)

    @property
    def size(self):
        return self.topology.num_points


class L2(Space):
    """Discontinuous: each cell's own values, at ``offsets[cell]``. Order 1 holds a
    value at each of the cell's corners: a linear DG field."""

    mixin = views._L2Field
    interpolated = True

    @classmethod
    def _params(cls, order=1):
        return _only_order_one("L2", order)

    def _layout(self):
        """``(offsets, size)``, made on first use and kept."""
        if "_offsets" not in self.__dict__:
            topology = self.topology
            if hasattr(topology, "offsets"):
                self._offsets = topology.offsets          # the connectivity's: shared
                self._size = int(topology.connectivity.shape[0])
            else:
                k = topology.groups()[0].shape.NUM_POINTS
                offsets = tack.field(tack.i32, shape=(topology.num_cells + 1,))
                offsets.from_numpy(np.arange(topology.num_cells + 1, dtype=np.int32) * k)
                self._offsets = offsets
                self._size = topology.num_cells * k
        return self._offsets, self._size

    @property
    def offsets(self):
        """Where each cell's values start: corner ``j`` of cell ``c`` is value
        ``offsets[c] + j``, and ``offsets[num_cells]`` is ``size``."""
        return self._layout()[0]

    @property
    def size(self):
        return self._layout()[1]

    def attributes(self):
        return {"offsets": self.offsets}


class Constant(Space):
    """One value per cell (L2, order 0). Today's cell data."""

    mixin = views._ConstantField
    interpolated = True

    @property
    def size(self):
        return self.topology.num_cells


class Values(Space):
    """One value per entity of kind ``on`` -- points, edges, faces or cells -- with no
    basis: data about the entity, not a function to interpolate."""

    mixin = views._ValuesField

    @classmethod
    def _params(cls, on):
        if on not in ("points", "edges", "faces", "cells"):
            raise ValueError(f"values live on points, edges, faces or cells, not {on!r}")
        return {"on": on}

    @property
    def size(self):
        topology = self.topology
        return {"points": lambda: topology.num_points,
                "cells": lambda: topology.num_cells,
                "faces": lambda: topology.faces().num_faces,
                "edges": lambda: topology.edges().num_edges}[self.on]()
