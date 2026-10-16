"""Sets: named selections of a dataset's points, cells or faces (``data.sets``).

A ``Set`` knows what it selects (``on``), so a cell or face set is an
iteration domain, filters carry sets as they carry fields, and an id outside
the topology is refused when the set is made. Its ids are sorted and unique.
"""

import numpy as np

import tack
from tack.algorithms.scan import exclusive_scan
from tack.dataset import ids as id_types
from tack.runtime.dispatch import as_device, get_backend, on_inputs_device, same_device

__all__ = ["Set"]

#: Names a set may not take: the iteration domains themselves.
DOMAINS = ("points", "cells", "faces", "edges")


@tack.kernel
def _mark(ids, flags, bad, n):
    for i in range(ids.shape[0]):
        k = ids[i]
        if k < 0 or k >= n:
            tack.atomic_add(bad, 0, 1)
        else:
            tack.atomic_max(flags, k, 1)


@tack.kernel
def _members(flags, slots, out):
    for i in range(flags.shape[0]):
        if flags[i] == 1:
            out[slots[i]] = i


@tack.kernel
def _side_faces(cells, masks, flags, bad):
    # The faces named by (cell, local face) pairs, a bit per local face.
    for c in cells:
        m = masks[cells.entity_id(c)]
        if m != 0:
            n = cells.num_faces(c)
            if n < 64 and (m >> n) != 0:
                tack.atomic_add(bad, 0, 1)
            for k in range(n):
                if (m >> k) & 1:
                    tack.atomic_max(flags, cells.face_id(c, k), 1)


@tack.kernel
def _listed_faces(cells, local, cell_offsets, cell_faces, flags, bad):
    # The faces named by (cell, local face) pairs, where each cell lists its faces.
    for i in range(cells.shape[0]):
        first = cell_offsets[cells[i]]
        if local[i] >= cell_offsets[cells[i] + 1] - first:
            tack.atomic_add(bad, 0, 1)
        else:
            tack.atomic_max(flags, cell_faces[first + local[i]], 1)


def _count(topology, on):
    if on == "points":
        return topology.num_points
    if on == "cells":
        return topology.num_cells
    return topology.faces().num_faces


class Set:
    """Some of a topology's ``on`` entities -- ``"points"``, ``"cells"`` or
    ``"faces"`` -- by id: ``Set(data, on, ids)``, ``ids`` a field, an array or a
    list. Duplicates collapse and the ids are kept sorted (``ids``, a field of
    the topology's ``id_dtype``); an id outside the topology is refused.

    A face set is also a side set: on the boundary a face has one side, so its
    face names its cell's side; ``Set.from_sides`` takes (cell, local face)
    pairs, as Exodus side sets give them. Put a set in ``data.sets`` to name it.
    """

    @on_inputs_device
    def __init__(self, data, on, ids):
        topology = getattr(data, "topology", data)
        if on not in ("points", "cells", "faces"):
            raise ValueError(f"a set selects points, cells or faces, not {on!r}")
        self.topology = topology
        self.on = on
        n = _count(topology, on)
        idt = topology.id_dtype
        given = id_types.as_ids(ids, id_types.at_least(idt, n))
        flags = tack.zeros((n,), tack.i32)
        if given.shape[0]:
            bad = tack.zeros((1,), tack.i32)
            _mark(given, flags, bad, n)
            if bad[0]:
                raise ValueError(f"{bad[0]} ids are outside the topology's {n} {on}")
        self._adopt(flags)

    @classmethod
    def _from_sorted_ids(cls, topology, on, ids):
        """A set of ids already sorted, unique and in range (as ``boundary()``
        gives them): taken as they are, its flags made only when asked for."""
        out = cls.__new__(cls)
        out.topology = topology
        out.on = on
        out.ids = ids
        out._flags = None
        return out

    @classmethod
    def _from_flags(cls, topology, on, flags):
        """A set from a 0/1 i32 array over the entities."""
        out = cls.__new__(cls)
        out.topology = topology
        out.on = on
        out._adopt(flags)
        return out

    def _adopt(self, flags):
        n = flags.shape[0]
        idt = self.topology.id_dtype
        slots = tack.empty((n,), idt)
        count = exclusive_scan(flags, slots, n) if n else 0
        self.ids = tack.empty((count,), idt)
        if count:
            _members(flags, slots, self.ids)
        self._flags = flags

    @classmethod
    @on_inputs_device
    def from_sides(cls, data, cells, local_faces):
        """The faces named by (cell, local face) pairs -- local faces numbered as the
        cell's view numbers them (VTK's order on the shape path, the cell's own
        face list on the polyhedral one). Which side was named is not kept: on
        the boundary, a face has only the one."""
        from tack.dataset.dataset import for_each

        cells = np.asarray(cells, np.int64).reshape(-1)
        local = np.asarray(local_faces, np.int64).reshape(-1)
        if cells.shape != local.shape:
            raise ValueError("one local face per cell")
        n = data.num_cells
        if cells.size and (cells.min() < 0 or cells.max() >= n):
            raise ValueError(f"cells outside the topology's {n}")
        if local.size and local.min() < 0:
            raise ValueError("local faces are numbered from 0")
        topology = data.topology
        faces = topology.faces()
        flags = tack.zeros((faces.num_faces,), tack.i32)
        bad = tack.zeros((1,), tack.i32)
        if hasattr(topology, "cell_faces"):
            # Cells list their faces, as many as they like: look each pair up.
            if cells.size:
                _listed_faces(tack.array(cells), tack.array(local),
                              topology.cell_offsets, topology.cell_faces, flags, bad)
        else:
            # Shapes have at most six faces: a bit per local face, a mask per cell.
            if local.size and local.max() >= 63:
                bad = tack.full((1,), int((local >= 63).sum()), tack.i32)
            else:
                host = np.zeros(n, np.int64)
                np.bitwise_or.at(host, cells, np.left_shift(np.int64(1), local))
                masks = tack.empty((n,), tack.i64)
                if n:
                    tack.copyto(masks, host)
                if cells.size:
                    for_each(_side_faces, data, "cells", masks, flags, bad)
        if bad[0]:
            raise ValueError(f"{bad[0]} cells are given a local face they do not have")
        return cls._from_flags(data.topology, "faces", flags)

    @property
    def _device(self):
        """Its topology's device, which its ids are on."""
        return same_device("Set", (self.__dict__.get("topology"), self.__dict__.get("ids")))

    @property
    def device(self):
        """The `tack.Device` its ids are on."""
        return as_device(self._device or get_backend())

    @on_inputs_device
    def flags(self):
        """Membership: an i32 array, 1 for each entity in the set, else 0."""
        if self._flags is None:
            n = _count(self.topology, self.on)
            flags = tack.zeros((n,), tack.i32)
            if self.ids.shape[0]:
                _mark(self.ids, flags, tack.zeros((1,), tack.i32), n)
            self._flags = flags
        return self._flags

    def cell_keys(self):
        """Each cell's key, for splitting the cells' groups: its membership."""
        return self.flags()

    def __len__(self):
        return int(self.ids.shape[0])

    def numpy(self):
        """The ids, sorted, on the host."""
        return self.ids.numpy()

    def __array__(self, dtype=None, copy=None):
        """``np.asarray(set)``: the ids, sorted, on the host."""
        if copy is False:
            raise ValueError("a Set cannot be viewed by NumPy without a copy")
        ids = self.numpy()
        return ids if dtype is None else ids.astype(dtype, copy=False)

    def __repr__(self):
        return f"Set(on={self.on!r}, {len(self)} ids)"


def check(name, value, topology):
    """``value`` as the set ``name`` of a dataset on ``topology``, or refused."""
    if not isinstance(value, Set):
        raise TypeError(f"set {name!r} is a {type(value).__name__}, not a tack.dataset.Set: "
                        f"Set(data, 'faces', ids) names faces, and so on")
    if name in DOMAINS:
        raise ValueError(f"a set may not be named {name!r}, an iteration domain's name")
    if value.topology is not topology:
        raise ValueError(f"set {name!r} selects from another topology")
    return value


class Sets(dict):
    """``data.sets``: names to ``Set``s of the dataset's topology, checked as they
    are put in."""

    def __init__(self, topology, sets=None):
        super().__init__()
        self._topology = topology
        self.update(sets or {})

    def __setitem__(self, name, value):
        super().__setitem__(name, check(name, value, self._topology))

    def update(self, *args, **kwargs):
        # Every entry checked before any goes in: a refused update leaves the
        # sets as they were.
        new = dict(*args, **kwargs)
        for name, value in new.items():
            check(name, value, self._topology)
        for name, value in new.items():
            super().__setitem__(name, value)

    def __ior__(self, other):
        # dict's |= merges in C, past __setitem__ and update.
        self.update(other)
        return self

    def setdefault(self, name, value=None):
        if name not in self:
            self[name] = value
        return self[name]


def rebind(sets, topology):
    """``sets`` on ``topology``, whose entities are numbered as theirs were."""
    return {name: Set._from_flags(topology, s.on, s.flags()) for name, s in sets.items()}
