"""``subdivide``: a dataset of curved or high-order cells as linear sub-cells -- each
cell's lattice of ``level`` steps along each axis -- with its fields evaluated at
the lattice points. ``contour`` and ``slice`` cut high-order data this way, on
only the cells that can hold the cut (``_refined``).

A lattice point on a cell's edge, face or corner is shared with the cells around
it: its id is a node of the equispaced ``H1`` space of order ``level``, numbered
by entity, so every cell names it alike. Each is evaluated once, by the first of
the cells refined that holds it (a stable sort of the ids), so a continuous
field and the geometry have one value there, to the bit, and cuts through
neighbouring cells meet without cracks. No atomics, no tolerance.
"""

import numpy as np

import tack
from tack import arrays
from tack.algorithms.scan import exclusive_scan
from tack.algorithms.sort import argsort
from tack.arrays import ConstantArray, size_of
from tack.dataset import bases, ids, shapes
from tack.dataset.dataset import Field, for_each
from tack.dataset.spaces import H1, L2, Constant, Values
from tack.dataset.topology import UnstructuredTopology, uniform_cells
from tack.runtime.dispatch import on_inputs_device

__all__ = ['subdivide']


# ── Which cells ─────────────────────────────────────────────────────

@tack.kernel
def _all_cells(cells, flags):
    for c in cells:
        flags[cells.entity_id(c)] = 1


@tack.kernel
def _straddling(cells, u, levels, nlevels, flags):
    # A cell whose field's Bernstein bound holds a level: it may be cut.
    for c in cells:
        lo, hi = u.value_range(c)
        hit = 0
        for v in range(nlevels):
            if lo <= levels[v] and levels[v] <= hi:
                hit = 1
        flags[cells.entity_id(c)] = hit


@tack.kernel
def _flatness(cells, u, flags, out):
    # What linearizing each flagged cell at its corners loses, at most.
    for c in cells:
        e = cells.entity_id(c)
        out[e] = u.flatness(c) if flags[e] == 1 else u.flatness(c) - u.flatness(c)


#: The most lattice steps a tolerance chooses: 32^3 sub-cells a hexahedron.
MAX_LEVEL = 32


@tack.kernel
def _place_chosen(flags, slots, slot_of, chosen):
    for e in range(flags.shape[0]):
        if flags[e] == 1:
            slot_of[e] = slots[e]
            chosen[slots[e]] = e
        else:
            slot_of[e] = -1


# ── The lattice ─────────────────────────────────────────────────────

@tack.kernel
def _lattice_keys(cells, t, slot_of, per_cell, node_of, keys):
    # Each refined cell's lattice points' ids: the nodes of the equispaced H1 space
    # of the lattice's order (node_of: a lattice node's number in that space's
    # cell order, which at order 2 is the quadratic shapes').
    for c in cells:
        slot = slot_of[cells.entity_id(c)]
        if slot >= 0:
            for n in range(per_cell):
                keys[slot * per_cell + n] = t.dof_index(c, node_of[n])


@tack.kernel
def _corner_keys(cells, slot_of, corner_of, keys):
    # Level 1: the lattice is the corners.
    for c in cells:
        slot = slot_of[cells.entity_id(c)]
        if slot >= 0:
            for n in range(cells.NUM_POINTS):
                keys[slot * cells.NUM_POINTS + n] = cells.point_id(c, corner_of[n])


@tack.kernel
def _runs(keys, order, first):
    # first[k]: whether element k comes first among those of its id (in the
    # stable order: the lowest slot).
    for i in range(order.shape[0]):
        k = order[i]
        first[k] = 1 if i == 0 or keys[order[i - 1]] != keys[k] else 0


@tack.kernel
def _ranks(order, starts, numbers, rank):
    # The runs before position i, counting its own if it starts one, less one.
    for i in range(order.shape[0]):
        rank[order[i]] = numbers[i] + starts[i] - 1


@tack.kernel
def _starts(order, first, starts):
    for i in range(order.shape[0]):
        starts[i] = first[order[i]]


@tack.func
def _lattice_point(n, level, dimension, steps):
    """Lattice node ``n``'s parametric coordinates, ``steps[i]`` (``i / level`` in the
    dataset's float type) along each axis, the first axis fastest."""
    q = level + 1
    pc = tack.vec3(steps[n % q], 0.0, 0.0)
    if dimension >= 2:
        pc[1] = steps[(n // q) % q]
    if dimension == 3:
        pc[2] = steps[n // (q * q)]
    return pc


@tack.kernel
def _positions(cells, slot_of, per_cell, level, steps, corner_at, first, rank, out):
    # A lattice corner is the cell's corner, its stored position to the bit.
    for c in cells:
        slot = slot_of[cells.entity_id(c)]
        if slot >= 0:
            for n in range(per_cell):
                k = slot * per_cell + n
                if first[k] == 1:
                    if corner_at[n] >= 0:
                        out[rank[k]] = cells.corner_position(c, corner_at[n])
                    else:
                        out[rank[k]] = cells.position(c, _lattice_point(n, level,
                                                                       cells.DIMENSION, steps))


@tack.kernel
def _values(cells, u, slot_of, per_cell, level, steps, corner_at, first, rank, out):
    for c in cells:
        slot = slot_of[cells.entity_id(c)]
        if slot >= 0:
            for n in range(per_cell):
                k = slot * per_cell + n
                if first[k] == 1:
                    if corner_at[n] >= 0:
                        out[rank[k]] = u.corner_value(c, corner_at[n])
                    else:
                        out[rank[k]] = u.value(c, _lattice_point(n, level, cells.DIMENSION,
                                                                 steps))


@tack.kernel
def _sub_cells(cells, slot_of, per_cell, level, rank, sub, conn, parent):
    # Each refined cell's level^d sub-cells, in its frame: corners in VTK's order
    # of the cell's shape (a hexahedron's, a quad's, a line's).
    for c in cells:
        e = cells.entity_id(c)
        slot = slot_of[e]
        if slot >= 0:
            q = level + 1
            subs = 1
            for _ in range(cells.DIMENSION):
                subs *= level
            for k in range(subs):
                i = k % level
                j = (k // level) % level
                m = k // (level * level)
                out = (slot * subs + k) * sub
                parent[slot * subs + k] = e
                for corner in range(sub):
                    a = i + ((corner ^ (corner >> 1)) & 1)
                    b = j + ((corner >> 1) & 1)
                    h = m + ((corner >> 2) & 1)
                    if cells.DIMENSION == 1:
                        a = i + corner
                        b = 0
                        h = 0
                    node = a + q * (b + q * h)
                    conn[out + corner] = rank[slot * per_cell + node]


# ── The refined dataset ─────────────────────────────────────────────

def _tensor_cells(data, what):
    for group in data.topology.groups():
        if group.count and not bases.tensor_shape(group.shape):
            raise NotImplementedError(
                f"{what} is on segments, quads, pixels, hexahedra and voxels; not yet on "
                f"{group.shape.__name__} cells")
    dims = {g.shape.DIMENSION for g in data.topology.groups() if g.count}
    if len(dims) > 1:
        raise NotImplementedError(f"{what} takes cells of one dimension")
    return dims.pop() if dims else 3


def _order(field):
    space = field.space
    if isinstance(space, (H1, L2)) and not space.varies:
        return space.order
    return 1


def default_level(data, *fields):
    """The lattice steps a cell is cut into by default: the largest order of the
    geometry and ``fields`` (1 when every one is linear)."""
    return max([_order(data.geometry)] + [_order(f) for f in fields])


def _refined(data, flags, level, fields):
    """The cells ``flags`` marks, cut into ``level`` steps along each axis: an
    unstructured linear dataset of their sub-cells, its geometry and ``fields`` (a
    dict of names to fields) evaluated at the lattice points (``H1`` and ``L2`` fields of any order;
    an ``L2`` field is evaluated from the first refined cell at a shared point,
    so its jumps are not kept), cell fields taken from each sub-cell's parent.
    Returns ``(refined, parent)``, ``parent`` each sub-cell's cell."""
    dimension = _tensor_cells(data, "subdivision")
    topology = data.topology
    n = data.num_cells
    slots = tack.empty((n,), tack.i64)
    count = exclusive_scan(flags, slots, n) if n else 0
    slot_of = tack.empty((n,), tack.i64)
    chosen = tack.empty((max(count, 1),), tack.i64)
    if n:
        _place_chosen(flags, slots, slot_of, chosen)
    per_cell = (level + 1) ** dimension
    subs = level ** dimension
    sub = 2 ** dimension
    total = count * per_cell
    keys = tack.empty((total,), tack.i64)
    if count:
        if level == 1:
            corner_of = tack.array(np.asarray(bases.dof_tables(
                next(g.shape for g in topology.groups() if g.count), 1)[1], np.int32))
            for group in topology.groups():
                if group.count:
                    _corner_keys(group.view(), slot_of, corner_of, keys)
        else:
            space = H1(data, order=level, nodes="equispaced")
            lattice = Field(space, ConstantArray(0.0, space.size, dtype=data.dtype))
            for group in data.launch_groups("cells", [lattice]):
                if group.count:
                    node_of = tack.array(bases.view_order(group.shape, level))
                    _lattice_keys(data.domain_view("cells", group), lattice.view(group),
                                  slot_of, per_cell, node_of, keys)
    order = argsort(keys) if total else tack.empty((0,), tack.i64)
    first = tack.empty((total,), tack.i32)
    numbers = tack.empty((total,), tack.i64)
    rank = tack.empty((total,), tack.i64)
    points = 0
    if total:
        _runs(keys, order, first)
        starts = tack.empty((total,), tack.i32)
        _starts(order, first, starts)
        points = exclusive_scan(starts, numbers, total)
        _ranks(order, starts, numbers, rank)
    geometry = tack.empty((points,), tack.vec(3, data.dtype))
    idt = ids.at_least(data.id_dtype, max(points, count * subs * sub))
    conn = tack.empty((count * subs * sub,), idt)
    parent = tack.empty((count * subs,), data.id_dtype)
    steps = tack.empty((level + 1,), data.dtype)
    tack.copyto(steps, (np.arange(level + 1) / level).astype(data.dtype.numpy_dtype))
    corner_at = {}
    for group in topology.groups():
        if group.count:
            kind, entity, _, _, _ = bases.dof_tables(group.shape, level)
            corner_at[group.root] = tack.array(np.where(kind == bases.CORNER, entity, -1)
                                               .astype(np.int32))
    if count:
        for group in data.launch_groups("cells", []):
            if group.count:
                cells = data.domain_view("cells", group)
                _positions(cells, slot_of, per_cell, level, steps, corner_at[group.root], first,
                           rank, geometry)
                _sub_cells(cells, slot_of, per_cell, level, rank, sub, conn, parent)
    shape = {3: shapes.HEXAHEDRON, 2: shapes.QUAD, 1: shapes.LINE}[dimension]
    types, offsets = uniform_cells(int(shape), count * subs, idt)
    linear = UnstructuredTopology(types, offsets, conn, num_points=points, id_dtype=idt)
    point_fields = {}
    for name, field in fields.items():
        if not isinstance(field.space, (H1, L2)) or field.space.varies:
            continue
        out = tack.empty((points,), _element(field.values))
        if count:
            for group in data.launch_groups("cells", [field]):
                if group.count:
                    _values(data.domain_view("cells", group), field.view(group), slot_of,
                            per_cell, level, steps, corner_at[group.root], first, rank, out)
        point_fields[name] = out
    from tack.dataset.dataset import DataSet

    refined = DataSet(linear, geometry, **data._frame())
    if data.ghosts is not None or data._ghosts_lost:
        refined._ghosts_lost = (data._ghosts_lost or "subdivided cells' lattice points have "
                                "no owners among the pieces yet")
    for name, values in point_fields.items():
        refined.fields[name] = Field(H1(refined), values, fields[name].unit)
    for name, field in fields.items():
        if isinstance(field.space, Constant) or (isinstance(field.space, Values)
                                                 and field.space.on == "cells"):
            refined.fields[name] = _taken(field, parent, refined)
    return refined, parent


def _element(values):
    """``values``' element type: a scalar's, or a vector's."""
    width = arrays.width_of(values)
    dtype = arrays.dtype_of(values)
    return dtype if not width or width == 1 else tack.vec(width, dtype)


@tack.kernel
def _take_values(source, parent, out):
    for i in range(out.shape[0]):
        out[i] = source[parent[i]]


def _taken(field, parent, refined):
    """A cell field of the parents, one value a sub-cell."""
    source = arrays.materialize(field.values)
    out = tack.empty((size_of(parent),), _element(source))
    if size_of(parent):
        _take_values(source, parent, out)
    space = Constant(refined) if isinstance(field.space, Constant) else Values(refined, "cells")
    return Field(space, out, field.unit)


@on_inputs_device
def subdivide(data, level=None, fields=None):
    """``data``'s cells cut into ``level`` steps along each parametric axis -- linear
    hexahedra, quads or lines on each cell's lattice -- and its fields evaluated at
    the lattice points: a curved or high-order dataset as a linear one, as VTK's
    ``vtkCellGridToUnstructuredGrid`` gives one (for export, rendering, or a filter
    that takes only linear cells). ``level`` is by default the largest order of the
    geometry and fields. A lattice point shared by cells is one point, its value
    computed once. ``H1`` and ``L2`` fields become point data (an ``L2`` field's
    value at a shared point from one of its cells); cell fields go to every
    sub-cell of their cell. Segments, quads, pixels, hexahedra and voxels."""
    names = list(data.fields) if fields is None else list(fields)
    level = default_level(data, *(data.fields[name] for name in names)) if level is None \
        else int(level)
    if level < 1:
        raise ValueError(f"subdivide's level is at least 1, not {level}")
    flags = tack.empty((data.num_cells,), tack.i32)
    if data.num_cells:
        for_each(_all_cells, data, "cells", flags)
    refined, _ = _refined(data, flags, level, {name: data.fields[name] for name in names})
    return refined


def level_for(data, field, flags, tolerance):
    """The fewest lattice steps that cut ``field``'s flagged cells within ``tolerance``
    (in the field's values): each cell's Bernstein coefficients' largest distance
    from the corners' multilinear interpolation, ``D``, bounds what linearizing it
    loses, and a lattice of ``s`` steps divides that by about ``s^2`` (exactly for a
    quadratic), so ``s = ceil(sqrt(D / tolerance))``, at most ``MAX_LEVEL``."""
    tolerance = float(tolerance)
    if not tolerance > 0:
        raise ValueError(f"tolerance is a positive number, not {tolerance}")
    out = tack.empty((data.num_cells,), arrays.dtype_of(field.values))
    for group in data.launch_groups("cells", [field]):
        if group.count:
            _flatness(data.domain_view("cells", group), field.view(group), flags, out)
    worst = float(out.max()) if data.num_cells else 0.0
    return int(min(MAX_LEVEL, max(1, np.ceil(np.sqrt(worst / tolerance)))))


def subdivides(data, field):
    """Whether cutting ``field`` on ``data`` takes subdivision: a curved geometry or a
    field above order 1, on tensor cells (segments, quads, pixels, hexahedra,
    voxels)."""
    nonlinear = _order(data.geometry) > 1 or (isinstance(field.space, (H1, L2))
                                              and not field.space.varies
                                              and (field.space.order > 1
                                                   or getattr(field.space, "tensor", False)))
    tensor = all(bases.tensor_shape(g.shape) for g in data.topology.groups() if g.count)
    return nonlinear and tensor


def subdivided_contour(data, field, value, levels, nlevels, subdivision, cut, tolerance=None):
    """``cut`` (contour with its other arguments bound) of the cells whose ``field``
    may hold one of ``levels``, cut into ``subdivision`` lattice steps (by default
    the largest order of the field and geometry, or with ``tolerance`` the fewest
    whose estimated error is within it): only those cells are refined, and every
    field carried evaluated at their lattice points."""
    if subdivision is not None and tolerance is not None:
        raise ValueError("give subdivision or tolerance, not both")
    flags = tack.empty((data.num_cells,), tack.i32)
    for group in data.launch_groups("cells", [field]):
        if group.count:
            _straddling(data.domain_view("cells", group), field.view(group), levels, nlevels,
                        flags)
    if tolerance is not None:
        level = level_for(data, field, flags, tolerance)
    else:
        level = default_level(data, field) if subdivision is None else int(subdivision)
    if level < 1:
        raise ValueError(f"subdivision is at least 1, not {level}")
    name = next((n for n, f in data.fields.items() if f is field), "__cut__")
    carried = dict(data.fields)
    carried[name] = field
    refined, _ = _refined(data, flags, level, carried)
    out = cut(refined, name)
    if name == "__cut__":
        out.fields.pop(name, None)
    return out
