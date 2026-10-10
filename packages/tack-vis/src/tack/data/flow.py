"""Particle advection and streamlines through a steady vector field, as Viskores'.

``advect(data, field, seeds, step_size, steps)`` moves each seed through the
velocity ``field`` -- point data, interpolated in the cell holding the
particle, or cell data, constant in it -- by fixed steps of RK4 (or Euler),
as Viskores' ParticleAdvection does, and returns the particles where they
stopped. ``streamlines`` returns the paths, as Viskores' Streamline does.

The rules are Viskores' (``filter/flow/worklet``):

- a step is taken whole when every stage of it lies in the mesh;
- a step that leaves the mesh is bisected to the boundary (until the
  remaining step is shorter than ten machine epsilons), and the particle is
  then pushed out with one Euler step of the length still left, so it ends
  just outside; its status gets ``SPATIAL_BOUNDS``;
- a particle stops after ``steps`` steps (``TERMINATE``), outside the mesh,
  or when its velocity's squared length falls to machine epsilon
  (``ZERO_VELOCITY``); a seed outside the mesh does not move and loses
  ``SUCCESS``.

Every particle advances in one thread, its whole path in one launch: the
point location of each stage runs from the particle (``CellLocator.cell_bins``)
and reaches its cell by id through the topology's flat arrays, with each
linear shape's own inversion and interpolation chosen by the cell's type. The
kernel is a loop with one point location per iteration, whatever it is for
-- a stage, a bisection, the push out -- so the shapes' code appears in it
once.
"""

import numpy as np

import tack
from tack.data import arrays, shapes
from tack.data.dataset import DataSet, Field
from tack.data.locator import _TOLERANCE, CellLocator, _bin, _vertices
from tack.data.spaces import H1, Constant, Values

__all__ = ["SPATIAL_BOUNDS", "SUCCESS", "TERMINATE", "TOOK_ANY_STEPS", "ZERO_VELOCITY",
           "advect", "streamlines"]

# A particle's status bits, Viskores' ParticleStatus.
SUCCESS = tack.constant(1, tack.i32)
TERMINATE = tack.constant(2, tack.i32)
SPATIAL_BOUNDS = tack.constant(4, tack.i32)
TOOK_ANY_STEPS = tack.constant(16, tack.i32)
ZERO_VELOCITY = tack.constant(64, tack.i32)

# What the next point location is for.
_STEP = tack.constant(0, tack.i32)       # a stage of a whole step
_START = tack.constant(1, tack.i32)      # the particle itself, starting the bisection
_CHECK = tack.constant(2, tack.i32)      # a stage of a bisection's trial step
_TRIAL = tack.constant(3, tack.i32)      # where the trial step ends
_LAST = tack.constant(4, tack.i32)       # the last point inside, for the push out
_OUT = tack.constant(5, tack.i32)        # where the push out ends


# ── Cells by id ─────────────────────────────────────────────────────

@tack.data_oriented
class _FlatCells:
    """An unstructured topology's cells by id: type and points from its arrays."""

    def __init__(self, types, offsets, connectivity):
        self.types = types
        self.offsets = offsets
        self.connectivity = connectivity

    @tack.func
    def cell_type(self, e):
        return tack.i32(self.types[e])

    @tack.func
    def num_points(self, e):
        return self.offsets[e + 1] - self.offsets[e]

    @tack.func
    def point_id(self, e, j):
        return self.connectivity[self.offsets[e] + j]


@tack.data_oriented
class _GridCells:
    """A structured grid's cells by id: hexahedra (quads in 2D), x fastest, corners
    in VTK's order."""

    def __init__(self, point_dims):
        dims = tuple(point_dims) + (1,) * (3 - len(point_dims))
        self.nx, self.ny = dims[0], dims[1]
        self.cx, self.cy = dims[0] - 1, max(dims[1] - 1, 1)
        self.dimension = len(point_dims)

    @tack.func
    def cell_type(self, e):
        return shapes.HEXAHEDRON if self.dimension == 3 else shapes.QUAD

    @tack.func
    def num_points(self, e):
        return 8 if self.dimension == 3 else 4

    @tack.func
    def point_id(self, e, j):
        i = e % self.cx
        rest = e // self.cx
        k = rest // self.cy
        return (i + ((j ^ (j >> 1)) & 1)) + ((rest - k * self.cy + ((j >> 1) & 1))
                                             + (k + (j >> 2)) * self.ny) * self.nx


# ── The kernel ──────────────────────────────────────────────────────

@tack.kernel
def _advect(mesh, kinds, tetra, hexa, wedge, pyramid, voxel, triangle, quad, pixel,
            positions, velocity, on_cells, lo, hi, bin_offsets, bin_cells,
            ox, oy, oz, ix, iy, iz, nx, ny, nz, slack, tolerance, ncells,
            h, max_steps, rk4, eps, xs, times, steps, status, trace, trace_times, lengths,
            capacity, record):
    for q in range(xs.shape[0]):
        x = xs[q]
        t = times[q]
        n = steps[q]
        st = status[q]
        took = 0
        length = 0
        if record == 1:
            trace[q * capacity] = x
            trace_times[q * capacity] = t
            length = 1
        pts = tack.local_array_like(lo, 24)
        k1 = x - x
        k2 = k1
        k3 = k1
        vel = k1
        curr = x
        out = x
        mode = _STEP
        stage = 0
        sub = h
        r0 = h - h
        r1 = h
        div = 1.0
        running = 1
        while running == 1:
            # Where to locate the particle next.
            p = x
            if mode == _STEP or mode == _CHECK:
                if stage == 1:
                    p = x + (0.5 * sub) * k1
                elif stage == 2:
                    p = x + (0.5 * sub) * k2
                elif stage == 3:
                    p = x + sub * k3
            elif mode == _TRIAL:
                p = x + sub * vel
            elif mode == _START or mode == _LAST:
                p = curr
            elif mode == _OUT:
                p = out

            # Its cell -- the smallest id among those holding it -- and the
            # velocity there.
            best = ncells
            best_type = 0
            best_pc = p - p
            bi = _bin(p[0], ox, ix, nx, slack)
            bj = _bin(p[1], oy, iy, ny, slack)
            bk = _bin(p[2], oz, iz, nz, slack)
            if bi >= 0 and bj >= 0 and bk >= 0:
                b = bi + nx * (bj + ny * bk)
                for m in range(bin_offsets[b], bin_offsets[b + 1]):
                    e = bin_cells[m]
                    a = lo[e]
                    c = hi[e]
                    s = tolerance * (c - a).norm()
                    if (e < best and p[0] >= a[0] - s and p[0] <= c[0] + s
                            and p[1] >= a[1] - s and p[1] <= c[1] + s
                            and p[2] >= a[2] - s and p[2] <= c[2] + s):
                        kind = mesh.cell_type(e)
                        for j in range(mesh.num_points(e)):
                            y = positions[mesh.point_id(e, j)]
                            pts[3 * j] = y[0]
                            pts[3 * j + 1] = y[1]
                            pts[3 * j + 2] = y[2]
                        pc = p - p
                        converged = 0
                        inside = 0
                        near = p - p
                        if (kinds.PRESENT & 1) != 0 and kind == shapes.TETRA:
                            pc, converged = tetra.world_to_parametric(pts, p)
                            inside = tetra.is_inside(pc, 0.0)
                            near = tetra.nearest_point(pts, p, pc)
                        elif (kinds.PRESENT & 2) != 0 and kind == shapes.HEXAHEDRON:
                            pc, converged = hexa.world_to_parametric(pts, p)
                            inside = hexa.is_inside(pc, 0.0)
                            near = hexa.nearest_point(pts, p, pc)
                        elif (kinds.PRESENT & 4) != 0 and kind == shapes.WEDGE:
                            pc, converged = wedge.world_to_parametric(pts, p)
                            inside = wedge.is_inside(pc, 0.0)
                            near = wedge.nearest_point(pts, p, pc)
                        elif (kinds.PRESENT & 8) != 0 and kind == shapes.PYRAMID:
                            pc, converged = pyramid.world_to_parametric(pts, p)
                            inside = pyramid.is_inside(pc, 0.0)
                            near = pyramid.nearest_point(pts, p, pc)
                        elif (kinds.PRESENT & 16) != 0 and kind == shapes.VOXEL:
                            pc, converged = voxel.world_to_parametric(pts, p)
                            inside = voxel.is_inside(pc, 0.0)
                            near = voxel.nearest_point(pts, p, pc)
                        elif (kinds.PRESENT & 32) != 0 and kind == shapes.TRIANGLE:
                            pc, converged = triangle.world_to_parametric(pts, p)
                            near = triangle.nearest_point(pts, p, pc)
                        elif (kinds.PRESENT & 64) != 0 and kind == shapes.QUAD:
                            pc, converged = quad.world_to_parametric(pts, p)
                            near = quad.nearest_point(pts, p, pc)
                        elif (kinds.PRESENT & 128) != 0 and kind == shapes.PIXEL:
                            pc, converged = pixel.world_to_parametric(pts, p)
                            near = pixel.nearest_point(pts, p, pc)
                        # As the locator: inside a solid, or within the tolerance of
                        # the cell's nearest point.
                        d = near - p
                        if converged == 1 and (inside == 1 or d.dot(d) <= s * s):
                            best = e
                            best_type = kind
                            best_pc = pc
            found = 1 if best < ncells else 0
            v = p - p
            if found == 1:
                if on_cells == 1:
                    v = velocity[best]
                else:
                    count = mesh.num_points(best)
                    for j in range(count):
                        w = 0.0
                        if (kinds.PRESENT & 1) != 0 and best_type == shapes.TETRA:
                            w = tetra.shape_function(j, best_pc)
                        elif (kinds.PRESENT & 2) != 0 and best_type == shapes.HEXAHEDRON:
                            w = hexa.shape_function(j, best_pc)
                        elif (kinds.PRESENT & 4) != 0 and best_type == shapes.WEDGE:
                            w = wedge.shape_function(j, best_pc)
                        elif (kinds.PRESENT & 8) != 0 and best_type == shapes.PYRAMID:
                            w = pyramid.shape_function(j, best_pc)
                        elif (kinds.PRESENT & 16) != 0 and best_type == shapes.VOXEL:
                            w = voxel.shape_function(j, best_pc)
                        elif (kinds.PRESENT & 32) != 0 and best_type == shapes.TRIANGLE:
                            w = triangle.shape_function(j, best_pc)
                        elif (kinds.PRESENT & 64) != 0 and best_type == shapes.QUAD:
                            w = quad.shape_function(j, best_pc)
                        elif (kinds.PRESENT & 128) != 0 and best_type == shapes.PIXEL:
                            w = pixel.shape_function(j, best_pc)
                        v += w * velocity[mesh.point_id(best, j)]

            # What it means for the particle.
            bisect = 0
            if mode == _STEP or mode == _CHECK:
                if found == 0:
                    if mode == _STEP:
                        # Viskores' SmallStep: from the particle, bisect the step.
                        mode = _START
                        curr = x
                        r0 = h - h
                        r1 = h
                        div = 1.0
                    else:
                        r1 = sub
                        bisect = 1
                elif rk4 == 1 and stage < 3:
                    if stage == 0:
                        k1 = v
                    elif stage == 1:
                        k2 = v
                    else:
                        k3 = v
                    stage += 1
                else:
                    vel = v
                    if rk4 == 1:
                        vel = (k1 + 2.0 * k2 + 2.0 * k3 + v) / 6.0
                    else:
                        k1 = v
                    stage = 0
                    if mode == _STEP:
                        x = x + h * vel
                        t += h
                        n += 1
                        took = 1
                        if record == 1:
                            trace[q * capacity + length] = x
                            trace_times[q * capacity + length] = t
                            length += 1
                        if vel.dot(vel) <= eps:
                            st = st | ZERO_VELOCITY | TERMINATE
                            running = 0
                        elif n == max_steps:
                            st = st | TERMINATE
                            running = 0
                    else:
                        mode = _TRIAL
            elif mode == _START:
                if found == 0:
                    # The seed itself is outside: it cannot move.
                    st = (st & ~SUCCESS) | SPATIAL_BOUNDS
                    running = 0
                else:
                    bisect = 1
            elif mode == _TRIAL:
                if found == 1:
                    curr = x + sub * vel
                    r0 = sub
                else:
                    r1 = sub
                bisect = 1
            elif mode == _LAST:
                vel = v
                out = curr + r1 * vel
                t += r1
                mode = _OUT
            else:
                # Where the push out ends: the step is taken, in or out.
                x = out
                n += 1
                took = 1
                if record == 1:
                    trace[q * capacity + length] = x
                    trace_times[q * capacity + length] = t
                    length += 1
                if found == 0:
                    st = st | SPATIAL_BOUNDS
                if vel.dot(vel) <= eps:
                    st = st | ZERO_VELOCITY | TERMINATE
                if n == max_steps:
                    st = st | TERMINATE
                if (st & (SPATIAL_BOUNDS | ZERO_VELOCITY | TERMINATE)) != 0:
                    running = 0
                else:
                    mode = _STEP
            if bisect == 1:
                # The next trial step, or the push out once the bracket is small.
                if r1 - r0 > 10.0 * eps:
                    div = div * 2.0
                    sub = r0 + h / div
                    mode = _CHECK
                    stage = 0
                else:
                    mode = _LAST
        xs[q] = x
        times[q] = t
        steps[q] = n
        status[q] = st | (TOOK_ANY_STEPS if took == 1 else 0)
        lengths[q] = length


# ── Python side ─────────────────────────────────────────────────────

_kinds_classes = {}


def _kinds(present):
    """A template whose class constant ``PRESENT`` has a bit for each of ``_SHAPES``
    the mesh has, so the kernel compiles only those shapes' code."""
    if present not in _kinds_classes:
        _kinds_classes[present] = tack.data_oriented(
            type(f"_Kinds{present}", (), {"PRESENT": present}))
    return _kinds_classes[present]()

_SHAPES = (shapes.Tetra, shapes.Hexahedron, shapes.Wedge, shapes.Pyramid, shapes.Voxel,
           shapes.Triangle, shapes.Quad, shapes.Pixel)


def _velocity(data, field):
    """The velocity's values and whether they are on the cells."""
    from tack.data.filters import _field

    field = _field(data, field)
    space = field.space
    if (isinstance(space, H1) and space.order == 1) or space is Values(data, "points"):
        on_cells = 0
    elif isinstance(space, Constant) or space is Values(data, "cells"):
        on_cells = 1
    else:
        raise TypeError(f"advection takes point data (H1 order 1, values on points) or "
                        f"cell data, not {space!r}")
    if arrays.width_of(field.values) != 3:
        raise TypeError("advection needs a field of 3-vectors")
    return _as_vectors(arrays.materialize(field.values), data.dtype), on_cells


def _as_vectors(values, dtype):
    if values.dtype == dtype:
        return values
    out = tack.Vector.field(3, dtype, shape=(arrays.size_of(values),))
    if arrays.size_of(values):
        out.from_numpy(values.to_numpy(vectors=True).astype(dtype.numpy_dtype))
    return out


def _mesh(data):
    """The cells by id, and the bits of the shapes present."""
    from tack.data.topology import StructuredTopology, UnstructuredTopology

    t = data.topology
    present = 0
    for group in t.groups() if hasattr(t, "groups") else ():
        if group.count and group.shape in _SHAPES:
            present |= 1 << _SHAPES.index(group.shape)
    if isinstance(t, UnstructuredTopology):
        return _FlatCells(t.types, t.offsets, t.connectivity), present
    if isinstance(t, StructuredTopology) and len(t.point_dims) >= 2:
        return _GridCells(t.point_dims), present
    raise NotImplementedError("advection takes unstructured and 2D or 3D structured "
                              "topologies; not polyhedra, whose cells have no parametric "
                              "coordinates, nor lines")


def _run(data, field, seeds, step_size, steps, integrator, locator, record):
    if int(steps) < 1:
        raise ValueError(f"steps must be at least 1, not {steps!r}")
    if integrator not in ("rk4", "euler"):
        raise ValueError(f"integrator is 'rk4' or 'euler', not {integrator!r}")
    geometry = data.geometry
    if not isinstance(geometry.space, H1) or geometry.space.order != 1:
        raise TypeError("advection needs an order-1 H1 geometry: one position per point")
    mesh, present = _mesh(data)
    velocity, on_cells = _velocity(data, field)
    locator = locator or CellLocator(data)
    bin_offsets, bin_cells = locator.cell_bins()
    dtype = data.dtype
    host = np.ascontiguousarray(np.asarray(
        seeds.to_numpy(vectors=True) if hasattr(seeds, "to_numpy") else seeds,
        dtype.numpy_dtype).reshape(-1, 3))
    m = len(host)
    xs = tack.Vector.field(3, dtype, shape=(m,))
    times = tack.field(dtype, shape=(m,))
    taken = tack.field(tack.i32, shape=(m,))
    status = tack.field(tack.i32, shape=(m,))
    lengths = tack.field(tack.i32, shape=(m,))
    capacity = int(steps) + 1 if record else 1
    trace = tack.Vector.field(3, dtype, shape=(m * capacity,))
    trace_times = tack.field(dtype, shape=(m * capacity,))
    if m:
        xs.from_numpy(host)
        times.from_numpy(np.zeros(m, dtype.numpy_dtype))
        taken.from_numpy(np.zeros(m, np.int32))
        status.from_numpy(np.full(m, SUCCESS, np.int32))
        positions = _as_vectors(arrays.materialize(geometry.values), dtype)
        eps = float(np.finfo(dtype.numpy_dtype).eps)
        _advect(mesh, _kinds(present), *(shape() for shape in _SHAPES), positions, velocity, on_cells,
                locator.lo, locator.hi, bin_offsets, bin_cells, *locator._grid(),
                float(locator.slack), _TOLERANCE, data.num_cells, float(step_size), int(steps),
                1 if integrator == "rk4" else 0, eps, xs, times, taken, status, trace,
                trace_times, lengths, capacity, 1 if record else 0)
    return xs, times, taken, status, (trace, trace_times), lengths, capacity


def advect(data, field, seeds, step_size, steps, integrator="rk4", locator=None):
    """Seeds moved through the velocity ``field`` (a 3-vector field or its name: point
    data, or cell data) by up to ``steps`` fixed steps of ``step_size``, as
    Viskores' ParticleAdvection: a dataset of the particles where they stopped,
    one vertex cell each, with ``steps`` (taken), ``time`` and ``status`` (the
    bits ``SUCCESS``, ``TERMINATE``, ``SPATIAL_BOUNDS``, ``TOOK_ANY_STEPS``,
    ``ZERO_VELOCITY``). ``seeds`` is an ``(n, 3)`` array or a field of
    3-vectors; ``integrator`` ``"rk4"`` or ``"euler"``; ``locator`` reuses a
    ``CellLocator`` of ``data``."""
    xs, times, taken, status, _, _, _ = _run(data, field, seeds, step_size, steps,
                                             integrator, locator, record=False)
    out = _vertices(xs.to_numpy(vectors=True) if xs.size else np.zeros((0, 3)))
    points = Values(out, "points")
    for name, values in (("steps", taken), ("time", times), ("status", status)):
        out.fields[name] = Field(points, values)
    return out


@tack.kernel
def _segments(trace, trace_times, lengths, starts, capacity, points, times, types, offsets,
              connectivity, seed_of, segment_starts):
    # Each path's points in order, and a line from each to the next.
    for q in range(lengths.shape[0]):
        first = starts[q]
        count = lengths[q]
        lines = segment_starts[q]
        for k in range(count):
            points[first + k] = trace[q * capacity + k]
            times[first + k] = trace_times[q * capacity + k]
        for k in range(count - 1):
            types[lines + k] = tack.u8(shapes.LINE)
            offsets[lines + k] = 2 * (lines + k)
            connectivity[2 * (lines + k)] = first + k
            connectivity[2 * (lines + k) + 1] = first + k + 1
            seed_of[lines + k] = q
        if q == lengths.shape[0] - 1:
            offsets[lines + max(count - 1, 0)] = 2 * (lines + max(count - 1, 0))


def streamlines(data, field, seeds, step_size, steps, integrator="rk4", locator=None):
    """The paths of ``advect``'s particles, as Viskores' Streamline: each seed's
    positions after every step, the seed first, as consecutive line cells (one
    per step; a polyline per seed in Viskores, until Tack has a line topology),
    with ``seed`` (which seed's path) on the cells and ``time`` on the points.
    A seed that takes no step has a point and no line."""
    from tack.algorithms.scan import exclusive_scan
    from tack.data.topology import UnstructuredTopology

    _, _, _, _, (trace, trace_times), lengths, capacity = _run(
        data, field, seeds, step_size, steps, integrator, locator, record=True)
    m = int(lengths.shape[0])
    host_lengths = lengths.to_numpy() if m else np.zeros(0, np.int32)
    total = int(host_lengths.sum())
    nlines = int(np.maximum(host_lengths - 1, 0).sum())
    idt = tack.i64 if max(total, 2 * nlines) > 2**31 - 1 else tack.i32
    starts = tack.field(idt, shape=(m,))
    segment_starts = tack.field(idt, shape=(m,))
    if m:
        exclusive_scan(lengths, starts, m)
        segment_starts.from_numpy(np.concatenate(
            [[0], np.cumsum(np.maximum(host_lengths - 1, 0))[:-1]]).astype(idt.numpy_dtype))
    points = tack.Vector.field(3, data.dtype, shape=(total,))
    point_times = tack.field(data.dtype, shape=(total,))
    types = tack.field(tack.u8, shape=(nlines,))
    offsets = tack.field(idt, shape=(nlines + 1,))
    connectivity = tack.field(idt, shape=(2 * nlines,))
    seed_of = tack.field(tack.i32, shape=(nlines,))
    offsets.fill(0)
    if m:
        _segments(trace, trace_times, lengths, starts, capacity, points, point_times, types,
                  offsets, connectivity, seed_of, segment_starts)
    topology = UnstructuredTopology(types, offsets, connectivity, num_points=total)
    out = DataSet(topology, points)
    out.fields["seed"] = Field(Values(out, "cells"), seed_of)
    out.fields["time"] = Field(Values(out, "points"), point_times)
    return out
