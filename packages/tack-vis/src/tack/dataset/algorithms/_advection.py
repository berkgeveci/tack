"""The advection kernel ``advect`` and ``streamlines`` share, by Viskores' rules:
RK4 or Euler steps; a step leaving the mesh bisected to the boundary and the
particle pushed out; a stop after its steps, outside, or at zero velocity.
Each particle's whole path is one thread of one launch, its stages located
from the particle's side (``CellLocator.cell_bins``), its cells reached by id
through the topology's flat arrays, each shape's code compiled in only when
the mesh has that shape. Polyhedra are located in their sub-tetrahedra, by the
code their views use (``views._PolyhedralLocation``)."""

import enum

import numpy as np

import tack
from tack import arrays
from tack.dataset import shapes, views
from tack.dataset.algorithms._common import _field
from tack.dataset.locator import _TOLERANCE, CellLocator, _bin
from tack.dataset.spaces import H1, Constant, Values
from tack.lang.array import Array

# A particle's status bits, Viskores' ParticleStatus: the kernel's constants,
# and the flag users compare against.
SUCCESS = tack.constant(1, tack.i32)
TERMINATE = tack.constant(2, tack.i32)
SPATIAL_BOUNDS = tack.constant(4, tack.i32)
TOOK_ANY_STEPS = tack.constant(16, tack.i32)
ZERO_VELOCITY = tack.constant(64, tack.i32)


class ParticleStatus(enum.IntFlag):
    """The bits of ``advect``'s ``status`` field, Viskores' ParticleStatus:
    ``SUCCESS`` (cleared when the seed was outside the mesh), ``TERMINATE`` (it
    stopped after its steps or at zero velocity), ``SPATIAL_BOUNDS`` (it left
    the mesh), ``TOOK_ANY_STEPS``, ``ZERO_VELOCITY``."""

    SUCCESS = 1
    TERMINATE = 2
    SPATIAL_BOUNDS = 4
    TOOK_ANY_STEPS = 16
    ZERO_VELOCITY = 64


# What the next point location is for.
_STEP = tack.constant(0, tack.i32)       # a stage of a whole step


_START = tack.constant(1, tack.i32)      # the particle itself, starting the bisection


_CHECK = tack.constant(2, tack.i32)      # a stage of a bisection's trial step


_TRIAL = tack.constant(3, tack.i32)      # where the trial step ends


_LAST = tack.constant(4, tack.i32)       # the last point inside, for the push out


_OUT = tack.constant(5, tack.i32)        # where the push out ends


class _NoPolyhedra:
    """For meshes of shapes: the polyhedral methods the kernel names, never called
    (their branch is compiled out)."""

    @tack.func
    def locate(self, e, p, slack):
        return -1, p - p

    @tack.func
    def velocity_at(self, e, t, pc):
        return pc - pc


@tack.device_class
class _PolyhedralMesh(views._PolyhedralLocation, views._PolyhedralWalk, views._PolyhedralCells):
    """A polyhedral topology's cells by id, located in their sub-tetrahedra by the
    same code as their views (``views._PolyhedralLocation``), with the positions
    and the point velocities it reads."""

    def __init__(self, topology, positions, velocity):
        point_offsets, point_ids = topology.cell_points()
        super().__init__(topology.cell_offsets, topology.cell_faces, topology.cell_face_sides,
                         topology.face_offsets, topology.face_points, point_offsets,
                         point_ids, topology.num_cells)
        self.positions = positions
        self.velocity = velocity

    @tack.func
    def point_position(self, p):
        return self.positions[p]

    @tack.func
    def cell_type(self, e):
        return shapes.POLYHEDRON

    @tack.func
    def velocity_at(self, e, t, pc):
        """The point velocity at ``pc`` in sub-tetrahedron ``t``: linear there, the
        cell and face points taking the means of their points' velocities."""
        k, j = self.tet_face(e, t)
        n = self.side_num_points(e, k)
        nxt = j + 1 if j + 1 < n else 0
        count = self.num_points(e)
        v0 = self.velocity[self.point_id(e, 0)]
        for i in range(1, count):
            v0 += self.velocity[self.point_id(e, i)]
        v0 = v0 / count
        vf = self.velocity[self.side_point_id(e, k, 0)]
        for i in range(1, n):
            vf += self.velocity[self.side_point_id(e, k, i)]
        vf = vf / n
        return (v0 + pc[0] * (vf - v0) + pc[1] * (self.velocity[self.side_point_id(e, k, j)] - v0)
                + pc[2] * (self.velocity[self.side_point_id(e, k, nxt)] - v0))


@tack.device_class
class _FlatCells(_NoPolyhedra):
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


@tack.device_class
class _GridCells(_NoPolyhedra):
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

            # Its cell, ranked as the locator ranks them -- holding the point
            # ahead of coming within the tolerance, then the smaller id -- and
            # the velocity there.
            best = ncells
            best_rank = 2 * ncells
            best_type = 0
            best_pc = p - p
            best_tet = -1
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
                    if (e < best_rank and p[0] >= a[0] - s and p[0] <= c[0] + s
                            and p[1] >= a[1] - s and p[1] <= c[1] + s
                            and p[2] >= a[2] - s and p[2] <= c[2] + s):
                        kind = mesh.cell_type(e)
                        if (kinds.PRESENT & 256) != 0 and kind == shapes.POLYHEDRON:
                            # No parametric coordinates: a sub-tetrahedron and the
                            # point's place in it, within the cell's tolerance.
                            tet, tpc = mesh.locate(e, p, s)
                            if tet >= 0:
                                best = e
                                best_type = kind
                                best_tet = tet
                                best_pc = tpc
                            continue
                        for j in range(mesh.num_points(e)):
                            y = positions[mesh.point_id(e, j)]
                            pts[3 * j] = y[0]
                            pts[3 * j + 1] = y[1]
                            pts[3 * j + 2] = y[2]
                        pc = p - p
                        converged = 0
                        inside = 0
                        near = p - p
                        surface = 0
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
                            inside = triangle.is_inside(pc, 0.0)
                            surface = 1
                            near = triangle.nearest_point(pts, p, pc)
                        elif (kinds.PRESENT & 64) != 0 and kind == shapes.QUAD:
                            pc, converged = quad.world_to_parametric(pts, p)
                            inside = quad.is_inside(pc, 0.0)
                            surface = 1
                            near = quad.nearest_point(pts, p, pc)
                        elif (kinds.PRESENT & 128) != 0 and kind == shapes.PIXEL:
                            pc, converged = pixel.world_to_parametric(pts, p)
                            inside = pixel.is_inside(pc, 0.0)
                            surface = 1
                            near = pixel.nearest_point(pts, p, pc)
                        # As the locator: inside a solid, or within the tolerance of
                        # the cell's nearest point; holding it (a surface: over it)
                        # ranks first.
                        d = near - p
                        if converged == 1 and ((inside == 1 and surface == 0)
                                               or d.dot(d) <= s * s):
                            rank = e if inside == 1 else e + ncells
                            if rank < best_rank:
                                best_rank = rank
                                best = e
                                best_type = kind
                                best_pc = pc
            found = 1 if best < ncells else 0
            v = p - p
            if found == 1:
                if on_cells == 1:
                    v = velocity[best]
                elif (kinds.PRESENT & 256) != 0 and best_type == shapes.POLYHEDRON:
                    v = mesh.velocity_at(best, best_tet, best_pc)
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


_kinds_classes = {}


def _kinds(present):
    """A template whose class constant ``PRESENT`` has a bit for each of ``_SHAPES``
    the mesh has, so the kernel compiles only those shapes' code."""
    if present not in _kinds_classes:
        _kinds_classes[present] = tack.device_class(
            type(f"_Kinds{present}", (), {"PRESENT": present}))
    return _kinds_classes[present]()


_SHAPES = (shapes.Tetra, shapes.Hexahedron, shapes.Wedge, shapes.Pyramid, shapes.Voxel,
           shapes.Triangle, shapes.Quad, shapes.Pixel)


def _velocity(data, field):
    """The velocity's values and whether they are on the cells."""

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
    if values.scalar_dtype == dtype:
        return values
    out = tack.empty((arrays.size_of(values),), tack.vec(3, dtype))
    if arrays.size_of(values):
        tack.copyto(out, values.numpy().astype(dtype.numpy_dtype))
    return out


# The kinds bit for polyhedra, past the shapes'.
_POLYHEDRA = 256


def _mesh(data, positions, velocity):
    """The cells by id, and the bits of the shapes present."""
    from tack.dataset.polyhedra import PolyhedralTopology
    from tack.dataset.topology import StructuredTopology, UnstructuredTopology

    t = data.topology
    present = 0
    for group in t.groups() if hasattr(t, "groups") else ():
        if group.count and group.shape in _SHAPES:
            present |= 1 << _SHAPES.index(group.shape)
    if isinstance(t, UnstructuredTopology):
        return _FlatCells(t.types, t.offsets, t.connectivity), present
    if isinstance(t, StructuredTopology) and len(t.point_dims) >= 2:
        return _GridCells(t.point_dims), present
    if isinstance(t, PolyhedralTopology) and t.dimension == 3:
        return _PolyhedralMesh(t, positions, velocity), _POLYHEDRA if t.num_cells else 0
    raise NotImplementedError("advection takes unstructured, polyhedral and 2D or 3D "
                              "structured topologies; not polygons or polylines, nor lines")


def _run(data, field, seeds, step_size, steps, integrator, locator, record):
    if int(steps) < 1:
        raise ValueError(f"steps must be at least 1, not {steps!r}")
    if integrator not in ("rk4", "euler"):
        raise ValueError(f"integrator is 'rk4' or 'euler', not {integrator!r}")
    geometry = data.geometry
    if not isinstance(geometry.space, H1) or geometry.space.order != 1:
        raise TypeError("advection needs an order-1 H1 geometry: one position per point")
    velocity, on_cells = _velocity(data, field)
    positions = _as_vectors(arrays.materialize(geometry.values), data.dtype)
    mesh, present = _mesh(data, positions, velocity)
    locator = locator or CellLocator(data)
    if locator.data is not data:
        raise ValueError("the locator was built for another dataset")
    bin_offsets, bin_cells = locator.cell_bins()
    dtype = data.dtype
    # Seeds in an array of 3-vectors are copied on the device; anything else
    # comes from the host.
    stored = isinstance(seeds, Array) and seeds._vector_n == 3
    host = None if stored else np.ascontiguousarray(
        np.asarray(seeds.numpy() if hasattr(seeds, "numpy") else seeds,
                   dtype.numpy_dtype).reshape(-1, 3))
    m = int(seeds.shape[0]) if stored else len(host)
    xs = tack.empty((m,), tack.vec(3, dtype))
    times = tack.empty((m,), dtype)
    taken = tack.empty((m,), tack.i32)
    status = tack.empty((m,), tack.i32)
    lengths = tack.empty((m,), tack.i32)
    capacity = int(steps) + 1 if record else 1
    trace = tack.empty((m * capacity,), tack.vec(3, dtype))
    trace_times = tack.empty((m * capacity,), dtype)
    if m:
        tack.copyto(xs, seeds.to_device(xs.device) if stored else host)
        times.fill(0)
        taken.fill(0)
        status.fill(int(SUCCESS))
        eps = float(np.finfo(dtype.numpy_dtype).eps)
        _advect(mesh, _kinds(present), *(shape() for shape in _SHAPES), positions, velocity, on_cells,
                locator.lo, locator.hi, bin_offsets, bin_cells, *locator._grid(),
                float(locator.slack), _TOLERANCE, data.num_cells, float(step_size), int(steps),
                1 if integrator == "rk4" else 0, eps, xs, times, taken, status, trace,
                trace_times, lengths, capacity, 1 if record else 0)
    return xs, times, taken, status, (trace, trace_times), lengths, capacity
