"""Implicit functions: regions of space by the sign of a function.

``value(p)`` is negative inside, zero on the surface and positive outside,
in the forms VTK (``vtkImplicitFunction``) and Viskores
(``viskores::ImplicitFunction``) use, so a slice by one lands on the same
points as theirs: a sphere and a cylinder are quadrics (squared distance
minus squared radius), a plane and a set of planes signed distances (a set's
the largest of its planes'), a box the signed distance to it.

Each is a ``@tack.data_oriented`` object whose parameters are instance
values: a kernel takes it as a template and calls ``f.value(p)`` on a
3-vector, and moving a sphere or resizing a box does not compile anything
anew. ``algorithms.implicit_values`` evaluates one on a dataset's geometry;
``slice``, ``extract_geometry`` and ``extract_points`` take one.
"""

import numpy as np

import tack

__all__ = ["Box", "Cylinder", "Plane", "Planes", "Sphere"]


def _vector(value, name):
    v = np.asarray(value, float).reshape(-1)
    if v.size != 3 or not np.isfinite(v).all():
        raise ValueError(f"{name} must be three finite numbers, not {value!r}")
    return v


def _unit(value, name):
    v = _vector(value, name)
    length = np.linalg.norm(v)
    if length == 0:
        raise ValueError(f"{name} must not be zero")
    return v / length


@tack.data_oriented
class Plane:
    """The half-space below a plane: ``n . (p - origin)``, ``n`` the unit normal."""

    def __init__(self, origin, normal):
        self.ox, self.oy, self.oz = map(float, _vector(origin, "origin"))
        self.nx, self.ny, self.nz = map(float, _unit(normal, "normal"))

    @tack.func
    def value(self, p):
        return ((p[0] - self.ox) * self.nx + (p[1] - self.oy) * self.ny
                + (p[2] - self.oz) * self.nz)


@tack.data_oriented
class Sphere:
    """A ball: ``|p - center|^2 - radius^2``."""

    def __init__(self, center, radius):
        self.cx, self.cy, self.cz = map(float, _vector(center, "center"))
        if not radius > 0:
            raise ValueError(f"radius must be positive, not {radius!r}")
        self.r2 = float(radius) ** 2

    @tack.func
    def value(self, p):
        dx = p[0] - self.cx
        dy = p[1] - self.cy
        dz = p[2] - self.cz
        return dx * dx + dy * dy + dz * dz - self.r2


@tack.data_oriented
class Cylinder:
    """An infinite cylinder about the line through ``center`` along ``axis``: the
    squared distance from that line minus ``radius^2``."""

    def __init__(self, center, axis, radius):
        self.cx, self.cy, self.cz = map(float, _vector(center, "center"))
        self.ax, self.ay, self.az = map(float, _unit(axis, "axis"))
        if not radius > 0:
            raise ValueError(f"radius must be positive, not {radius!r}")
        self.r2 = float(radius) ** 2

    @tack.func
    def value(self, p):
        dx = p[0] - self.cx
        dy = p[1] - self.cy
        dz = p[2] - self.cz
        t = dx * self.ax + dy * self.ay + dz * self.az
        rx = dx - t * self.ax
        ry = dy - t * self.ay
        rz = dz - t * self.az
        return rx * rx + ry * ry + rz * rz - self.r2


@tack.data_oriented
class Box:
    """An axis-aligned box from ``lower`` to ``upper``: the signed distance to it
    (negative inside, by the distance to the nearest face)."""

    def __init__(self, lower, upper):
        lo, hi = _vector(lower, "lower"), _vector(upper, "upper")
        if not (lo < hi).all():
            raise ValueError(f"lower {lo} must be below upper {hi} on every axis")
        self.cx, self.cy, self.cz = map(float, (lo + hi) / 2)
        self.hx, self.hy, self.hz = map(float, (hi - lo) / 2)

    @tack.func
    def value(self, p):
        qx = abs(p[0] - self.cx) - self.hx
        qy = abs(p[1] - self.cy) - self.hy
        qz = abs(p[2] - self.cz) - self.hz
        ox = max(qx, 0.0)
        oy = max(qy, 0.0)
        oz = max(qz, 0.0)
        return sqrt(ox * ox + oy * oy + oz * oz) + min(max(qx, max(qy, qz)), 0.0)


@tack.data_oriented
class Planes:
    """The convex region below every one of several planes (a frustum, a clipping
    box at any angle): the largest of their signed distances."""

    def __init__(self, origins, normals):
        origins = np.asarray(origins, float).reshape(-1, 3)
        normals = np.asarray(normals, float).reshape(-1, 3)
        if len(origins) == 0 or origins.shape != normals.shape:
            raise ValueError("Planes needs one origin and one normal per plane, at least one")
        lengths = np.linalg.norm(normals, axis=1)
        if not (lengths > 0).all():
            raise ValueError("a plane's normal must not be zero")
        rows = np.hstack([origins, normals / lengths[:, None]]).astype(np.float32)
        self.planes = tack.field(tack.f32, shape=(rows.size,))
        self.planes.from_numpy(rows.reshape(-1))

    @tack.func
    def value(self, p):
        most = -3.0e38
        for k in range(self.planes.shape[0] // 6):
            b = 6 * k
            d = ((p[0] - self.planes[b]) * self.planes[b + 3]
                 + (p[1] - self.planes[b + 1]) * self.planes[b + 4]
                 + (p[2] - self.planes[b + 2]) * self.planes[b + 5])
            most = max(most, d)
        return most
