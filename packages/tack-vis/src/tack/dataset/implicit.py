"""Implicit functions: regions of space by the sign of a function.

``value(p)`` is negative inside, zero on the surface and positive outside,
in the forms VTK (``vtkImplicitFunction``) and Viskores
(``viskores::ImplicitFunction``) use, so a slice by one lands on the same
points as theirs: a sphere and a cylinder are quadrics (squared distance
minus squared radius), a plane and a set of planes signed distances (a set's
the largest of its planes'), a box the signed distance to it.

Each is a ``@tack.device_class`` object whose parameters are instance
values: a kernel takes it as a template and calls ``f.value(p)`` on a
3-vector, and moving a sphere or resizing a box does not compile anything
anew. ``algorithms.implicit_values`` evaluates one on a dataset's geometry;
``slice``, ``clip``, ``extract_geometry`` and ``extract_points`` take one.
``Union``, ``Intersection`` and ``Difference`` combine them, as VTK's
``vtkImplicitBoolean`` does (the smaller value, the larger, the first's or the
second's negated), on the device: a kernel inlines the whole tree.

Parameters are positions in world coordinates. On a dataset with an origin
(``DataSet.origin``) a filter evaluates the function at the geometry's values
-- positions relative to that origin -- through ``relative_to``: each
function here moves its parameters by the origin in double precision
(``_relative_to``), so nothing is lost far from zero; any other object with a
``value`` gets the origin added to the position it is given, in the
geometry's type.

Every product in a ``value`` is named, so the float contract rounds it on its
own and no backend fuses it into a sum: a function gives the same bits on
every backend. A slice through mesh points -- whose values are then exactly
the level, and merge (``contour``) -- cuts alike everywhere; with a fused
multiply-add, Metal's noise around zero cut a plane through grid points into
1015 triangles where the CPU made 883.
"""

import numpy as np

import tack

__all__ = ["Box", "Cylinder", "Difference", "Intersection", "Plane", "Planes", "Sphere", "Union"]


def relative_to(function, origin):
    """``function`` for positions relative to ``origin`` (three floats): itself at
    a zero origin, a copy with its parameters moved where the function has
    ``_relative_to``, else ``function`` evaluated at the position plus
    ``origin``."""
    if not any(origin):
        return function
    moved = getattr(type(function), "_relative_to", None)
    if moved is not None:
        return moved(function, np.asarray(origin, np.float64))
    return _Shifted(function, origin)


def _copy(function, **values):
    """A copy of ``function`` with some instance values replaced."""
    out = object.__new__(type(function))
    out.__dict__.update(function.__dict__)
    out.__dict__.update(values)
    return out


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


@tack.device_class
class Plane:
    """The half-space below a plane: ``n . (p - origin)``, ``n`` the unit normal."""

    def __init__(self, origin, normal):
        self.ox, self.oy, self.oz = map(float, _vector(origin, "origin"))
        self.nx, self.ny, self.nz = map(float, _unit(normal, "normal"))

    def _relative_to(self, origin):
        return _copy(self, ox=self.ox - origin[0], oy=self.oy - origin[1],
                     oz=self.oz - origin[2])

    @tack.func
    def value(self, p):
        # Each product named, so rounded on its own (the module's docstring).
        a = (p[0] - self.ox) * self.nx
        b = (p[1] - self.oy) * self.ny
        c = (p[2] - self.oz) * self.nz
        return a + b + c


@tack.device_class
class Sphere:
    """A ball: ``|p - center|^2 - radius^2``."""

    def __init__(self, center, radius):
        self.cx, self.cy, self.cz = map(float, _vector(center, "center"))
        if not radius > 0:
            raise ValueError(f"radius must be positive, not {radius!r}")
        self.r2 = float(radius) ** 2

    def _relative_to(self, origin):
        return _copy(self, cx=self.cx - origin[0], cy=self.cy - origin[1],
                     cz=self.cz - origin[2])

    @tack.func
    def value(self, p):
        dx = p[0] - self.cx
        dy = p[1] - self.cy
        dz = p[2] - self.cz
        xx = dx * dx
        yy = dy * dy
        zz = dz * dz
        return xx + yy + zz - self.r2


@tack.device_class
class Cylinder:
    """An infinite cylinder about the line through ``center`` along ``axis``: the
    squared distance from that line minus ``radius^2``."""

    def __init__(self, center, axis, radius):
        self.cx, self.cy, self.cz = map(float, _vector(center, "center"))
        self.ax, self.ay, self.az = map(float, _unit(axis, "axis"))
        if not radius > 0:
            raise ValueError(f"radius must be positive, not {radius!r}")
        self.r2 = float(radius) ** 2

    def _relative_to(self, origin):
        return _copy(self, cx=self.cx - origin[0], cy=self.cy - origin[1],
                     cz=self.cz - origin[2])

    @tack.func
    def value(self, p):
        dx = p[0] - self.cx
        dy = p[1] - self.cy
        dz = p[2] - self.cz
        tx = dx * self.ax
        ty = dy * self.ay
        tz = dz * self.az
        t = tx + ty + tz
        sx = t * self.ax
        sy = t * self.ay
        sz = t * self.az
        rx = dx - sx
        ry = dy - sy
        rz = dz - sz
        xx = rx * rx
        yy = ry * ry
        zz = rz * rz
        return xx + yy + zz - self.r2


@tack.device_class
class Box:
    """An axis-aligned box from ``lower`` to ``upper``: the signed distance to it
    (negative inside, by the distance to the nearest face)."""

    def __init__(self, lower, upper):
        lo, hi = _vector(lower, "lower"), _vector(upper, "upper")
        if not (lo < hi).all():
            raise ValueError(f"lower {lo} must be below upper {hi} on every axis")
        self.cx, self.cy, self.cz = map(float, (lo + hi) / 2)
        self.hx, self.hy, self.hz = map(float, (hi - lo) / 2)

    def _relative_to(self, origin):
        return _copy(self, cx=self.cx - origin[0], cy=self.cy - origin[1],
                     cz=self.cz - origin[2])

    @tack.func
    def value(self, p):
        qx = abs(p[0] - self.cx) - self.hx
        qy = abs(p[1] - self.cy) - self.hy
        qz = abs(p[2] - self.cz) - self.hz
        ox = max(qx, 0.0)
        oy = max(qy, 0.0)
        oz = max(qz, 0.0)
        xx = ox * ox
        yy = oy * oy
        zz = oz * oz
        return sqrt(xx + yy + zz) + min(max(qx, max(qy, qz)), 0.0)


@tack.device_class
class Planes:
    """The convex region below every one of several planes (a frustum, a clipping
    box at any angle): the largest of their signed distances.

    The planes follow the data's precision, as the other functions' Python
    numbers do: `implicit_values`, and every filter that takes an implicit
    function, evaluate a copy holding them in the geometry's floating type (f64
    for f64 data) on its device (`_matching`). Made, it holds them in f32 on
    the default device, which a kernel of one's own calling ``value`` reads."""

    def __init__(self, origins, normals):
        origins = np.asarray(origins, float).reshape(-1, 3)
        normals = np.asarray(normals, float).reshape(-1, 3)
        if len(origins) == 0 or origins.shape != normals.shape:
            raise ValueError("Planes needs one origin and one normal per plane, at least one")
        lengths = np.linalg.norm(normals, axis=1)
        if not (lengths > 0).all():
            raise ValueError("a plane's normal must not be zero")
        # Kept as given (underscored: no kernel receives them), and stored per
        # precision and device as they are asked for.
        self._rows = np.hstack([origins, normals / lengths[:, None]]).reshape(-1)
        self._kept = {}
        self.planes = self._stored(tack.f32, None)

    def _stored(self, dtype, device):
        planes = tack.empty((self._rows.size,), dtype, device=device)
        tack.copyto(planes, self._rows.astype(dtype.numpy_dtype))
        return planes

    def _relative_to(self, origin):
        rows = self._rows.reshape(-1, 6)
        return Planes(rows[:, :3] - origin, rows[:, 3:])

    def _matching(self, dtype, device):
        """These planes in ``dtype`` on ``device``: itself, or a copy made once."""
        if self.planes.dtype is dtype and self.planes.device == device:
            return self
        key = (dtype, device)
        if key not in self._kept:
            copy = object.__new__(Planes)
            copy._rows, copy._kept = self._rows, self._kept
            copy.planes = self._stored(dtype, device)
            self._kept[key] = copy
        return self._kept[key]

    @tack.func
    def value(self, p):
        most = -3.0e38
        for k in range(self.planes.shape[0] // 6):
            b = 6 * k
            x = (p[0] - self.planes[b]) * self.planes[b + 3]
            y = (p[1] - self.planes[b + 1]) * self.planes[b + 4]
            z = (p[2] - self.planes[b + 2]) * self.planes[b + 5]
            most = max(most, x + y + z)
        return most


def _operand(function):
    if not callable(getattr(function, "value", None)):
        raise TypeError(f"an implicit function has a value(p): {function!r} has none")
    return function


class _Boolean:
    """Two functions combined, as VTK's ``vtkImplicitBoolean`` combines them;
    more than two fold into a chain of the same operation. The operands are
    device classes held as attributes, so a kernel inlines the whole tree."""

    def __init__(self, first, second, *more):
        self.a = _operand(first)
        self.b = type(self)(second, *more) if more else _operand(second)

    def _relative_to(self, origin):
        return _copy(self, a=relative_to(self.a, origin), b=relative_to(self.b, origin))

    def _matching(self, dtype, device):
        """The operands at ``dtype`` on ``device`` (a ``Planes`` holds arrays)."""
        def matched(f):
            m = getattr(type(f), "_matching", None)
            return m(f, dtype, device) if m is not None else f

        a, b = matched(self.a), matched(self.b)
        return self if a is self.a and b is self.b else _copy(self, a=a, b=b)


@tack.device_class
class Union(_Boolean):
    """Inside either function's region: the smaller of their values."""

    @tack.func
    def value(self, p):
        return min(self.a.value(p), self.b.value(p))


@tack.device_class
class Intersection(_Boolean):
    """Inside both functions' regions: the larger of their values."""

    @tack.func
    def value(self, p):
        return max(self.a.value(p), self.b.value(p))


@tack.device_class
class Difference(_Boolean):
    """Inside the first function's region and outside the others': the first's
    value or the second's negated, whichever is larger."""

    def __init__(self, first, second, *more):
        # first - (second | more...): the rest as one union.
        self.a = _operand(first)
        self.b = Union(second, *more) if more else _operand(second)

    @tack.func
    def value(self, p):
        return max(self.a.value(p), -self.b.value(p))


@tack.device_class
class _Shifted:
    """A function given positions relative to an origin: its own ``value`` at the
    position plus the origin, added in the position's type (``relative_to``,
    for functions without ``_relative_to``)."""

    def __init__(self, function, origin):
        self.function = function
        self.ox, self.oy, self.oz = (float(v) for v in origin)

    @tack.func
    def value(self, p):
        return self.function.value([p[0] + self.ox, p[1] + self.oy, p[2] + self.oz])

