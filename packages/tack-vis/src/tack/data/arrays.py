"""Arrays: what a field's values are, apart from how they are stored.

A ``Field``'s ``values`` is any array: a device field (``tack.field`` or
``tack.Vector.field``), stored explicitly, or an implicit array, which
computes each value from its index and stores next to nothing. Implicit
arrays compose: a permutation of a concatenation of a counting array is an
array, to any depth.

=======================================  ========================  ====================================
Tack                                     VTK                       Viskores
=======================================  ========================  ====================================
``ConstantArray(value, size)``           ``vtkConstantArray``      ``ArrayHandleConstant``
``CountingArray(size, start, step)``     ``vtkAffineArray``        ``ArrayHandleCounting``
``CartesianProduct(x, y, z)``            ``vtkStructuredPointArray``  ``ArrayHandleCartesianProduct``
``UniformCoordinates(dims, origin,       ``vtkStructuredPointArray``  ``ArrayHandleUniformPointCoordinates``
spacing, direction)``                    (image data)
``Permutation(ids, base)``               ``vtkIndexedArray``       ``ArrayHandlePermutation``
``Concatenate(a, b, ...)``               ``vtkCompositeArray``     ``ArrayHandleConcatenate``
``Strided(values, stride, offset)``      ``vtkStridedArray``       ``ArrayHandleStride``
``View(base, start, size)``              --                        ``ArrayHandleView``
``ExtractComponent(base, c)``            --                        ``ArrayHandleExtractComponent``
``Components(a, b, ...)``                --                        ``ArrayHandleCompositeVector``
``Function(func, size, dtype)``          ``vtkStdFunctionArray``   ``ArrayHandleImplicit``
``Transform(func, base)``                --                        ``ArrayHandleTransform``
``Cast(base, dtype)``                    --                        ``ArrayHandleCast``
``RandomUniform``, ``RandomNormal``      --                        ``ArrayHandleRandomUniformReal``,
                                                                   ``ArrayHandleRandomStandardNormal``
=======================================  ========================  ====================================

An implicit array is a ``@tack.data_oriented`` template with ``get(k)``,
value ``k``, and grid points also with ``get_ijk(ijk)``, the point at
``(i, j, k)``, which a structured grid's cells use to skip splitting a flat
index. An array holding arrays reaches them as ``self.base.get(k)``; the
templates nest, so a composed array compiles to the expression one would
write by hand. A field's view holds the array (``views._ArrayStorage``) and
reads it through ``self.array.get(k)``; kernels never see the difference.
Unlike VTK's ``std::function`` arrays, a ``Function`` or ``Transform`` is a
``@tack.func``, so it runs in the kernel on every backend.

An array's size, width (components per value, or None) and dtype are kept
beside it, not as attributes a kernel would receive: ``size_of``,
``width_of``, ``dtype_of`` and ``to_host`` answer for any array, explicit or
implicit, and ``materialize`` stores one, by a kernel.
"""

import numpy as np

import tack
from tack import random
from tack.lang.field import Field as _DeviceField
from tack.lang.func import Func

__all__ = ["CartesianProduct", "Cast", "Components", "Concatenate", "ConstantArray",
           "CountingArray", "ExtractComponent", "Function", "Permutation", "RandomNormal",
           "RandomUniform", "Strided", "Transform", "UniformCoordinates", "View", "dtype_of",
           "materialize", "size_of", "to_host", "width_of"]


class _Implicit:
    """What every implicit array has: ``size``, ``width`` (components per value, or
    None for scalars) and ``dtype``, kept as underscored attributes so that a
    kernel holding the array does not receive them."""

    _width = None

    @property
    def size(self):
        return self._size

    @property
    def width(self):
        return self._width

    @property
    def dtype(self):
        return self._dtype

    def to_numpy(self, vectors=False):
        """The values as a host array: ``(size,)``, or ``(size, width)`` for vectors
        (also when ``vectors`` is false: there is no flat layout to return)."""
        return to_host(materialize(self))

    def __repr__(self):
        return type(self).__name__


def _metadata(array, size, width, dtype):
    array._size, array._width, array._dtype = int(size), width, dtype


@tack.data_oriented
class _Explicit(_Implicit):
    """A device field, where an array holds an array: value ``k`` is ``values[k]``."""

    def __init__(self, values):
        self.values = values
        _metadata(self, size_of(values), width_of(values), values.dtype)

    @tack.func
    def get(self, k):
        return self.values[k]


def _operand(values):
    """``values`` as an array another array can hold: an implicit array as it is, a
    device field wrapped, anything else stored first."""
    if isinstance(values, _Implicit):
        return values
    if isinstance(values, _DeviceField):
        return _Explicit(values)
    host = np.asarray(values)
    dtype = tack.i64 if host.dtype.kind in "iu" and host.size and \
        np.abs(host).max() > 2**31 - 1 else (tack.i32 if host.dtype.kind in "iu" else tack.f32)
    return _Explicit(_store(host, dtype))


def _store(host, dtype):
    host = np.ascontiguousarray(host, dtype=dtype.numpy_dtype)
    if host.ndim == 2:
        field = tack.Vector.field(host.shape[1], dtype, shape=(host.shape[0],))
    else:
        field = tack.field(dtype, shape=host.shape)
    if host.size:
        field.from_numpy(host)
    return field


# ── Constants and sequences ─────────────────────────────────────────

@tack.data_oriented
class ConstantArray(_Implicit):
    """``size`` copies of ``value``: a scalar, or a vector (a sequence of components),
    as VTK's constant arrays of several components. ``dtype`` defaults to i32 for
    integers and f32 otherwise; the value is kept in a one-element field of it,
    so it is exactly that type in the kernel."""

    def __init__(self, value, size, dtype=None):
        host = np.asarray(value)
        if dtype is None:
            dtype = tack.i32 if host.dtype.kind in "iu" else tack.f32
        self.value = _store(host.reshape(1, -1) if host.ndim else host.reshape(1), dtype)
        _metadata(self, size, host.shape[0] if host.ndim else None, dtype)

    @tack.func
    def get(self, k):
        return self.value[0]


@tack.data_oriented
class CountingArray(_Implicit):
    """``start + step * index`` for ``index < size`` (VTK's affine array): integers are
    ``dtype``, by default i32 unless a value needs i64 (ids past ``2**31 - 1``),
    else f32."""

    def __init__(self, size, start=0, step=1, dtype=None):
        self.start = start
        self.step = step
        integral = all(isinstance(v, (int, np.integer)) for v in (start, step))
        if dtype is None and integral:
            reach = max(abs(start), abs(start + step * max(int(size) - 1, 0)))
            dtype = tack.i64 if reach > 2**31 - 1 else tack.i32
        _metadata(self, size, None, dtype or tack.f32)

    @tack.func
    def get(self, k):
        return self.start + self.step * k

    def to_numpy(self, vectors=False):
        return (self.start + self.step * np.arange(self.size)).astype(self.dtype.numpy_dtype)


# ── Grid points ─────────────────────────────────────────────────────

@tack.data_oriented
class CartesianProduct(_Implicit):
    """The points of a grid whose lines run along the axes: ``(x[i], y[j], z[k])`` at
    index ``i + nx * (j + ny * k)``. Only the three axes are stored."""

    def __init__(self, x, y=(0.0,), z=(0.0,), dtype=tack.f32):
        self.xs, self.ys, self.zs = (self._axis(a, dtype) for a in (x, y, z))
        self.dims = (self.xs.shape[0], self.ys.shape[0], self.zs.shape[0])
        self.px, self.py = self.dims[0], self.dims[1]
        _metadata(self, np.prod(self.dims), 3, dtype)

    @staticmethod
    def _axis(values, dtype):
        if isinstance(values, _DeviceField):
            return values
        return _store(np.asarray(values).reshape(-1), dtype)

    x = property(lambda self: self.xs)
    y = property(lambda self: self.ys)
    z = property(lambda self: self.zs)

    @property
    def point_dims(self):
        """``dims`` without trailing ones: the dimensions of the grid of cells."""
        return _point_dims(self.dims)

    def __repr__(self):
        return f"CartesianProduct of axes of {' x '.join(map(str, self.dims))}"

    @tack.func
    def get(self, p):
        rest = p // self.px
        return [self.xs[p - rest * self.px], self.ys[rest % self.py], self.zs[rest // self.py]]

    @tack.func
    def get_ijk(self, at):
        return [self.xs[at[0]], self.ys[at[1]], self.zs[at[2]]]

    def to_numpy(self, vectors=False):
        z, y, x = np.meshgrid(self.zs.to_numpy(), self.ys.to_numpy(), self.xs.to_numpy(),
                              indexing="ij")
        return np.stack([x, y, z], axis=-1).reshape(-1, 3)


def _point_dims(dims):
    dims = list(dims)
    while len(dims) > 1 and dims[-1] == 1:
        dims.pop()
    return tuple(dims)


@tack.data_oriented
class UniformCoordinates(_Implicit):
    """The points of an image: ``origin + direction @ ((i, j, k) * spacing)`` at index
    ``i + nx * (j + ny * k)``, as VTK's image data (with its direction matrix) and
    Viskores' uniform point coordinates. Nothing per point is stored."""

    ORIENTED = 0                              # compile-time: apply ``direction``

    def __init__(self, dims, origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0), direction=None,
                 dtype=tack.f32):
        dims = tuple(int(d) for d in dims) + (1,) * (3 - len(dims))
        self.dims = dims
        self.px, self.py = dims[0], dims[1]
        self.ox, self.oy, self.oz = (float(v) for v in origin)
        self.sx, self.sy, self.sz = (float(v) for v in spacing)
        matrix = np.eye(3) if direction is None else np.asarray(direction, float).reshape(3, 3)
        if direction is not None and not np.allclose(matrix, np.eye(3)):
            self.ORIENTED = 1
        # Always present, so an f64 grid's scalars are f64 in the kernel.
        self.direction = _store(matrix.reshape(-1), dtype)
        self._origin, self._spacing, self._matrix = tuple(origin), tuple(spacing), matrix
        _metadata(self, np.prod(dims), 3, dtype)

    @property
    def point_dims(self):
        return _point_dims(self.dims)

    def __repr__(self):
        return f"UniformCoordinates of {' x '.join(map(str, self.dims))} points"

    @tack.func
    def get_ijk(self, at):
        x = [at[0] * self.sx, at[1] * self.sy, at[2] * self.sz]
        if self.ORIENTED == 1:
            d = self.direction
            x = [d[0] * x[0] + d[1] * x[1] + d[2] * x[2],
                 d[3] * x[0] + d[4] * x[1] + d[5] * x[2],
                 d[6] * x[0] + d[7] * x[1] + d[8] * x[2]]
        return [self.ox + x[0], self.oy + x[1], self.oz + x[2]]

    @tack.func
    def get(self, p):
        rest = p // self.px
        return self.get_ijk([p - rest * self.px, rest % self.py, rest // self.py])

    def to_numpy(self, vectors=False):
        k, j, i = np.meshgrid(*(np.arange(n) for n in self.dims[::-1]), indexing="ij")
        ijk = np.stack([i, j, k], axis=-1).reshape(-1, 3) * np.asarray(self._spacing)
        points = np.asarray(self._origin) + ijk @ self._matrix.T
        return points.astype(self.dtype.numpy_dtype)


# ── Arrays of arrays ────────────────────────────────────────────────

@tack.data_oriented
class Permutation(_Implicit):
    """``base[ids[k]]``: ``base``'s values in the order ``ids`` (integers, explicit or
    not) gives, as VTK's indexed array."""

    def __init__(self, ids, base):
        self.ids = _operand(ids)
        self.base = _operand(base)
        _metadata(self, size_of(self.ids), width_of(self.base), dtype_of(self.base))

    @tack.func
    def get(self, k):
        return self.base.get(self.ids.get(k))


@tack.data_oriented
class _Concatenation(_Implicit):
    def __init__(self, first, second):
        self.first, self.second = first, second
        self.split = size_of(first)
        _metadata(self, size_of(first) + size_of(second), width_of(first), dtype_of(first))

    @tack.func
    def get(self, k):
        return self.first.get(k) if k < self.split else self.second.get(k - self.split)


def Concatenate(*arrays):
    """The arrays end to end, as VTK's composite array: of one width and type."""
    if not arrays:
        raise ValueError("Concatenate needs at least one array")
    parts = [_operand(a) for a in arrays]
    for part in parts[1:]:
        if (width_of(part), dtype_of(part)) != (width_of(parts[0]), dtype_of(parts[0])):
            raise TypeError("concatenated arrays must have one width and one type")
    joined = parts[0]
    for part in parts[1:]:
        joined = _Concatenation(joined, part)
    return joined


@tack.data_oriented
class Strided(_Implicit):
    """``values[offset + stride * k]`` of a scalar device field, as VTK's strided array:
    one column of interleaved data, or every n-th value."""

    def __init__(self, values, stride, offset=0, size=None):
        if width_of(values):
            raise TypeError("Strided reads a scalar field; ExtractComponent reads a vector's")
        self.values = values
        self.stride, self.offset = int(stride), int(offset)
        count = -(-(size_of(values) - self.offset) // self.stride) if size is None else size
        _metadata(self, count, None, values.dtype)

    @tack.func
    def get(self, k):
        return self.values[self.offset + self.stride * k]


@tack.data_oriented
class View(_Implicit):
    """``base[start + k]`` for ``k < size``: a stretch of another array."""

    def __init__(self, base, start, size):
        self.base = _operand(base)
        self.start = int(start)
        _metadata(self, size, width_of(self.base), dtype_of(self.base))

    @tack.func
    def get(self, k):
        return self.base.get(self.start + k)


@tack.data_oriented
class ExtractComponent(_Implicit):
    """Component ``c`` of a vector array's values."""

    COMPONENT = 0                             # compile-time: the component read

    def __init__(self, base, component):
        self.base = _operand(base)
        if not width_of(self.base) or not 0 <= component < width_of(self.base):
            raise ValueError(f"component {component} of an array of width "
                             f"{width_of(self.base)}")
        self.COMPONENT = int(component)
        _metadata(self, size_of(self.base), None, dtype_of(self.base))

    @tack.func
    def get(self, k):
        return self.base.get(k)[self.COMPONENT]


@tack.data_oriented
class _Components2(_Implicit):
    @tack.func
    def get(self, k):
        return [self.c0.get(k), self.c1.get(k)]


@tack.data_oriented
class _Components3(_Implicit):
    @tack.func
    def get(self, k):
        return [self.c0.get(k), self.c1.get(k), self.c2.get(k)]


@tack.data_oriented
class _Components4(_Implicit):
    @tack.func
    def get(self, k):
        return [self.c0.get(k), self.c1.get(k), self.c2.get(k), self.c3.get(k)]


def Components(*arrays):
    """Two to four scalar arrays as the components of a vector array, as Viskores'
    composite vector."""
    parts = [_operand(a) for a in arrays]
    if not 2 <= len(parts) <= 4 or any(width_of(p) for p in parts):
        raise TypeError("Components takes two to four scalar arrays")
    if len({size_of(p) for p in parts}) != 1:
        raise ValueError("the component arrays differ in size")
    out = {2: _Components2, 3: _Components3, 4: _Components4}[len(parts)]()
    for c, part in enumerate(parts):
        setattr(out, f"c{c}", part)
    _metadata(out, size_of(parts[0]), len(parts), dtype_of(parts[0]))
    return out


# ── Functions ───────────────────────────────────────────────────────

def _device_function(func):
    if not isinstance(func, Func):
        raise TypeError("the function must be a @tack.func")
    return func


@tack.data_oriented
class Function(_Implicit):
    """``func(k)`` for ``k < size``: any ``@tack.func`` of the index, as VTK's
    ``vtkStdFunctionArray`` -- but evaluated in the kernel, on every backend. Its
    values are ``dtype`` (and vectors of ``width``) as the function returns them."""

    def __init__(self, func, size, dtype=tack.f32, width=None):
        self.func = _device_function(func)
        _metadata(self, size, width, dtype)

    @tack.func
    def get(self, k):
        return self.func(k)


@tack.data_oriented
class Transform(_Implicit):
    """``func(base[k])``: another array's values through a ``@tack.func``. ``dtype``
    and ``width`` are the function's results, by default ``base``'s."""

    def __init__(self, func, base, dtype=None, width=None):
        self.func = _device_function(func)
        self.base = _operand(base)
        _metadata(self, size_of(self.base), width if width is not None else width_of(self.base),
                  dtype or dtype_of(self.base))

    @tack.func
    def get(self, k):
        return self.func(self.base.get(k))


@tack.func
def _to_i32(x):
    return tack.i32(x)


@tack.func
def _to_i64(x):
    return tack.i64(x)


@tack.func
def _to_u32(x):
    return tack.u32(x)


@tack.func
def _to_f32(x):
    return tack.f32(x)


@tack.func
def _to_f64(x):
    return tack.f64(x)


_CASTS = {tack.i32: _to_i32, tack.i64: _to_i64, tack.u32: _to_u32, tack.f32: _to_f32,
          tack.f64: _to_f64}


def Cast(base, dtype):
    """``base``'s values converted to ``dtype``, component by component."""
    if dtype not in _CASTS:
        raise TypeError(f"Cast converts to {', '.join(t.name for t in _CASTS)}")
    return Transform(_CASTS[dtype], base, dtype=dtype)


@tack.data_oriented
class RandomUniform(_Implicit):
    """``size`` uniform draws in ``[low, high)``, value ``k`` a pure function of ``k``
    and ``seed`` (``tack.random``: the same on every backend, and from
    ``tack.random.np_uniform`` on the host). Indices repeat past ``2**32``."""

    def __init__(self, size, seed=0, low=0.0, high=1.0):
        self.seed = int(seed)
        self.low, self.span = float(low), float(high) - float(low)
        _metadata(self, size, None, tack.f32)

    @tack.func
    def get(self, k):
        u, _ = random.uniform(random.seed(k, self.seed))
        return self.low + self.span * u


@tack.data_oriented
class RandomNormal(_Implicit):
    """``size`` normal draws of ``mean`` and ``stddev``, as ``RandomUniform``."""

    def __init__(self, size, seed=0, mean=0.0, stddev=1.0):
        self.seed = int(seed)
        self.mean, self.stddev = float(mean), float(stddev)
        _metadata(self, size, None, tack.f32)

    @tack.func
    def get(self, k):
        z, _ = random.normal(random.seed(k, self.seed))
        return self.mean + self.stddev * z


# ── Any array ───────────────────────────────────────────────────────

def size_of(values):
    """The number of values."""
    if isinstance(values, _Implicit):
        return values.size
    return values.shape[0] if not width_of(values) else values.shape[0] // width_of(values)


def width_of(values):
    """Components per value: 3 for 3-vectors, None for scalars."""
    if isinstance(values, _Implicit):
        return values.width
    return getattr(values, "_vector_n", None)


def dtype_of(values):
    """The type of each component."""
    return values.dtype


def to_host(values):
    """The values as a host array: ``(size,)`` for scalars, ``(size, width)`` for vectors."""
    if isinstance(values, _Implicit):
        return values.to_numpy(vectors=True)
    return values.to_numpy(vectors=True) if width_of(values) else values.to_numpy()


@tack.kernel
def _fill(array, out):
    for k in range(out.shape[0]):
        out[k] = array.get(k)


def materialize(values):
    """``values`` as a device field: itself if it is one, else computed by a kernel."""
    if not isinstance(values, _Implicit):
        return values
    if isinstance(values, _Explicit):
        return values.values
    if values.width:
        out = tack.Vector.field(values.width, values.dtype, shape=(values.size,))
    else:
        out = tack.field(values.dtype, shape=(values.size,))
    if values.size:
        _fill(values, out)
    return out


def storage(values):
    """The storage mixin that reads ``values``, and the attributes it needs."""
    from tack.data import views

    if isinstance(values, _Implicit):
        mixin = views._ArrayIJKStorage if hasattr(values, "get_ijk") else views._ArrayStorage
        return mixin, {"array": values}
    if isinstance(values, _DeviceField):
        return views._ExplicitStorage, {"values": values}
    raise TypeError(f"field values must be a tack field or an implicit array, "
                    f"not {type(values).__name__}")
