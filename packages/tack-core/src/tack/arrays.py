"""Arrays: values apart from how they are stored -- stored, or computed from the index.

An array is a Tack array (``tack.empty``, ``tack.array``), stored
explicitly, or an implicit array, which computes each value from its index
and stores next to nothing. Implicit arrays compose: a permutation of a
concatenation of a counting array is an array, to any depth. Every one is
indexable in kernels as a stored array is -- ``a[k]``, ``len(a)`` -- so a
kernel written for stored arrays takes them as they are, compiled for each;
``tack.dataset`` fields hold them as their values.

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

An implicit array is a ``@tack.device_class`` template with ``get(k)``,
value ``k`` (``__getitem__`` calls it, and ``__len__`` reads ``length``),
and grid points also with ``get_ijk(ijk)``, the point at
``(i, j, k)``, which a structured grid's cells use to skip splitting a flat
index. An array holding arrays reaches them as ``self.base.get(k)``; the
templates nest, so a composed array compiles to the expression one would
write by hand. A field's view holds the array (``views._ArrayStorage``) and
reads it through ``self.array.get(k)``; kernels never see the difference.
Unlike VTK's ``std::function`` arrays, a ``Function`` or ``Transform`` is a
``@tack.func``, so it runs in the kernel on every backend.

An array's size, width (components per value, or None) and dtype are kept
beside it, not as attributes a kernel would receive (only ``length``, for
``len(a)``, is one): ``size_of``,
``width_of``, ``dtype_of`` and ``to_host`` answer for any array, explicit or
implicit, and ``materialize`` stores one, by a kernel.
"""

import math

import numpy as np

import tack
from tack import random
from tack.lang.array import Array as _DeviceArray
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

    # Python's indexing protocol, so a kernel indexes any array as it does a
    # stored one: ``a[k]`` is ``get(k)``, ``len(a)`` its length.
    @tack.func
    def __getitem__(self, k):
        return self.get(k)

    @tack.func
    def __len__(self):
        return self.length

    @property
    def _device(self):
        """The backend of the arrays this one holds, or None for one that holds
        none: where it is read, and where `materialize` stores it."""
        for value in vars(self).values():
            dev = getattr(value, "_device", None)
            if dev is not None:
                return dev
        return None

    @property
    def device(self):
        """The `tack.Device` of the arrays this one holds; the default device for
        one that holds none."""
        from tack.runtime.dispatch import as_device, get_backend
        return as_device(self._device or get_backend())

    def to_device(self, device):
        """This array, reading from ``device``: a copy with every array it holds
        copied there (an implicit array it holds moved the same way); the
        parameters it keeps as numbers are the same. One that holds no array
        belongs to no device and is returned as it is."""
        if self._device is None:
            return self
        import copy
        held = {name: value.to_device(device) for name, value in vars(self).items()
                if isinstance(value, (_Implicit, _DeviceArray)) and value._device is not None}
        if all(held[name] is getattr(self, name) for name in held):
            return self                       # already readable there
        moved = copy.copy(self)
        for name, value in held.items():
            setattr(moved, name, value)
        return moved

    # Reductions compute the values into a temporary array and reduce it as a
    # stored array is reduced, so they share its semantics exactly: float
    # extrema propagate NaNs, integer sums wrap in 64 bits.
    def sum(self):
        """The sum of every component, as ``Array.sum``."""
        return materialize(self).sum()

    def min(self):
        """The least component, as ``Array.min``."""
        return materialize(self).min()

    def max(self):
        """The greatest component, as ``Array.max``."""
        return materialize(self).max()

    def mean(self):
        """The mean of every component, as ``Array.mean``."""
        return materialize(self).mean()

    def numpy(self):
        """The values as a host array: ``(size,)``, or ``(size, width)`` for vectors."""
        return to_host(materialize(self))

    def __array__(self, dtype=None, copy=None):
        """``np.asarray(values)``: the values computed on the host."""
        if copy is False:
            raise ValueError("an implicit array cannot be viewed by NumPy without a copy")
        values = self.numpy()
        return values if dtype is None else values.astype(dtype, copy=False)

    def __repr__(self):
        return type(self).__name__


def _metadata(array, size, width, dtype):
    from tack.runtime.dispatch import same_device

    # An implicit array lives where the arrays it holds do, so they must
    # share one device's memory (what `materialize` and kernels read).
    same_device(type(array).__name__, vars(array).values())
    array._size, array._width, array._dtype = int(size), width, dtype
    # The length again, as an attribute a kernel receives, for `len(a)`.
    array.length = int(size)


@tack.device_class
class _Explicit(_Implicit):
    """A device array, where an array holds an array: value ``k`` is ``values[k]``."""

    def __init__(self, values):
        self.values = values
        _metadata(self, size_of(values), width_of(values), dtype_of(values))

    @tack.func
    def get(self, k):
        return self.values[k]

    @tack.func
    def __setitem__(self, k, v):
        self.values[k] = v


def _operand(values, device=None):
    """``values`` as an array another array can hold: an implicit array as it is, a
    device array wrapped, anything else stored first (on ``device``, by default
    the default device)."""
    if isinstance(values, _Implicit):
        return values
    if isinstance(values, _DeviceArray):
        return _Explicit(values)
    host = np.asarray(values)
    dtype = tack.i64 if host.dtype.kind in "iu" and host.size and \
        np.abs(host).max() > 2**31 - 1 else (tack.i32 if host.dtype.kind in "iu" else tack.f32)
    return _Explicit(_store(host, dtype, device))


def _operands(*values):
    """Several operands of one array: host data is stored where the device arrays
    among them are, so an array of arrays on another device can take a list."""
    device = next((v._device for v in values if getattr(v, "_device", None) is not None),
                  None)
    return [_operand(v, device) for v in values]


def _store(host, dtype, device=None):
    host = np.ascontiguousarray(host, dtype=dtype.numpy_dtype)
    if host.ndim == 2:
        field = tack.empty((host.shape[0],), tack.vec(host.shape[1], dtype), device=device)
    else:
        field = tack.empty(host.shape, dtype, device=device)
    if host.size:
        tack.copyto(field, host)
    return field


# ── Constants and sequences ─────────────────────────────────────────

@tack.device_class
class ConstantArray(_Implicit):
    """``size`` copies of ``value``: a scalar, or a vector (a sequence of components),
    as VTK's constant arrays of several components. ``dtype`` defaults to i32 for
    integers and f32 otherwise; the value is kept in a one-element field of it,
    so it is exactly that type in the kernel, on ``device`` (by default the
    default device)."""

    def __init__(self, value, size, dtype=None, device=None):
        host = np.asarray(value)
        if dtype is None:
            dtype = tack.i32 if host.dtype.kind in "iu" else tack.f32
        self.value = _store(host.reshape(1, -1) if host.ndim else host.reshape(1), dtype,
                            device)
        _metadata(self, size, host.shape[0] if host.ndim else None, dtype)

    @tack.func
    def get(self, k):
        return self.value[0]


@tack.device_class
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

    def numpy(self):
        return (self.start + self.step * np.arange(self.size)).astype(self.dtype.numpy_dtype)


# ── Grid points ─────────────────────────────────────────────────────

@tack.device_class
class CartesianProduct(_Implicit):
    """The points of a grid whose lines run along the axes: ``(x[i], y[j], z[k])`` at
    index ``i + nx * (j + ny * k)``. Only the three axes are stored."""

    def __init__(self, x, y=(0.0,), z=(0.0,), dtype=tack.f32, device=None):
        if device is None:      # host axes go where the device axes are
            device = next((a._device for a in (x, y, z) if isinstance(a, _DeviceArray)), None)
        self.xs, self.ys, self.zs = (self._axis(a, dtype, device) for a in (x, y, z))
        self.dims = (self.xs.shape[0], self.ys.shape[0], self.zs.shape[0])
        self.px, self.py = self.dims[0], self.dims[1]
        _metadata(self, math.prod(self.dims), 3, dtype)

    @staticmethod
    def _axis(values, dtype, device=None):
        if isinstance(values, _DeviceArray):
            return values
        return _store(np.asarray(values).reshape(-1), dtype, device)

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

    def numpy(self):
        z, y, x = np.meshgrid(self.zs.numpy(), self.ys.numpy(), self.xs.numpy(),
                              indexing="ij")
        return np.stack([x, y, z], axis=-1).reshape(-1, 3)


def _point_dims(dims):
    dims = list(dims)
    while len(dims) > 1 and dims[-1] == 1:
        dims.pop()
    return tuple(dims)


@tack.device_class
class UniformCoordinates(_Implicit):
    """The points of an image: ``origin + direction @ ((i, j, k) * spacing)`` at index
    ``i + nx * (j + ny * k)``, as VTK's image data (with its direction matrix) and
    Viskores' uniform point coordinates. Nothing per point is stored."""

    ORIENTED = 0                              # compile-time: apply ``direction``

    def __init__(self, dims, origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0), direction=None,
                 dtype=tack.f32, device=None):
        dims = tuple(int(d) for d in dims) + (1,) * (3 - len(dims))
        self.dims = dims
        self.px, self.py = dims[0], dims[1]
        self.ox, self.oy, self.oz = (float(v) for v in origin)
        self.sx, self.sy, self.sz = (float(v) for v in spacing)
        matrix = np.eye(3) if direction is None else np.asarray(direction, float).reshape(3, 3)
        if direction is not None and not np.allclose(matrix, np.eye(3)):
            self.ORIENTED = 1
        # Always present, so an f64 grid's scalars are f64 in the kernel; it
        # puts the grid on ``device`` (by default the default device).
        self.direction = _store(matrix.reshape(-1), dtype, device)
        self._origin, self._spacing, self._matrix = tuple(origin), tuple(spacing), matrix
        _metadata(self, math.prod(dims), 3, dtype)

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

    def numpy(self):
        k, j, i = np.meshgrid(*(np.arange(n) for n in self.dims[::-1]), indexing="ij")
        ijk = np.stack([i, j, k], axis=-1).reshape(-1, 3) * np.asarray(self._spacing)
        points = np.asarray(self._origin) + ijk @ self._matrix.T
        return points.astype(self.dtype.numpy_dtype)


# ── Arrays of arrays ────────────────────────────────────────────────

@tack.device_class
class Permutation(_Implicit):
    """``base[ids[k]]``: ``base``'s values in the order ``ids`` (integers, explicit or
    not) gives, as VTK's indexed array."""

    def __init__(self, ids, base):
        self.ids, self.base = _operands(ids, base)
        _metadata(self, size_of(self.ids), width_of(self.base), dtype_of(self.base))

    @tack.func
    def get(self, k):
        return self.base.get(self.ids.get(k))

    @tack.func
    def __setitem__(self, k, v):
        self.base[self.ids[k]] = v           # a scatter: repeated ids race


@tack.device_class
class _Concatenation(_Implicit):
    def __init__(self, first, second):
        self.first, self.second = first, second
        self.split = size_of(first)
        _metadata(self, size_of(first) + size_of(second), width_of(first), dtype_of(first))

    @tack.func
    def get(self, k):
        return self.first.get(k) if k < self.split else self.second.get(k - self.split)

    @tack.func
    def __setitem__(self, k, v):
        if k < self.split:
            self.first[k] = v
        else:
            self.second[k - self.split] = v


def Concatenate(*arrays):
    """The arrays end to end, as VTK's composite array: of one width and type."""
    if not arrays:
        raise ValueError("Concatenate needs at least one array")
    parts = _operands(*arrays)
    for part in parts[1:]:
        if (width_of(part), dtype_of(part)) != (width_of(parts[0]), dtype_of(parts[0])):
            raise TypeError("concatenated arrays must have one width and one type")
    joined = parts[0]
    for part in parts[1:]:
        joined = _Concatenation(joined, part)
    return joined


@tack.device_class
class Strided(_Implicit):
    """``values[offset + stride * k]`` of a scalar device array, as VTK's strided array:
    one column of interleaved data, or every n-th value."""

    def __init__(self, values, stride, offset=0, size=None):
        values = _operand(values)
        if width_of(values):
            raise TypeError("Strided reads a scalar field; ExtractComponent reads a vector's")
        self.values = values
        self.stride, self.offset = int(stride), int(offset)
        count = -(-(size_of(values) - self.offset) // self.stride) if size is None else size
        _metadata(self, count, None, dtype_of(values))

    @tack.func
    def get(self, k):
        return self.values[self.offset + self.stride * k]

    @tack.func
    def __setitem__(self, k, v):
        self.values[self.offset + self.stride * k] = v


@tack.device_class
class View(_Implicit):
    """``base[start + k]`` for ``k < size``: a stretch of another array."""

    def __init__(self, base, start, size):
        self.base = _operand(base)
        self.start = int(start)
        _metadata(self, size, width_of(self.base), dtype_of(self.base))

    @tack.func
    def get(self, k):
        return self.base.get(self.start + k)

    @tack.func
    def __setitem__(self, k, v):
        self.base[self.start + k] = v


@tack.device_class
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

    @tack.func
    def __setitem__(self, k, v):
        element = self.base[k]
        element[self.COMPONENT] = v
        self.base[k] = element


@tack.device_class
class _Components2(_Implicit):
    @tack.func
    def get(self, k):
        return [self.c0.get(k), self.c1.get(k)]

    @tack.func
    def __setitem__(self, k, v):
        self.c0[k] = v[0]
        self.c1[k] = v[1]


@tack.device_class
class _Components3(_Implicit):
    @tack.func
    def get(self, k):
        return [self.c0.get(k), self.c1.get(k), self.c2.get(k)]

    @tack.func
    def __setitem__(self, k, v):
        self.c0[k] = v[0]
        self.c1[k] = v[1]
        self.c2[k] = v[2]


@tack.device_class
class _Components4(_Implicit):
    @tack.func
    def get(self, k):
        return [self.c0.get(k), self.c1.get(k), self.c2.get(k), self.c3.get(k)]

    @tack.func
    def __setitem__(self, k, v):
        self.c0[k] = v[0]
        self.c1[k] = v[1]
        self.c2[k] = v[2]
        self.c3[k] = v[3]


def Components(*arrays):
    """Two to four scalar arrays as the components of a vector array, as Viskores'
    composite vector."""
    parts = _operands(*arrays)
    if not 2 <= len(parts) <= 4 or any(width_of(p) for p in parts):
        raise TypeError("Components takes two to four scalar arrays")
    if len({size_of(p) for p in parts}) != 1:
        raise ValueError("the component arrays differ in size")
    if len({dtype_of(p) for p in parts}) != 1:
        raise TypeError("the component arrays differ in type: Cast them to one")
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


@tack.device_class
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


@tack.device_class
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


@tack.func
def _to_i8(x):
    return tack.i8(x)


@tack.func
def _to_u8(x):
    return tack.u8(x)


@tack.func
def _to_i16(x):
    return tack.i16(x)


@tack.func
def _to_u16(x):
    return tack.u16(x)


@tack.func
def _to_u64(x):
    return tack.u64(x)


_CASTS = {tack.i8: _to_i8, tack.u8: _to_u8, tack.i16: _to_i16, tack.u16: _to_u16,
          tack.i32: _to_i32, tack.u32: _to_u32, tack.i64: _to_i64, tack.u64: _to_u64,
          tack.f32: _to_f32, tack.f64: _to_f64}


def Cast(base, dtype):
    """``base``'s values converted to ``dtype``, component by component."""
    if dtype not in _CASTS:
        raise TypeError(f"Cast converts to {', '.join(t.name for t in _CASTS)}")
    return Transform(_CASTS[dtype], base, dtype=dtype)


@tack.device_class
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


@tack.device_class
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
    return values.size


def width_of(values):
    """Components per value: 3 for 3-vectors, None for scalars."""
    if isinstance(values, _Implicit):
        return values.width
    return getattr(values, "_vector_n", None)


def dtype_of(values):
    """The type of each component."""
    return values.scalar_dtype if isinstance(values, _DeviceArray) else values.dtype


def element_type(values):
    """The element type an array of these values stores: ``tack.vec(3, f32)``
    for 3-vectors of f32, the component type for scalars (a stored array's
    ``dtype``)."""
    if isinstance(values, _Implicit):
        return tack.vec(values.width, values.dtype) if values.width else values.dtype
    return values.dtype


def to_host(values):
    """The values as a host array: ``(size,)`` for scalars, ``(size, width)`` for vectors."""
    return values.numpy()


@tack.kernel
def _fill(array, out):
    for k in range(out.shape[0]):
        out[k] = array.get(k)


def materialize(values):
    """``values`` as a device array: itself if it is one, else computed by a kernel."""
    if not isinstance(values, _Implicit):
        return values
    if isinstance(values, _Explicit):
        return values.values
    out = tack.empty((values.size,), element_type(values), device=values._device)
    if values.size:
        _fill(values, out)
    return out
