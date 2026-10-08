"""Arrays: what a field's values are, apart from how they are stored.

A ``Field``'s ``values`` is any array: a device field (``tack.field`` or
``tack.Vector.field``), stored explicitly, or one of the implicit arrays
here, which compute each value from its index and store next to nothing:

- ``CartesianProduct(x, y, z)``: 3-vectors ``(x[i], y[j], z[k])`` at index
  ``i + nx * (j + ny * k)`` -- a rectilinear grid's points, as Viskores'
  ``ArrayHandleCartesianProduct``;
- ``ConstantArray(value, size)``: the same value everywhere;
- ``CountingArray(size, start, step)``: ``start + step * index``.

Kernels never see the difference. A field's view carries a *storage* mixin
(``tack.data.views``) whose ``get(k)`` reads value ``k``; spaces read
through it. A storage may also offer ``get_ijk(ijk)``, which a structured
grid's cells use to skip splitting a flat index (``CartesianProduct`` does).

The helpers ``size_of``, ``width_of``, ``dtype_of`` and ``to_host`` answer
for any array, explicit or implicit.
"""

import numpy as np

import tack
from tack.data import views
from tack.lang.field import Field as _DeviceField

__all__ = ["CartesianProduct", "ConstantArray", "CountingArray", "dtype_of", "materialize",
           "size_of", "to_host", "width_of"]


class _Implicit:
    """What every implicit array has: ``size``, ``width`` (components per value, or
    None for scalars), ``dtype``, a storage ``mixin`` and the ``attributes`` it reads."""

    width = None
    mixin = None

    def attributes(self):
        raise NotImplementedError

    def to_numpy(self, vectors=False):
        """The values as a host array: ``(size,)``, or ``(size, width)`` for vectors
        (also when ``vectors`` is false: there is no flat layout to return)."""
        raise NotImplementedError

    def __repr__(self):
        return type(self).__name__


class CartesianProduct(_Implicit):
    """The points of a grid whose lines run along the axes: ``(x[i], y[j], z[k])`` at
    index ``i + nx * (j + ny * k)``. Only the three axes are stored."""

    width = 3
    mixin = views._CartesianStorage

    def __init__(self, x, y=(0.0,), z=(0.0,), dtype=tack.f32):
        self.dtype = dtype
        self.x, self.y, self.z = (self._axis(a) for a in (x, y, z))
        self.dims = (self.x.shape[0], self.y.shape[0], self.z.shape[0])
        self.size = int(np.prod(self.dims))

    def _axis(self, values):
        if isinstance(values, _DeviceField):
            return values
        values = np.ascontiguousarray(values, dtype=self.dtype.numpy_dtype).reshape(-1)
        field = tack.field(self.dtype, shape=values.shape)
        field.from_numpy(values)
        return field

    @property
    def point_dims(self):
        """``dims`` without trailing ones: the dimensions of the grid of cells."""
        dims = list(self.dims)
        while len(dims) > 1 and dims[-1] == 1:
            dims.pop()
        return tuple(dims)

    def __repr__(self):
        return f"CartesianProduct of axes of {' x '.join(map(str, self.dims))}"

    def attributes(self):
        return {"xs": self.x, "ys": self.y, "zs": self.z,
                "px": self.dims[0], "py": self.dims[1]}

    def to_numpy(self, vectors=False):
        z, y, x = np.meshgrid(self.z.to_numpy(), self.y.to_numpy(), self.x.to_numpy(),
                              indexing="ij")
        return np.stack([x, y, z], axis=-1).reshape(-1, 3)


class ConstantArray(_Implicit):
    """``size`` copies of the scalar ``value``. Kernels read it as a runtime scalar: an
    int is i32 (or wider when it must be), a float f32 unless an f64 field is in
    the same call."""

    mixin = views._ConstantStorage

    def __init__(self, value, size):
        self.value = value
        self.size = int(size)
        self.dtype = tack.i32 if isinstance(value, (int, np.integer)) else tack.f32

    def attributes(self):
        return {"constant": self.value}

    def to_numpy(self, vectors=False):
        return np.full(self.size, self.value, dtype=self.dtype.numpy_dtype)


class CountingArray(_Implicit):
    """``start + step * index`` for ``index < size``; integers stay i32."""

    mixin = views._CountingStorage

    def __init__(self, size, start=0, step=1):
        self.size = int(size)
        self.start = start
        self.step = step
        integral = all(isinstance(v, (int, np.integer)) for v in (start, step))
        self.dtype = tack.i32 if integral else tack.f32

    def attributes(self):
        return {"start": self.start, "step": self.step}

    def to_numpy(self, vectors=False):
        return (self.start + self.step * np.arange(self.size)).astype(self.dtype.numpy_dtype)


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


def materialize(values):
    """``values`` as a device field: itself if it is one, else filled from the host."""
    if not isinstance(values, _Implicit):
        return values
    host = values.to_numpy(vectors=True)
    if values.width:
        out = tack.Vector.field(values.width, values.dtype, shape=(values.size,))
    else:
        out = tack.field(values.dtype, shape=(values.size,))
    if values.size:
        out.from_numpy(host)
    return out


def storage(values):
    """The storage mixin that reads ``values``, and the attributes it needs."""
    if isinstance(values, _Implicit):
        return values.mixin, values.attributes()
    if isinstance(values, _DeviceField):
        return views._ExplicitStorage, {"values": values}
    raise TypeError(f"field values must be a tack field or an implicit array, "
                    f"not {type(values).__name__}")
