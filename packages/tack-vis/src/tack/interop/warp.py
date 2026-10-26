"""Arrays between Tack and NVIDIA Warp, without copies.

With the Warp device (``tack.device("warp")``) a Tack array's memory is a
``wp.array`` on Warp's device, and Tack launches on that device's current
Warp stream, the stream a caller's ``wp.launch`` uses. So a program can run
its own Warp kernels and Tack's filters on the same arrays:

    tack.set_default_device("warp")
    wp.launch(my_kernel, n, inputs=[f])
    data.fields["f"] = td.Field(td.H1(data), from_warp(f))
    iso = alg.contour(data, "f", 0.5)
    points = to_warp(iso.geometry.values)          # a wp.array of wp.vec3

`from_warp` and `to_warp` map Warp's vector and matrix types to Tack's
(``wp.vec3`` and ``tack.vec(3, tack.f32)``, ``wp.mat33`` and
``tack.mat(3, 3, tack.f32)``), keep the other side's memory alive for as
long as the result lives, and do not mark the memory as handed outside
Tack: on one device and stream, Tack's launches need not complete before
returning. Under ``wp.ScopedStream`` Tack follows the scope's stream too.

Warp arrays have no read-only flag: a read-only Tack array given to Warp
can be written there.
"""

import math

import numpy as np

import tack
from tack.arrays import _Implicit, materialize
from tack.lang.array import Array, SoaArray, _array_view
from tack.lang.types import ElementType, mat, vec


def _warp():
    import warp as wp
    return wp


def _scalar(warp_scalar):
    """The Tack scalar type of a Warp scalar type."""
    from tack.codegen.warp_gen import _WARP_TYPES
    for dtype, name in _WARP_TYPES.items():
        if name == f"wp.{warp_scalar.__name__}":
            return dtype
    raise TypeError(f"no Tack type for Warp's {warp_scalar.__name__}")


def _tack_dtype(warp_dtype):
    """``wp.vec3`` as ``tack.vec(3, tack.f32)``, ``wp.mat33`` as ``tack.mat(3, 3,
    tack.f32)``, a scalar as itself."""
    wp = _warp()
    if wp.types.type_is_vector(warp_dtype):
        return vec(warp_dtype._length_, _scalar(warp_dtype._wp_scalar_type_))
    if wp.types.type_is_matrix(warp_dtype):
        rows, columns = warp_dtype._shape_
        return mat(rows, columns, _scalar(warp_dtype._wp_scalar_type_))
    return _scalar(warp_dtype)


_named = {}


def _warp_dtype(dtype):
    """The Warp type of a Tack scalar, vector or matrix type: Warp's named type
    where it has one (``wp.vec3f``, ``wp.mat33d``), which Warp code compares by
    identity, else the type ``wp.types.vector``/``matrix`` builds."""
    from tack.runtime.warp_backend import _warp_dtype as scalar
    wp = _warp()
    if not isinstance(dtype, ElementType):
        return scalar(dtype)
    if dtype in _named:
        return _named[dtype]
    if len(dtype.shape) == 1:
        built = wp.types.vector(length=dtype.shape[0], dtype=scalar(dtype.scalar_dtype))
    else:
        built = wp.types.matrix(shape=dtype.shape, dtype=scalar(dtype.scalar_dtype))
    prefix = "vec" if len(dtype.shape) == 1 else "mat"
    named = (getattr(wp, n) for n in dir(wp) if n.startswith(prefix))
    _named[dtype] = next((t for t in named if isinstance(t, type)
                          and wp.types.types_equal(t, built)), built)
    return _named[dtype]


def _warp_backend():
    return tack.device("warp")._backend


def from_warp(array, *, writable=True):
    """``array``, a contiguous ``wp.array``, as a Tack array on the Warp device,
    sharing its memory: a write through either is seen by the other.

    The element type follows Warp's (``wp.vec3`` gives ``tack.vec(3,
    tack.f32)``) and the shape is the array's. The Tack array keeps
    ``array`` alive. ``array`` must be on the device Tack's Warp device
    drives (Warp's default device).
    """
    from tack.runtime.warp_backend import WarpBuffer
    wp = _warp()
    if not isinstance(array, wp.array):
        raise TypeError(f"from_warp takes a wp.array, not {type(array).__name__}")
    if not array.is_contiguous:
        raise ValueError("from_warp wraps a contiguous wp.array; copy it with "
                         "wp.clone first")
    backend = _warp_backend()
    if str(array.device) != str(backend.wp_device):
        raise ValueError(f"the array is on Warp's {array.device}, but Tack's Warp device "
                         f"drives {backend.wp_device} (Warp's default device)")
    dtype = _tack_dtype(array.dtype)
    shape = tuple(array.shape)
    scalar = dtype.scalar_dtype if isinstance(dtype, ElementType) else dtype
    count = math.prod(shape) * (dtype.size if isinstance(dtype, ElementType) else 1)
    storage = (count,) if isinstance(dtype, ElementType) else shape
    if count:
        flat = wp.array(ptr=array.ptr, dtype=_warp_dtype(scalar), shape=(count,),
                        capacity=count * np.dtype(scalar.numpy_dtype).itemsize,
                        device=array.device)
    else:
        flat = None                     # nothing to share; WarpBuffer makes its own
    buffer = WarpBuffer(backend, scalar, storage, flat)
    buffer._owner = array               # the memory lives as long as the Tack array
    return _array_view(dtype, shape, buffer, writable=writable, device=backend)


def to_warp(array):
    """``array`` as a ``wp.array`` sharing its memory, with Warp's element type
    (``tack.vec(3, tack.f32)`` gives ``wp.vec3``) at its logical shape.

    An array on the Warp device is its own memory, kept alive by the result.
    An array on another Tack device goes through DLPack, which completes its
    queued launches. An implicit array (``tack.arrays``: ``ConstantArray``,
    ``UniformCoordinates``, ...) has no memory, and is materialized on its
    device first. Structure-of-arrays storage has no single ``wp.array``;
    give its components one at a time.
    """
    wp = _warp()
    if isinstance(array, _Implicit):
        array = materialize(array)
    if isinstance(array, SoaArray):
        raise TypeError("a structure-of-arrays array has no single wp.array; "
                        "give its components (array.components) one at a time")
    if not isinstance(array, Array):
        raise TypeError(f"to_warp takes a Tack array, not {type(array).__name__}")
    shape, dtype, nbytes = tuple(array.shape), array.dtype, array._buffer.nbytes
    backend = array._device
    if backend.name == "warp":
        address, device = array._buffer.address, backend.wp_device
    else:
        source = wp.from_dlpack(array)          # Tack's export completes its launches
        address, device = source.ptr, source.device
        array = (array, source)                 # keep both alive
    if not math.prod(shape):
        return wp.empty(shape, dtype=_warp_dtype(dtype), device=device)

    def keep(_ptr, _size, _held=array):
        """Warp calls this when the result is freed; it holds the Tack array till then."""

    return wp.array(ptr=address, dtype=_warp_dtype(dtype), shape=shape, capacity=nbytes,
                    device=device, deleter=keep)
