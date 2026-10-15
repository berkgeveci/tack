"""Tack type system — scalar types and type annotations for kernels."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np


class ScalarType:
    """Represents a scalar data type for arrays and kernel arguments."""

    def __init__(self, name: str, numpy_dtype: np.dtype, llvm_name: str, bits: int):
        self.name = name
        self.numpy_dtype = numpy_dtype
        self.llvm_name = llvm_name
        self.bits = bits

    def __repr__(self):
        return f"tack.{self.name}"

    def __deepcopy__(self, memo):
        # ScalarTypes are singletons — preserve identity through deepcopy
        return self

    def __reduce__(self):
        # ...and through pickle (the disk cache's IR): by module-level name.
        return self.name


# Scalar types
i8  = ScalarType("i8",  np.dtype(np.int8),    "i8",  8)
u8  = ScalarType("u8",  np.dtype(np.uint8),   "i8",  8)
i16 = ScalarType("i16", np.dtype(np.int16),   "i16", 16)
u16 = ScalarType("u16", np.dtype(np.uint16),  "i16", 16)
i32 = ScalarType("i32", np.dtype(np.int32),   "i32", 32)
u32 = ScalarType("u32", np.dtype(np.uint32),  "i32", 32)
i64 = ScalarType("i64", np.dtype(np.int64),   "i64", 64)
u64 = ScalarType("u64", np.dtype(np.uint64),  "i64", 64)
f32 = ScalarType("f32", np.dtype(np.float32), "float", 32)
f64 = ScalarType("f64", np.dtype(np.float64), "double", 64)

INTEGER_TYPES = frozenset((i8, i16, i32, i64, u8, u16, u32, u64))
UNSIGNED_TYPES = frozenset((u8, u16, u32, u64))


def integer_type_for_value(value: int) -> ScalarType:
    """Choose a representable type for an integer literal or scalar argument."""
    if -(2**31) <= value < 2**31:
        return i32
    if -(2**63) <= value < 2**63:
        return i64
    if 0 <= value < 2**64:
        return u64
    raise TypeError(f"Integer value {value} is outside Tack's 64-bit range")


# Map numpy dtypes back to tack types
_numpy_to_tack = {
    np.dtype(np.int8):    i8,
    np.dtype(np.uint8):   u8,
    np.dtype(np.int16):   i16,
    np.dtype(np.uint16):  u16,
    np.dtype(np.int32):   i32,
    np.dtype(np.uint32):  u32,
    np.dtype(np.int64):   i64,
    np.dtype(np.uint64):  u64,
    np.dtype(np.float32): f32,
    np.dtype(np.float64): f64,
}


# Map type name strings to ScalarType singletons
_NAME_TO_TYPE = {
    "i8": i8, "u8": u8, "i16": i16, "u16": u16,
    "i32": i32, "u32": u32, "i64": i64, "u64": u64,
    "f32": f32, "f64": f64,
}


def from_numpy_dtype(dtype: np.dtype) -> ScalarType:
    """Convert a numpy dtype to a Tack scalar type."""
    dtype = np.dtype(dtype)
    if dtype not in _numpy_to_tack:
        raise TypeError(f"Unsupported numpy dtype: {dtype}")
    return _numpy_to_tack[dtype]


def as_dtype(dtype):
    """A Tack scalar or element type for anything a dtype argument accepts.

    Tack types pass through. Anything NumPy reads as a dtype maps to the
    matching scalar type, following NumPy: ``np.float32``, ``"float32"``,
    ``np.dtype("int64")``, and ``float`` (which NumPy reads as float64).
    """
    if isinstance(dtype, (ScalarType, ElementType)):
        return dtype
    try:
        numpy_dtype = np.dtype(dtype)
    except TypeError:
        raise TypeError(
            f"dtype must be a Tack type (tack.f32, tack.vec3(), ...) or a NumPy dtype, "
            f"not {dtype!r}") from None
    return from_numpy_dtype(numpy_dtype)


@dataclass(frozen=True)
class ElementType:
    """A fixed-size vector or matrix element, independent of an array's shape.

    Components are stored together, without padding, in row-major order.
    ``scalar_dtype`` is the component type; ``shape`` describes one element.
    """

    scalar_dtype: ScalarType
    shape: tuple[int, ...]

    def __post_init__(self):
        if not isinstance(self.scalar_dtype, ScalarType):
            raise TypeError("an element's component dtype must be a Tack scalar type")
        if len(self.shape) not in (1, 2) or any(
                isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in self.shape):
            raise ValueError("an element needs one or two positive integer dimensions")
        if len(self.shape) == 2 and max(self.shape) > 4:
            raise ValueError("matrix elements are at most 4x4")

    @property
    def size(self):
        return int(np.prod(self.shape))

    def __repr__(self):
        if len(self.shape) == 1:
            return f"tack.vec({self.shape[0]}, {self.scalar_dtype})"
        return f"tack.mat({self.shape[0]}, {self.shape[1]}, {self.scalar_dtype})"


@lru_cache(maxsize=None, typed=True)
def vec(n: int, dtype: ScalarType = f32) -> ElementType:
    """The type of an n-component vector; no storage is allocated."""
    return ElementType(dtype, (n,))


@lru_cache(maxsize=None, typed=True)
def mat(rows: int, columns: int, dtype: ScalarType = f32) -> ElementType:
    """The type of a row-major matrix element, up to 4x4.

    In a kernel, ``tack.mat(row, ...)`` builds a matrix from its rows instead.
    """
    return ElementType(dtype, (rows, columns))


def vec2(dtype: ScalarType = f32) -> ElementType:
    """A two-component vector element type; in a kernel, ``tack.vec2(...)`` builds one.

    In kernels it takes 2 scalars, one scalar for every component, or one
    2-vector.
    """
    return vec(2, dtype)


def vec3(dtype: ScalarType = f32) -> ElementType:
    """A three-component vector element type; in a kernel, ``tack.vec3(...)`` builds one.

    In kernels it takes 3 scalars, one scalar for every component, or one
    3-vector.
    """
    return vec(3, dtype)


def vec4(dtype: ScalarType = f32) -> ElementType:
    """A four-component vector element type; in a kernel, ``tack.vec4(...)`` builds one.

    In kernels it takes 4 scalars, one scalar for every component, or one
    4-vector.
    """
    return vec(4, dtype)


def mat2(dtype: ScalarType = f32) -> ElementType:
    """A 2x2 matrix element type; in a kernel, ``tack.mat2(...)`` builds one.

    In kernels it takes 2 rows (lists of scalars or vectors) or 4
    scalars in row-major order.
    """
    return mat(2, 2, dtype)


def mat3(dtype: ScalarType = f32) -> ElementType:
    """A 3x3 matrix element type; in a kernel, ``tack.mat3(...)`` builds one.

    In kernels it takes 3 rows (lists of scalars or vectors) or 9
    scalars in row-major order.
    """
    return mat(3, 3, dtype)


def mat4(dtype: ScalarType = f32) -> ElementType:
    """A 4x4 matrix element type; in a kernel, ``tack.mat4(...)`` builds one.

    In kernels it takes 4 rows (lists of scalars or vectors) or 16
    scalars in row-major order.
    """
    return mat(4, 4, dtype)
