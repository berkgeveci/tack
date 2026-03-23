"""PGC type system — scalar types and type annotations for kernels."""

import numpy as np


class ScalarType:
    """Represents a scalar data type for fields and kernel arguments."""

    def __init__(self, name: str, numpy_dtype: np.dtype, llvm_name: str, bits: int):
        self.name = name
        self.numpy_dtype = numpy_dtype
        self.llvm_name = llvm_name
        self.bits = bits

    def __repr__(self):
        return f"pgc.{self.name}"

    def __deepcopy__(self, memo):
        # ScalarTypes are singletons — preserve identity through deepcopy
        return self


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

# Map numpy dtypes back to pgc types
_numpy_to_pgc = {
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
    """Convert a numpy dtype to a PGC scalar type."""
    dtype = np.dtype(dtype)
    if dtype not in _numpy_to_pgc:
        raise TypeError(f"Unsupported numpy dtype: {dtype}")
    return _numpy_to_pgc[dtype]


class TemplateType:
    """Marker for template (generic) kernel arguments — resolved at call time."""
    pass


def template() -> TemplateType:
    """Type annotation for kernel arguments that are resolved at call time."""
    return TemplateType()
