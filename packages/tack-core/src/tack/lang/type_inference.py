"""Tack type inference — resolves types for kernel parameters and expressions.

Type inference is performed at call time, when actual arguments are available.
This module infers scalar types for IR nodes based on the concrete argument types.
"""

import numpy as np

from tack.lang import ir
from tack.lang.field import Field, Texture3D
from tack.lang.types import ScalarType, i8, u8, i16, u16, i32, u32, i64, u64, f32, f64, from_numpy_dtype


def infer_param_types(ir_func: ir.IRFunction, args: tuple) -> list[ScalarType]:
    """Infer parameter types from actual call arguments.

    Returns a list of ScalarType for each parameter, based on the runtime
    arguments passed to the kernel.
    """
    if len(args) != len(ir_func.params):
        raise TypeError(
            f"Kernel '{ir_func.name}' expects {len(ir_func.params)} arguments, "
            f"got {len(args)}"
        )

    # Determine the float context from field arguments: if any field is f64,
    # Python float scalars should be f64 to avoid silent precision loss.
    float_context = f32
    for arg in args:
        if isinstance(arg, (Field, Texture3D)) and arg.dtype is f64:
            float_context = f64
            break

    types = []
    for param, arg in zip(ir_func.params, args):
        if isinstance(arg, Texture3D):
            param.type_annotation = arg.dtype
            param._is_field = True
            param._is_texture = True
            types.append(arg.dtype)
        elif isinstance(arg, Field):
            param.type_annotation = arg.dtype
            param._is_field = True
            types.append(arg.dtype)
        elif isinstance(arg, (float, np.floating)):
            param.type_annotation = float_context
            param._is_field = False
            types.append(float_context)
        elif isinstance(arg, (int, np.integer)):
            val = int(arg)
            if val > 2**31 - 1 or val < -(2**31):
                param.type_annotation = i64
                param._is_field = False
                types.append(i64)
            else:
                param.type_annotation = i32
                param._is_field = False
                types.append(i32)
        else:
            raise TypeError(
                f"Unsupported argument type for parameter '{param.name}': {type(arg)}"
            )
    return types


def check_dispatch_types(ir_func: ir.IRFunction, args: tuple,
                         supported_dtypes: set[ScalarType] | None = None,
                         backend_name: str = ""):
    """Validate field and scalar types at dispatch time.

    Checks:
    - Field dtypes are supported by the target backend
    - All arguments have valid Tack types

    Raises TypeError with a clear message if validation fails.
    """
    for param, arg in zip(ir_func.params, args):
        if isinstance(arg, (Field, Texture3D)):
            dtype = arg.dtype
            if supported_dtypes is not None and dtype not in supported_dtypes:
                raise TypeError(
                    f"Kernel '{ir_func.name}': parameter '{param.name}' has dtype "
                    f"{dtype}, which is not supported on {backend_name}. "
                    f"Supported dtypes: {', '.join(str(t) for t in sorted(supported_dtypes, key=lambda t: t.name))}"
                )


def infer_types(ir_module: ir.IRModule, args: tuple):
    """Run type inference on an IR module given concrete arguments.

    Mutates the IR by filling in type annotations on parameters.
    """
    if not ir_module.functions:
        return
    func = ir_module.functions[0]
    infer_param_types(func, args)


# Type promotion rules for binary operations
_PROMOTION_ORDER = {i8: 0, u8: 0, i16: 1, u16: 1, i32: 2, u32: 2, i64: 3, u64: 3, f32: 4, f64: 5}

# Unsigned types and their signed counterpart at the same width
_IS_UNSIGNED = {u8, u16, u32, u64}

# When mixing signed + unsigned of the same width, promote to the next wider signed type
_MIXED_SIGN_PROMOTE = {
    0: i16,   # i8 + u8 → i16
    1: i32,   # i16 + u16 → i32
    2: i64,   # i32 + u32 → i64
    3: i64,   # i64 + u64 → i64 (no wider signed type; best we can do)
}


def promote_types(a: ScalarType, b: ScalarType) -> ScalarType:
    """Return the promoted type for a binary operation between types a and b.

    When mixing signed and unsigned integers of the same width, promotes to
    the next wider signed type to avoid unsigned overflow surprises
    (e.g., u8 + i8 → i16, u32 + i32 → i64).
    """
    if a is b:
        return a
    rank_a = _PROMOTION_ORDER.get(a, -1)
    rank_b = _PROMOTION_ORDER.get(b, -1)
    if rank_a < 0 or rank_b < 0:
        raise TypeError(f"Cannot promote types: {a}, {b}")
    if rank_a == rank_b:
        # Same width — check for signed/unsigned mismatch
        a_unsigned = a in _IS_UNSIGNED
        b_unsigned = b in _IS_UNSIGNED
        if a_unsigned != b_unsigned:
            return _MIXED_SIGN_PROMOTE[rank_a]
        # Same signedness, same width but different types shouldn't happen
        # (caught by a is b above), but return the higher-ranked one
        return a
    return a if rank_a > rank_b else b
