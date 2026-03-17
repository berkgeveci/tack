"""PGC type inference — resolves types for kernel parameters and expressions.

Type inference is performed at call time, when actual arguments are available.
This module infers scalar types for IR nodes based on the concrete argument types.
"""

import numpy as np

from pgc.lang import ir
from pgc.lang.field import Field, Texture3D
from pgc.lang.types import ScalarType, f32, f64, i32, i64, from_numpy_dtype


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
            param.type_annotation = f32
            param._is_field = False
            types.append(f32)
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


def infer_types(ir_module: ir.IRModule, args: tuple):
    """Run type inference on an IR module given concrete arguments.

    Mutates the IR by filling in type annotations on parameters.
    """
    if not ir_module.functions:
        return
    func = ir_module.functions[0]
    infer_param_types(func, args)


# Type promotion rules for binary operations
_PROMOTION_ORDER = {i32: 0, i64: 1, f32: 2, f64: 3}


def promote_types(a: ScalarType, b: ScalarType) -> ScalarType:
    """Return the promoted type for a binary operation between types a and b."""
    if a is b:
        return a
    rank_a = _PROMOTION_ORDER.get(a, -1)
    rank_b = _PROMOTION_ORDER.get(b, -1)
    if rank_a < 0 or rank_b < 0:
        raise TypeError(f"Cannot promote types: {a}, {b}")
    return a if rank_a >= rank_b else b
