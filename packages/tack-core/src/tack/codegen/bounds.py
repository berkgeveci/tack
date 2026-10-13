"""Debug mode's bounds checks in the C-family generators (CUDA, HIP, OpenCL, MSL).

One helper per kernel, ``__tack_in_bounds__``, tests an index against its
extent and, out of bounds, records the first violation in the errors field
(the layout of ``lang/ir_bounds.py``): an atomic add elects the first, which
stores the rest. An ``IRBoundsCheck`` is ``&&`` over its dimensions, so the
first failing dimension is the one recorded, then the flat index or -1.
"""

from tack.lang import ir_bounds

# qualifiers, the 64-bit integer, the errors pointer, the atomic add on it
_DIALECTS = {
    'cuda': ('__device__ inline', 'long long', 'int*', 'atomicAdd(errors, 1)'),
    'opencl': ('static inline', 'long', '__global int*',
               'atomic_fetch_add_explicit((volatile __global atomic_int*)errors, 1, '
               'memory_order_relaxed, memory_scope_device)'),
    'metal': ('inline', 'long', 'device int*',
              'atomic_fetch_add_explicit((volatile device atomic_int*)errors, 1, '
              'memory_order_relaxed)'),
}


def bounds_helpers(dialect: str, needed: bool) -> list:
    """The helper's definition, if a check was emitted."""
    if not needed:
        return []
    qualifiers, int64, pointer, add = _DIALECTS[dialect]
    b = ir_bounds
    return [
        f"{qualifiers} int __tack_in_bounds__({int64} index, {int64} extent, int site, "
        f"int dim, {pointer} errors) {{",
        "    if (index >= 0 && index < extent) return 1;",
        f"    if ({add} == 0) {{",
        f"        errors[{b.SITE}] = site;",
        f"        errors[{b.DIM}] = dim;",
        f"        errors[{b.INDEX_LO}] = (int)index;",
        f"        errors[{b.INDEX_HI}] = (int)(index >> 32);",
        f"        errors[{b.EXTENT_LO}] = (int)extent;",
        f"        errors[{b.EXTENT_HI}] = (int)(extent >> 32);",
        "    }",
        "    return 0;",
        "}",
        "",
    ]


def bounds_check_expr(dialect: str, node, expr) -> str:
    """An ``IRBoundsCheck`` as an expression; ``expr`` emits its children."""
    int64 = _DIALECTS[dialect][1]
    errors = expr(node.errors)
    tests = " && ".join(
        f"__tack_in_bounds__(({int64})({expr(index)}), ({int64})({expr(extent)}), "
        f"{node.site}, {dim}, {errors})"
        for dim, (index, extent) in enumerate(zip(node.indices, node.extents)))
    return f"(({tests}) ? ({int64})({expr(node.flat)}) : ({int64})-1)"
