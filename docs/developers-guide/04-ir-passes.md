# IR Passes

PGC runs several IR passes between AST transformation and codegen. Each
pass walks the IR tree and mutates it in place.

## Pass Order

```
1. ir_resolve      — Replace IRDimSize with constants, set texture shapes, resolve shared_like
2. type_inference   — Annotate params with types from actual arguments
3. check_dispatch_types — Validate field dtypes against backend capabilities
4. ir_optimize      — LICM, copy propagation, CSE
5. ir_type_annotate — Annotate all expression nodes with dtype (ScalarType)
6. ir_pack_scalars  — Group scalar params into field buffers (GPU only)
```

## IR Resolve (`ir_resolve.py`, 139 lines)

Replaces `IRDimSize(field_name, dim)` with `IRConstant(shape[dim])` using
the actual field shapes passed at call time. Also sets
`IRTextureSample.shape` from the `Texture3D` object.

This pass runs early because subsequent passes (optimization, codegen) need
concrete sizes.

## Type Inference (`type_inference.py`, 81 lines)

Annotates each `IRParam` with:
- `type_annotation` — a `ScalarType` (f32, i32, i64, etc.)
- `_is_field` — True if the argument is a `Field` or `Texture3D`
- `_is_texture` — True if the argument is a `Texture3D`

Type rules:
- `Field` → dtype of the field
- `Texture3D` → dtype of the underlying field
- Python `float` or `np.floating` → `f32` by default; auto-promotes to `f64` if any field argument uses `f64`
- Python `int` or `np.integer` → `i32` (auto-promotes to `i64` if value > 2^31)

## Dispatch-Time Type Checking (`type_inference.py`)

`check_dispatch_types()` validates that all field argument dtypes are
supported by the target backend. This runs after type inference in every
backend's `execute()` method.

Each backend defines its supported dtypes (e.g., Metal excludes `f64`).
Unsupported dtypes produce a clear `TypeError` naming the kernel, parameter,
dtype, and backend.

## IR Optimize (`ir_optimize.py`, 539 lines)

Three sub-passes run in sequence:

### Loop-Invariant Code Motion (LICM)

Hoists `IRAssign` nodes out of loops when their RHS depends only on values
defined outside the loop. This is critical after `@pgc.func` inlining —
inlined function bodies often re-load field values every iteration that
could be loaded once.

Algorithm:
1. Collect all variables assigned inside the loop body
2. For each assignment, check if its RHS references only variables defined
   outside the loop (parameters, or variables assigned before the loop)
3. Move qualifying assignments before the loop

### Copy Propagation

Resolves chains of `a = b` assignments by replacing references to `a` with
`b`. This is common after `@pgc.func` inlining, which creates parameter
assignments like `__func_x_0__ = x`.

Handles field alias propagation: if `a = x` where `x` is a field parameter,
subsequent `a[i]` loads are rewritten to `x[i]`.

### Common Subexpression Elimination (CSE)

Deduplicates identical `IRFieldLoad` expressions within a basic block. Two
loads are considered identical if they read from the same field at the same
index (structurally compared). The second load is replaced with a reference
to the first load's result variable.

## IR Type Annotate (`ir_type_annotate.py`)

Walks the IR and annotates **every expression node** with a `dtype` attribute
(a `ScalarType`). This also sets `_resolved_type` on each `IRAssign`.

After this pass, codegen backends read `node.dtype` directly instead of
reimplementing type inference heuristics. This eliminated ~40 lines of
duplicated `_infer_c_type` / `_infer_expr_type` logic per codegen backend.

Key rules:
- `IRConstant(3.14)` → `f32`, `IRConstant(42)` → `i32`, `IRConstant(2**31)` → `i64`
- `IRFieldLoad` → element type of the field
- `IRBinOp` → `promote_types(left, right)` (f64 > f32 > i64 > i32)
- `IRCast(value, ScalarType)` → the target ScalarType
- `IRCall("sqrt", ...)` → `f32` (or `f64` if any arg is f64)
- `IRCall("abs", [int_arg])` → preserves integer type
- `IRCall("min"/"max", ...)` → promoted type of arguments
- `IRCompare`, `IRBoolOp` → `i32`
- `IRName` referencing a field param → `None` (field pointers aren't scalars)

The pass tracks a type environment (`var_name → ScalarType`) and propagates
types through assignment chains.

## Scalar Packing (`ir_pack_scalars.py`, 204 lines)

GPU backends only. Groups scalar parameters by type into packed field
buffers to reduce buffer binding count (critical for Metal's 31-binding
limit).

Algorithm:
1. Collect all params where `_is_field == False`, grouped by type
2. Create new `IRParam` for each group: `__pack_f32__`, `__pack_i32__`, etc.
3. Rewrite `IRName("scalar_param")` → `IRFieldLoad(IRName("__pack_f32__"), IRConstant(idx))`
4. Remove original scalar params, append pack params

The pass runs on a `deepcopy` of the IR (to preserve the cached original)
and stores `pack_info` metadata for the dispatch layer. Pack field buffers
are allocated once and cached alongside the compiled kernel — subsequent
calls just update the scalar values via `from_numpy`.

`ScalarType.__deepcopy__` returns `self` to preserve singleton identity
through the deep copy (type map lookups rely on object identity).
