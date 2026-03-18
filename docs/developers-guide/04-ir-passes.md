# IR Passes

PGC runs several IR passes between AST transformation and codegen. Each
pass walks the IR tree and mutates it in place.

## Pass Order

```
1. ir_resolve      — Replace IRDimSize with constants, set texture shapes
2. type_inference   — Annotate params with types from actual arguments
3. ir_optimize      — LICM, copy propagation, CSE
4. ir_type_annotate — Annotate IRAssign nodes with resolved types
5. ir_pack_scalars  — Group scalar params into field buffers (GPU only)
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
- Python `float` or `np.floating` → `f32` (not f64, for GPU compatibility)
- Python `int` or `np.integer` → `i32` (auto-promotes to `i64` if value > 2^31)

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

## IR Type Annotate (`ir_type_annotate.py`, 188 lines)

Walks the IR and annotates each `IRAssign` with a `_resolved_type`
attribute — a `ScalarType` that tells codegen what C type to emit for the
variable declaration.

This replaces the fragile `_infer_c_type` heuristic in codegens, which
previously defaulted unknown variables to `_INT` (causing bugs like the
tuple swap type mismatch).

Key rules:
- `IRConstant(3.14)` → `f32`, `IRConstant(42)` → `i32`
- `IRFieldLoad` → element type of the field
- `IRBinOp(float, int)` → `f32` (float wins)
- `IRCast("int", ...)` → `i32`
- `IRName` referencing a field param → `None` (field pointers can't be
  typed as scalars — codegen falls back to its own logic)

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
