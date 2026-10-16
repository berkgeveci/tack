"""The parts of lowering (Python AST to Tack IR) that `KernelTransformer` is built from.

`tack.lang.ast_transform.KernelTransformer` walks a kernel's AST and emits
IR. Its methods are grouped by job into mixins, one module each, each
setting up the state it owns in its own ``_init_...`` method:

- `inlining` — device functions inlined at call sites (`Inlining`);
- `vectors` — vector and matrix values, scalarized (`VectorValues`);
- `arrays` — indexing, and the elements of vector and structure-of-arrays
  arrays (`ArrayAccess`);
- `loops` — the parallel loop and sequential loops (`Loops`);
- `checks` — checks on what lowering emitted (`LoweringChecks`);
- `local_arrays` — shared and local arrays (`LocalArrays`);
- `bounds` — debug mode's bounds checks (`BoundsChecks`);
- `values` — the lowering-time value types they share (`VectorValue`,
  `TupleValue`).

The transformer keeps the ``visit_*`` walk and the expression machinery
every mixin uses (``visit``, ``_capture_value``, ``_source_error``, ...).
A mixin reaches another's state and methods through ``self``; each one's
docstring says what it owns and what it reads.
"""
