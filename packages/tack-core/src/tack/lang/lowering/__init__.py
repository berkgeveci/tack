"""The parts of lowering (Python AST to Tack IR) that `KernelTransformer` is built from.

`tack.lang.ast_transform.KernelTransformer` walks a kernel's AST and emits
IR. Its methods are grouped by job into mixins here, each in its own
module, with the state it owns set up by its own ``_init_...`` method:
`values` holds the lowering-time value types they share. The transformer
keeps the core ``visit_*`` methods and the expression machinery every
mixin uses (``visit``, ``_capture_value``, ``_source_error``, ...).
"""
