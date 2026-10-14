# Adding a New Feature

This chapter walks through the process of adding a new IR node to Tack,
using `tack.local_array()` as the example. This was an actual feature
added in a single session.

## 1. Define the IR Node

Add a new class to `ir.py`:

```python
class IRLocalAlloc(IRNode):
    """Allocate a per-thread local array (private memory on GPU, stack on CPU)."""

    def __init__(self, name: str, dtype: str, size):
        self.name = name
        self.dtype = dtype   # "float", "int", etc.
        self.size = size     # IRNode expression for number of elements
```

Also add it to the `dump()` function for debugging:

```python
if isinstance(node, IRLocalAlloc):
    return f"{prefix}LocalAlloc {node.name}: {node.dtype}[{dump(node.size)}]"
```

## 2. Add the Python-Side API

In `__init__.py`, add a placeholder that raises at runtime (the actual
handling happens at AST level):

```python
def local_array(dtype, size):
    """Allocate a per-thread local array. Only usable inside a @tack.kernel."""
    raise RuntimeError("local_array() can only be used inside a @tack.kernel")
```

## 3. Handle in AST Transform

In `ast_transform.py`, add recognition in two places:

**Assignment form** (the normal usage):

```python
# In visit_Assign, after shared memory check:
if isinstance(target, ast.Name) and isinstance(value, ast.Call):
    local_result = self._try_parse_local_alloc(target.id, value)
    if local_result is not None:
        return local_result
```

**Standalone call error** (helpful message if used wrong):

```python
if func_name == "local_array":
    raise NotImplementedError(
        "tack.local_array() must be used in an assignment: "
        "arr = tack.local_array(tack.f32, 8)")
```

**Parser method** (mirrors `_try_parse_shared_alloc`):

```python
def _try_parse_local_alloc(self, target_name, value_node):
    # Parse tack.local_array(tack.f32, 8) → IRLocalAlloc
    name = self._resolve_call_name(value_node)
    if name != "local_array":
        return None
    # ... parse dtype, size ...
    self._shared_vars.add(target_name)  # treat as array for indexing
    return ir.IRLocalAlloc(name=target_name, dtype=c_type, size=size)
```

Note: adding to `_shared_vars` ensures that `arr[i]` is recognized as
array indexing (field load/store) rather than a scalar subscript.

## 4. Update IR Passes

Register the node's structural children in `ir_traversal.py`. Resolution,
packing, and copy substitution will then recurse into them automatically:

```python
ir.IRLocalAlloc: (('size', 'expr'),),
```

Add the node to the appropriate statement/expression set in `ir_verify.py`,
declare required metadata there, and enforce its stage postconditions.
Unknown nodes fail verification and traversal until explicitly supported.
Type annotation and control-flow-sensitive analyses still need semantic
handling. For an allocation, annotate the size and record the element type:

```python
if isinstance(node, ir.IRLocalAlloc):
    _annotate_expr(node.size, env, field_params)
    env[node.name] = node.dtype
    return
```

Test resolution, packing, copy substitution, and verification through the
new child slots, including malformed IR and missing required annotations.

## 5. Add to Each Codegen

### C-Like Backends (CUDA, MSL, OpenCL)

Simple — emit a C array declaration:

```python
# cuda_gen.py (also inherited by HIP):
elif isinstance(node, ir.IRLocalAlloc):
    self._emit(f"{node.dtype} {node.name}[{self._expr(node.size)}];")

# msl_gen.py:
elif isinstance(node, ir.IRLocalAlloc):
    self._emit(f"{node.dtype} {node.name}[{self._expr(node.size)}];")

# opencl_gen.py (overrides _emit_stmt):
elif isinstance(node, ir.IRLocalAlloc):
    self._emit(f"{node.dtype} {node.name}[{self._expr(node.size)}];")
```

### LLVM Backend

Use private stack allocation for local arrays on CPU. Shared memory requires
workgroup execution and is rejected by the CPU target:

```python
elif isinstance(node, ir.IRLocalAlloc):
    self._emit_local_alloc(node)
```

## 6. Write Tests

Test on all available backends:

```python
def _available_backends():
    backends = ["cpu"]
    for arch in ["metal", "cuda", "hip", "level_zero"]:
        try:
            tack.init(arch=arch)
            backends.append(arch)
        except (ImportError, RuntimeError, OSError):
            pass
    tack.init(arch="cpu")
    return backends

@pytest.fixture(params=_available_backends())
def backend(request):
    tack.init(arch=request.param)
    return request.param

def test_local_array_store_load(backend):
    @tack.kernel
    def use_local(out, n):
        for i in range(n):
            arr = tack.local_array(tack.f32, 4)
            arr[0] = 1.0
            arr[1] = 2.0
            out[i] = arr[0] + arr[1]
    # ...
```

A new test module for a new area joins the quick tier
(`pytestmark = pytest.mark.quick` at module level) if it runs in under ten
seconds; see "Tests, in three tiers" in `CLAUDE.md`. The suite runs with the
disk cache off (`TACK_CACHE=off`) unless the environment says otherwise, so
a test that counts compilations is not hidden by a cache hit.

## 7. What Else a Feature Can Touch

The IR node is most of the work, but several runtime mechanisms assume they
know everything that shapes the generated code:

- **Debug mode.** A node that indexes a field or array must have its index
  checked in debug mode: lowering wraps it through `_checked` in
  `ast_transform.py` (an `IRBoundsCheck`), `ir_bounds.guard_accesses`
  must skip the access when the check is negative, and every generator must
  emit the check. Run the new tests with `TACK_DEBUG=1` too.
- **Variant key and fast-path signature.** A new fact about the arguments
  that changes the generated code goes into the variant key, and the
  dispatch fast path's `_call_signature` must distinguish it as well, or a
  warm call reuses the wrong variant.
- **Disk cache.** What lowering reads from Python must be resolved through
  `CallBindings` or added to the lowered key (`disk_cache.lowered_key`).
  What compiled code depends on besides the IR and argument types (a
  device property, a compiler version or option) must be named by the
  backend's `cache_target()`; keep compiler options in the module constant
  the target reads.
- **Metal objects.** Every `new...` selector `metal.py` calls must be
  registered in `_OWNED_RESULTS` before it is first called, or PyObjC
  retains its result once too often and it is never freed.
- **Queued launches.** On Metal, CUDA, HIP and Level Zero a launch returns
  before it runs. Host code that reads device memory without going through
  a `DeviceBuffer` method, or hands it to another library, must call
  `tack.sync()` (or `Backend.synchronize()`) first.

## 8. Incidental Fixes

New features often expose pre-existing bugs. `tack.local_array` revealed
that reusing a loop variable name in sibling `for` loops caused
"undeclared identifier" errors on GPU backends due to C block scoping.
The fix (always re-declare loop variables in the for-header) was unrelated
to local arrays but was caught by the new tests.

## Summary Checklist

For any new IR node (all files under `packages/tack-core/src/tack/`):

- [ ] `lang/ir.py`: Node class + dump entry
- [ ] `__init__.py`: Python-side API placeholder
- [ ] `lang/ast_transform.py`: AST recognition + parser (and `_checked` for an indexed access)
- [ ] `lang/ir_traversal.py`: Structural children in `CHILD_FIELDS`
- [ ] `lang/ir_verify.py`: Statement/expression role, required metadata
- [ ] `lang/ir_resolve.py`: Recurse into child expressions
- [ ] `lang/ir_pack_scalars.py`: Recurse into child expressions
- [ ] `lang/ir_type_annotate.py`: Annotate dtype or skip
- [ ] `lang/ir_optimize.py`: Handle if it affects optimization
- [ ] `codegen/llvm_gen.py`: LLVM emission
- [ ] `codegen/cuda_gen.py`: CUDA C emission (inherited by HIP)
- [ ] `codegen/msl_gen.py`: MSL emission
- [ ] `codegen/opencl_gen.py`: OpenCL C emission (if different from CUDA)
- [ ] `lang/ir_bounds.py`: Guard the access in debug mode, if it indexes memory
- [ ] Tests on all backends (in `packages/tack-core/tests/`), with and without `TACK_DEBUG=1`; a new module marked quick if it is fast
