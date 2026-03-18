# Adding a New Feature

This chapter walks through the process of adding a new IR node to PGC,
using `pgc.local_array()` as the example. This was an actual feature
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
    """Allocate a per-thread local array. Only usable inside a @pgc.kernel."""
    raise RuntimeError("local_array() can only be used inside a @pgc.kernel")
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
        "pgc.local_array() must be used in an assignment: "
        "arr = pgc.local_array(pgc.f32, 8)")
```

**Parser method** (mirrors `_try_parse_shared_alloc`):

```python
def _try_parse_local_alloc(self, target_name, value_node):
    # Parse pgc.local_array(pgc.f32, 8) → IRLocalAlloc
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

Each IR pass that walks the tree needs to handle the new node. For
`IRLocalAlloc`, the passes only need to recurse into the `size`
expression:

```python
# ir_resolve.py:
if isinstance(node, ir.IRLocalAlloc):
    node.size = _resolve(node.size, fields)
    return node

# ir_pack_scalars.py:
if isinstance(node, ir.IRLocalAlloc):
    node.size = _rewrite(node.size, replace_map)
    return node

# ir_type_annotate.py:
if isinstance(node, ir.IRLocalAlloc):
    return  # no type to annotate
```

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

Reuse the shared memory alloca pattern (both are stack allocations on CPU):

```python
elif isinstance(node, ir.IRLocalAlloc):
    self._emit_shared_alloc(node)  # same as shared on CPU
```

## 6. Write Tests

Test on all available backends:

```python
def _available_backends():
    backends = ["cpu"]
    for arch in ["metal", "cuda", "hip", "vulkan", "level_zero"]:
        try:
            pgc.init(arch=arch)
            backends.append(arch)
        except (ImportError, RuntimeError, OSError):
            pass
    pgc.init(arch="cpu")
    return backends

@pytest.fixture(params=_available_backends())
def backend(request):
    pgc.init(arch=request.param)
    return request.param

def test_local_array_store_load(backend):
    @pgc.kernel
    def use_local(out, n):
        for i in range(n):
            arr = pgc.local_array(pgc.f32, 4)
            arr[0] = 1.0
            arr[1] = 2.0
            out[i] = arr[0] + arr[1]
    # ...
```

## 7. Incidental Fixes

New features often expose pre-existing bugs. `pgc.local_array` revealed
that reusing a loop variable name in sibling `for` loops caused
"undeclared identifier" errors on GPU backends due to C block scoping.
The fix (always re-declare loop variables in the for-header) was unrelated
to local arrays but was caught by the new tests.

## Summary Checklist

For any new IR node:

- [ ] `ir.py`: Node class + dump entry
- [ ] `__init__.py`: Python-side API placeholder
- [ ] `ast_transform.py`: AST recognition + parser
- [ ] `ir_resolve.py`: Recurse into child expressions
- [ ] `ir_pack_scalars.py`: Recurse into child expressions
- [ ] `ir_type_annotate.py`: Handle or skip
- [ ] `ir_optimize.py`: Handle if it affects optimization
- [ ] `llvm_gen.py`: LLVM emission
- [ ] `cuda_gen.py`: CUDA C emission (inherited by HIP)
- [ ] `msl_gen.py`: MSL emission
- [ ] `opencl_gen.py`: OpenCL C emission (if different from CUDA)
- [ ] Tests on all backends
