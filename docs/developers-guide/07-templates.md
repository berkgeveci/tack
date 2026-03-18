# Template System

The template system enables zero-cost polymorphism: the same kernel source
compiles to different GPU code depending on the concrete types passed.

## Components

Three pieces work together:

1. **`@pgc.data_oriented`** (`data_oriented.py`, 29 lines) — marks a class
   as passable to kernels. Collects `@pgc.func` methods into
   `cls._pgc_func_methods`.

2. **`@pgc.func`** (`func.py`, 53 lines) — captures a function's AST for
   inlining. Non-method functions go into the global `_func_registry`.
   Class methods are stored on the class by `@pgc.data_oriented`.

3. **Template rewrite** (`template_rewrite.py`, 207 lines) — AST pre-pass
   that resolves template parameters before IR transformation.

## Template Rewrite Flow

When `backend.execute()` detects a `@pgc.data_oriented` argument:

```python
# In cpu.py / metal.py / etc.:
template_args = _detect_template_args(kernel, args)
# → {0: ("grid", grid_obj)}  (param index → (name, object))

effective_args = _expand_template_args(args, template_args)
# → removes template objects, inserts their field attributes

ir_module = kernel.get_ir(..., template_args=template_args)
# → calls template_rewrite.rewrite_templates() before AST transform
```

### What rewrite_templates Does

Given a kernel:

```python
@pgc.kernel
def avg(cs: pgc.template(), data, out, n):
    for c in range(n):
        total = 0.0
        for v in range(cs.points_per_cell):
            total = total + data[cs.get_point_id(c, v)]
        out[c] = total / float(cs.points_per_cell)
```

Called with `CellSetStructured3D(nx=50, ny=50, nz=50)`:

1. **Remove template param** from signature: `cs` is removed
2. **Add field params**: `cs.connectivity` (if it exists) becomes a new
   parameter `__tmpl_CellSetStructured3D_connectivity_<id>__`
3. **Replace scalar attribute access**: `cs.points_per_cell` →
   `ast.Constant(8)` (baked in)
4. **Replace method calls**: `cs.get_point_id(c, v)` → inline the
   method body with `self` references resolved

### Attribute Classification

`classify_template_attrs(obj)` splits an object's attributes:

```python
scalars = {"nx": 50, "ny": 50, "points_per_cell": 8, ...}  # int/float
fields = {"connectivity": <Field>}                           # pgc.Field
```

Scalars become AST constants. Fields become kernel parameters.

## @pgc.func Inlining

When the AST transformer encounters a `@pgc.func` call, it runs
`_inline_func_call()`:

1. **Generate unique suffix**: `__{func_name}_{param}_{counter}__`
2. **Build rename map**: callee param names → unique names
3. **Deep-copy callee AST** and rename all variables
4. **Propagate metadata**: vector variables, texture fields
5. **Emit parameter assignments**: `__func_x_0__ = caller_arg`
   (skipped for texture params — they reference the original field)
6. **Visit renamed body** to produce IR statements
7. **Return result variable** as `IRName`

The inlined statements are collected in `_pre_stmts` and hoisted before
the statement containing the call.

### Multi-Return Inlining

For functions returning tuples:

```python
@pgc.func
def minmax(a, b):
    if a < b:
        return a, b
    return b, a
```

The inliner detects `n_returns > 1` and creates multiple result variables:
`__minmax_ret_0_0__`, `__minmax_ret_1_0__`. Return statements become
assignments to these variables.

### Texture Propagation

When a texture field is passed through `@pgc.func`, the inliner must:

1. **Propagate `_texture_fields`**: renamed param → same shape metadata
2. **Track origin**: `_texture_origin[renamed] = original_param_name`
3. **Skip assignment**: texture params can't be assigned to pointer variables
4. **Use original name**: `IRTextureSample.field_name` uses the original
   kernel param name (via `_texture_origin`), not the renamed inline name

### Local Array Propagation

When a `pgc.local_array` is passed to a `@pgc.func`, arrays can't be
assigned in C (`int arr2[8] = arr1;` is invalid). The inliner handles
this by aliasing the caller's array name directly:

1. **Before the renamer runs**, check if the caller arg is in `_shared_vars`
2. If so, override the rename map: `rename_map[param_name] = caller_arg_name`
3. The renamer substitutes the callee's param name with the caller's array
   name throughout the inlined body
4. **Skip the parameter assignment** (like textures)

This means the inlined body's `pts[v] = ...` compiles to `caller_array[v] = ...`
with no intermediate variable. The same mechanism works for `pgc.shared()`
arrays.

## Cache Keys

Each unique combination of template types produces a different compiled
kernel. The cache key includes:

- Kernel name and identity (`id(kernel)`)
- Type signature of all params
- Template class names and their scalar/field attribute values
- Vector field metadata
- Texture shapes

This means `avg(structured_cs, ...)` and `avg(explicit_cs, ...)` produce
two separately cached compilations with different generated code.
