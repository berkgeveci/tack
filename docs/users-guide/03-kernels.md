# Kernels

The [kernel language contract (draft)](../reference/language-contract.md)
distinguishes intended guarantees from current limitations and numerical
policies still being decided. Consult it when depending on ordering,
aliasing, or Python-compatible expression semantics.

## Parallel Loops

The outermost `for` loop in a kernel is the parallel loop — each iteration
maps to one GPU thread (or one chunk of work on CPU):

```python
@tack.kernel
def fill(data, val, n):
    for i in range(n):
        data[i] = val
```

A kernel has exactly one parallel loop, a `for` statement directly in its
body (not inside an `if` or `while`). Tack uses it to determine how many
threads to launch. A kernel without one, or with a second, is rejected with
an `UnsupportedSyntaxError` that gives the line and column.

Code before or after the parallel loop may run any number of times per
launch (once per GPU thread, once per CPU chunk), so it may only assign
local variables, read fields, and declare `tack.shared` or
`tack.local_array` arrays. Field stores, atomics, barriers, block
reductions and `print` there are rejected; move them into the loop body.
Each iteration starts from the values the code before the loop assigned:

```python
@tack.kernel
def scale(x, out, k):
    s = 2.0 * k                    # fine: a local, set up before the loop
    for i in range(x.shape[0]):
        out[i] = x[i] * s
```

The loop bound can come from:
- A scalar argument: `range(n)`
- A field shape: `range(data.shape[0])`
- `len(data)`

## Constants

A kernel is compiled from its own source and does not see Python variables
around it: a module-level `dt = 0.01` could change after the kernel was
compiled, and the kernel would silently keep the old value. Reading such a
name raises `NameError`. Declare the value with `tack.constant` instead:

```python
DT = tack.constant(0.01)
GRID = tack.constant(128)

@tack.func
def decay(v):
    return v * (1.0 - DT)

@tack.kernel
def step(x, v):
    for i in range(GRID):
        v[i] = decay(v[i])
        x[i] += v[i] * DT
```

The kernel reads `DT` as the literal `0.01`. The constant is still an
ordinary Python number, so host code uses the same name: `range(GRID)`,
`np.float32(DT)`, `tack.field(dtype=tack.f32, shape=(GRID,))`. Constants
work from another module too (`from params import DT`, or `params.DT`),
and inside device functions.

A second argument gives the constant a type, which matters in two places:

```python
MULTIPLIER = tack.constant(747796405, tack.u32)   # x * MULTIPLIER wraps at 32 bits
TENTH = tack.constant(0.1, tack.f64)              # the exact double, not a widened f32
```

Without it, a `u32` value times an integer literal is computed in `i64`.

The value is fixed where the constant is declared. Arithmetic on constants
gives plain numbers, so a derived value is declared again:
`H = tack.constant(1.0 / (GRID - 2))`. Binding the name to a different
constant after a kernel has been compiled does not change that kernel.
For values that change between calls, pass an argument. `math.pi`,
`math.e` and `math.tau` can be written directly.

## Field Dimensions

`field.shape[k]` and `len(field)` work anywhere in a kernel — not only as
the loop bound:

```python
@tack.kernel
def reverse(x, out):
    for i in range(x.shape[0]):
        out[i] = x[x.shape[0] - 1 - i]

@tack.kernel
def normalize_rows(a, out):
    for i, j in tack.ndrange(a.shape[0], a.shape[1]):
        out[i, j] = a[i, j] / float(a.shape[1])
```

The dimension index must be a literal: `x.shape[0]` is fine, `x.shape[d]`
with a runtime `d` is not.

One thing to know about compilation. A dimension used in the loop bound is
passed to the launch, so one compiled kernel serves every array length.
A dimension used *inside* the kernel becomes part of the generated code,
so Tack compiles a separate version per shape. That is usually what you
want — it lets the compiler fold and vectorize around a known size. If you
would rather not specialize, pass the length as a scalar argument:

```python
@tack.kernel
def reverse(x, out, n):     # one compiled kernel for every n
    for i in range(n):
        out[i] = x[n - 1 - i]
```

## Multi-Dimensional Iteration

Use `tack.ndrange` for 2D or 3D parallel iteration:

```python
@tack.kernel
def fill_2d(grid, width, height):
    for i, j in tack.ndrange(width, height):
        grid[j * width + i] = float(i + j)
```

This launches `width * height` threads. Each thread gets its `(i, j)` pair
via index decomposition.

An argument can also be a `(start, end)` pair, which is how a stencil
visits the interior of a grid and leaves the boundary alone:

```python
@tack.kernel
def smooth(u, out, n, m):
    for i, j in tack.ndrange((1, n - 1), (1, m - 1)):
        out[i, j] = 0.25 * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1])
```

Sizes and pairs can be mixed. An empty or reversed pair makes the loop
run no iterations.

## Sequential Inner Loops

Loops nested inside the parallel loop run sequentially per thread:

```python
@tack.kernel
def reduction_per_row(data, out, width, height):
    for row in range(height):          # parallel (one thread per row)
        total = 0.0
        for col in range(width):       # sequential (per thread)
            total = total + data[row * width + col]
        out[row] = total
```

Sequential loops support:
- `range(n)` with runtime bounds (including field loads)
- `range(start, end)`
- `range(start, end, step)`

### Variable-Length Inner Loops

The loop bound can come from a field load, enabling variable-length
iteration patterns like Viskores-style connectivity:

```python
@tack.kernel
def sum_segments(offsets, data, output, n_cells):
    for c in range(n_cells):
        total = 0.0
        for i in range(offsets[c], offsets[c + 1]):
            total = total + data[i]
        output[c] = total
```

## Kernel Caching

Kernels are compiled on first call. A compiled *variant* is reused by later
calls that agree on everything the generated code depends on: argument
dtypes, which arguments are fields, scalars or textures, vector widths,
texture extents, `@tack.data_oriented` class constants and layout, any field
dimensions the kernel bakes in (for example `x.shape[1]` used as a row
stride), and on CPU whether the fields overlap. Anything else is a runtime
parameter.

So changing scalar values (passing `alpha=2.5`, then `alpha=3.0`) does
**not** recompile, and neither does changing the length of a 1-D field
used only as the parallel loop bound. Changing a dtype, a vector width or a
baked-in dimension does. If a kernel recompiles for every array size, pass
the size as a scalar argument instead of reading it from `shape` inside the
body. [Specialization and Caching](../design/specialization-and-caching.md)
lists exactly what goes into the key, with worked examples.

## Inspecting Generated Code

`tack.inspect()` shows the generated code for a kernel without executing it.
This is useful for debugging, learning, and understanding what the compiler
produces:

```python
@tack.kernel
def saxpy(x, y, out, a):
    for i in range(len(x)):
        out[i] = a * x[i] + y[i]

n = 1024
x = tack.field(dtype=tack.f32, shape=(n,))
y = tack.field(dtype=tack.f32, shape=(n,))
out = tack.field(dtype=tack.f32, shape=(n,))

# Tack intermediate representation
print(tack.inspect(saxpy, x, y, out, 2.0, mode="ir"))

# Backend-specific source (MSL, CUDA C, LLVM IR, etc.)
print(tack.inspect(saxpy, x, y, out, 2.0, mode="source"))

# Post-optimization LLVM IR (CPU backend only — runs LLVM O3 passes)
print(tack.inspect(saxpy, x, y, out, 2.0, mode="optimized"))
```

The three modes are:

| Mode | Output |
|------|--------|
| `"ir"` | Tack intermediate representation |
| `"source"` | Backend source code: LLVM IR (CPU), MSL (Metal), CUDA C, HIP C, OpenCL C |
| `"optimized"` | Post-optimization LLVM IR (loop vectorization, unrolling, etc.). CPU only; other backends raise `ValueError` |

You must pass the same arguments the kernel would receive at runtime, since
type inference, dimension resolution, and template expansion all depend on
them. Inspection applies the same checks as a call, so arguments a call
would reject (an `f64` field on Metal, for example) raise here too, and
on GPUs without texture hardware the source shows the software sampling
the call would use. Templates and scalar arguments work as expected:

```python
@tack.data_oriented
class Sim:
    GRAVITY = -9.81

    def __init__(self, n):
        self.pos = tack.field(dtype=tack.f32, shape=(n,))
        self.vel = tack.field(dtype=tack.f32, shape=(n,))
        self.dt = 0.01

    @tack.func
    def step(self, i):
        self.vel[i] = self.vel[i] + self.GRAVITY * self.dt
        self.pos[i] = self.pos[i] + self.vel[i] * self.dt

@tack.kernel
def update(sim):
    for i in range(len(sim.pos)):
        sim.step(i)

sim = Sim(1024)
print(tack.inspect(update, sim))  # shows template expansion + inlined methods
```
