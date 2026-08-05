# Kernels

## Parallel Loops

The outermost `for` loop in a kernel is the parallel loop — each iteration
maps to one GPU thread (or one chunk of work on CPU):

```python
@tack.kernel
def fill(data, val, n):
    for i in range(n):
        data[i] = val
```

The parallel loop must be the first (and only top-level) loop. Tack uses it
to determine how many threads to launch.

The loop bound can come from:
- A scalar argument: `range(n)`
- A field shape: `range(data.shape[0])`
- `len(data)`

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

Kernels are compiled on first call and cached by name and argument type
signature. Subsequent calls with the same types reuse the compiled kernel.
Changing scalar values (e.g., passing `alpha=2.5` then `alpha=3.0`) does
**not** trigger recompilation — only type changes do.

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
| `"optimized"` | Post-optimization LLVM IR on CPU (loop vectorization, unrolling, etc.) |

You must pass the same arguments the kernel would receive at runtime, since
type inference, dimension resolution, and template expansion all depend on
them. Templates and scalar arguments work as expected:

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
