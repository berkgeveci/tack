# Kernels

## Parallel Loops

The outermost `for` loop in a kernel is the parallel loop — each iteration
maps to one GPU thread (or one chunk of work on CPU):

```python
@pgc.kernel
def fill(data, val, n):
    for i in range(n):
        data[i] = val
```

The parallel loop must be the first (and only top-level) loop. PGC uses it
to determine how many threads to launch.

The loop bound can come from:
- A scalar argument: `range(n)`
- A field shape: `range(data.shape[0])`
- `len(data)`

## Multi-Dimensional Iteration

Use `pgc.ndrange` for 2D or 3D parallel iteration:

```python
@pgc.kernel
def fill_2d(grid, width, height):
    for i, j in pgc.ndrange(width, height):
        grid[j * width + i] = float(i + j)
```

This launches `width * height` threads. Each thread gets its `(i, j)` pair
via index decomposition.

## Sequential Inner Loops

Loops nested inside the parallel loop run sequentially per thread:

```python
@pgc.kernel
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
@pgc.kernel
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

`pgc.inspect()` shows the generated code for a kernel without executing it.
This is useful for debugging, learning, and understanding what the compiler
produces:

```python
@pgc.kernel
def saxpy(x, y, out, a):
    for i in range(len(x)):
        out[i] = a * x[i] + y[i]

n = 1024
x = pgc.field(dtype=pgc.f32, shape=(n,))
y = pgc.field(dtype=pgc.f32, shape=(n,))
out = pgc.field(dtype=pgc.f32, shape=(n,))

# PGC intermediate representation
print(pgc.inspect(saxpy, x, y, out, 2.0, mode="ir"))

# Backend-specific source (MSL, CUDA C, LLVM IR, etc.)
print(pgc.inspect(saxpy, x, y, out, 2.0, mode="source"))

# Post-optimization LLVM IR (CPU backend only — runs LLVM O3 passes)
print(pgc.inspect(saxpy, x, y, out, 2.0, mode="optimized"))
```

The three modes are:

| Mode | Output |
|------|--------|
| `"ir"` | PGC intermediate representation |
| `"source"` | Backend source code: LLVM IR (CPU), MSL (Metal), CUDA C, HIP C, OpenCL C |
| `"optimized"` | Post-optimization LLVM IR on CPU (loop vectorization, unrolling, etc.) |

You must pass the same arguments the kernel would receive at runtime, since
type inference, dimension resolution, and template expansion all depend on
them. Templates and scalar arguments work as expected:

```python
@pgc.data_oriented
class Sim:
    GRAVITY = -9.81

    def __init__(self, n):
        self.pos = pgc.field(dtype=pgc.f32, shape=(n,))
        self.vel = pgc.field(dtype=pgc.f32, shape=(n,))
        self.dt = 0.01

    @pgc.func
    def step(self, i):
        self.vel[i] = self.vel[i] + self.GRAVITY * self.dt
        self.pos[i] = self.pos[i] + self.vel[i] * self.dt

@pgc.kernel
def update(sim):
    for i in range(len(sim.pos)):
        sim.step(i)

sim = Sim(1024)
print(pgc.inspect(update, sim))  # shows template expansion + inlined methods
```
