# From Python to a Tack program

A Tack application contains two kinds of code. **Host code** is ordinary Python:
it allocates fields, prepares data, chooses which kernels to run, repeats timesteps,
and displays the results. **Device code** is the body of a `@tack.kernel` or
`@tack.func`: Tack reads that source and compiles its supported numerical operations
for the selected backend. On the CPU, device code still means compiled code.

Understanding that boundary makes it much easier to turn a NumPy calculation into
a program that runs on a GPU.

## Start with one operation per element

Suppose you want `out = alpha * x + y`. NumPy expresses the whole array operation
at once. A Tack kernel expresses what happens to one element and provides the
iteration space:

```python
import numpy as np
import tack

tack.init(arch=tack.cpu)
x = tack.field_like(np.arange(8, dtype=np.float32))
y = tack.field_like(np.ones(8, dtype=np.float32))
out = tack.field(dtype=tack.f32, shape=(8,))

@tack.kernel
def saxpy(x, y, out, alpha):
    for i in range(x.shape[0]):
        out[i] = alpha * x[i] + y[i]

saxpy(x, y, out, 2.0)
np.testing.assert_array_equal(out.to_numpy(), np.arange(8, dtype=np.float32) * 2 + 1)
```

The allocation, call and assertion run in Python. The arithmetic in `saxpy` runs
as compiled code. The loop has eight independent iterations. Tack may schedule
them in any order; the program is correct because iteration `i` owns `out[i]`.

## A call describes a whole launch

A kernel has one parallel `for` directly in its body. Nested loops run sequentially
within an iteration. For a row sum, assign one iteration to a row and let that
iteration walk its columns:

```python
@tack.kernel
def row_sums(matrix, sums, rows, columns):
    for row in range(rows):
        total = 0.0
        for column in range(columns):
            total += matrix[row, column]
        sums[row] = total
```

Here `total` belongs to one iteration. A local variable does not communicate
between threads. Sequential statement order inside an iteration is preserved;
it does not establish an order between rows.

For a serial recurrence, use `for _ in range(1)` with an ordinary loop inside it.
For multidimensional work, use `tack.ndrange`. See [Kernels](03-kernels.md) for
stepped ranges, interior ranges and variable-length inner loops.

## Separate stages into separate kernels

Consider particle motion: first compute every force from the old positions,
then update every position. Writing positions while other iterations still read
them mixes old and new state. Put the two stages in separate kernels:

```python
@tack.kernel
def compute_force(pos, force, n):
    for i in range(n):
        force[i] = -pos[i]                 # independent restoring force

@tack.kernel
def integrate(pos, vel, force, dt, n):
    for i in range(n):
        vel[i] += dt * force[i]
        pos[i] += dt * vel[i]

for step in range(100):                    # Python controls the simulation
    compute_force(pos, force, n)
    integrate(pos, vel, force, 0.01, n)
```

Tack dispatch is synchronous: a call completes before Python continues. The
second kernel therefore sees the completed force field. Host reads after the
call also see completed results. There is no public asynchronous launch/stream
API to manage in these examples.

A `tack.barrier()` synchronizes only a GPU workgroup. It cannot turn two stages
over a whole grid into one ordered launch. Use completed kernel calls for that.
The [memory chapter](13-memory-and-parallelism.md) explains stencil buffers,
scatter operations, and the cases where atomics are appropriate.

## Choose what changes between calls

There are three common ways to give device code a numerical value:

| Kind of value | How to supply it | Example |
|---|---|---|
| A coefficient changing each call | Scalar argument | `advance(pos, vel, dt, n)` |
| A fixed model parameter | `tack.constant` | `GRAVITY = tack.constant(9.8)` |
| An array of state | Field argument or data-oriented attribute | Particle positions, a grid, a counter |

An ordinary Python global such as `gravity = 9.8` is not captured in device code.
Declare a constant or pass it. Changing a runtime argument changes the next call
without changing the kernel source. Constants are fixed when the specialization
is first lowered; rebinding a Python name later does not update cached code.

`@tack.data_oriented` gathers fields and parameters into an object. Its instance
scalars are runtime arguments; its class constants and method structure help
define the compiled specialization. The [Templates](06-templates.md) chapter
shows how to use this to organize a solver.

## Shapes, values and compilation

The first call includes compilation. Later compatible calls reuse the compiled
variant. Changing a scalar value usually reuses it. Changing the scalar's dtype,
field dtype, vector width, template structure or a baked-in dimension may create
another variant.

This distinction matters for a grid. A 1-D length used only as the launch bound
can vary without recompilation; multidimensional indexing uses row strides that
depend on the shape. Two grids with different shapes may need different code
even when the Python kernel is identical. Prefer fixed working fields during a
simulation; allocate once and update them each frame.

See [Kernel Caching](03-kernels.md#kernel-caching) for examples and
[Debugging and timing](15-debugging-and-timing.md) for how to measure warm calls.

## The kernel language uses Python syntax

Many familiar constructs work: arithmetic, `if`, sequential `for` and `while`,
vectors, matrices and device-function calls. Arbitrary Python objects, NumPy
calls and file operations do not run inside a kernel. For example, generate an
initial state with NumPy on the host, upload it, then work on the field:

```python
rng = np.random.default_rng(42)
initial = rng.normal(size=1024).astype(np.float32)  # Python / NumPy
data = tack.field_like(initial)
```

Use `tack.random` for random draws within a kernel. Use `@tack.func` for reusable
compiled helpers. Such helpers can be called from kernels and other device
functions, rather than as ordinary host functions.

Write complete examples in a `.py` file: Tack needs the decorated function's
source to compile it. Dynamically generated functions and interactive contexts
that cannot supply their source are poor starting points for learning the API.

Before tackling a larger program, run [Heat diffusion](tutorials/heat.md). It
combines allocation, a multidimensional kernel, a Python timestep loop and a
NumPy reference in one small script.
