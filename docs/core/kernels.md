# Kernels

A kernel is a Python function, decorated with `@tack.kernel`, whose outermost
loop runs in parallel on a device. This page teaches the parts of a kernel:
the parallel loop, loops over two or three dimensions, ordinary loops inside
it, array dimensions, constants, returning a value, and seeing the code Tack
generates. It is for readers who have run a first kernel and know
[how a program runs](how-programs-run.md). After it you can write the kernels
of a typical simulation or image calculation.

## The parallel loop

Every kernel has exactly one parallel loop. It is a `for` statement directly
in the kernel's body, and each of its iterations runs on its own GPU thread, or
in a share of the CPU's threads:

<!-- tack-check: parallel-loop -->
```python
import numpy as np
import tack

tack.set_default_device("cpu")

@tack.kernel
def fill(data, value):
    for i in range(len(data)):
        data[i] = value

data = tack.empty((1000,))
fill(data, 3.0)
assert np.all(data.numpy() == 3.0)
```

The loop's range sets how many iterations a launch runs. It can be a scalar
argument, `range(n)`, an array dimension, `range(data.shape[0])`, or
`len(data)`, the length of the first dimension. A start and a step work too, as
in `range(1, n - 1)` and `range(0, n, 2)`.

The loop must be at the top of the body, not inside an `if` or another loop.
A kernel with no parallel loop, or with two, raises `UnsupportedSyntaxError`
naming the line and column. Two independent loops belong in two kernels.

### Code before the loop

Statements before the parallel loop may run more than once per launch: once
per GPU thread, or once per share of the CPU's work. So they may only assign
local variables, read arrays and declare scratch arrays. Every iteration
starts from the values they set:

```python
@tack.kernel
def scale(x, out, k):
    s = 2.0 * k                    # computed before the loop, read by every iteration
    for i in range(x.shape[0]):
        out[i] = x[i] * s
```

A store to an array, an atomic or a `print` before or after the loop raises
`UnsupportedSyntaxError`, with the remedy: move it into the loop.

### Work that must run in order

Some calculations are a recurrence, where each value needs the one before it.
Run them in a parallel loop of one iteration, with an ordinary loop inside:

```python
@tack.kernel
def cumulative_trapezoid(rho, e, n, dx):
    for _ in range(1):                 # one iteration: the loop below runs in order
        e[0] = rho[0] * dx * 0.5
        for i in range(1, n):
            e[i] = e[i - 1] + (rho[i - 1] + rho[i]) * dx * 0.5
```

This uses one thread of the device, so keep it for short or rare work. A
running sum over a large array is a *scan*, which
[Reductions, scans and sorts](reductions-scans-sorts.md) computes in parallel.

## Loops over two or three dimensions

`tack.ndindex` makes a parallel loop over a grid. Each iteration gets one
index per dimension, the slowest first:

<!-- tack-check: ndindex -->
```python
import numpy as np
import tack

tack.set_default_device("cpu")

@tack.kernel
def smooth(u, out):
    for i, j in tack.ndindex((1, u.shape[0] - 1), (1, u.shape[1] - 1)):
        out[i, j] = 0.25 * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1])

u = tack.array(np.arange(16, dtype=np.float32).reshape(4, 4))
out = tack.zeros((4, 4))
smooth(u, out)
print(out.numpy())
# [[ 0.  0.  0.  0.]
#  [ 0.  5.  6.  0.]
#  [ 0.  9. 10.  0.]
#  [ 0.  0.  0.  0.]]
assert out.numpy()[1, 1] == 5.0 and out.numpy()[0, 0] == 0.0
```

Each argument of `ndindex` is a size, such as `u.shape[0]`, or a
`(start, end)` pair. The pairs here visit the interior of the grid and leave
the boundary alone, as a stencil needs. Sizes and pairs can be mixed. An empty
or reversed pair gives no iterations. The launch runs the product of the
extents, here 2 times 2.

`ndindex` takes two or three dimensions as one parallel loop. It writes the
index of a two-dimensional grid as `(row, column)`, matching NumPy's
`[row, column]` and its `(height, width)` shapes.

## Ordinary loops inside the parallel loop

Loops inside the parallel loop are ordinary loops. Each iteration runs them in
order, on its own:

```python
@tack.kernel
def row_sums(data, out):
    for row in range(data.shape[0]):          # parallel: one iteration a row
        total = 0.0
        for column in range(data.shape[1]):   # in order, within the row
            total += data[row, column]
        out[row] = total
```

An inner loop's bounds may be anything the iteration computes, including
values read from arrays. That is how a kernel walks lists of different
lengths, stored one after another with an offsets array, as VTK stores a
cell's points:

<!-- tack-check: segments -->
```python
import numpy as np
import tack

tack.set_default_device("cpu")

@tack.kernel
def sum_segments(offsets, data, sums):
    for c in range(sums.shape[0]):
        total = 0.0
        for i in range(offsets[c], offsets[c + 1]):
            total += data[i]
        sums[c] = total

offsets = tack.array([0, 3, 4, 8])                 # three lists: 3, 1 and 4 long
data = tack.array(np.arange(8, dtype=np.float32))
sums = tack.empty((3,))
sum_segments(offsets, data, sums)
assert np.allclose(sums.numpy(), [3.0, 3.0, 22.0])
```

Inner loops take `range` with a start, an end and a step, and `while`.
[Control flow and math](control-flow-and-math.md) covers `while`, `break` and
`continue`.

## Index arrays

An array takes one index per dimension, `grid[i, j]`, or a single flat index
in row-major order, `grid[k]`, as a NumPy array's `ravel()` would number it.
Indexing with another number of indices raises `TypeError` when the kernel is
called, naming the line. Without that check, `grid[i, j]` on a
three-dimensional array would silently read some other element.

A vector of integers supplies one index per component, so `grid[cell]` is
`grid[cell[0], cell[1]]` when `cell` is a 2-vector.
[Vectors and matrices](vectors-and-matrices.md) introduces vectors.

## Array dimensions inside a kernel

`a.shape[k]` and `len(a)` work anywhere in a kernel: in conditions, in
arithmetic and in indices, not only as the loop's range. The dimension's
number must be written as a literal, as in `x.shape[0]`. A variable `d` in
`x.shape[d]` raises `UnsupportedSyntaxError`, because the shape is read when
the kernel compiles.

```python
@tack.kernel
def reverse(x, out):
    for i in range(x.shape[0]):
        out[i] = x[x.shape[0] - 1 - i]
```

Where a dimension appears decides whether the kernel compiles again for
another shape. A dimension used only as the parallel loop's range is passed
to the launch, so one compiled kernel serves every length. A dimension used
inside the body is compiled in as a number. The `reverse` above therefore
compiles once per array length. A two-dimensional index such as
`grid[row, column]` uses the row length to find the element, so grids of
different widths compile separately too.

Compiling the dimension in lets the compiler simplify the arithmetic around a
known size, and for a simulation that keeps its arrays it costs nothing after
the first step. When the shapes change often, pass the length as a scalar
argument instead, which is a runtime value:

```python
@tack.kernel
def reverse(x, out, n):            # one compiled kernel for every n
    for i in range(n):
        out[i] = x[n - 1 - i]
```

## Declare constants

A kernel does not see the Python variables around it. If it did, a
module-level `dt = 0.01` could change after the kernel compiled, and the
kernel would silently keep the old value. Reading such a name raises
`NameError`. Declare a fixed value with `tack.constant` instead:

<!-- tack-check: constants -->
```python
import numpy as np
import tack

tack.set_default_device("cpu")

DT = tack.constant(0.01)
GRID = tack.constant(128)

@tack.kernel
def step(x, v):
    for i in range(GRID):
        v[i] = v[i] * (1.0 - DT)
        x[i] += v[i] * DT

x = tack.zeros((GRID,))                # the host uses the same names
v = tack.ones((GRID,))
step(x, v)
assert np.allclose(x.numpy(), 0.99 * 0.01)
```

The kernel reads `DT` as if it were the literal `0.01`. A constant is still a
Python number, so host code uses it too, as `tack.zeros((GRID,))` does.
Constants work from other modules (`from params import DT`) and inside
[device functions](device-functions.md). `math.pi`, `math.e` and `math.tau`
work without declaring them.

A second argument gives a constant a type. That matters for integer
arithmetic that must wrap at a width, and for double-precision values:

```python
MULTIPLIER = tack.constant(747796405, tack.u32)   # x * MULTIPLIER wraps at 32 bits
TENTH = tack.constant(0.1, tack.f64)              # exactly the double nearest 0.1
```

A tuple of numbers declares a vector constant, and a tuple of equal-length
tuples a matrix constant:

```python
SUN = tack.constant((0.5, 0.5, 0.0))
ROTATE = tack.constant(((0.0, -1.0), (1.0, 0.0)))
```

In a kernel they are a vector and a matrix. On the host they index like tuples,
and `np.asarray(SUN)` gives an array to compute with.

A constant is fixed where you declare it, and arithmetic on constants gives
plain numbers. So declare a derived value again, as in
`H = tack.constant(1.0 / (GRID - 2))`. Binding the name to another constant
later does not change a kernel already compiled. For a value that changes
between calls, pass an argument.

## Return a value

A kernel whose last statement is `return`, after the parallel loop, hands a
value back to Python. The return annotation names its type: `-> int` for
`i32`, `-> float` for `f32`, or any Tack type such as `-> tack.f64`.

<!-- tack-check: return-value -->
```python
import numpy as np
import tack

tack.set_default_device("cpu")

@tack.kernel
def count_above(values, count, threshold) -> int:
    for i in range(len(values)):
        if values[i] > threshold:
            tack.atomic_add(count, 0, 1)
    return count[0]

values = tack.array(np.array([0.1, 0.9, 0.7, 0.2], np.float32))
count = tack.zeros((1,), tack.i32)
live = count_above(values, count, 0.5)
assert live == 2 and isinstance(live, int)        # a Python int
```

The returned expression is evaluated once, after the loop has finished. It may
read arrays, arguments, constants and variables assigned before the loop. It
may not read a variable the loop assigns, which has one value per iteration.
The cost is one tiny extra launch and the transfer of one number.

`tack.atomic_add` adds to an element safely when many iterations add to the
same one. It returns the element's value before the addition, which gives each
iteration a slot of its own. That makes a common pattern: append the values
that pass a test to an output array, and return how many there are.

```python
--8<-- "docs/examples/kernel_results.py:compact"
```

Reset the counter before each call, and give the output room for every input:

```python
--8<-- "docs/examples/kernel_results.py:run"
```

The returned length is exact, but the order of the appended values is not
fixed, so compare them sorted. When the order matters, count, scan and write
instead, as [marching squares](../users-guide/tutorials/contours.md) does.
[Workgroups, atomics, textures and random numbers](workgroups-atomics-textures.md)
covers atomics in full.

## See the generated code

`tack.inspect` compiles a kernel for the arguments you give it and returns the
code as a string, without running it:

<!-- tack-check: inspect-code -->
```python
import tack

tack.set_default_device("cpu")

@tack.kernel
def saxpy(x, y, out, a):
    for i in range(len(x)):
        out[i] = a * x[i] + y[i]

x, y, out = tack.empty((1024,)), tack.empty((1024,)), tack.empty((1024,))
ir = tack.inspect(saxpy, x, y, out, 2.0, mode="ir")
print(ir)
# Function saxpy(x, y, out, a):
#   ParallelFor i in [0, DimSize(x, 0)):
#     out[i] = ((a * x[i]) + y[i])
assert "ParallelFor" in ir
source = tack.inspect(saxpy, x, y, out, 2.0, mode="source")
assert "fmul" in source                           # LLVM IR on the CPU
```

| Mode | Shows |
|---|---|
| `"ir"` | Tack's intermediate representation, after inlining |
| `"source"` | The code given to the device's compiler: LLVM IR on the CPU, Metal Shading Language, CUDA C, HIP C, or OpenCL C for Level Zero |
| `"optimized"` | LLVM IR after LLVM's optimizations. CPU only; other devices raise `ValueError` |

Pass the same arguments a call would get, because the types and dimensions
the code is compiled for come from them. Inspection makes the same checks as a
call, so arguments a call would refuse raise here too. The code is for the
device the call would run on, chosen from the arguments' arrays as a call
chooses. To see another device's code, pass arrays made on it, such as
`tack.empty((n,), device="cuda")`.

## Details

The exact rules of the kernel language are in the
[kernel language contract](../reference/language-contract.md): which Python
constructs are accepted, the order in which expressions are evaluated, and
what may and may not run outside the parallel loop. Two of them come up in
practice:

- A local variable assigned before the parallel loop and reassigned inside it
  gets a fresh copy in each iteration. Iterations never see each other's
  values.
- A `break` that leaves the parallel loop itself is refused, because the
  iterations are not in sequence. `break` and `continue` work in inner loops.

## Where to go next

- [Control flow and math](control-flow-and-math.md) covers conditions, inner
  loops and the math functions kernels can call.
- [Device functions](device-functions.md) shows how to factor a kernel's body
  into helpers.
- [Heat diffusion](../users-guide/tutorials/heat.md) is a complete stencil
  simulation built from these pieces.
