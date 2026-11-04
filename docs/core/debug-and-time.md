# Debug and time a kernel

A new kernel can give wrong values, crash, or run slower than it should. The
first step is to check it against an independent reference, with a tolerance
that suits the comparison. Debug mode then finds an index out of bounds,
`print` shows values from inside a kernel, and the generated code shows what
Tack made of your source. Timing comes last, measured so that the number
means what you think.

The most common bug is an index one past the end of an array. Kernels do not
check their indices, so the read quietly takes a neighbouring value. Debug
mode checks every index and names the first bad one:

<!-- tack-check: debug-first -->
```python
import numpy as np
import tack

tack.set_default_device("cpu")


@tack.kernel
def forward_difference(a, out):
    for i in range(a.shape[0]):
        out[i] = a[i + 1] - a[i]           # reads a[n] at the last i


a = tack.array(np.arange(8, dtype=np.float32) ** 2)
out = tack.empty(8)

tack.set_debug(True)                       # or TACK_DEBUG=1 in the environment
try:
    forward_difference(a, out)
except IndexError as error:
    print(error)
    # Kernel 'forward_difference': a[i + 1] at ...: index 8 is outside [0, 8).
    # The access was skipped (a load gives 0).
    assert "a[i + 1]" in str(error) and "index 8 is outside [0, 8)" in str(error)
else:
    raise AssertionError("debug mode missed the bad index")
finally:
    tack.set_debug(False)
```

## Check a kernel against a reference

Start a new kernel with a small input and a result you can work out another
way. Once the two agree, grow the problem and change the device. Taking these
steps one at a time separates layout mistakes, unsupported operations and
numerical differences, which look alike when they arrive together.

The first check should include the whole calculation: copying the input in,
running the kernel, and reading the output back.

<!-- tack-check: debug-reference -->
```python
import numpy as np
import tack

tack.set_default_device("cpu")


@tack.kernel
def transform(x, out):
    for i in range(x.shape[0]):
        out[i] = sqrt(abs(x[i]))


host_x = np.random.default_rng(42).normal(size=32).astype(np.float32)
x = tack.array(host_x)
out = tack.empty(32)
transform(x, out)

actual = out.numpy()
expected = np.sqrt(np.abs(host_x))
assert np.isfinite(actual).all()
np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-7)
```

Write the reference so that it states the mathematics directly. A second
kernel with the same indexing would repeat the same mistake. For a stencil, a
NumPy slice expression makes a good contrast with the kernel's loads one cell
at a time. For a surface, check that its points lie near the level set they
should and that its connectivity holds valid indices.

Every tutorial has such a check, which runs without its figure:

```bash
uv run python docs/examples/validate.py --arch cpu
uv run python docs/examples/validate.py --arch metal    # on an Apple GPU
```

The `--arch` you name must start. The checks never fall back to another
device.

## Choose a tolerance

Compare small integer results exactly, where the arithmetic does not
overflow. Floating-point results need a tolerance that allows for their
precision, the order of their operations, and how sensitive the problem is.

`np.testing.assert_allclose` accepts an element when
`abs(actual - expected) <= atol + rtol * abs(expected)`. The absolute part
matters near zero, where a relative tolerance alone accepts nothing. Keep the
reference's element type explicit. An accidental `float64` reference answers a
different question than an `f32` kernel does. For `f32` inputs, a `float64`
reference measures the error that accumulates, but the `f32` result will not
match it in every bit.

Several things move the low bits legitimately:

- Math functions such as `sin` and `exp` differ slightly between vendors'
  libraries.
- A multiply and an add may be fused into one operation, which rounds once
  instead of twice.
- Parallel floating-point sums by atomics can add in a different order from
  one run to the next.

A simulation can amplify such differences over many steps. Check an early
step as well as the last, and check physical properties too, such as mass
conservation, a residual or a temperature that stays in bounds.

Do not raise a tolerance to hide a difference. First rule out the usual
causes: the array layout, the boundaries, initialization, conversions between
types, and iterations that race on one element. [Arrays and
types](arrays-and-types.md) covers the precision of literals and the integer
rules, and [Reductions, scans and sorts](reductions-scans-sorts.md) the
precision of sums.

## Find an index out of bounds with debug mode

An index out of bounds usually shows as a crash, as wrong values near the
edges of an array, or as garbage that moves when the input size changes. In
a two-dimensional array, `a[i, j + 1]` at the last column reads the first
element of the next row, so the result looks plausible.

Turn on debug mode with `tack.set_debug(True)`, or with `TACK_DEBUG=1` in the
environment. `TACK_DEBUG` is read once, when Tack is imported.
`tack.set_debug(False)` turns it off again.

In debug mode every array index is checked, one dimension at a time. An
access out of bounds is skipped, so a load gives 0 and a store does nothing.
After the launch, the call raises `IndexError` naming the first bad access,
its source line, the index and the bounds:

```text
IndexError: Kernel 'blur': a[i, j + 1] at blur.py:14, column 19: index 64 of
dimension 1 is outside [0, 64). The access was skipped (a load gives 0).
63 more out-of-bounds accesses in the launch were skipped too.
```

An access inside a device function names the function and its line. Debug
mode works on every device. It is for finding bugs, not for timing. Kernels
compile again with the checks, run slower, and every launch waits to finish.

## Print from a kernel

`print` works inside kernels on the CPU, CUDA and HIP. It is useful for a few
iterations at a time:

<!-- tack-check: debug-print -->
```python
import tack

tack.set_default_device("cpu")


@tack.kernel
def show(values):
    for i in range(values.shape[0]):
        if i < 2:
            print("value:", values[i])


show(tack.arange(8, dtype=tack.f32))
# value: 0.000000
# value: 1.000000
```

On Metal, `print` does nothing, since Metal has no `printf`. On CUDA and HIP
launches are queued, so the output may appear only when the host next waits
for the device. Remove `print` before timing. For a check that works on
every device, write diagnostic values to an array and read it after the call.

## Read the generated code

`tack.inspect` compiles a kernel for the given arguments and returns the code
as a string, without running it. Pass the real arguments, since their types
and shapes decide what is compiled:

<!-- tack-check: debug-inspect -->
```python
import tack

tack.set_default_device("cpu")


@tack.kernel
def transform(x, out):
    for i in range(x.shape[0]):
        out[i] = sqrt(abs(x[i]))


x = tack.ones(8)
out = tack.empty(8)
print(tack.inspect(transform, x, out, mode="ir"))
# Function transform(x, out):
#   ParallelFor i in [0, DimSize(x, 0)):
#     out[i] = sqrt(abs(x[i]))
source = tack.inspect(transform, x, out, mode="source")    # LLVM IR on the CPU
assert "define void" in source
```

There are three modes:

- `mode="ir"` shows Tack's own intermediate form, the same on every device.
- `mode="source"` shows what the device compiles: LLVM IR on the CPU, Metal
  Shading Language on Metal, CUDA C, HIP C, or OpenCL C on Level Zero.
- `mode="optimized"` shows LLVM IR after LLVM's optimizations. It works on
  the CPU only.

Inspection applies the device's checks, so a kernel the device would reject
is rejected here too, with the same message. [Kernels](kernels.md) says more
about what specializes a kernel.

## Diagnose a failure by its symptom

| symptom | look first at |
|---|---|
| A name in device code is unknown | pass the value as an argument or declare it with `tack.constant`; check the module the kernel was defined in |
| An array has unexpected values or shape | the NumPy shape and element type; whether it holds vectors or flat components |
| Results change between runs | two iterations storing to one element, a missing atomic, or floating-point sums by atomics |
| Only boundary cells are wrong | bounds, and the initialization of both time-step buffers: run in debug mode |
| A crash, or garbage that moves with the input size | an index out of bounds: run in debug mode |
| The kernel is rejected on the CPU | shared memory, barriers or `thread_id`, which need a GPU: see [workgroups](workgroups-atomics-textures.md#keep-a-path-for-the-cpu) |
| The kernel is rejected on Metal | `f64`, an atomic on an unsupported type, or an array size not known when compiling |
| Every call seems slow | compilation, allocation in the loop, a read-back after every launch, or many small launches |

## Time a kernel

Two things make a naive timing wrong. On a GPU, a kernel call returns once the
launch is queued, so the clock must wait with `tack.sync()`. And the first
call with new argument types compiles the kernel. Measure the first call and
warm calls separately:

<!-- tack-check: debug-timing -->
```python
from time import perf_counter

import numpy as np
import tack

tack.set_default_device("cpu")


@tack.kernel
def transform(x, out):
    for i in range(x.shape[0]):
        out[i] = sqrt(abs(x[i]))


x = tack.array(np.random.default_rng(0).normal(size=1_000_000).astype(np.float32))
out = tack.empty_like(x)                   # allocate outside the timed loop

start = perf_counter()
transform(x, out)                          # compiles, unless the disk cache has it
tack.sync()                                # wait for a queued GPU launch
first_call = perf_counter() - start

for _ in range(5):                         # let the CPU's threading settle
    transform(x, out)
tack.sync()

times = []
for _ in range(30):
    start = perf_counter()
    transform(x, out)
    tack.sync()
    times.append(perf_counter() - start)

print(f"first call {first_call * 1e3:.2f} ms, warm median {np.median(times) * 1e3:.3f} ms")
assert np.median(times) <= first_call
```

Keep the shape and element type fixed while you measure, since a new shape
can compile a new variant. A *variant* is one compiled form of a kernel, for
one set of argument types and shapes. On the CPU, the backend measures for
itself whether worker threads help, and several warm calls let that settle. A
run with one thread, `tack.set_default_device("cpu", num_threads=1)`,
separates the cost of the arithmetic from the cost of scheduling.

The first call in a process may find the kernel in Tack's disk cache rather
than compile it. To time a cold compile, run with `TACK_CACHE=off`. See
[Keep compiled kernels between runs](kernel-cache.md).

Measure the whole application separately, including the copies in,
preprocessing, every kernel stage and the read-back, and say which costs a
reported number includes. No speedup factor holds for every problem size and
machine.

## Make it faster, in order

Try these in order, and measure after each:

1. Keep intermediate data in Tack arrays, and read back only what you need,
   at checkpoints.
2. Allocate working arrays once and reuse them. Swap time-step buffers
   instead of copying them.
3. Combine independent operations on the same elements into one kernel, to
   save launches and temporary arrays. Keep stages that depend on each other's
   whole output in separate kernels.
4. Once a measurement shows that memory traffic is the limit, consider
   tiling with shared memory. The [N-body tutorial](../users-guide/tutorials/nbody.md)
   does this while keeping a path for the CPU.

Compare a change with identical inputs, device, shape and procedure, and run
the numerical check again. The [performance page](../performance.md) records
measurements of Tack against other tools.

## Details

- **Debug mode and the cache.** A kernel compiled in debug mode is a
  different kernel from the plain one. Switching modes never reuses the
  other's compiled code, in memory or on disk.
- **Metal source on disk.** On Metal, `TACK_DUMP_MSL=1` also writes each
  compiled kernel's source to `/tmp`. Other devices ignore it.
- **Compilation errors.** A kernel that fails to compile reports its name,
  the device, and the relevant lines of the compiler's message, without the
  whole generated source.

## Where to go next

- [Keep compiled kernels between runs](kernel-cache.md), so a program's
  first call is fast.
- [Memory and parallel correctness](memory-and-correctness.md), for the
  races behind results that change between runs.
- The [language contract](../reference/language-contract.md), for exactly
  which numerical results are guaranteed.
