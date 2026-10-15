# Debugging, numerical checks and timing

Start a new kernel with a small input and a result you can calculate independently.
Once it agrees, increase the problem size and change the backend. This makes
layout errors, unsupported operations and numerical differences much easier to
separate.

## Compare a complete calculation

The first check should include input upload, execution and output download:

```python
import numpy as np
import tack

tack.set_default_device("cpu")
rng = np.random.default_rng(42)
host_x = rng.normal(size=32).astype(np.float32)
x = tack.array(host_x)
out = tack.empty(host_x.shape)

@tack.kernel
def transform(x, out):
    for i in range(x.shape[0]):
        out[i] = sqrt(abs(x[i]))

transform(x, out)
actual = out.numpy()
expected = np.sqrt(np.abs(host_x))
assert np.isfinite(actual).all()
np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-7)
```

An independent reference should express the mathematics directly. Copying the
same indexing expressions into a second kernel can reproduce the same mistake.
For a stencil, a NumPy slice expression is a useful contrast with the kernel's
per-cell loads. For a surface, check that vertices lie near the analytic level
set and that connectivity indices are valid.

The tutorials include such checks, executable without image dependencies:

```bash
uv run python docs/examples/validate.py --arch cpu
uv run python docs/examples/validate.py --arch metal  # on an Apple GPU
```

These commands require the selected backend; they do not silently fall back to
another one. The CPU run checks the portable algorithms, while a GPU run also
executes N-body's shared-memory path.

## Choose a tolerance for the computation

Use exact comparisons for small integer results where arithmetic does not
overflow. Floating computations need a tolerance that accounts for precision,
operation order and the problem's sensitivity.

`assert_allclose` accepts an element when
`abs(actual - expected) <= atol + rtol * abs(expected)`. Absolute tolerance is
particularly useful around zero. Keep reference and input dtypes explicit; an
accidental `float64` reference may answer a different precision question.

Transcendentals can differ slightly across vendor libraries. Multiply-add
contraction can alter rounding. Parallel floating reductions and atomic scatters
can change summation order even between runs on one backend. A simulation may
amplify those differences over many steps. Check an early timestep as well as
the final result, and consider physical properties such as mass conservation,
residuals or bounded temperatures when a long trajectory is sensitive.

Do not increase a tolerance merely to hide a discrepancy. First check array
layout, boundaries, initialization, dtype conversions and cross-iteration races.
Use [Arrays and Types](02-arrays-and-types.md) for literal precision and integer
rules, and [Reductions and Scans](11-reductions-and-scans.md) for accumulator
precision. Statistics such as `dot` form their terms in `f64` if either input
is `f64`, and otherwise in `f32`, then use array reductions. For `f32` inputs,
a `float64` reference is useful for measuring accumulated error; it is not a
promise that the `f32` result will agree in every bit.

## Diagnose the failing stage

| Symptom | First thing to inspect |
|---|---|
| Unknown name in device code | Pass a runtime value or declare `tack.constant`; check the defining namespace |
| Array has unexpected values or shape | Compare NumPy shapes and dtype; distinguish vectors from flat components |
| Results change between runs | Conflicting stores, missing atomics, or floating atomic summation |
| Only boundary cells are wrong | Bounds and initialization of both timestep buffers: run in debug mode (below) |
| A crash, or garbage that moves with the input size | An index out of bounds: run in debug mode (below) |
| Kernel rejected on CPU | Workgroup primitives need a GPU; use a portable path or local arrays |
| Kernel rejected on Metal | Look for `f64`, unsupported atomic types, or a runtime array size |
| Every call seems slow | Compilation/specialization, allocation, readback after every launch, or many small launches |

`print` in a kernel is useful on CPU, CUDA and HIP. Limit it to a few iterations
and remove it when timing. Metal does not provide this printing facility. For
portable inspection, write diagnostic values to an array and read it after the
call.

## Check every index with debug mode

An index out of bounds is the usual cause of a kernel that crashes, or that
gives wrong values near the edges of an array. Kernels do not check indices,
so `a[i, j + 1]` at the last column quietly reads the first element of the
next row. Turn on debug mode to find it:

```python
tack.set_debug(True)    # or run with TACK_DEBUG=1 in the environment
```

Every array index is then checked, one dimension at a time. An
access out of bounds is skipped (a load gives 0) and the call raises
`IndexError` naming the first one:

```text
IndexError: Kernel 'blur': a[i, j + 1] at blur.py:14, column 19: index 64 of
dimension 1 is outside [0, 64). The access was skipped (a load gives 0).
63 more out-of-bounds accesses in the launch were skipped too.
```

An access in a `@tack.func` names the device function and its line. Debug
mode works on every backend and compiles kernels anew, so it is for finding
bugs, not for timing: each launch waits for its result.
`tack.set_debug(False)` turns it off again. `TACK_DEBUG` is read once, when
Tack is imported.

To inspect compilation without running the kernel:

```python
print(tack.inspect(transform, x, out, mode="ir"))
print(tack.inspect(transform, x, out, mode="source"))
```

Pass the actual argument types and shapes. Inspection uses them to prepare the
same specialization and applies the backend's capability checks. `mode="optimized"`
shows optimized LLVM IR on CPU only. See [Inspecting Generated Code](03-kernels.md#inspecting-generated-code).

## Measure first-call and warm time separately

On the CPU a call returns when the kernel has finished. On Metal, CUDA, HIP
and Level Zero it returns once the launch is queued, and the host waits only
when it reads an array or calls `tack.sync()`. Call `tack.sync()` before
reading the clock, or a GPU time measures only the launch. The first call
also includes compilation, unless an earlier process left the kernel in the
[disk cache](17-kernel-cache.md); run with `TACK_CACHE=off` to time a cold
compile. Record these separately:

```python
from time import perf_counter

@tack.kernel
def timed_transform(x, out):
    for i in range(x.shape[0]):
        out[i] = sqrt(abs(x[i]))

start = perf_counter()
timed_transform(x, out)                    # this kernel has not been called yet
tack.sync()                                # wait for a queued GPU launch
first_call = perf_counter() - start

for _ in range(5):
    timed_transform(x, out)                 # warm compilation and CPU scheduling
tack.sync()

times = []
for _ in range(30):
    start = perf_counter()
    timed_transform(x, out)
    tack.sync()
    times.append(perf_counter() - start)

print(f"First call: {first_call * 1e3:.3f} ms")
print(f"Warm median: {np.median(times) * 1e3:.3f} ms")
```

The 32-element diagnostic example is intentionally small; use a larger array
when studying throughput. Keep its shape and dtype fixed during measurement.
On CPU the scheduler measures whether worker threads help; several warm calls
allow that policy to settle. A single-thread run can help distinguish scheduling
cost from compiled arithmetic: `tack.set_default_device("cpu", num_threads=1)`.

Measure end-to-end application time separately, including upload, preprocessing,
kernel stages and readback. Allocate working arrays outside the timed loop when
studying kernel execution, and describe which costs a reported number includes.
There is no portable speedup factor for all problem sizes and machines.

## Improvements to try in order

Keep intermediate data in arrays, reuse allocations, swap timestep buffers, and
read back only at checkpoints. Combine independent per-element operations in one
kernel when that avoids launches and temporary arrays. Keep dependent global
stages separate. After measuring a clear memory bottleneck, consider tiling;
[N-body](tutorials/nbody.md) shows how to preserve both barriers and a CPU path.

Compare a change using identical inputs, backend, shape and measurement procedure.
Re-run the numerical check too. The repository's [performance page](../performance.md)
contains historical measurements, while the [design pages](../design/index.md)
explain specialization and CPU scheduling in more detail.
