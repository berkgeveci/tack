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

tack.init(arch=tack.cpu)
rng = np.random.default_rng(42)
host_x = rng.normal(size=32).astype(np.float32)
x = tack.field_like(host_x)
out = tack.field(dtype=tack.f32, shape=host_x.shape)

@tack.kernel
def transform(x, out):
    for i in range(x.shape[0]):
        out[i] = sqrt(abs(x[i]))

transform(x, out)
actual = out.to_numpy()
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

Do not increase a tolerance merely to hide a discrepancy. First check field
layout, boundaries, initialization, dtype conversions and cross-iteration races.
Use [Fields and Types](02-fields-and-types.md) for literal precision and integer
rules, and [Reductions and Scans](11-reductions-and-scans.md) for accumulator
precision. For example, a statistics `dot` currently accumulates in `f32` even
when its inputs are `f64`.

## Diagnose the failing stage

| Symptom | First thing to inspect |
|---|---|
| Unknown name in device code | Pass a runtime value or declare `tack.constant`; check the defining namespace |
| Field has unexpected values or shape | Compare NumPy shapes and dtype; distinguish vectors from flat components |
| Results change between runs | Conflicting stores, missing atomics, or floating atomic summation |
| Only boundary cells are wrong | Bounds and initialization of both timestep buffers |
| Kernel rejected on CPU | Workgroup primitives need a GPU; use a portable path or local arrays |
| Kernel rejected on Metal | Look for `f64`, unsupported atomic types, or a runtime array size |
| Every call seems slow | Compilation/specialization, allocation, readback, or small synchronous launches |

`print` in a kernel is useful on CPU, CUDA and HIP. Limit it to a few iterations
and remove it when timing. Metal does not provide this printing facility. For
portable inspection, write diagnostic values to a field and read it after the
call.

To inspect compilation without running the kernel:

```python
print(tack.inspect(transform, x, out, mode="ir"))
print(tack.inspect(transform, x, out, mode="source"))
```

Pass the actual argument types and shapes. Inspection uses them to prepare the
same specialization and applies the backend's capability checks. `mode="optimized"`
shows optimized LLVM IR on CPU only. See [Inspecting Generated Code](03-kernels.md#inspecting-generated-code).

## Measure first-call and warm time separately

Tack launches synchronously, so elapsed time around a completed call includes its
execution. The first call also includes compilation. Record these separately:

```python
from time import perf_counter

@tack.kernel
def timed_transform(x, out):
    for i in range(x.shape[0]):
        out[i] = sqrt(abs(x[i]))

start = perf_counter()
timed_transform(x, out)                    # this kernel has not been called yet
first_call = perf_counter() - start

for _ in range(5):
    timed_transform(x, out)                 # warm compilation and CPU scheduling

times = []
for _ in range(30):
    start = perf_counter()
    timed_transform(x, out)
    times.append(perf_counter() - start)

print(f"First call: {first_call * 1e3:.3f} ms")
print(f"Warm median: {np.median(times) * 1e3:.3f} ms")
```

The 32-element diagnostic example is intentionally small; use a larger field
when studying throughput. Keep its shape and dtype fixed during measurement.
On CPU the scheduler measures whether worker threads help; several warm calls
allow that policy to settle. A single-thread run can help distinguish scheduling
cost from compiled arithmetic: `tack.init(arch=tack.cpu, num_threads=1)`.

Measure end-to-end application time separately, including upload, preprocessing,
kernel stages and readback. Allocate working fields outside the timed loop when
studying kernel execution, and describe which costs a reported number includes.
There is no portable speedup factor for all problem sizes and machines.

## Improvements to try in order

Keep intermediate data in fields, reuse allocations, swap timestep buffers, and
read back only at checkpoints. Combine independent per-element operations in one
kernel when that avoids launches and temporary fields. Keep dependent global
stages separate. After measuring a clear memory bottleneck, consider tiling;
[N-body](tutorials/nbody.md) shows how to preserve both barriers and a CPU path.

Compare a change using identical inputs, backend, shape and measurement procedure.
Re-run the numerical check too. The repository's [performance page](../performance.md)
contains historical measurements, while the [design pages](../design/index.md)
explain specialization and CPU scheduling in more detail.
