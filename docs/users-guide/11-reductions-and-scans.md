# Reductions and Scans

Tack has four ways to combine many values into a few:

| Family | Entry points | Where it runs |
|---|---|---|
| [Field reductions](#field-reductions) | `field.sum()`, `.min()`, `.max()`, `.mean()` | Called from Python; the device or NumPy, depending on backend and dtype |
| [Statistics](#statistics) | `var`, `std`, `norm`, `absmax`, `count_nonzero`, `dot`, `histogram` in `tack.algorithms` | Called from Python; kernels with atomic accumulators on the active backend |
| [Prefix scans](#prefix-scans) | `exclusive_scan`, `inclusive_scan` (plus `copy`, `fill_value`) in `tack.algorithms` | Called from Python; a sequence of kernels on the active backend |
| [Block reductions](#block-reductions-inside-kernels) | `tack.block_sum`, `block_min`, `block_max` | Inside a kernel; GPU backends only |

Every family runs on all five backends except block reductions, which need
a workgroup execution model and are rejected on CPU. All of them return to
Python only after the work has finished: dispatch is synchronous.

The behavior below is stated per backend. Where it is the current
implementation rather than a promise, the page says so; the
[reduction contract](../reference/language-contract.md#field-and-parallel-reductions)
is the normative text.

## Field reductions

```python
import numpy as np
import tack

tack.init(arch=tack.cuda)

data = tack.field(dtype=tack.f32, shape=(1024,))
data.from_numpy(np.linspace(-1.0, 1.0, 1024, dtype=np.float32))

total = data.sum()       # Python float
lo = data.min()
hi = data.max()
average = data.mean()    # data.sum() / data.size
```

Each method reduces **every element** of the field, whatever its shape; a
reshaped field reduces the same elements, and there is no axis argument.
The result is always a Python `float`, including for integer fields.

**Where it runs.** CPU always reduces with NumPy. Metal, CUDA, HIP and Level
Zero reduce `f32` fields on the device with a 256-lane tree per group and an
atomic combine of the group results. Every other dtype is copied to the
host with `to_numpy()` and reduced with NumPy. Two further host fallbacks
exist for `f32`: Metal reduces fields of 2^32 or more elements with NumPy
over its shared buffer, and Level Zero uses NumPy when the device cannot
run 256-lane groups.

**Accumulator and result.**

| Input dtype | `sum()` accumulates in | `min()` / `max()` |
|---|---|---|
| `f32` | `f32` | exact `f32` value |
| `f64` (not Metal) | `f64` | exact `f64` value |
| `i8` … `i64` | `i64`, wrapping modulo 2^64 | exact integer value |
| `u8` … `u64` | `u64`, wrapping modulo 2^64 | exact integer value |

The accumulated value is then converted to a Python `float`, so integer
results beyond 2^53 can round. `mean()` divides the returned sum by
`field.size` in Python (binary64); it does not use a wider accumulator, and
an integer mean inherits any wrapping of the integer sum.

**NaN, infinities and zeros.** A NaN anywhere makes `sum()`, `min()` and
`max()` return NaN. Sums combine infinities by ordinary floating addition:
`inf + 1` is `inf`, `inf + -inf` is NaN. `min()` of `0.0` and `-0.0` is
`-0.0`, `max()` is `0.0`, whatever their order.

**Empty fields.** `sum()` returns `0.0`, `mean()` returns NaN, and `min()`
and `max()` raise `ValueError` ("min reduction requires a nonempty field").
None of these launches anything. Whether a zero-element field can be
allocated at all depends on the backend: CUDA's allocator refuses it with a
`RuntimeError`.

**Run-to-run repeatability.**

| | CPU | Metal / CUDA / HIP / Level Zero |
|---|---|---|
| `f32` `sum()` / `mean()` | NumPy | Low bits can differ between calls: group results are combined atomically in whatever order the groups finish |
| other dtypes, `sum()` / `mean()` | NumPy | NumPy |
| `min()` / `max()` | Order-independent | Order-independent |
| integer `sum()` | Exact (with wrapping) | Exact (with wrapping) |

The NumPy paths currently give the same answer on every call, but the
contract does not promise that floating sums are reproducible on any
backend. The accuracy that *is* promised is an absolute error budget,
stated in the
[reduction contract](../reference/language-contract.md#field-and-parallel-reductions).

!!! note "What crosses to the host"
    On a GPU backend an `f32` reduction reads back one value. Any other
    dtype copies the whole field to the host first, on every call.

## Statistics

```python
from tack.algorithms import var, std, norm, absmax, count_nonzero, dot, histogram

v = var(data)                         # population variance
s = std(data)                         # sqrt(var(data))
l1 = norm(data, ord=1)                # sum of |x|
l2 = norm(data)                       # ord=2: Euclidean norm
linf = norm(data, ord=float("inf"))   # max |x|
m = absmax(data)                      # same as norm(data, ord=float("inf"))
nz = count_nonzero(data)
d = dot(data, data)
counts, edges = histogram(data, bins=50, range=(-1.0, 1.0))
```

These functions live in the `tack.algorithms` package, which `import tack`
does not load. Import them as above, or `import tack.algorithms` before
writing `tack.algorithms.var(...)`.

Each one allocates a one-element accumulator, launches a kernel whose
iterations combine into it with `tack.atomic_add` or `tack.atomic_max`, and
reads the accumulator back with `to_numpy()`. `var` and `std` also call
`data.sum()`, and `histogram` without `range` calls `data.min()` and
`data.max()`, so they inherit the field-reduction host traffic described
above.

### What each function computes

| Function | Accumulator | Returns | Notes |
|---|---|---|---|
| `var(data, n=None)` | `f32`, `atomic_add` | NumPy `float32` scalar | Mean is `data.sum() / n`, then Σ(x − mean)² / n |
| `std(data, n=None)` | as `var` | Python `float` | `math.sqrt(var(data, n))` |
| `norm(data, ord=2, n=None)` | `f32`; `atomic_add` for `ord` 1 and 2, `atomic_max` for infinity | Python `float` | `ord` must equal `1`, `2` or `float("inf")`; anything else raises `ValueError: Unsupported norm order` before any launch |
| `absmax(data, n=None)` | `f32`, `atomic_max`, starting at `0.0` | Python `float` | |
| `count_nonzero(data, n=None)` | `i32`, `atomic_add` | Python `int` | Tests `x != 0.0`: `-0.0` counts as zero, NaN as nonzero. Wraps past 2^31 − 1 |
| `dot(a, b, n=None)` | `f32`, `atomic_add` | Python `float` | Σ a[i]·b[i] |
| `histogram(data, bins=10, range=None, n=None)` | `i32` per bin, `atomic_add` | `(counts, edges)`: an `i32` **field** of `bins` counts, still on the device, and a NumPy `float64` array of `bins + 1` edges | See below |

**`n`** is the number of elements to read, in flat (row-major) order, so
multi-dimensional fields work. `n=None` means `data.size` (for `dot`,
`a.size`). Every function checks `n` against each field it reads and
raises `ValueError` when `n` is negative or larger than a field, so
`dot(a, b)` with a shorter `b` is refused rather than reading past it.

With an `n` smaller than the field, everything covers the first `n`
elements only: `var` and `std` take the mean of those elements, and
`histogram` without `range` takes its range from them. To do that they copy
the first `n` elements into a temporary field and reduce it with
`Field.sum()`, `min()` and `max()`, so the mean and range follow the
[field-reduction rules](#field-reductions).

### Dtypes on each backend

Every input dtype the backend can allocate works. The accumulators are
always `f32` or `i32`, which are inside every backend's
[atomic domain](../reference/language-contract.md#atomic-field-updates),
so no backend rejects a statistics call because of the input dtype —
Metal and Level Zero, which have no 64-bit atomics, included.

| Input | CPU | CUDA / HIP | Metal | Level Zero |
|---|---|---|---|---|
| `f32` | supported | supported | supported | supported |
| `f64` | runs, accumulates in `f32` | runs, accumulates in `f32` | no `f64` fields | runs where the device has `f64`, accumulates in `f32`; elsewhere the field cannot be allocated |
| integers | runs | runs | runs | runs |

So an `f64` input does not give an `f64` result: each term is rounded to
`f32` before it is added. For example `dot` of `[1.0, 1e-9, 1e-9, 1e-9]`
with itself returns exactly `1.0`.

Integer inputs follow the kernel language's integer rules before the value
reaches the accumulator. Products in `dot` and `norm(ord=2)` are computed
in the input's integer type and wrap: for an `i32` field holding `100000`,
`dot(x, x)` is `1410065408.0`, not `1e10`. `var`, `norm(ord=1)` and
`absmax` subtract or negate in floating point and do not wrap.

### NaN, infinity and empty inputs

| Function | NaN in input | Infinity in input | `n == 0` |
|---|---|---|---|
| `var`, `std` | NaN | NaN | NaN, as `Field.mean()` of an empty field |
| `norm(ord=1)`, `norm(ord=2)`, `dot` | NaN | `inf`, or NaN when opposite infinities are added or, in `dot`, an infinity meets a zero | `0.0` |
| `norm(ord=inf)`, `absmax` | not supported (see below) | not supported (see below) | `0.0` |
| `count_nonzero` | counted as nonzero | counted | `0` |
| `histogram` | not supported (see below) | not supported (see below) | all-zero counts with an explicit `range`; `ValueError` without one |

`absmax` and `norm(ord=inf)` use atomic maximum, whose portable domain is
finite values: NaNs and infinities are outside it. In probes on CPU and
CUDA a NaN was skipped and an infinity gave `inf`, but no backend promises
that.
Use `field.max()` on `abs` values written by a kernel when NaNs must
propagate.

An empty field (`data.size == 0`) behaves like `n == 0`, where the backend
can allocate one.

### Histogram binning

The bin of each value is `int((x - lo) * (1 / width))`, computed in `f32`
(`f64` for an `f64` field), where `width = (hi - lo) / bins`, then clamped
into `[0, bins - 1]`:

- values below `lo` are counted in the **first** bin and values above `hi`
  in the **last** — unlike `numpy.histogram`, which drops them;
- a value equal to `hi` is counted in the last bin, as in NumPy;
- `range=None` uses the minimum and maximum of the first `n` elements; if
  they are equal, `hi` becomes `lo + 1`;
- a value whose offset `(x - lo) / width` does not fit a 32-bit integer, and
  a NaN, is outside the kernel language's float-to-integer domain. Its bin is
  backend-dependent: a value of `1e12` with `range=(0, 1)` lands in the first
  bin on CPU and in the last on CUDA. Filter such values first.

`bins` must be at least 1 (`bins=0` raises `ZeroDivisionError`), and
`range` is not checked: `lo > hi` produces meaningless counts. Because the
bin index is computed in `f32` while `edges` is `float64`, a value lying
exactly on an interior edge can land on either side of it. `counts` stays
on the device; call `counts.to_numpy()` to read it.

### Accuracy and repeatability

Every term is added into a single `f32` value, in whatever order the
iterations reach the atomic. Two consequences:

- **The low bits can change from run to run** for `var`, `std`,
  `norm(ord=1)`, `norm(ord=2)` and `dot`, on CPU when it uses worker
  threads as well as on GPUs. `count_nonzero`, `histogram` counts,
  `absmax` and `norm(ord=inf)` on finite inputs are exact and repeatable.
- **Error grows with `n`.** These are not compensated or tree sums. In a
  2^20-element `dot` of normally distributed `f32` values, the relative error
  against a `float64` reference was about 4·10⁻⁴. When accuracy matters,
  write the terms into a field with a kernel and call `field.sum()`, whose
  error budget is stated in the contract.

## Prefix scans

```python
from tack.algorithms import exclusive_scan, inclusive_scan

counts = tack.field(dtype=tack.i32, shape=(n,))
offsets = tack.field(dtype=tack.i32, shape=(n,))
# ... a kernel fills counts ...

total = exclusive_scan(counts, offsets, n)  # offsets[i] = counts[0] + ... + counts[i-1]
total = inclusive_scan(counts, offsets, n)  # offsets[i] = counts[0] + ... + counts[i]
```

The classic use is stream compaction: one kernel counts how many outputs
each item produces, an exclusive scan turns the counts into write offsets,
and the returned total sizes the output field.

Both functions run on every backend. They use no shared memory or
barriers, only ordinary kernels: an up-sweep and a down-sweep with doubling
and halving strides (a Blelloch-style scan), about 2·log₂ n launches in
all. Each launch is synchronous, so for small arrays — up to roughly a
million elements — copying to NumPy, calling `np.cumsum` and copying back
can be faster.

**Return value.** Both return the sum of the first `n` inputs as a Python
`int`. It is read by a one-element kernel that copies the last inclusive
sum into a one-element field, so only four bytes come back to the host,
not the whole output.

**Behavior.**

- The supported dtype is `i32` for both fields. Sums wrap at 32 bits, and
  so does the returned total.
- The input is not modified (unless it is also the output). Both functions
  work **in place**: `exclusive_scan(f, f, n)` and `inclusive_scan(f, f, n)`
  are correct.
- Only the first `n` elements are read and written; later output elements
  are left alone. `n` must be at most the size of both fields; a larger or
  negative `n` raises `ValueError`. `n = 0` writes nothing and returns `0`,
  the empty sum.
- `exclusive_scan` allocates an `n`-element `i32` work buffer per call.
- Integer results are exact (modulo wrapping) and identical on every
  backend and every run.

With other dtypes the two functions are not symmetrical, so keep both
fields `i32`. `exclusive_scan` copies the input into its `i32` work buffer,
truncating floats and wrapping wider integers, then converts the results
on store into the output dtype. `inclusive_scan` scans in the output
field's own dtype. In both, the returned total is read through an `i32`:
an `inclusive_scan` of `i64` values summing to `2**40 + 1` writes the right
output but returns `1`.

### Copy and fill utilities

```python
from tack.algorithms import copy, fill_value

copy(src, dst, n)          # dst[i] = src[i] for i < n
fill_value(dst, 0, n)      # dst[i] = 0 for i < n
```

Both are single kernels on the active backend and return `None`. They work
on the first `n` elements in flat order and do not check `n` against either
field.

- `copy` converts on store into `dst`'s dtype with the kernel language's
  rules: floats truncate toward zero (`2.7` → `2`, `-2.7` → `-2`) and must
  fit the target; integers wrap. Unlike `field.copy()`, it writes into an
  existing field rather than allocating one.
- `fill_value` passes `value` as a kernel scalar argument (a Python float is
  `f32`, or `f64` when `dst` is `f64`; an int is `i32`, or `i64`/`u64` by
  magnitude) and converts it on store: `2.5` fills an `i32` field with `2`,
  and `300` fills a `u8` field with `44`. Unlike `field.fill(value)`, which
  fills the whole field through a host-side NumPy array, it fills a prefix
  with a kernel.
- `copy_with_offset(src, dst, dst_offset, n)` writes
  `dst[dst_offset + i] = src[i]`. It is not re-exported from
  `tack.algorithms`; write `from tack.algorithms.copy import copy_with_offset`.
  (The attribute `tack.algorithms.copy` is the `copy` function, not the
  submodule, so `tack.algorithms.copy.copy_with_offset` does not work.)

## Block reductions inside kernels

`tack.block_sum(v)`, `tack.block_min(v)` and `tack.block_max(v)` combine one
`f32` value from every lane of a 256-lane workgroup and give **every lane**
the result. Use them to reduce within a kernel without a separate pass:

```python
@tack.kernel
def group_sums(data, partial, n):
    for i in range(n):
        s = tack.block_sum(data[i])     # every lane of the group gets the same s
        if tack.thread_id() == 0:
            partial[i // 256] = s       # one lane writes the group's result

n = 1 << 20                             # a multiple of 256
data = tack.field(dtype=tack.f32, shape=(n,))
partial = tack.field(dtype=tack.f32, shape=(n // 256,))
group_sums(data, partial, n)
total = partial.sum()
```

With `for i in range(n)`, group `g` holds iterations `256*g` to
`256*g + 255`, and `tack.thread_id()` is the lane's position in it.

**Backends.** Metal, CUDA, HIP and Level Zero. CPU has no workgroups and
rejects any kernel that contains a block reduction, even in a branch that
never runs: calling it raises `RuntimeError` ("Kernel 'group_sums' failed
on CPUBackend: CPU backend does not support workgroup execution required
by block_sum, thread_id ..."), and `tack.inspect` raises
`NotImplementedError`. Check `get_backend().supports_workgroups` to choose
a path, and use the field reductions or statistics above on CPU.

**Rules.** The [workgroup contract](../reference/language-contract.md#workgroups-and-synchronization)
has the full conservative domain; in short:

- **`f32` only.** The argument must be `f32` and the result is `f32`.
  Anything else raises `TypeError` ("block_sum requires f32 input; use an
  explicit f32 cast") on the first call. Write `tack.block_sum(tack.f32(x))`
  for an integer or `f64` value.
- **Complete groups.** A positive iteration count must be a multiple of
  256, including stepped ranges and the total of a `tack.ndrange`.
  Otherwise the call raises `ValueError` ("... require complete 256-lane
  groups; iteration count 1000 would create a partial workgroup") before
  anything runs. A count of zero runs nothing. Tack does not pad; pad the
  data to a multiple of 256 with the reduction's identity (`0.0` for a sum,
  `inf` for a minimum, `-inf` for a maximum).
- **Every lane participates.** The reduction must be reached by all lanes
  of the group: not under a condition that depends on the loop index, a
  field value or `thread_id()`, not after a lane-dependent `break` or
  `continue`, and not in a conditional expression, the second operand of
  `and`/`or`, or a `while` condition. Conditions on scalar arguments and
  on earlier block-reduction results are fine. Violations raise
  `ValueError` naming the kernel and IR location. To reduce a subset,
  have every lane contribute, using the identity where it should not count:
  `tack.block_sum(x if x > 0.0 else 0.0)`.
- The generated reduction includes its own barriers, including one after
  the result is read, so no extra `tack.barrier()` is needed around it.

**Values.** `block_min` and `block_max` follow the field-reduction rules: a
NaN in the group gives NaN, and zero ties give `-0.0` for the minimum and
`0.0` for the maximum. `block_sum` adds in a tree whose order may vary, with
the same error budget as field sums over the group's terms. Combining the
group sums with `tack.atomic_add` adds in arrival order, so such a total
can change in its low bits from run to run.
