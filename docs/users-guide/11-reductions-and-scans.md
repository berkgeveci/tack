# Reductions and Scans

Tack has five ways to combine many values into a few:

| Family | Entry points | Where it runs |
|---|---|---|
| [Array reductions](#array-reductions) | `array.sum()`, `.min()`, `.max()`, `.mean()` | Called from Python; the array's device or NumPy, depending on backend and dtype |
| [Statistics](#statistics) | `var`, `std`, `norm`, `absmax`, `count_nonzero`, `dot`, `histogram` in `tack.algorithms` | Called from Python; terms formed on the inputs' device and reduced as arrays are |
| [Prefix scans](#prefix-scans) | `exclusive_scan`, `inclusive_scan` (plus `copy`, `fill_value`) in `tack.algorithms` | Called from Python; a sequence of kernels on the inputs' device |
| [Sorting and segmented reductions](#sorting-and-segmented-reductions) | `argsort`, `sort_by_key`, `gather`, `unique`, `reduce_by_key` in `tack.algorithms` | Called from Python; a sequence of kernels on the inputs' device |
| [Segmented operations](#segmented-operations) | `segment_reduce`, `segment_scan`, `segment_broadcast`, `segment_ids` in `tack.algorithms` | Called from Python; a sequence of kernels on the inputs' device |
| [Block reductions](#block-reductions-inside-kernels) | `tack.block_sum`, `block_min`, `block_max` | Inside a kernel; GPU backends only |

Every family runs on all five backends except block reductions, which need
a workgroup execution model and are rejected on CPU. A call that returns a
Python value (an array reduction, a statistic, a scan's total) waits for the
work it reads. On GPU backends launches are queued, so a scan without its
total, a sort, or a kernel using block reductions may return before the
device has finished; the next read of an array waits for it (see
[When a call returns](08-backends.md#when-a-call-returns)).

The behavior below is stated per backend. Where it is the current
implementation rather than a promise, the page says so; the
[reduction contract](../reference/language-contract.md#array-and-parallel-reductions)
is the normative text.

## Array reductions

```python
import numpy as np
import tack

tack.set_default_device("cpu")   # or a configured GPU backend

data = tack.array(np.arange(8, dtype=np.float32))

total = data.sum()       # Python float
lo = data.min()
hi = data.max()
average = data.mean()    # data.sum() / data.size
assert (total, lo, hi, average) == (28.0, 0.0, 7.0, 3.5)
```

Each method reduces **every scalar component**, whatever the array's shape
or element dtype. A reshape preserves the result; there is no axis argument.
For a `(n,)` array with dtype `vec3(f32)`, `size` is `n`, but the reduction
includes `3 * n` components.
The result is always a Python `float`, including for integer arrays.

**Where it runs.** CPU always reduces with NumPy. Metal, CUDA, HIP and Level
Zero reduce `f32` arrays on the device with a 256-lane tree per group and an
atomic combine of the group results. Every other dtype is copied to the
host with `numpy()` and reduced with NumPy. Two further host fallbacks
exist for `f32`: Metal reduces arrays of 2^32 or more elements with NumPy
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
`array.scalar_size` in Python (binary64); it does not use a wider accumulator, and
an integer mean inherits any wrapping of the integer sum.

**NaN, infinities and zeros.** A NaN anywhere makes `sum()`, `min()` and
`max()` return NaN. Sums combine infinities by ordinary floating addition:
`inf + 1` is `inf`, `inf + -inf` is NaN. `min()` of `0.0` and `-0.0` is
`-0.0`, `max()` is `0.0`, whatever their order.

**Empty arrays.** `sum()` returns `0.0`, `mean()` returns NaN, and `min()`
and `max()` raise `ValueError` ("min reduction requires a nonempty array").
None of these launches anything. Every backend allocates a zero-element
array.

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
[reduction contract](../reference/language-contract.md#array-and-parallel-reductions).

!!! note "What crosses to the host"
    On a GPU backend an `f32` reduction reads back one value. Any other
    dtype copies the whole array to the host first, on every call.

The implicit arrays of `tack.arrays` reduce the same way, with the same
semantics: `tack.arrays.CountingArray(8).sum()` is 28. Their values are
computed into a temporary array first, which is then reduced as a stored
array is. The algorithms below also take implicit arrays as inputs, read
in their kernels without a temporary: `exclusive_scan(CountingArray(n),
offsets, n)` scans 0, 1, 2, ... without storing them.

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
counts, edges = histogram(data, bins=8, range=(0.0, 8.0))
```

These functions live in the `tack.algorithms` package, which `import tack`
does not load. Import them as above, or `import tack.algorithms` before
writing `tack.algorithms.var(...)`.

Each statistic forms its terms -- |x|, x², (x − mean)², a·b -- in a
temporary array of the inputs' precision, f64 if any input is f64 and
otherwise f32 (an integer input converts before any arithmetic), and
reduces that array with `Array.sum()` or `Array.max()`. So the
[array-reduction rules](#array-reductions) apply to them: their accuracy,
repeatability and host traffic are those of array reductions.
`count_nonzero` scans its 0/1 flags into `i64` on the device and reads only
the total. `histogram` bins with `tack.atomic_add`, which gives exact
counts in any order.

### What each function computes

| Function | Accumulator | Returns | Notes |
|---|---|---|---|
| `var(data, n=None)` | the data's precision, `Array.sum()` | Python `float` | Mean of the first `n` values, then Σ(x − mean)² / n; integers convert to floating point first; NaN when empty |
| `std(data, n=None)` | as `var` | Python `float` | `math.sqrt(var(data, n))` |
| `norm(data, ord=2, n=None)` | the data's precision; `Array.sum()` for `ord` 1 and 2, `Array.max()` for infinity | Python `float` | `ord` must equal `1`, `2` or `float("inf")`; anything else raises `ValueError: Unsupported norm order` before any launch. `0.0` when empty |
| `absmax(data, n=None)` | as `norm(data, float("inf"))` | Python `float` | `0.0` when empty; a NaN propagates, as in `Array.max()` |
| `count_nonzero(data, n=None)` | `i64`, a scan of 0/1 flags | Python `int` | Tests `x != 0.0`: `-0.0` counts as zero, NaN as nonzero. Exact past 2^31 |
| `dot(a, b, n=None)` | f64 if either input is f64, otherwise f32; `Array.sum()` | Python `float` | Σ a[i]·b[i]; `0.0` when empty |
| `histogram(data, bins=10, range=None, n=None)` | `i32` per bin, `atomic_add` | `(counts, edges)`: an `i32` **array** of `bins` counts, still on the device, and a NumPy `float64` array of `bins + 1` edges | See below |

**`n`** is the number of elements to read, in flat (row-major) order, so
multi-dimensional arrays work. `n=None` means `data.size` (for `dot`,
`a.size`). Every function checks `n` against each array it reads and
raises `ValueError` when `n` is negative or larger than an array, so
`dot(a, b)` with a shorter `b` is refused rather than reading past it.

With an `n` smaller than the array, everything covers the first `n`
elements only: `var` and `std` take the mean of those elements, and
`histogram` without `range` takes its range from them. To do that they copy
the first `n` elements into a temporary array and reduce it with
`Array.sum()`, `min()` and `max()`, so the mean and range follow the
[array-reduction rules](#array-reductions).

### Dtypes on each backend

Every input dtype the backend can allocate works. `var`, `std`, `norm`,
`absmax` and `dot` form terms in a temporary array, then use array reductions.
They do not require floating-point atomics in the input's dtype.

| Input | CPU | CUDA / HIP | Metal | Level Zero |
|---|---|---|---|---|
| `f32` | supported | supported | supported | supported |
| `f64` | `f64` terms and sums | `f64` terms and sums; host reduction | no `f64` arrays | `f64` terms and sums where supported; host reduction |
| integers | runs | runs | runs | runs |

Integer inputs convert to the accumulation precision **before** products,
subtraction or absolute values. This avoids integer overflow in a product
such as `100000 * 100000`, though large integers can round on conversion
to floating point. `dot` uses `f64` if either input is `f64`; otherwise these
statistics use `f32`.

### Complete example: check accumulation precision

This CPU example checks both conversion before multiplication and double-precision
accumulation. Save it as a `.py` file:

<!-- tack-check: statistics-precision -->
```python
import numpy as np
import tack
from tack.algorithms import absmax, dot

tack.set_default_device("cpu")
integers = tack.array(np.array([100000], dtype=np.int32))
assert dot(integers, integers) == 10_000_000_000.0

precise = tack.array(np.array([1.0, 1e-4, 1e-4, 1e-4], dtype=np.float64))
actual = dot(precise, precise)
np.testing.assert_allclose(actual, 1.00000003, rtol=0, atol=1e-15)
print(actual)                              # approximately 1.00000003

with_nan = tack.array(np.array([1.0, np.nan], dtype=np.float32))
assert np.isnan(absmax(with_nan))
```

The same `f64` calculation requires a backend that supports `f64` arrays.
On a GPU it currently copies the temporary `f64` terms to the host for the sum,
as described under [array reductions](#array-reductions).

### NaN, infinity and empty inputs

| Function | NaN in input | Infinity in input | `n == 0` |
|---|---|---|---|
| `var`, `std` | NaN | NaN | NaN, as `Array.mean()` of an empty array |
| `norm(ord=1)`, `norm(ord=2)`, `dot` | NaN | `inf`, or NaN when opposite infinities are added or, in `dot`, an infinity meets a zero | `0.0` |
| `norm(ord=inf)`, `absmax` | NaN | `inf` | `0.0` |
| `count_nonzero` | counted as nonzero | counted | `0` |
| `histogram` | not supported (see below) | not supported (see below) | all-zero counts with an explicit `range`; `ValueError` without one |

`absmax` and `norm(ord=inf)` reduce absolute values with `Array.max()`, including
its NaN propagation. Histogram inputs still need finite, representable bin
coordinates. With no explicit range, a nonfinite autodetected range raises
`ValueError`; passing a finite range does not make a NaN input valid.

An empty array (`data.size == 0`) behaves like `n == 0`.

### Histogram binning

The bin of each value is `int((x - lo) * (1 / width))`, computed in `f32`
(`f64` for an `f64` array), where `width = (hi - lo) / bins`, then clamped
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

Use an integer `bins` of at least 1. `bins=0` raises `ValueError`, as does a
range with a nonfinite endpoint or `lo > hi`. Equal endpoints are expanded
by setting `hi = lo + 1`. For an `f32` input the bin arithmetic uses `f32`
while `edges` is `float64`, so a value lying exactly on an interior edge can
land on either side of it. `counts` stays on the device; call
`counts.numpy()` to read it. Counts must fit in each bin's `i32` accumulator.

### Accuracy and repeatability

The sum's order and error budget are those of [array reductions](#array-reductions).
On GPUs, the `f32` tree sums combine group results atomically, so their low bits
can vary. The current CPU and `f64` host reductions use NumPy; portable code
should still compare floating sums with a suitable tolerance.

The reduction budget covers summing the terms. Rounding when forming products,
squares or deviations from a mean also contributes to a statistic's error.
Check against an independent reference in the precision appropriate for the
problem. `count_nonzero` is exact, and histogram counts are exact while they fit
their `i32` bins. Extrema are order-independent and propagate NaNs as above.

## Prefix scans

```python
from tack.algorithms import exclusive_scan, inclusive_scan

counts = tack.empty((n,), tack.i32)
offsets = tack.empty((n,), tack.i32)
# ... a kernel fills counts ...

total = exclusive_scan(counts, offsets, n)  # offsets[i] = counts[0] + ... + counts[i-1]
total = inclusive_scan(counts, offsets, n)  # offsets[i] = counts[0] + ... + counts[i]
```

The classic use is stream compaction: one kernel counts how many outputs
each item produces, an exclusive scan turns the counts into write offsets,
and the returned total sizes the output array. When the output can exceed
2^31 − 1 elements, keep the counts `i32` but make the offsets `i64`: the
scan runs in the output's dtype, so the running sums are formed in `i64`
and the total comes back exact.

```python
offsets = tack.empty((n,), tack.i64)
total = exclusive_scan(counts, offsets, n)  # i32 counts, i64 offsets and total
```

Both functions run on every backend, by chunks: each chunk is copied into
the output and totalled, the chunk totals are scanned the same way,
recursively, and each chunk then writes its running sums from its offset.
That is two launches per level, and a level small enough for one thread
-- 4096 elements on the CPU, one tile on GPU backends -- is one launch
that scans it in order. On the CPU a chunk is 256 elements, one thread
each: a million elements take a chunked level and a serial one, three
launches. On GPU backends a chunk is a tile of 2048 elements scanned by
one workgroup -- eight elements to each of its 256 lanes, the lane totals
scanned in shared memory -- so a million elements take a tiled level and
a serial one. Launches on the CPU complete before returning; Metal, CUDA,
HIP and Level Zero queue them and wait when the host next reads an array.
For small arrays, copying to NumPy, calling `np.cumsum` and copying
back can still be faster.

**Return value.** Both return the sum of the first `n` inputs, in the
output array's dtype, as a Python `int` (integer outputs) or `float`
(floating outputs). It is read as one element from the device, so only
that element comes back to the host, not the whole output.

**Behavior.**

- Any dtype the backend allocates works. The scan runs in the **output**
  array's dtype; input values convert to it on the copy, as
  `numpy.cumsum` does with an output dtype (an `i32` input scanned into an
  `f32` output gives `f32` sums). Integer sums wrap at the output's width,
  and so does the returned total.
- The input is not modified (unless it is also the output). Both functions
  work **in place**: `exclusive_scan(f, f, n)` and `inclusive_scan(f, f, n)`
  are correct.
- Only the first `n` elements are read and written; later output elements
  are left alone. `n` must be at most the size of both arrays; a larger or
  negative `n` raises `ValueError`. `n = 0` writes nothing and returns `0`,
  the empty sum.
- Returning the total means reading it to the host, which waits for the
  device (on GPU backends, for every launch queued so far). A caller that needs
  only the offsets passes `total=False`: the scan returns `None`, reads
  nothing, and stays queued with whatever follows.
- Each call allocates two arrays of the output's dtype per chunked level,
  the chunk totals and their offsets, one element per chunk: about
  `n / 128` elements in all on the CPU, `n / 1024` on GPUs, not a copy of
  the input, and a one-element array for the total.
- Integer results are exact (modulo wrapping) and identical on every
  backend and every run.
- Floating results are deterministic for a given `n` and backend. Through
  the serial level's size -- 4096 elements on the CPU, 2048 on GPUs --
  they are added in `np.cumsum`'s order and match it bit for bit; beyond
  it a chunk starts from the sum of the earlier chunks' totals rather than
  from the previous element's running sum (on GPUs a tile's lanes from
  their predecessors' totals, combined in tree order), so results can
  differ from `np.cumsum` in the last bits.

### Copy and fill utilities

```python
from tack.algorithms import copy, fill_value

copy(src, dst, n)          # dst[i] = src[i] for i < n
fill_value(dst, 0, n)      # dst[i] = 0 for i < n
```

Both are single kernels on their arrays' device and return `None`. They work
on the first `n` elements in flat order and do not check `n` against either
array.

- `copy` converts on store into `dst`'s dtype with the kernel language's
  rules: floats truncate toward zero (`2.7` → `2`, `-2.7` → `-2`) and must
  fit the target; integers wrap. Unlike `array.copy()`, it writes into an
  existing array rather than allocating one.
- `fill_value` passes `value` as a kernel scalar argument (a Python float is
  `f32`, or `f64` when `dst` is `f64`; an int is `i32`, or `i64`/`u64` by
  magnitude) and converts it on store: `2.5` fills an `i32` array with `2`,
  and `300` fills a `u8` array with `44`. Unlike `array.fill(value)`, which
  converts the value as NumPy's `ndarray.fill` does and fills the whole
  array, it fills a prefix.
- `copy_with_offset(src, dst, dst_offset, n)` writes
  `dst[dst_offset + i] = src[i]`. It is not re-exported from
  `tack.algorithms`; write `from tack.algorithms.copy import copy_with_offset`.
  (The attribute `tack.algorithms.copy` is the `copy` function, not the
  submodule, so `tack.algorithms.copy.copy_with_offset` does not work.)

## Sorting and segmented reductions

```python
from tack.algorithms import argsort, gather, reduce_by_key, sort_by_key, unique

perm = argsort(keys)                           # index array: keys[perm] ascending, stable
skeys, svals = sort_by_key(keys, values)       # new arrays, values carried along
ukeys, counts = unique(skeys)                  # each distinct key once, and its run length
ukeys, sums = reduce_by_key(skeys, svals)      # one value per distinct key; op="sum"|"min"|"max"
picked = gather(values, perm)                  # picked[i] = values[perm[i]]
```

These are the building blocks for everything on unstructured data that is
free on a structured grid: inverting a connectivity array into the cells
around each point, finding the external faces of a mesh (sort the faces,
keep the ones that appear once), merging duplicate points. The pattern is
always *make a key per item, sort, then walk the runs of equal keys*:

```python
# The cells around each point, as a CSR structure. conn holds the point
# ids of each cell's corners; cell_of holds the cell id of each entry.
points, cells = sort_by_key(conn, cell_of)     # entries grouped by point
upoints, counts = unique(points)               # each point and its cell count
offsets = tack.empty((upoints.size,), tack.i32)
exclusive_scan(counts, offsets, upoints.size)  # cells[offsets[p]:offsets[p]+counts[p]]
```

All five functions allocate and return new arrays rather than writing into
the caller's, and take an optional `n` (default: the whole array) with the
same bounds check as the scans: `n` past any array involved raises
`ValueError`, and `n = 0` returns empty arrays.

**Sorting** (`argsort`, `sort_by_key`). Keys are `i32`, `u32`, `i64` or
`u64` arrays; any other key dtype raises `TypeError`. The sort is
**stable**, so equal keys keep their original order, and `sort_by_key`'s
values keep theirs. Values can be any dtype; they are gathered once by the
final permutation, so `sort_by_key(keys, values)` costs an `argsort` plus
two gathers. The permutation is `i32` up to 2^31 − 1 elements and `i64`
beyond; `argsort(keys, index_dtype=tack.i64)` asks for `i64` at any size,
as a mesh with 64-bit ids does. `unique`'s counts follow the same rule.

The implementation is a least-significant-digit radix sort over 8-bit
digits, built from kernels and the exclusive scan. On the CPU, one thread
per chunk of 256 elements counts digits into a private histogram, a scan
turns the counts into output slots, and a second kernel scatters each chunk
in order. On GPU backends, which have workgroups, each pass instead sorts
tiles of 4096 keys in shared memory by their digit (eight stable one-bit
splits), scans the tiles' digit counts, and scatters each digit's run; a
private histogram per thread is slow on a GPU. Both give the same
permutation. The
number of passes follows the **spread** of the keys, not their width:
the smallest key is subtracted on the fly, so point ids below 2^16 take
two passes whatever the key dtype, and all-equal keys take none. Results
are identical on every backend and every run.

**Runs of equal keys** (`unique`, `reduce_by_key`). Both take keys that
are already grouped — sorted, or merely with equal keys adjacent — and
work on any key dtype that supports `!=`, floats included. They return the
first key of each run in order of appearance, so an unsorted input such as
`[7, 7, 2, 2, 7]` gives `[7, 2, 7]`; sort first for a true set of distinct
keys. `unique` returns the run lengths as an `i32` array. `reduce_by_key`
reduces each run's values with `"sum"`, `"min"` or `"max"` into an array of
the values' dtype, running one thread per run **serially in element
order**: integer sums are exact modulo the values' width, and floating
results are reproducible and equal to a sequential sum of the run. The
cost is load imbalance when a few runs are very long; that is the price of
not using atomics, whose floating sums would vary between runs.

## Segmented operations

```python
from tack.algorithms import segment_broadcast, segment_ids, segment_reduce, segment_scan

# Segment k is values[offsets[k]:offsets[k + 1]]: CSR offsets, empty segments allowed.
sums = segment_reduce(values, offsets)               # one per segment; op="min"|"max"|"mean"
prefix = segment_scan(values, offsets)               # running sums, restarting per segment
before = segment_scan(values, offsets, inclusive=False)
spread = segment_broadcast(sums, offsets, n)         # each segment's value on its elements
ids = segment_ids(offsets, n)                        # each element's segment, -1 outside
```

A collection of variable-length segments -- each cell's points, each
point's cells, the degrees of freedom of each element, the rows of a sparse
matrix -- is flat values and CSR offsets, `size(offsets) - 1` segments.
These are its reductions, prefix sums and broadcasts, where `reduce_by_key`
needs keys and gives one thread to each run.

**Shapes.** A segment of up to 256 elements is one thread's, in element
order; a longer one is cut into chunks of 256, each a thread's, and a
segment over several chunks is combined a level up, the same way. So many
short segments cost one pass, and one enormous segment a few launches
rather than one thread walking it. On an M1 Max's GPU, a 10M-element
segment reduces in 4 ms where a thread per segment takes 1.2 s; 2.5M
segments of about 4 elements take the same 0.6-0.7 ms either way.

**Results.** Integer sums wrap at the values' width; min and max are exact.
Floating sums group by those fixed chunks, so they are the same bits on
every backend and every run, though a long segment's sum is not the
sequential one; `segment_reduce(..., ordered=True)` sums each segment in
element order, one thread a segment, as `reduce_by_key` does. An empty
segment reduces to `empty=`, by default 0 for a sum, NaN for a floating
mean and the type's greatest or least (an infinity for floats) for min and
max; `empty="keep"` leaves its output as it was. tack-vis averages over
each point's cells this way (point gradients, cell-to-point projection,
surface normals), reading the values through the point links without
gathering them: a point at the apex of 403,200 tetrahedra took 294 ms on
an M1 Max's GPU, one thread walking its links, and takes 5.7 ms. Vectors reduce
componentwise. Elements outside every segment (offsets need not start at 0
or end at the last value) are left alone. In debug mode (`tack.set_debug`)
the offsets are checked to be non-decreasing and within the values.

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
data = tack.empty((n,))
partial = tack.empty((n // 256,))
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
a path, and use the array reductions or statistics above on CPU.

**Rules.** The [workgroup contract](../reference/language-contract.md#workgroups-and-synchronization)
has the full conservative domain; in short:

- **`f32` only.** The argument must be `f32` and the result is `f32`.
  Anything else raises `TypeError` ("block_sum requires f32 input; use an
  explicit f32 cast") on the first call. Write `tack.block_sum(tack.f32(x))`
  for an integer or `f64` value.
- **Complete groups.** A positive iteration count must be a multiple of
  256, including stepped ranges and the total of a `tack.ndindex`.
  Otherwise the call raises `ValueError` ("... require complete 256-lane
  groups; iteration count 1000 would create a partial workgroup") before
  anything runs. A count of zero runs nothing. Tack does not pad; pad the
  data to a multiple of 256 with the reduction's identity (`0.0` for a sum,
  `inf` for a minimum, `-inf` for a maximum).
- **Every lane participates.** The reduction must be reached by all lanes
  of the group: not under a condition that depends on the loop index, an
  array value or `thread_id()`, not after a lane-dependent `break` or
  `continue`, and not in a conditional expression, the second operand of
  `and`/`or`, or a `while` condition. Conditions on scalar arguments and
  on earlier block-reduction results are fine. Violations raise
  `ValueError` naming the kernel and IR location. To reduce a subset,
  have every lane contribute, using the identity where it should not count:
  `tack.block_sum(x if x > 0.0 else 0.0)`.
- The generated reduction includes its own barriers, including one after
  the result is read, so no extra `tack.barrier()` is needed around it.

**Values.** `block_min` and `block_max` follow the array-reduction rules: a
NaN in the group gives NaN, and zero ties give `-0.0` for the minimum and
`0.0` for the maximum. `block_sum` adds in a tree whose order may vary, with
the same error budget as array sums over the group's terms. Combining the
group sums with `tack.atomic_add` adds in arrival order, so such a total
can change in its low bits from run to run.
