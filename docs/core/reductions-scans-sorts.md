# Reductions, scans and sorts

Many parallel programs need the same few steps between their own kernels:
add up an array, turn counts into offsets, sort by a key, or combine the
values of each group. This page shows Tack's ready-made versions of those
steps. It is for anyone writing kernels who would otherwise copy an array
to NumPy for a sum, a `cumsum` or an `argsort`. After it you can reduce
arrays, compute statistics, scan, sort, group and reduce segments on the
device, and you will know which results are reproducible.

<!-- tack-check: reductions-first -->
```python
import numpy as np
import tack
from tack.algorithms import exclusive_scan, std

tack.set_default_device("cpu")

data = tack.array(np.array([3.0, 1.0, 4.0, 1.0, 5.0], dtype=np.float32))
assert data.sum() == 14.0                     # a Python float
assert data.max() == 5.0
assert abs(std(data) - np.std([3, 1, 4, 1, 5])) < 1e-6

counts = tack.array(np.array([2, 0, 3, 1], dtype=np.int32))
offsets = tack.empty((4,), tack.i32)
total = exclusive_scan(counts, offsets, 4)    # offsets[i] = counts[0] + ... + counts[i - 1]
assert total == 6
np.testing.assert_array_equal(offsets.numpy(), [0, 2, 2, 5])
```

The sum and maximum are methods of every array. The standard deviation and
the scan come from `tack.algorithms`, which runs them as kernels on the
device that holds the arrays.

## Choose the right tool

| You want | Use | Returns |
|---|---|---|
| one number from an array | `a.sum()`, `a.min()`, `a.max()`, `a.mean()` | a Python `float` |
| a statistic | `var`, `std`, `norm`, `absmax`, `count_nonzero`, `dot`, `histogram` | a Python number; `histogram` an array |
| running sums, or offsets from counts | `exclusive_scan`, `inclusive_scan` | the total, and the sums in your output array |
| a sorted order | `argsort`, `sort_by_key`, `gather` | new arrays |
| one value per run of equal keys | `unique`, `reduce_by_key` | new arrays |
| one value per segment of a CSR layout | `segment_reduce`, `segment_scan`, `segment_broadcast`, `segment_ids` | arrays |
| a sum inside a kernel, over a workgroup | `tack.block_sum`, `block_min`, `block_max` | a value in every lane |

`import tack` does not load `tack.algorithms`. Import its functions as in
the examples, or `import tack.algorithms` first.

Everything here runs on every backend, except the block reductions, which
need a GPU. The algorithms take stored arrays and the implicit arrays of
`tack.arrays` alike. For example,
`exclusive_scan(tack.arrays.CountingArray(n), offsets, n)` scans 0, 1, 2
and so on without storing them.

## Reduce an array to one number

`a.sum()`, `a.min()`, `a.max()` and `a.mean()` reduce **every scalar
component** of the array, whatever its shape or element type. There is no
axis argument. A vector array of `n` elements of type `vec3` reduces
`3 * n` components. The result is always a Python `float`, also for
integer arrays.

The CPU reduces with NumPy. The GPU backends reduce `f32` arrays on the
device and read back one value.

!!! note "What crosses to the host"
    On a GPU, a reduction of any element type other than `f32` copies the
    whole array to the host and reduces it with NumPy, on every call.

The accumulator depends on the element type.

| Element type | `sum()` adds in | `min()`, `max()` |
|---|---|---|
| `f32` | `f32` | exact |
| `f64` | `f64` | exact |
| signed integers | `i64`, wrapping modulo 2^64 | exact |
| unsigned integers | `u64`, wrapping modulo 2^64 | exact |

Converting the sum to a Python `float` can round integer sums past 2^53.
`mean()` divides the returned sum by the number of components, in
double precision.

## Compute statistics

<!-- tack-check: reductions-statistics -->
```python
import numpy as np
import tack
from tack.algorithms import count_nonzero, dot, histogram, norm, var

tack.set_default_device("cpu")

values = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
data = tack.array(values)
assert abs(var(data) - np.var(values)) < 1e-6          # population variance
assert abs(norm(data) - np.linalg.norm(values)) < 1e-5
assert norm(data, ord=1) == 21.0
assert dot(data, data) == 91.0
assert count_nonzero(data) == 6

counts, edges = histogram(data, bins=3, range=(0.0, 6.0))
np.testing.assert_array_equal(counts.numpy(), [1, 2, 3])   # counts stay on the device
np.testing.assert_allclose(edges, [0, 2, 4, 6])            # edges are NumPy
```

| Function | Computes |
|---|---|
| `var(data)`, `std(data)` | the population variance and standard deviation |
| `norm(data, ord=2)` | the Euclidean norm; `ord=1` the sum of absolute values; `ord=float("inf")` the largest |
| `absmax(data)` | the largest absolute value |
| `count_nonzero(data)` | the number of nonzero components, exact as a Python `int` |
| `dot(a, b)` | the sum of `a[i] * b[i]` |
| `histogram(data, bins, range)` | the counts per bin as an `i32` array, and the edges as NumPy |

Each function takes an optional `n`, the number of elements to read in
flat order. A negative `n`, or one larger than an array, raises
`ValueError`.

The statistics compute in `f64` if an input is `f64`, and otherwise in
`f32`. Integer inputs convert before any product, so `dot` of two `i32`
arrays holding 100000 gives 10^10 rather than an overflow. Each statistic
forms its terms in a temporary array and reduces it as `a.sum()` or
`a.max()` would. Their accuracy and host traffic are therefore those of
the array reductions above.

`histogram` differs from `numpy.histogram` in one way. Values below the
range are counted in the first bin and values above it in the last, where
NumPy drops them. Details has the rest of its rules.

## Turn counts into offsets with a scan

A **prefix scan** turns a list of numbers into running sums.
`exclusive_scan` writes, at position `i`, the sum of the inputs before
`i`. `inclusive_scan` includes input `i` itself. Both return the total.

The classic use is **stream compaction**, in three steps. A kernel counts
what each input will write, a scan turns the counts into write offsets and
the total into the output's size, and a second kernel writes each input's
output at its offset:

<!-- tack-check: reductions-compaction -->
```python
import numpy as np
import tack
from tack.algorithms import exclusive_scan

tack.set_default_device("cpu")

@tack.kernel
def count_positive(values, counts):
    for i in range(values.shape[0]):
        counts[i] = 1 if values[i] > 0.0 else 0

@tack.kernel
def write_positive(values, offsets, out):
    for i in range(values.shape[0]):
        if values[i] > 0.0:
            out[offsets[i]] = values[i]

values = tack.array(np.array([0.5, -1.0, 2.0, -3.0, 4.0, 0.0], dtype=np.float32))
n = values.size
counts = tack.empty((n,), tack.i32)
offsets = tack.empty((n,), tack.i32)

count_positive(values, counts)
total = exclusive_scan(counts, offsets, n)
positive = tack.empty((total,))
write_positive(values, offsets, positive)
np.testing.assert_allclose(positive.numpy(), [0.5, 2.0, 4.0])   # in input order
```

Unlike claiming slots with an atomic counter, the output keeps the inputs'
order and has exactly the right size. The
[marching squares tutorial](../users-guide/tutorials/contours.md) builds a
whole contouring algorithm this way.

Some options cover common needs:

- **Wide offsets.** The scan adds in the **output's** element type. Keep
  the counts `i32` and make the offsets `i64` when the total may pass
  2^31 − 1.
- **CSR offsets.** `exclusive_scan(counts, offsets, n, close=True)` also
  writes the total at `offsets[n]`, so `offsets` needs `n + 1` elements.
  Compressed sparse row (CSR) offsets of this kind describe variable-length
  groups, as in the next section.
- **No wait.** Returning the total reads it from the device, which waits
  for the queued work on a GPU. Pass `total=False` when you need only the
  offsets. The scan then returns `None` and stays queued.
- **In place.** `exclusive_scan(a, a, n)` is correct.

Only the first `n` elements are read and written. Integer results are
exact, apart from wrapping at the output's width, and identical on every
backend and every run.

`tack.algorithms` also has `copy(src, dst, n)` and `fill_value(dst, value,
n)`, single kernels that copy or fill the first `n` elements.

## Sort and group by key

<!-- tack-check: reductions-sort -->
```python
import numpy as np
import tack
from tack.algorithms import argsort, gather, reduce_by_key, sort_by_key, unique

tack.set_default_device("cpu")

keys = tack.array(np.array([5, 3, 5, 1], dtype=np.int32))
values = tack.array(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))

order = argsort(keys)                              # stable: equal keys keep their order
np.testing.assert_array_equal(order.numpy(), [3, 1, 0, 2])
np.testing.assert_allclose(gather(values, order).numpy(), [4, 2, 1, 3])

sorted_keys, sorted_values = sort_by_key(keys, values)
distinct, sums = reduce_by_key(sorted_keys, sorted_values)    # op="sum", "min" or "max"
np.testing.assert_array_equal(distinct.numpy(), [1, 3, 5])
np.testing.assert_allclose(sums.numpy(), [4, 2, 4])
distinct, counts = unique(sorted_keys)
np.testing.assert_array_equal(counts.numpy(), [1, 1, 2])
```

`argsort` returns the permutation that sorts the keys, and `gather` applies
it. `sort_by_key` sorts keys and carries values along. `unique` and
`reduce_by_key` walk the runs of equal keys, giving one key and one count
or one reduced value per run. All of them return new arrays.

Sorting is how unstructured data gets what a structured grid has for free.
The pattern is always the same: make a key per item, sort, then walk the
runs. This example inverts a mesh's connectivity, from the points of each
cell to the cells around each point:

<!-- tack-check: reductions-cells-around-points -->
```python
import numpy as np
import tack
from tack.algorithms import exclusive_scan, sort_by_key, unique

tack.set_default_device("cpu")

# Three triangles; connectivity lists each cell's points, cell_of each entry's cell.
connectivity = tack.array(np.array([0, 1, 2, 1, 2, 3, 2, 3, 4], dtype=np.int32))
cell_of = tack.array(np.repeat(np.arange(3, dtype=np.int32), 3))

points, cells = sort_by_key(connectivity, cell_of)     # entries grouped by point
point_ids, counts = unique(points)                     # each point and its cell count
offsets = tack.empty((point_ids.size + 1,), tack.i32)
exclusive_scan(counts, offsets, point_ids.size, close=True)

# The cells around point p are cells[offsets[p]:offsets[p + 1]].
o, c = offsets.numpy(), cells.numpy()
np.testing.assert_array_equal(c[o[2]:o[3]], [0, 1, 2])   # point 2 is in every triangle
np.testing.assert_array_equal(c[o[4]:o[5]], [2])
```

The same pattern finds a mesh's boundary faces, which appear once after
sorting, and merges duplicate points.

**Keys and order.** Sort keys are `i32`, `u32`, `i64` or `u64`; other key
types raise `TypeError`. The sort is **stable**, so equal keys keep their
original order and their values keep theirs. Values may be any element
type. The permutation is `i32` up to 2^31 − 1 elements and `i64` beyond;
`argsort(keys, index_dtype=tack.i64)` asks for `i64` at any size.

**Runs.** `unique` and `reduce_by_key` need equal keys next to each other,
which sorting guarantees. They work on any key type that supports `!=`,
floats included. They return each run's first key in order of appearance,
so the unsorted input `[7, 7, 2, 2, 7]` gives `[7, 2, 7]`.

**Reproducible results.** The sort gives the same permutation on every
backend and every run. `reduce_by_key` reduces each run on one thread, in
element order, so its floating sums are the same on every run and equal a
sequential sum. Very long runs make that one thread the slowest.

## Reduce segments of a CSR layout

Variable-length groups, such as each cell's points, each point's cells or
the rows of a sparse matrix, are often stored as one flat array of values
and CSR offsets. Segment `k` is `values[offsets[k]:offsets[k + 1]]`, and
there are `offsets.size - 1` segments. The segmented operations work on
that layout directly, with no keys:

<!-- tack-check: reductions-segments -->
```python
import numpy as np
import tack
from tack.algorithms import segment_broadcast, segment_ids, segment_reduce, segment_scan

tack.set_default_device("cpu")

values = tack.array(np.array([1, 2, 3, 4, 5, 6], dtype=np.float32))
offsets = tack.array(np.array([0, 2, 2, 6], dtype=np.int32))    # segment 1 is empty

sums = segment_reduce(values, offsets)                  # op="min", "max" or "mean"
np.testing.assert_allclose(sums.numpy(), [3, 0, 18])
np.testing.assert_allclose(segment_reduce(values, offsets, op="mean").numpy(),
                           [1.5, np.nan, 4.5])
np.testing.assert_allclose(segment_scan(values, offsets).numpy(), [1, 3, 3, 7, 12, 18])
np.testing.assert_allclose(segment_broadcast(sums, offsets, 6).numpy(), [3, 3, 18, 18, 18, 18])
np.testing.assert_array_equal(segment_ids(offsets, 6).numpy(), [0, 0, 2, 2, 2, 2])
```

| Function | Gives |
|---|---|
| `segment_reduce(values, offsets, op)` | one value per segment |
| `segment_scan(values, offsets, inclusive=True)` | running sums that restart at every segment; `op="min"` or `"max"` for running extremes |
| `segment_broadcast(segment_values, offsets, n)` | each segment's value on each of its elements |
| `segment_ids(offsets, n)` | each element's segment number, -1 outside every segment |

**Empty segments.** An empty segment reduces to the `empty=` argument. By
default that is 0 for a sum, NaN for a floating mean, and the type's
greatest or least value for min and max. `empty="keep"` leaves the output
as it was, which needs an `out=` array you filled.

**Long and short segments.** A segment of up to 256 elements is reduced by
one thread, in element order. A longer one is cut into chunks of 256 that
are combined afterwards. So many short segments cost one pass, and one huge
segment costs a few launches rather than one thread walking it all.

**Reproducible results.** Floating sums are grouped by those fixed chunks,
so they are the same bits on every backend and every run. A long segment's
sum is then not exactly the sequential one. `segment_reduce(...,
ordered=True)` sums each segment in element order, one thread per segment,
as `reduce_by_key` does. Vector values reduce component by component.

## Reduce inside a kernel on a GPU

On a GPU, iterations run in **workgroups** of 256 lanes that can share
memory. `tack.block_sum(v)`, `tack.block_min(v)` and `tack.block_max(v)`
combine one `f32` value from every lane of a workgroup, and give every lane
the result. They save a separate pass when each workgroup needs its own
partial result.

The CPU has no workgroups and refuses any kernel that uses them. This
example runs the kernel where it can and shows the refusal on the CPU:

<!-- tack-check: reductions-block -->
```python
import numpy as np
import tack
from tack.runtime.dispatch import get_backend

tack.set_default_device("cpu")

@tack.kernel
def group_sums(data, partial):
    for i in range(data.shape[0]):
        s = tack.block_sum(data[i])         # every lane of the group gets the same s
        if tack.thread_id() == 0:
            partial[i // 256] = s           # one lane writes the group's result

data = tack.ones((1024,))                   # a multiple of 256
partial = tack.empty((4,))
if get_backend().supports_workgroups:
    group_sums(data, partial)
    np.testing.assert_allclose(partial.numpy(), [256, 256, 256, 256])
else:
    try:
        group_sums(data, partial)
    except RuntimeError as error:
        assert "does not support workgroup execution" in str(error)
    else:
        raise AssertionError("the CPU ran a block reduction")
```

With `for i in range(n)`, workgroup `g` holds iterations `256 * g` to
`256 * g + 255`, and `tack.thread_id()` is the lane's position in it. Three
rules apply:

- **`f32` only.** Cast other values first: `tack.block_sum(tack.f32(x))`.
- **Complete workgroups.** The iteration count must be a multiple of 256,
  or the call raises `ValueError` before anything runs. Tack does not pad.
  Pad the data with the reduction's identity, such as `0.0` for a sum.
- **Every lane takes part.** The reduction must not sit under a condition
  that depends on the loop index, an array value or `thread_id()`. To leave
  values out, have every lane contribute the identity instead:
  `tack.block_sum(x if x > 0.0 else 0.0)`. Conditions on scalar arguments
  are fine.

The reduction includes its own barriers, so it needs no `tack.barrier()`
around it. [Workgroups, atomics and textures](workgroups-atomics-textures.md)
explains workgroups in full.

## Details

**Repeatability.** Integer results and extremes are exact everywhere.
Floating sums differ by family:

| Operation | CPU | GPUs |
|---|---|---|
| `f32` `a.sum()`, `mean()`, and statistics on `f32` | NumPy | the last bits can change from run to run |
| other `a.sum()` and statistics | NumPy | NumPy, after a copy to the host |
| scans | the same on every run | the same on every run |
| `reduce_by_key`, `segment_*` | the same bits everywhere | the same bits everywhere |
| `block_sum` combined with `atomic_add` | not available | the last bits can change from run to run |

The GPUs combine their workgroups' `f32` partial sums with atomics, in the
order the workgroups finish, which is why those sums vary. The contract
promises an error bound, not reproducibility, for array sums on any
backend. Compare floating sums with a tolerance.

Scans of floating values match `np.cumsum` bit for bit up to 4096 elements
on the CPU and 2048 on GPUs. Beyond that, each chunk starts from the sum of
the earlier chunks' totals, so the last bits can differ from `np.cumsum`.

**NaN, infinity and empty arrays.** A NaN anywhere makes `sum()`, `min()`
and `max()` return NaN. Infinities add by ordinary floating addition, so
`inf + -inf` is NaN. `min()` of `0.0` and `-0.0` is `-0.0` and `max()` is
`0.0`, in either order. On an empty array `sum()` is `0.0`, `mean()` is
NaN, and `min()` and `max()` raise `ValueError`. `var` and `std` of
nothing are NaN; `norm`, `absmax` and `dot` of nothing are `0.0`.
`count_nonzero` counts NaN as nonzero and `-0.0` as zero.

**Histogram bins.** A value's bin is `int((x - lo) * (1 / width))`, computed in
the data's precision and clamped to the first and last bins. A value equal
to `hi` lands in the last bin, as in NumPy. Without `range`, the range is
the data's minimum and maximum, and equal ones become `lo` to `lo + 1`.
`bins` must be at least 1, and the range must be finite with `lo` at most
`hi`. NaN, and values so far outside the range that their bin does not fit
a 32-bit integer, land in a backend-dependent bin; filter them first.
Counts are exact while they fit an `i32`.

**Copy and fill.** `copy` converts into the destination's type by the
kernel language's rules, so floats truncate toward zero and integers wrap.
`fill_value` converts its value the same way, so `300` fills a `u8` array
with `44`. Neither checks `n` against the arrays. `array.fill(value)`, by
contrast, fills the whole array and converts as NumPy does.
`copy_with_offset(src, dst, offset, n)` writes `dst[offset + i] = src[i]`;
import it with `from tack.algorithms.copy import copy_with_offset`.

**When calls return.** A call that returns a Python value, such as a
reduction, a statistic or a scan's total, waits for the work it reads. On
GPUs, a scan with `total=False`, a sort or a segmented operation may
return before the device finishes. The next read of an array waits for it,
as [How a Tack program runs](how-programs-run.md) explains.

The [language contract](../reference/language-contract.md#array-and-parallel-reductions)
states the reduction rules precisely, and the
[parallel execution notes](../design/parallel-execution.md#library-algorithms-scans-and-sorts)
explain how the scans and sorts are built.

## Where to go next

- [Memory and parallel correctness](memory-and-correctness.md) explains
  when a kernel needs an atomic, a scan or a sort to stay correct.
- The [marching squares tutorial](../users-guide/tutorials/contours.md)
  builds a contouring algorithm on counts, a scan and writes.
- [Workgroups, atomics and textures](workgroups-atomics-textures.md) covers
  shared memory and atomics, the tools under the block reductions.
