# Marching squares: count, scan, write

A contour algorithm produces a variable amount of output: some cells emit no
segments, others one or two. This example extracts a circle from samples of
`x² + y² - 0.7²` and gives each cell a known, disjoint interval in the output.

![Contour segments on a grid](../../assets/tutorials/contours.png)

**Origin:** a teaching adaptation of [Taichi's marching-squares example](https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/algorithm/marching_squares.py),
Copyright © The Taichi Authors, Apache-2.0. This version replaces animated noise
with an analytic circle, uses paired edge crossings instead of a lookup table,
and uses an exclusive scan instead of appending to a dynamic list.

## Count crossings per cell

The input has `(n + 1, n + 1)` scalar samples and therefore `n * n` cells. Each
cell reads four corners in order around its perimeter. An edge crosses the level
when its two endpoints differ in sign:

```python
--8<-- "docs/examples/contours.py:count"
```

Two crossings form one segment; four form two. This example pairs crossings
in perimeter order. For general data, four-crossing saddle cells need a deliberate
connectivity decision, such as an asymptotic decider. A coarse arbitrary field
can also contain features that its corner samples miss. Those are algorithmic
issues rather than storage problems; the circle avoids saddle ambiguity.

## Turn counts into output intervals

For example:

| Cell | Count | Exclusive offset | Output slots |
|---|---:|---:|---|
| 0 | 0 | 0 | none |
| 1 | 1 | 0 | 0 |
| 2 | 2 | 1 | 1, 2 |
| 3 | 1 | 3 | 3 |

The scan returns the total, four, so the host can allocate exactly four segments:

```python
--8<-- "docs/examples/contours.py:allocate"
```

The return value reaches Python to choose an allocation size; the counts and
offsets remain fields. Handle `total == 0` before allocating an empty GPU output.
The general scan implementation performs multiple synchronous launches, so
this pattern is not free for small inputs.

## Write without competing for slots

The second cell visit recomputes crossings, interpolates their positions along
edges, and writes within that cell's interval:

```python
--8<-- "docs/examples/contours.py:write"
```

The two private arrays hold at most four crossings per iteration. Their sizes
are fixed, and every used element is written before it is read. Output is a
two-component vector field of shape `(total, 2)`: two endpoint vectors per
segment. No atomic is needed for those writes.

Unlike a fetch-and-add append, scanning fixes output order by input cell order.
The tradeoff is visiting each cell twice and performing a scan. Use the same
pattern for filtered particles, mesh faces, variable neighbour lists, or any
operation with a count known per input item.

## Run and check

```bash
uv run python docs/examples/contours.py --check
uv run --with vtk python docs/examples/contours.py --output contours.png
```

The check verifies scan offsets against NumPy cumulative sums, valid nonzero
segments, and the analytic circle residual within the interpolation error of
the grid. Try coarsening the grid, then refining it and observing how the
polygon approaches the circle.

## Complete program

[Download contours.py](../../examples/contours.py). The adapted code is
[Apache-2.0 licensed](../../examples/LICENSE-APACHE-2.0.txt).

```python
--8<-- "docs/examples/contours.py"
```
