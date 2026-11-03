# Write a filter

A filter is a Python function that takes a dataset and returns a field or a
new dataset, running kernels to do the work. The built-in filters are
written that way, with the same public API you have, so a filter of your own
can do anything theirs do. This chapter rewrites six of them. Each rewrite
gives the built-in's result in every part, including topology, geometry,
fields and their units, sets, ghost marks, global ids, origin, spatial unit
and time. The example program checks this, and so does the test suite
(`test_public_filters.py`).

A filter has to compute the right values, and it also has to keep what the
dataset means:

- the positions are relative to an origin;
- fields have units;
- a piece's ghost cells and points are copies that other pieces own;
- hidden cells hold no valid data;
- the arrays live on a device.

Each section below shows where a filter of that kind deals with one of
these. [Write Mesh Algorithms](05-mesh-kernels.md) covers the kernels
themselves (views, `for_each`, ids); read it first.

## The data

```python
--8<-- "docs/examples/data_filters.py:data"
```

The origin is far from the cells, so a filter that confuses world and
relative positions gives visibly wrong answers. The dataset has a unit and a
time, a cell set, and pieces with a layer of ghost cells and global ids. The
example runs every filter on the image, on its tetrahedra and on a ghosted
piece.

## A field from positions: `point_elevation`

```python
--8<-- "docs/examples/data_filters.py:field"
```

- **Positions are relative.** A dataset's positions are `origin + value`,
  and kernels only see the values, which stay precise in f32 however far the
  origin is (see [Understand a Dataset](01-datasets.md)). The two end
  points come in world positions, so the filter moves them to the origin on
  the host, in f64, before any kernel sees them.
- **Allocate on the dataset's device.** Use `device=data.device`, and
  `data.dtype` for values in the geometry's precision. Every public
  algorithm runs on its input's device. A filter that allocates on the
  default device instead fails when the dataset lives on another one: a
  kernel refuses arrays from two devices.
- **Arrays may be implicit.** An image's geometry is a `UniformCoordinates`
  array, which computes positions instead of storing them, and has no
  `.shape`: `size_of` or `len` gives its length, and `positions[i]` reads one
  position on the host as it does from a stored array. Pass it to a kernel as it is, since
  kernels index implicit arrays (`positions[i]`, `len(positions)`); the
  kernel computes each position as it reads it. `materialize` stores every
  value, which an image of a billion points cannot afford: keep it for code
  that needs stored memory, such as a pointer handed to another library.
- The result is a `Field` on the geometry's space, one value per point.

## A subset of cells: `threshold`

```python
--8<-- "docs/examples/data_filters.py:subset"
```

- **Parameters take quantities.** `units.magnitude` turns `"300 K"`,
  `(26.85, "degC")` or a plain `300.0` into a number in the field's unit, so
  `threshold(data, "T", "-123.15 degC", (250, "K"))` works as the built-in's
  does. A plain number passes through untouched. For a length, use
  `units.magnitude(value, data.unit, ...)`.
- **A kernel's choice becomes a set.** The kernel writes one flag per cell
  at `cells.entity_id(c)`, and `Set.from_flags` turns the flags into a
  `Set`. `extract_cells` then keeps those cells with all their fields,
  sets, ghost marks, global ids and frame, and drops hidden cells.

## New geometry on the same cells: `shrink`

```python
--8<-- "docs/examples/data_filters.py:geometry"
```

`shrink` keeps the input's cells and points but builds a new `DataSet`
around them. A dataset built by hand starts with nothing, so the filter
hands it the input's ghost marks, global ids, origin, unit and time with
`keep_marks`.

Forgetting them is a silent error. Contours of pieces without ghost cells
cannot place their points' ghost marks, so they carry a note that the marks
were lost, and reductions over such pieces refuse to run rather than count
shared points twice. A filter that drops the note turns that refusal into a
wrong answer. When a filter's output cells and points don't correspond to
the input's at all, use `drop_marks(data, out, reason)` to record the loss
instead.

## Fields on faces, and units of results: `jump`

```python
--8<-- "docs/examples/data_filters.py:faces"
```

A field a filter computes gets the unit of what it is, so a difference gets
`units.delta`: a difference of degrees Celsius is in kelvin. For other
results:

- `units.derived((unit, power), ...)` gives the unit of a product of powers
  (a gradient is the field's unit over the spatial unit);
- `units.named` folds a unit to its SI name.

An unknown unit (`None`) gives `None`, never a guess, and passing no unit at
all also gives `None`. The kernel here is a face kernel: `side_cell(f, 0)`
is the cell on the face's inner side, and a boundary face has one side.

## New cells: `triangulate`

```python
--8<-- "docs/examples/data_filters.py:cells"
```

- **Start from `visible(data)`.** It leaves out hidden cells (`HIDDENCELL`,
  data not valid there) and AMR cells a finer level covers. Every filter
  that makes cells starts there.
- **Count, scan, fill.** One kernel counts each cell's output, an
  `exclusive_scan` turns the counts into each cell's first slot and the
  total, and a second kernel fills the slots. No atomics are needed, and the
  output is in input order on every backend.
- **A topology's arrays may be implicit.** Every output cell is a triangle,
  so the types are a `ConstantArray` and the offsets a `CountingArray`. The
  `ConstantArray` is made with `device=device`, like every array the filter
  allocates: an implicit array lives on the device it is made for.
- **`carry` brings everything else.** It takes the input dataset, the new
  topology, and a map for each kind of entity, saying where the output's
  entities came from, and carries fields, sets, ghost marks, global ids and
  the frame by those maps. Here the points are the input's (`Same()`), and
  each triangle is a piece of the cell `sources` names (`Pieces(sources)`).
  The next section, whose filter makes new points as well, takes `carry`
  apart.

## New points and lines: `contour` of triangles

```python
--8<-- "docs/examples/data_filters.py:contour"
```

The lines where a point field crosses a level on a mesh of triangles, which
is what `td.contour` gives for a surface. The filter makes new points, one on
each edge the level crosses, and new cells, each a line inside a triangle, so
`carry` gets a map for each.

- **Number the new points by the edges.** A 2D mesh's faces are its edges
  (`data.topology.faces()`), each stored once and shared by the triangles on
  either side.
  - One kernel over the faces flags the edges the level crosses, a scan
    numbers them, and a second kernel fills in each point's two ends and its
    weight, how far it lies from the first end, with `td.edge_crossing`, the
    rule every built-in cut uses.
  - The two triangles on an edge then name its point by the same number,
    `slots[face_id]`, so their lines join.
  - `edge_crossing` takes the end first by position (x, then y, then z) as
    the first, so every piece holding the edge computes the point to the same
    bits, and gets the built-in's bits.
- **Orient the lines.** Going round each triangle (`side_point_id(c, k, 0)` to
  `side_point_id(c, k, 1)`), its line runs from the edge where the values rise
  past the level to the edge where they fall back. The higher values then lie
  on its right, as with the built-in.
- **What it leaves out.** The built-in also:
  - makes the crossings of the edges that meet at a point into one point
    there, when the point's value is the level or the crossing lies within
    half an ulp of it (`edge_crossing` returns that point as both ends): a
    plane a hair from mesh points would otherwise put several points at one
    position;
  - contours several levels at once;
  - skips cells with a NaN.

  The example picks a level no crossing lands on a point at.

### `carry`, map by map

`carry(data, out, points=..., cells=...)` builds a dataset on the new
topology `out`, and brings everything of `data` that its maps say how to
bring:

| Map | Output entity `i` is… | Used for |
|---|---|---|
| `Same()` | input entity `i` | points or cells kept as they are |
| `Take(ids)` | input entity `ids[i]` | a subset, a reordering |
| `Pieces(ids)` | a part of input cell `ids[i]` | cells cut from cells: triangles, tetrahedra, contour lines |
| `Interpolate(ends, weights)` | the point `weights[i]` of the way from input point `ends[i][0]` to `ends[i][1]` | points made on edges: contours, clips |

Each kind of entity has its own keyword:
- `points=` and `cells=`;
- `faces=` and `edges=`, for values stored on a mesh's faces or edges;
- `faces_to_cells=` and `edges_to_cells=`, when the output's cells are the
  input's faces or edges (an extracted surface, extracted edges).

What goes with each map:

- **Point fields** (`H1`, values on points) go by the points map.
  `Interpolate` blends them as `va + w * (vb - va)`, so the contour's points
  get `RTData` (the level, to rounding) and every other point field. A field
  of integers cannot be blended and is left behind.
- **The geometry** is a point field like any other, so `Interpolate` places
  the new points: the filter never computes a position. A filter that
  computes positions some other way passes them as `geometry=`.
- **Cell fields** (`Constant`, values on cells) go by the cells map: each
  line takes its triangle's value. `L2` fields go with `Same` and `Take` only,
  since a piece of a cell is not the whole cell their basis describes.
- **Sets** go by the same maps, so a line is in a cell set when its triangle
  is. A point set cannot go through `Interpolate` and is left behind.
- **Ghost marks.** A line takes its triangle's mark. A new point takes the
  mark of its edge's first end by position, so the piece that owns that end
  owns the point, and each line or point of the whole's contour is owned
  once over the pieces.
  - Pieces without ghost cells cannot tell who owns what they make on the
    edges they share.
  - Their output records that its marks were lost, and reductions over it
    refuse to run (see `shrink` above) until `generate_global_ids` of the
    outputs decides the owners by matching positions, which works because
    every piece computes a shared point to the same bits.
- **Global ids** go with entities that are input entities (`Same`, `Take`).
  Made points and pieces of cells have none.
- **The frame:** the origin, spatial unit and time always come along.

A kind of entity with no map carries nothing: no fields, sets, marks or ids
of that kind. `fields=` (a list of names) limits which fields are carried.

## A reduction over pieces: `entropy`

```python
--8<-- "docs/examples/data_filters.py:pieces"
```

- **Count each value once.** `owned_values` gives the values of the cells
  and points a piece owns, leaving out its ghost copies and hidden cells, so
  a value shared by two pieces is counted once.
- **Reduce over the blocks.** `tack.communication.Blocks` holds this rank's
  pieces, and `all_reduce` combines one value per piece in global piece
  order, so a float sum has the same bits however the pieces are spread
  over ranks. The same function runs unchanged under MPI: pass a
  `World(comm)` to the `PartitionedDataSet`.
- Algorithms that need neighbours' data rather than a reduction (exchanges,
  ghost plans, routing points to pieces) are in
  [Pieces and Blocks](14-communication.md), which links the User's Guide's
  chapter on `tack.communication`.

## What to keep, and what forgetting it gives

| Concern | Do | Forgetting it gives |
|---|---|---|
| Origin | Convert world parameters with `data.origin` on the host; build datasets through `carry`, `keep_marks` or with `origin=` | Wrong positions, visible only far from zero |
| Device | Allocate with `device=data.device` | An error on a dataset on another device |
| Implicit arrays | `size_of`, `materialize`, or index in a kernel | `AttributeError` on `.shape` |
| Hidden cells | Start from `visible(data)` | Cells with invalid data processed |
| Ghost marks | `carry` maps, `keep_marks`, or `drop_marks` | Shared cells and points counted once per piece |
| Units | `units.magnitude` for parameters; `units.delta`/`derived` for results | A number in the wrong unit; a wrong unit on the result |
| Pieces | `owned_values`, `Blocks.all_reduce` | Ghost copies counted, or results that depend on the rank count |

The [Data Model reference](reference/data-model.md#filters-of-your-own)
documents each of these names.

## Run and check

```bash
uv run python docs/examples/data_filters.py --arch cpu --check
```

The check holds each rewrite to its built-in, on the image, its tetrahedra
and a ghosted piece, and a threshold with bounds in degrees Celsius and
kelvin to the built-in's. [Download the complete program](../examples/data_filters.py).

Continue with [Sample and Trace Fields](06-pipelines.md) to connect
filters into a workflow, or [Pieces and Blocks](14-communication.md)
to write one that exchanges data between pieces.
