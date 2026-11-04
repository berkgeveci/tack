# Write a filter

A filter is a Python function that takes a dataset and returns a new dataset
or a field, running kernels to do the work. Tack's built-in filters are
written that way, with the same public names you have, so a filter of your own
can do anything theirs do. This tutorial rewrites six built-ins, building on
the kernels of [Write a mesh algorithm](write-a-mesh-algorithm.md). Each
rewrite gives the built-in's result in every part: topology, geometry, fields
and their units, sets, ghost marks, global ids, origin, spatial unit and time.
Keeping all of them is what lets a filter's output mean what its input meant,
on one piece or on many.

## A first filter

This filter keeps the cells whose corners all lie in a range of a point field,
as `td.threshold` does:

<!-- tack-check: filter-threshold -->
```python
import numpy as np

import tack
import tack.dataset as td

tack.set_default_device("cpu")


@tack.kernel
def all_corners_inside(cells, u, lower, upper, keep):
    for c in cells:
        inside = 0
        for j in range(cells.NUM_POINTS):
            v = u.corner_value(c, j)
            if lower <= v and v <= upper:
                inside += 1
        keep[cells.entity_id(c)] = 1 if inside == cells.NUM_POINTS else 0


def threshold(data, name, lower, upper):
    """The cells whose corners all lie in [lower, upper]."""
    field = data.fields[name]
    keep = tack.empty((data.num_cells,), tack.i32, device=data.device)
    td.for_each(all_corners_inside, data, "cells", field, float(lower), float(upper), keep)
    return td.extract_cells(data, td.Set.from_flags(data, "cells", keep))


data = td.wavelet(6)                              # 13^3 points, the field "RTData"
mine = threshold(data, "RTData", 150.0, 250.0)
theirs = td.threshold(data, "RTData", 150.0, 250.0)
assert mine.num_cells == theirs.num_cells > 0
assert np.array_equal(mine.fields["RTData"].values.numpy(),
                      theirs.fields["RTData"].values.numpy())
```

The kernel decides, one flag per cell, written at the cell's global id.
`td.Set.from_flags` turns the flags into a cell [set](../reference/glossary.md),
and `td.extract_cells` keeps those cells. It brings along every field, set,
ghost mark and global id, and the dataset's origin, unit and time, so the
filter itself handles none of them.

The array of flags is made with `device=data.device`. Every public algorithm
runs on its input's [device](../data/devices.md), and a filter that allocated
on the default device would fail on a dataset that lives elsewhere: a kernel
refuses arrays from two devices.

## What a filter has to keep

A filter has to compute the right values. It also has to keep what the dataset
means:

- **Positions are relative to an origin.** A dataset stores its positions
  relative to an origin of three double-precision numbers, so that coordinates
  far from zero stay accurate in single precision. Kernels see only the
  relative values. See [Units, coordinates and time](../data/units-coordinates-time.md).
- **Fields have units.** A parameter may come with a unit, and a computed
  field has the unit of what it is.
- **Pieces have ghost marks.** A dataset in [pieces](../data/pieces.md) marks
  the cells and points that are copies another piece owns. A reduction must
  count each value once.
- **Hidden cells hold no valid data.** A filter never processes them.
- **Arrays live on a device**, and some arrays are implicit: computed when
  read, never stored.

The sections below rewrite one built-in each and show where a filter of that
kind deals with each of these.

## The data

```python
--8<-- "docs/examples/data_filters.py:data"
```

`td.wavelet` is VTK's wavelet source, an image with the point field `RTData`.
Its origin is far from the cells, so a filter that mixes world and relative
positions gives visibly wrong answers. It has a unit and a time, a cell set,
and pieces with a layer of [ghost cells](../reference/glossary.md) and global
ids. The example runs every filter on the image, on its tetrahedra and on a
ghosted piece.

## A field from positions: `point_elevation`

```python
--8<-- "docs/examples/data_filters.py:field"
```

The result is a `Field` on the geometry's space, one value per point. Three
things in it matter for every filter:

- **World parameters convert on the host.** The two end points are world
  positions. The filter subtracts the origin from them in double precision,
  on the host, before any kernel sees them. The kernel then works in relative
  positions throughout.
- **Allocate on the dataset's device and in its precision.** `data.device`
  is the device, and `data.dtype` is the geometry's floating-point type.
- **Arrays may be implicit.** An image's geometry is a `UniformCoordinates`
  array, which computes positions instead of storing them. It has no
  `.shape`, so the filter takes its length with `size_of`. Pass an implicit
  array to a kernel as it is: kernels index implicit arrays as they index
  stored ones. `tack.arrays.materialize` stores every value, which an image of
  a billion points cannot afford. Keep it for code that needs stored memory,
  such as a pointer handed to another library.

## A subset of cells: `threshold`

```python
--8<-- "docs/examples/data_filters.py:subset"
```

This is the first filter again, with one addition: its bounds take
quantities. `units.magnitude` turns `"300 K"`, `(26.85, "degC")` or a plain
`300.0` into a number in the field's unit. So
`threshold(data, "T", "-123.15 degC", (250, "K"))` works as the built-in's
does. A plain number passes through untouched. For a length, convert to the
spatial unit with `units.magnitude(value, data.unit, ...)`.

## New geometry on the same cells: `shrink`

```python
--8<-- "docs/examples/data_filters.py:geometry"
```

`shrink` keeps the input's cells and points, but builds a new `DataSet`
around them. Each cell gets its own copy of its corners, an `L2` field of the
positions, moved toward the cell's centre. A dataset built by hand starts with
nothing, so the filter hands it the input's ghost marks, global ids, origin,
unit and time with `td.keep_marks`.

Forgetting them is a silent error. Some outputs cannot place their ghost
marks: a contour of pieces without ghost cells cannot tell which piece owns a
point it makes on a shared edge. Such an output records that its marks were
lost, and reductions over it refuse to run rather than count shared points
twice. A filter that builds a dataset without `keep_marks` drops that record,
which turns the refusal into a wrong answer. When a filter's output cells and
points do not correspond to the input's at all, call
`td.drop_marks(data, out, reason)` instead. It records the loss, and `reason`
is the sentence the refusal will show.

## Fields on faces, and units of results: `jump`

```python
--8<-- "docs/examples/data_filters.py:faces"
```

The kernel runs over faces: `side_cell(f, 0)` is the cell on the face's inner
side, and a boundary face has one side. The result is a field of values on
faces.

A field a filter computes gets the unit of what it is. A difference gets
`units.delta` of the field's unit: a difference of degrees Celsius is in
kelvin. For other results:

- `units.derived((unit, power), ...)` gives the unit of a product of powers.
  A gradient, for example, is the field's unit over the spatial unit.
- `units.named` folds a unit to its SI name, where a filter names its result.

An unknown unit (`None`) gives `None`, never a guess. A filter that passes no
unit at all also gives `None`, so forgetting a unit loses it rather than
making it wrong.

## New cells: `triangulate`

```python
--8<-- "docs/examples/data_filters.py:cells"
```

- **Start from `visible(data)`.** `td.visible` leaves out hidden cells, whose
  data is not valid, and AMR cells a finer level covers. Every filter that
  makes cells starts there.
- **Count, scan, fill.** One kernel counts each cell's output, an
  `exclusive_scan` turns the counts into each cell's first slot and the total,
  and a second kernel fills the slots. No atomics are needed, and the output
  is in input order on every device. The scan is Core's; see
  [Reductions, scans and sorts](../core/reductions-scans-sorts.md).
- **A topology's arrays may be implicit.** Every output cell is a triangle, so
  the cell types are a `ConstantArray` and the offsets a `CountingArray`. The
  `ConstantArray` is made with `device=device`, like every array the filter
  makes: an implicit array lives on the device it is made for.
- **`carry` brings everything else.** `td.carry` takes the input dataset, the
  new topology, and a map for each kind of entity. A map says where the
  output's entities came from. Here the points are the input's (`Same()`), and
  each triangle is a piece of the cell `sources` names (`Pieces(sources)`).
  The next section, whose filter also makes new points, takes `carry` apart.

## New points and lines: `contour` of triangles

```python
--8<-- "docs/examples/data_filters.py:contour"
```

This filter makes the lines where a point field crosses a level on a mesh of
triangles, which is what `td.contour` gives for a surface. It makes new
points, one on each edge the level crosses, and new cells, each a line inside
a triangle. So `carry` gets a map for each.

- **Number the new points by the edges.** A 2D mesh's faces are its edges
  (`data.topology.faces()`), each stored once and shared by the triangles on
  either side. One kernel over the faces flags the edges the level crosses. A
  scan numbers them. A second kernel records each new point's two ends and its
  weight, how far it lies from the first end. The two triangles on an edge
  then name its point by the same number, `slots[face_id]`, so their lines
  join.
- **Place each crossing as the built-ins do.** `td.edge_crossing` is the rule
  every built-in cut uses. It interpolates from the end that comes first by
  position (x, then y, then z), so every piece holding the edge computes the
  point to the same bits.
- **Orient the lines.** Going round each triangle, its line runs from the edge
  where the values rise past the level to the edge where they fall back. The
  higher values then lie on its right, as with the built-in.

The built-in does three things this rewrite leaves out. It contours several
levels at once. It skips cells with a NaN. And when a point's value is the
level, or a crossing lies within half an ulp of a point, it makes the
crossings of every edge there into that one point. `edge_crossing` reports
that case by returning the point as both ends. The example picks a level no
crossing lands on a point at.

### `carry`, map by map

`td.carry(data, out, points=..., cells=...)` builds a dataset on the new
topology `out`, and brings everything of `data` that its maps say how to
bring:

| Map | Output entity `i` is | Used for |
|---|---|---|
| `Same()` | input entity `i` | points or cells kept as they are |
| `Take(ids)` | input entity `ids[i]` | a subset, a reordering |
| `Pieces(ids)` | a part of input cell `ids[i]` | cells cut from cells: triangles, tetrahedra, contour lines |
| `Interpolate(ends, weights)` | the point `weights[i]` of the way from input point `ends[i][0]` to `ends[i][1]` | points made on edges: contours, clips |

Each kind of entity has its own keyword. `points=` and `cells=` are the usual
two. `faces=` and `edges=` carry values stored on a mesh's faces or edges.
`faces_to_cells=` and `edges_to_cells=` say that the output's cells are the
input's faces or edges, as in an extracted surface or extracted edges.

What goes with each map:

- **Point fields** (`H1` fields and values on points) go by the points map.
  `Interpolate` blends them as `va + w * (vb - va)`, so the contour's points get
  `RTData` (the level, to rounding) and every other point field. A field of
  integers cannot be blended, and is left behind.
- **The geometry** is a point field like any other, so `Interpolate` places the
  new points. The filter never computes a position. A filter that computes
  positions some other way passes them as `geometry=`.
- **Cell fields** (`Constant` fields and values on cells) go by the cells map:
  each line takes its triangle's value. `L2` fields go only with `Same` and
  `Take`, since a piece of a cell is not the whole cell their basis describes.
  `H1` fields above order 1 keep their order under `Same` and `Take` too, and
  bring their corner values under other maps.
- **Sets** go by the same maps, so a line is in a cell set when its triangle
  is. A point set cannot go through `Interpolate`, and is left behind.
- **Ghost marks.** A line takes its triangle's mark. A new point takes the
  mark of its edge's first end by position, so the piece that owns that end
  owns the point, and each line or point of the whole's contour is owned once
  over the pieces. Pieces without ghost cells cannot tell who owns what they
  make on the edges they share. Their output records that its marks were lost,
  as `shrink` above describes. `td.generate_global_ids` of the outputs then
  decides the owners by matching positions, which works because every piece
  computes a shared point to the same bits.
- **Global ids** go with entities that are input entities (`Same`, `Take`).
  Made points and pieces of cells have none.
- **The frame**, the origin, spatial unit and time, always comes along.

A kind of entity with no map carries nothing: no fields, sets, marks or ids of
that kind. `fields=` (a list of names) limits which fields are carried.

## A reduction over pieces: `entropy`

```python
--8<-- "docs/examples/data_filters.py:pieces"
```

This filter takes a `PartitionedDataSet`, the [pieces](../data/pieces.md) of
a dataset, and returns one number for the whole.

- **Count each value once.** `td.owned_values` gives the values of the cells
  and points a piece owns. It leaves out ghost copies and hidden cells, so a
  value shared by two pieces is counted once. It refuses a piece whose marks
  were lost.
- **Reduce over the blocks.** `tack.communication.Blocks` holds this rank's
  pieces, one [block](../reference/glossary.md) each. `all_reduce` combines
  one value per block in global block order, so a floating-point sum has the
  same bits however the pieces are spread over ranks. The same function runs
  unchanged under MPI: give the `PartitionedDataSet` a `World(comm)`. See
  [Communication](../communication/index.md).

Algorithms that need their neighbours' data rather than a reduction use the
rest of the Communication layer: exchanges and ghost plans
([Exchange and plans](../communication/exchange-and-plans.md)) and sending
points to the pieces that hold them ([Routing](../communication/routing.md)).

## What to keep, and what forgetting it gives

| Concern | Do | Forgetting it gives |
|---|---|---|
| Origin | Convert world parameters with `data.origin` on the host; build datasets through `carry`, `keep_marks`, or with `origin=` | Wrong positions, visible only far from zero |
| Device | Allocate with `device=data.device` | An error on a dataset on another device |
| Implicit arrays | `size_of`, `materialize`, or index them in a kernel | `AttributeError` on `.shape` |
| Hidden cells | Start from `visible(data)` | Cells with invalid data processed |
| Ghost marks | `carry` maps, `keep_marks`, or `drop_marks` | Shared cells and points counted once per piece |
| Units | `units.magnitude` for parameters; `units.delta` or `units.derived` for results | A number in the wrong unit, or a wrong unit on the result |
| Pieces | `owned_values`, `Blocks.all_reduce` | Ghost copies counted, or results that depend on the rank count |

## Run it

```bash
uv run python docs/examples/data_filters.py --arch cpu --check
```

The check holds each rewrite to its built-in on the image, its tetrahedra and
a ghosted piece, and a threshold with bounds in degrees Celsius and kelvin to
the built-in's. The test suite holds the same rewrites, and a seventh
(`net_flux`, cells summing their faces), to the built-ins with a check that
they import nothing private. [Download the complete program](../examples/data_filters.py).

## Details

- **The public pieces.** Everything this page uses is public: `td.carry` and its
  maps (`Same`, `Take`, `Pieces`, `Interpolate`), `edge_crossing`,
  `keep_marks`, `drop_marks`, `visible`, `owned_values`, `Set.from_flags`,
  `td.units` and `tack.communication`. A built-in filter that comes to need
  anything else is a gap to report. The
  [Data Model reference](../vis-guide/reference/data-model.md#filters-of-your-own)
  documents each name.
- **Higher-order data.** A filter that reads corner values, as these do, sees an
  `H1` field above order 1 through its corners. The built-in cuts instead cut
  such cells on linear sub-cells, with each crossing placed on the field itself;
  see [Extract](extract.md).
- **Several fields of one name.** A point field and a cell field may share a
  name. `data.fields[name, td.Constant]` picks the cell one; see
  [Fields](../data/fields.md).

## Where to go next

- [VTK](vtk.md) puts a filter of your own into a VTK pipeline, beside VTK's
  filters.
- [Filters over pieces](over-pieces.md) shows what changes when the data
  comes in pieces: ghost cells, global ids, merging, MPI.
