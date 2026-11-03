# Connect a simulation's data

This page shows how to make a dataset on a running simulation's own memory:
its point positions, its cell connectivity and its fields, wrapped where they
are instead of copied. It is for simulation developers who want to analyze
data in situ, on the device the simulation already uses and in the pieces it
already has. After it you can build a dataset once over your solver's arrays,
let the solver update them step by step, and have Tack read the new values
each time. [Build meshes and fields](build-meshes-and-fields.md) shows the
same constructors with data Tack owns; this page is about the boundary
between the simulation and Tack.

Here the simulation's arrays are NumPy arrays, which stand for any memory on
any device. Two hexahedra share a face; the simulation has a pressure at each
point and a density in each cell:

<!-- tack-check: connect-first -->
```python
import numpy as np

import tack
import tack.dataset as td

tack.set_default_device("cpu")

# The simulation's memory. On a GPU these would be its device arrays.
points = np.array([[i * 0.5, j, k] for k in range(2) for j in range(2) for i in range(3)],
                  np.float32)                                  # 12 points, x y z interleaved
cell_types = np.array([12, 12], np.uint8)                      # VTK_HEXAHEDRON
offsets = np.array([0, 8, 16], np.int32)
connectivity = np.array([0, 1, 4, 3, 6, 7, 10, 9,
                         1, 2, 5, 4, 7, 8, 11, 10], np.int32)
pressure = np.zeros(12, np.float32)                            # one value per point
density = np.ones(2, np.float32)                               # one value per cell

# Wrap it: nothing is copied.
types_view, offsets_view, connectivity_view = (
    tack.from_dlpack(a) for a in (cell_types, offsets, connectivity))
topology = td.UnstructuredTopology(types_view, offsets_view, connectivity_view,
                                   num_points=len(points))
positions = tack.array_from_ptr(points, (len(points),), tack.vec3())   # 3-vectors
data = td.DataSet(topology, positions, unit="m")
data.fields["pressure"] = td.Field(td.H1(data), tack.from_dlpack(pressure), unit="Pa")
data.fields["density"] = td.Field(td.Constant(data), tack.from_dlpack(density),
                                  unit="kg/m**3")
assert topology.connectivity is connectivity_view and data.geometry.values is positions

# The simulation takes a step, writing its arrays in place.
pressure[:] = 100.0 * points[:, 0]

# The dataset sees the new values without being rebuilt.
print(data.fields["pressure"].values.sum())                    # 600.0
assert data.fields["pressure"].values.sum() == 600.0
```

The dataset is built once. Each time the simulation writes its arrays, the
fields hold the new values, and any analysis run on `data` reads them. The rest
of the page explains each wrapping call, what to do when more than values
change, how the simulation's work and Tack's are ordered, and how a
simulation's devices and pieces carry over.

## Choose how to share each array

Tack shares memory through two calls, and copies through a third. They are
the core layer's; [Share arrays](../core/share-arrays.md) covers them in full.

| The simulation has | Use | Tack holds the memory alive |
|---|---|---|
| An array that speaks DLPack: NumPy, CuPy, PyTorch, JAX | `tack.from_dlpack(array)` | yes |
| A raw address, from C, C++ or Fortran | `tack.array_from_ptr(ptr, shape, dtype, device=)` | no |
| Separate x, y and z arrays | `tack.soa(x, y, z)` of wrapped arrays | as its parts do |
| Anything in another layout | `tack.array(values)`, a copy | it owns the copy |

`from_dlpack` reads the shape and element type from the tensor and keeps the
tensor alive while the Tack array exists. `array_from_ptr` takes the address
and a description you give, and owns nothing. Keep the simulation's memory
alive for as long as the dataset, and any dataset made from it, may read it.

A wrapped array is shared both ways. Tack sees the simulation's writes, and the
simulation sees what Tack kernels write into it. `array_from_ptr` makes a
read-only array unless you pass `writable=True`. A kernel that may store into
a read-only array is refused before it runs, so a read-only wrap guarantees
that nothing in Tack changes the simulation's memory. A DLPack import keeps the
producer's own read-only flag.

## Wrap the positions

A dataset's geometry is one 3-vector per point (`tack.vec3()`). How to wrap
it depends on how the simulation lays the coordinates out.

**Interleaved coordinates**, x y z x y z and so on, are an array of 3-vectors
already. `array_from_ptr` with `dtype=tack.vec3()` wraps them as such. On the
CPU, pass the NumPy array itself, as in the example above. On a GPU, pass the
device address and the device, for example
`tack.array_from_ptr(tensor.data_ptr(), (n,), tack.vec3(), device="cuda")` for
a PyTorch tensor of shape `(n, 3)`. `from_dlpack` of the same tensor gives a
scalar array of shape `(n, 3)`, which is not a geometry.

**Separate coordinate arrays**, all the x first and then all the y, become
one vector array without a copy through `tack.soa`, a structure of arrays:
`tack.soa(tack.from_dlpack(x), tack.from_dlpack(y), tack.from_dlpack(z))`.
A 2D simulation has no z array. `tack.arrays.Components(x, y, zero)` combines
two wrapped arrays and `tack.arrays.ConstantArray(0.0, n)` into 3-vectors that
are computed as they are read, with nothing stored. [Storage and
layouts](storage-and-layouts.md) describes both forms.

**Structured grids** need no positions at all when they are images or
rectilinear grids. Make the geometry from the simulation's origin and spacing
(`td.uniform_grid`), or from its axis arrays (`td.rectilinear_grid` of wrapped
axes), and nothing per point exists.

Positions in double precision make a double-precision geometry. A simulation
whose domain is far from zero, such as one in geographic coordinates, can keep
single precision by storing positions relative to an origin. [Coordinates,
units and time](units-coordinates-time.md) explains the origin.

## Wrap the connectivity

A structured grid's topology is its point dimensions,
`td.StructuredTopology(dims)`, so there is nothing to wrap. An unstructured
grid's topology is VTK's three arrays, which
[Build meshes and fields](build-meshes-and-fields.md) describes. Tack keeps the
simulation's arrays as they are when their types fit:

- **Types** must be `u8` arrays of VTK cell types. A simulation that stores
  types in another integer type needs a conversion, which is a copy of one byte
  per cell (`types.astype(tack.u8)`). A mesh of one shape needs no array:
  `tack.arrays.ConstantArray(12, n, dtype=tack.u8)` is `n` hexahedra.
- **Offsets and connectivity** are kept as given when they are `i32` or `i64`.
  A simulation that stores a fixed number of points per cell, such as an
  `(n, 8)` array of hexahedra, has no offsets array. Give
  `tack.arrays.CountingArray(n + 1, step=8)` for them, and the connectivity
  flattened with `reshape(-1)`, which shares the storage.
- **Corner order** must be VTK's for each shape. If the simulation orders a
  cell's corners differently, the connectivity has to be reordered, which is a
  copy. [Cells and meshes](cells-and-meshes.md) lists VTK's orders.

Pass `num_points=`. Without it, Tack reads the connectivity back to the host
to find the largest point id, which on a GPU is a transfer and a wait.

## Wrap the fields

Each field is the simulation's array in the space that says what its values
mean. The array's length alone cannot say it: a nodal value, a cell average
and a face flux are all one number per something.

- **Nodal values** (one per point) are an `H1` field (point data):
  `td.Field(td.H1(data), values)`.
- **Cell-centred values**, as a finite-volume code keeps them, are a `Constant`
  field (cell data): `td.Field(td.Constant(data), values)`.
- **Discontinuous Galerkin coefficients** are an `L2` field, each cell's values
  at the space's `offsets`, in Tack's node order. [Higher-order and
  finite-element data](higher-order.md) gives the orders.
- **Vector values** follow the rules for positions above: interleaved through
  `array_from_ptr` with a vector `dtype`, separate components through
  `tack.soa`.

Values on faces need care. `td.Values(data, "faces")` holds one value per face,
but Tack numbers the faces itself, when it derives them from the cells. A
simulation's face array matches only if it numbers its faces the same way;
otherwise the values must be reordered, which is a copy.

## Keep the dataset from step to step

A typical in situ loop builds the dataset once, before the first step. The
simulation then writes its arrays in place, and the analysis runs every few
steps on the same `data`. Datasets are values, not a pipeline: nothing reruns
by itself when the simulation's arrays change. You call the analysis again.

What you have to do depends on what the simulation changed:

| What changed | What to do |
|---|---|
| A field's values, in place | Nothing. The next analysis reads the new values. |
| A field moved to a new buffer | Wrap the new buffer and set `data.fields[name]` again. |
| Point positions, in place | Wrap them again and set `data.geometry`, so that cached searches are rebuilt. |
| The connectivity | Make a new topology, then the dataset and fields on it. |
| The number of points or cells | The same: a new topology. |

Positions need the extra step because a topology keeps what it derives.
Faces and edges depend only on the connectivity. A cell or point locator,
which probes and streamlines use to find the cell holding a point, also
depends on the positions. Tack notices when its own kernels write the
positions, and rebuilds the locator then. It cannot see writes the simulation
makes through its own code. Setting `data.geometry` to a newly wrapped array
tells it that the positions are new, and the next search builds its locator
again.

## Order the simulation's work and Tack's

On a GPU, the simulation and Tack each queue work on their own streams. Two
rules keep them in order.

**Finish the simulation's writes before calling Tack.** Tack's launches do not
wait for another library's stream. Synchronize the simulation's stream, or its
device, before the analysis reads memory it just wrote. In PyTorch that is
`torch.cuda.synchronize()`.

**Tack finishes before it returns.** Any Tack launch that uses wrapped memory
or a DLPack import completes before the call returns. The simulation can
read what Tack wrote, or overwrite what Tack read, as soon as the call returns.
Launches that use only Tack's own arrays are queued and return at once, as
usual. [Choose a device](../core/choose-a-device.md) explains when calls
return.

## Use the simulation's device

A dataset lives on the device of its arrays, so wrapping device memory puts
the dataset on that device, and Tack's functions run there. `from_dlpack`
puts the array on the tensor's own device when the default device cannot use
that memory. A CUDA tensor is a CUDA array even when the default device is the
CPU. `array_from_ptr` uses the `device=` you give, or the default device.

All of a dataset's arrays must be on one device. A `Field` or a `DataSet` with
parts on two devices is refused, with an error that names both.
[Data on devices](devices.md) covers devices in full. Some devices need one
more step to share memory:

- **CUDA.** Tack works in each device's primary context, the one PyTorch,
  CuPy and Warp use. Their device pointers are valid in Tack without
  anything more.
- **Level Zero.** A pointer belongs to the context that allocated it. Make a
  device in the simulation's context with
  `tack.device("level_zero", external_context={"driver": ..., "device": ...,
  "context": ...})`, giving the simulation's three handles. Tack never destroys
  a context it adopted.
- **Metal.** There is no zero-copy DLPack import. `array_from_ptr` takes an
  `MTLBuffer`, or host memory that starts and ends on whole pages.

## Hand over the simulation's pieces

A distributed simulation holds its mesh in blocks, each with a layer of ghost
cells copied from its neighbours. Tack's pieces are the same thing: one
`DataSet` per block in a `td.PartitionedDataSet`. [Data in pieces](pieces.md)
explains the model. Your part is to tell Tack which cells and points each
block owns.

Every cell and point that several blocks hold has one owner. On the other
blocks it is a copy, marked in the block's ghost marks: `DUPLICATECELL` for a
cell and `DUPLICATEPOINT` for a point, VTK's `vtkGhostType` bits. Global ids
number each point and cell once across all blocks. Both come from the
simulation, which already knows them.

This line of six cells is shared by two ranks, each with one ghost cell:

<!-- tack-check: connect-pieces -->
```python
import numpy as np

import tack
import tack.dataset as td
from tack import communication as tc

tack.set_default_device("cpu")

COPY_CELL, COPY_POINT = td.Ghosts.DUPLICATECELL, td.Ghosts.DUPLICATEPOINT


def block(first_point, cell_marks, point_marks, temperature):
    """One block of a line of 7 points and 6 cells, as one rank holds it."""
    n = len(point_marks)
    topology = td.StructuredTopology((n,), offset=(first_point,))   # its place in the line
    geometry = tack.arrays.UniformCoordinates((n,), origin=(0.5 * first_point,),
                                              spacing=(0.5,))
    data = td.DataSet(topology, geometry,
                      ghosts=td.Ghosts(cells=cell_marks, points=point_marks),
                      global_ids=td.GlobalIds(cells=first_point + np.arange(n - 1),
                                              points=first_point + np.arange(n)))
    data.fields["temperature"] = td.Field(td.H1(data),
                                          tack.array(temperature, dtype=tack.f32))
    return data


# Rank 0 owns cells 0-2 and points 0-3. Its cell 3 and point 4 are rank 1's.
left = block(0, [0, 0, 0, COPY_CELL], [0, 0, 0, 0, COPY_POINT], [0, 1, 2, 3, 4])
# Rank 1 owns cells 3-5 and points 4-6. Its cell 2 and points 2 and 3 are rank 0's.
right = block(2, [COPY_CELL, 0, 0, 0], [COPY_POINT, COPY_POINT, 0, 0, 0], [2, 3, 4, 5, 6])

# One process holds both blocks here. Under MPI each rank passes its own blocks
# and tc.World(MPI.COMM_WORLD).
pieces = td.PartitionedDataSet([left, right], tc.World())

owned = [td.owned_values(piece, "temperature").numpy() for piece in pieces]
print(owned)            # [array([0., 1., 2., 3.], ...), array([4., 5., 6.], ...)]
assert sum(values.sum() for values in owned) == sum(range(7))   # each point once
```

`td.owned_values` gives a block's values without its copies, so sums over
the blocks count each point once. Algorithms that combine pieces, such as
statistics, rely on the same marks. `offset=` places a structured block in the
whole grid, as the start of a VTK extent does.

Under MPI, each rank makes a `PartitionedDataSet` of the blocks it holds and
the world they share, `tc.World(MPI.COMM_WORLD)`. Tack imports no MPI module
itself: you make the communicator with mpi4py. [Run under
MPI](../communication/mpi.md) covers the setup.

A simulation without ghost cells can still mark its shared points, so that
each has one owner. Filters that need neighbouring cells can then add a ghost
layer themselves ([Filters over pieces](../visualization/over-pieces.md)).

## Details

**Large meshes.** A topology keeps `i32` ids only when every array it may
derive from them fits in `i32`. It sizes them for the largest case, twelve
edges for each cell. Past about 178 million cells it uses `i64`, and `i32`
arrays given to it are converted, which is a copy. Give such a mesh `i64`
offsets and connectivity.

**What Tack keeps.** A topology holds what it derives from the connectivity
(faces, edges, cell groups) and a locator. These are Tack's own allocations,
made on first use and kept with the topology. They are the reason to build the
topology once and keep it while the connectivity stays the same.

**Readers and other libraries.** Data that arrives through VTK, MFEM or Warp's
simdata has its own conversions, some of which share memory too. See
[VTK](../visualization/vtk.md) and [MFEM, Warp and
others](../visualization/mfem-warp-and-others.md).

## Where to go next

- [Data on devices](devices.md) explains where a dataset lives and how to move
  it.
- [Data in pieces](pieces.md) explains ghost marks, owners and global ids.
- [Share arrays](../core/share-arrays.md) has the full rules for DLPack and
  wrapped pointers.
