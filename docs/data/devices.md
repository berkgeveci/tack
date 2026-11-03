# Data on devices

Running a filter on a GPU starts with putting the dataset on the GPU. You can
make a dataset on any device Tack supports, or move one there, and Tack's
functions then run where the dataset is. The pieces of a large dataset can be
spread over several GPUs. The sections below also explain what happens when a
dataset's parts are on different devices.
[Choose a device](../core/choose-a-device.md) explains devices themselves, for
arrays and kernels. Here the subject is datasets.

A dataset lives on one device. `data.to_device(device)` gives the dataset on
another, and everything Tack does with it then happens there:

<!-- tack-check: devices-first -->
```python
import tack
import tack.dataset as td

tack.set_default_device("cpu")

data = td.wavelet(extent=8)                  # made on the default device
print(data.device)                           # cpu

# The first GPU this machine has, else the CPU.
gpus = [kind for kind in ("cuda", "hip", "level_zero", "metal") if tack.device_count(kind)]
target = tack.device(gpus[0]) if gpus else tack.device("cpu")

moved = data.to_device(target)               # a copy there, or data itself
assert moved.device == target
assert moved.fields["RTData"].values.device == target
assert (moved is data) == (target == data.device)

# Tack's own functions run where the dataset is, whatever the default device.
boundary = td.boundary_faces(moved)
assert boundary.device == target

# Arrays you make for the dataset go on its device too.
flags = tack.zeros((moved.num_cells,), tack.i32, device=moved.device)
moved.sets["flagged"] = td.Set.from_flags(moved, "cells", flags)
print(moved.device, len(boundary), len(moved.sets["flagged"]))
```

On a machine without a GPU the example runs on the CPU, and `to_device`
returns the dataset itself. On one with a GPU it copies the dataset there.

## A dataset lives where its arrays are

A dataset is made of arrays: the topology's connectivity, the geometry's
positions, each field's values, each set's ids, and the ghost marks and global
ids when there are any. All of them must be on one device, and that device is
the dataset's, `data.device`. A field (`field.device`) and a set
(`set.device`) have one too.

Some parts hold no arrays of their own. A structured grid's topology is only
its dimensions, so it records the device it was made for. An implicit array,
such as an image's `UniformCoordinates`, lives where the arrays it holds live,
and the constructors of implicit arrays take `device=` for that reason.

Parts on two devices are refused when you put them together. A `Field` whose
values are on another device than its topology raises `ValueError`, and so
does a `DataSet` whose fields or sets are. The message names both devices,
for example "Field was given arrays on two devices (cuda:0 and cpu)". Move one
of the parts with `to_device` and try again.

The rule is about memory, not about device objects. Two devices that share
memory count as one: `tack.device("cpu", num_threads=1)` and the default CPU
device can use each other's arrays, so a dataset on one is usable on the other
and `to_device` between them returns it as it is.

## Make a dataset on a device

There are three ways to choose the device a new dataset is made on.

- **`device=` on a constructor.** `td.uniform_grid`, `td.rectilinear_grid`,
  `td.point_cloud`, `td.wavelet` and the other sources take it, as do
  `td.StructuredTopology` and every Tack array constructor.
- **`with tack.device("cuda"):`** makes that device the default inside the
  block, on this thread. Everything made inside it without a `device=` goes
  there.
- **`tack.set_default_device("cuda")`** makes it the default for the whole
  process.

A dataset made from existing arrays is on their device. An unstructured
topology, geometry and fields made from CUDA arrays make a CUDA dataset,
whatever the default device is. This is what [Connect a simulation's
data](connect-simulation-data.md) relies on: wrapping a simulation's device
memory puts the dataset on the simulation's device. A structured topology
holds no arrays, so give it the device: `td.StructuredTopology(dims,
device=...)`.

## Work on a dataset's device

Everything in `tack.dataset` runs on the dataset's device and allocates there.
That includes deriving faces and edges, building sets, interpolating fields,
and every algorithm and filter. The default device does not matter: a
function given a dataset on a GPU runs on that GPU even when the default
device is the CPU.

Your own code follows a simpler rule from the core layer. A kernel runs on
the device of the arrays it is given, and a new array goes on the default
device unless you say otherwise. So arrays you make for a dataset should name
its device. There are two ways:

- pass `device=data.device` to the array constructor, as the example does; or
- make them inside `with data.device:`, which makes the dataset's device the
  default for the block.

An array made on the default device and put in a dataset on a GPU is refused,
as described above. This mistake is the most common one with datasets on
GPUs, and the error says what to do.

## Move a dataset

`data.to_device(device)` returns the dataset on `device`. It is the dataset
itself when its memory is already usable there. Otherwise it is a copy, made
like this:

- the topology is made again from its arrays, copied there;
- the geometry and each field are copied, each in the same kind of space on
  the new topology;
- sets keep the same ids, and ghost marks and global ids are copied;
- the origin, spatial unit and time are kept.

What the topology derives, such as its faces, edges and cell locator, is not
copied. It is derived again on the new device the first time something needs
it, so the first operation after a move can take longer than later ones.

Implicit arrays stay implicit: an image's geometry is still a formula after the
move, and a constant field is still one number. A discontinuous (`L2`) field of
order 3 or more may be reordered on the way. Its layout follows the device by
default, as [Storage and layouts](storage-and-layouts.md) explains.

The copy goes through the host. For a large dataset that you will analyze
on a GPU, it is cheaper to make or read it there in the first place.

## Use devices without double precision

Metal has no double precision, and some Level Zero devices lack it too. An
array of `f64` cannot be made on such a device, so a dataset with `f64`
positions or fields cannot move there: `to_device` raises `TypeError`. Kernels
that compute in `f64` are refused there as well.

Make such datasets in single precision from the start. `dtype=tack.f32` on a
constructor does it for one dataset, and single precision is the default
unless `tack.set_default_float(tack.f64)` changed it. Coordinates far from
zero keep their precision in single precision through the dataset's origin
([Coordinates, units and time](units-coordinates-time.md)).

## Spread pieces over several devices

A dataset in pieces, a `td.PartitionedDataSet`, has no device of its own.
Each piece is a dataset with its own device, so the pieces of one partitioned
dataset can be on several GPUs. An algorithm given the partitioned dataset
runs each piece where it is.

<!-- tack-check: devices-pieces -->
```python
import tack
import tack.dataset as td

tack.set_default_device("cpu")

# Every CUDA device this machine has, else the CPU.
devices = [tack.device(f"cuda:{i}") for i in range(tack.device_count("cuda"))]
devices = devices or [tack.device("cpu")]

data = td.wavelet(extent=8)
pieces = td.PartitionedDataSet(piece.to_device(devices[i % len(devices)])
                               for i, piece in enumerate(td.split(data, 4)))
print([str(piece.device) for piece in pieces])     # ['cpu', 'cpu', 'cpu', 'cpu'] here
assert len(pieces) == 4
```

`td.split` cuts a structured grid into pieces, and each piece goes to one
device in turn. Over MPI, each rank usually holds the pieces of its own GPU.
Make that device before starting MPI, as [Run under
MPI](../communication/mpi.md) describes. [Data in pieces](pieces.md) explains
the pieces themselves.

## Details

**Reading values back.** `field.values.numpy()`, `data.positions()` and the
other host reads copy from the device. On a GPU they first wait for the work
queued before them, so a read is also a wait. Read at checkpoints and for
output, not inside a loop of device work.

**Waiting for every device.** `tack.sync()` waits until every kernel
launched so far, on every device, has finished. `data.device.synchronize()`
waits for one device.

**What has a device.** Datasets, fields, sets, ghost marks, global ids,
topologies and octrees each report their device as `.device`. A
`PartitionedDataSet` and an `AMRDataSet` do not, because their pieces may be
on different devices.

## Where to go next

- [Connect a simulation's data](connect-simulation-data.md) makes a dataset
  on a simulation's own device memory.
- [Choose a device](../core/choose-a-device.md) covers devices, defaults and
  what each backend supports.
- [Data in pieces](pieces.md) explains partitioned datasets, owners and ghost
  marks.
