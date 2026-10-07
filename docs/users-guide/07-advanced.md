# Advanced Features

## Local Arrays

`tack.local_array` allocates a per-thread private array. It maps to stack
memory on CPU and private/register memory on GPU:

```python
@tack.kernel
def cached_interp(cs: tack.template(), ct: tack.template(),
                  field1, field2, out1, out2, n_cells):
    for c in range(n_cells):
        # Compute weights once, reuse for both fields
        w = tack.local_array(tack.f32, ct.num_points)
        pid = tack.local_array(tack.i32, ct.num_points)
        for v in range(ct.num_points):
            w[v] = ct.weight(v, 0.5, 0.5, 0.5)
            pid[v] = cs.get_point_id(c, v)

        v1 = 0.0
        v2 = 0.0
        for v in range(ct.num_points):
            v1 = v1 + w[v] * field1[pid[v]]
            v2 = v2 + w[v] * field2[pid[v]]
        out1[c] = v1
        out2[c] = v2
```

The size must be known at compilation: a literal, a class constant, a
`tack.constant`, or an expression involving resolved field dimensions and
constants, such as `CAPACITY * 4`. An instance scalar is a runtime parameter
and is unsuitable for a portable GPU allocation. Initialize each element
before reading it; local scratch storage is not implicitly cleared.

### `local_array_like`

When the local array dtype should match a field parameter, use
`tack.local_array_like` instead of hardcoding the type:

```python
@tack.kernel
def generic_process(data, out):
    for i in range(data.shape[0]):
        buf = tack.local_array_like(data, 8)   # inherits dtype from data
        buf[0] = data[i]
        out[i] = buf[0]
```

This works with both `f32` and `i32` fields without separate kernels.

### Passing Local Arrays to @tack.func

Local arrays can be passed to `@tack.func` functions, enabling methods that
fill an array for the caller. This is useful for cell set abstractions
where `get_cell_points` populates a buffer in one call:

```python
@tack.data_oriented
class CellSetExplicit:
    points_per_cell = 4  # fixed topology; compile-time allocation size

    def __init__(self, connectivity):
        self.connectivity = connectivity

    @tack.func
    def get_cell_points(self, cell_id, pts):
        for v in range(self.points_per_cell):
            pts[v] = self.connectivity[cell_id * self.points_per_cell + v]

@tack.kernel
def cell_average(cs: tack.template(), data, out, n_cells):
    for c in range(n_cells):
        pts = tack.local_array(tack.i32, cs.points_per_cell)
        cs.get_cell_points(c, pts)      # fills pts in one call
        total = 0.0
        for v in range(cs.points_per_cell):
            total = total + data[pts[v]]
        out[c] = total / float(cs.points_per_cell)
```

The inliner aliases the local array name directly — no copy, no pointer
assignment. The `@tack.func` body accesses the caller's array in place.

## Random Numbers

`tack.random` draws random numbers in kernels from a counter-based
generator with explicit state: a kernel seeds a state from the element it
works on and a stream number, and every draw returns the value and the
next state.

```python
from tack import random

@tack.kernel
def init_particles(pos, vel, n, frame):
    for i in range(n):
        state = random.seed(i, frame)          # element i of this frame
        u, state = random.uniform(state)       # f32 in [0, 1)
        z, state = random.normal(state)        # standard normal
        d, state = random.direction3(state)    # a unit 3-vector; direction2 for 2-D
        pos[i] = d * (0.4 + 0.6 * u)
        vel[i] = d * z
```

The sequence is a pure function of the integers it was seeded with, so a
kernel draws the same numbers on every backend and for any thread order,
and a reference written with the module's NumPy mirrors (`np_seed`,
`np_uniform`, `np_normal`, ...) over an array of indices draws exactly
the same ones; `uniform` and the states match to the bit, `normal` and
the directions to the rounding of `log`, `sqrt`, `cos` and `sin`. Unlike
Taichi's `ti.random()`, there is no hidden per-thread state: a different
stream number (a frame counter, a pass, a purpose) gives a different
sequence, and reusing one repeats it.

## Atomic Operations

Atomic operations make a single field update indivisible when several iterations
contribute to it. Initialize the accumulator before the launch. Atomics do not
order arbitrary reads/stores or make a vector update indivisible as a whole; see
[Gather and scatter](13-memory-and-parallelism.md#gather-and-scatter).

```python
@tack.kernel
def histogram(data, bins, n):
    for i in range(n):
        idx = int(data[i] * 10.0)
        tack.atomic_add(bins, idx, 1)
```

Available atomics:
- `tack.atomic_add(field, index, value)` — atomic addition
- `tack.atomic_min(field, index, value)` — atomic minimum
- `tack.atomic_max(field, index, value)` — atomic maximum

Each one is also an expression whose value is the element's value just
before the update, which is how threads claim unique slots and append to
a shared list:

```python
@tack.kernel
def compact(values, out, count, n):
    for i in range(n):
        if values[i] > 0.0:
            slot = tack.atomic_add(count, 0, 1)    # the old count: this thread's slot
            out[slot] = values[i]
```

Reset `count` before each call, allocate enough output capacity, and read it
only after the call completes. Slot order depends on execution order. For ordered
output and exact allocation, use the [count/scan/write tutorial](tutorials/contours.md).

For a multi-dimensional field the index is a tuple with one index per
dimension, and a vector may supply several of them. On a vector field a
vector value updates every component of the element, one atomic per
component (and, as an expression, gives the vector of their old values):

```python
@tack.kernel
def scatter(pos, vel, mass, grid_m, grid_v, inv_dx, n):
    for p in range(n):
        cell = int(pos[p] * inv_dx)                     # a 2-vector of cell indices
        tack.atomic_add(grid_m, cell, mass[p])          # grid_m[cell[0], cell[1]] += ...
        tack.atomic_add(grid_v, (cell[0], cell[1]), mass[p] * vel[p])
```

A plain integer index is the flat, row-major index. On a vector field
with a scalar value it is the flat index of one component.

## Shared Memory

Shared memory is visible to all threads within a workgroup. Use it for
cooperative algorithms like parallel reductions.

These kernels require a GPU backend with `supports_workgroups=True`.
CPU rejects shared memory, barriers, `thread_id` and block reductions.
For scratch storage private to each iteration, use `tack.local_array`
or `tack.local_array_like`; both work on CPU and GPU.

Positive iteration counts must be divisible by 256. Scalar arguments can
control collective branches and loops; field-loaded conditions and
lane-dependent exits are rejected. See the
[workgroup contract](../reference/language-contract.md#workgroups-and-synchronization)
for the conservative supported domain and unsupported collective expressions.

For example, with fully participating 256-lane workgroups:

```python
@tack.kernel
def block_reduce(data, partial_sums, n):
    for i in range(n):
        smem = tack.shared(tack.f32, 256)
        tid = tack.thread_id()
        smem[tid] = data[i]
        tack.barrier()

        # Tree reduction within workgroup
        stride = 128
        while stride > 0:
            if tid < stride:
                smem[tid] = smem[tid] + smem[tid + stride]
            tack.barrier()
            stride = stride // 2

        if tid == 0:
            tack.atomic_add(partial_sums, 0, smem[0])
```

For an `f32` sum, minimum or maximum over the workgroup, `tack.block_sum`,
`tack.block_min` and `tack.block_max` do this in one call; see
[Reductions and Scans](11-reductions-and-scans.md#block-reductions-inside-kernels).

- `tack.shared(dtype, size)` — allocate threadgroup memory
- `tack.barrier()` — synchronize threads in the workgroup
- `tack.thread_id()` — thread index within the workgroup

### `shared_like`

When the shared memory dtype should match a field parameter, use `tack.shared_like`
instead of hardcoding the type:

```python
@tack.kernel
def generic_reduce(data, partial_sums):
    for i in range(data.shape[0]):
        smem = tack.shared_like(data, 256)   # inherits dtype from data
        tid = tack.thread_id()
        smem[tid] = data[i]
        tack.barrier()
        # ... reduction ...
```

This is especially useful for kernels that need to work with both `f32` and `i32`
fields without separate implementations.

## 3D Textures

Wrap a field as a 3D texture for hardware-accelerated trilinear interpolation:

```python
# Create texture from a field
data = tack.field(dtype=tack.f32, shape=(W * H * D,))
data.from_numpy(volume_data.ravel())
tex = tack.texture3d(data, shape=(W, H, D))

@tack.kernel
def sample_volume(tex, output, n):
    for i in range(n):
        u = float(i) / float(n)
        output[i] = tex.sample(u, 0.5, 0.5)  # normalized [0,1] coords
```

Sampling is always trilinear. Which unit does it depends on the backend:

| Backend | Sampling |
|---|---|
| CPU | Software trilinear interpolation, generated in the kernel |
| Metal | Hardware texture units, `texture3d<float>.sample()` |
| CUDA | Hardware, `tex3D` on a texture object |
| HIP | Hardware when the device reports image support; software otherwise (for example on CDNA parts such as the MI300) |
| Level Zero | Hardware `image3d_t` when the device has samplers; software on Xe-HPC (Ponte Vecchio) |

The texture holds a copy of the field taken by `tack.texture3d()`, on every
backend. Writes to the field afterwards are not visible to it until you call
`tex.update()`:

```python
data.from_numpy(next_frame.ravel())
tex.update()   # sampling now sees next_frame
```

The field must be `f32` and hold `W * H * D` elements, and `interp='linear'`
is the only interpolation mode. See [Backend Implementations](../design/backend-implementations.md#textures).

`tex.sample()` also works inside `@tack.func` — texture metadata is
propagated through inlining automatically.

## Vectors

Vectors are part of the everyday kernel language. See [Vectors and matrices](14-vectors-and-matrices.md#vectors)
for arithmetic, components, field layouts, masks and `tack.select`.

## Matrices

See [Matrices](14-vectors-and-matrices.md#matrices) for fixed-size matrix values,
products, methods and fields, and [MPM](tutorials/mpm.md) for a worked application.

## Printing (Debug)

`print()` works inside kernels on CPU, CUDA, and HIP for debugging:

```python
@tack.kernel
def debug_kernel(data, n):
    for i in range(n):
        if i < 3:
            print("val:", data[i])
```

This emits `printf` calls. On Metal, print is a no-op (Metal has no printf).
