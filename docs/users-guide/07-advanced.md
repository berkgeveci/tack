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

The size can be a literal or a template parameter (compile-time constant).

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
    def __init__(self, connectivity, points_per_cell):
        self.connectivity = connectivity
        self.points_per_cell = points_per_cell

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

## Atomic Operations

Atomic operations are safe for concurrent writes from multiple threads:

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

## Shared Memory

Shared memory is visible to all threads within a workgroup. Use it for
cooperative algorithms like parallel reductions:

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

On Metal, this uses hardware texture units with `texture3d<float>.sample()`.
On other backends, Tack generates a software trilinear interpolation fallback.
On Level Zero (Intel GPUs), hardware `image3d_t` sampling is used when the
device supports it.

`tex.sample()` also works inside `@tack.func` — texture metadata is
propagated through inlining automatically.

## Vectors

Tack provides a `Vector` type for multi-component fields. Vector operations
are scalarized at the IR level:

```python
v = tack.Vector.field(3, dtype=tack.f32, shape=(n,))

@tack.kernel
def normalize_vectors(v, n):
    for i in range(n):
        vec = v[i]                   # loads 3 components
        length = sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)
        v[i] = vec / length          # stores 3 components
```

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
