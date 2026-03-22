# Advanced Features

## Local Arrays

`pgc.local_array` allocates a per-thread private array. It maps to stack
memory on CPU and private/register memory on GPU:

```python
@pgc.kernel
def cached_interp(cs: pgc.template(), ct: pgc.template(),
                  field1, field2, out1, out2, n_cells):
    for c in range(n_cells):
        # Compute weights once, reuse for both fields
        w = pgc.local_array(pgc.f32, ct.num_points)
        pid = pgc.local_array(pgc.i32, ct.num_points)
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

### Passing Local Arrays to @pgc.func

Local arrays can be passed to `@pgc.func` functions, enabling methods that
fill an array for the caller. This is useful for cell set abstractions
where `get_cell_points` populates a buffer in one call:

```python
@pgc.data_oriented
class CellSetExplicit:
    def __init__(self, connectivity, points_per_cell):
        self.connectivity = connectivity
        self.points_per_cell = points_per_cell

    @pgc.func
    def get_cell_points(self, cell_id, pts):
        for v in range(self.points_per_cell):
            pts[v] = self.connectivity[cell_id * self.points_per_cell + v]

@pgc.kernel
def cell_average(cs: pgc.template(), data, out, n_cells):
    for c in range(n_cells):
        pts = pgc.local_array(pgc.i32, cs.points_per_cell)
        cs.get_cell_points(c, pts)      # fills pts in one call
        total = 0.0
        for v in range(cs.points_per_cell):
            total = total + data[pts[v]]
        out[c] = total / float(cs.points_per_cell)
```

The inliner aliases the local array name directly — no copy, no pointer
assignment. The `@pgc.func` body accesses the caller's array in place.

## Atomic Operations

Atomic operations are safe for concurrent writes from multiple threads:

```python
@pgc.kernel
def histogram(data, bins, n):
    for i in range(n):
        idx = int(data[i] * 10.0)
        pgc.atomic_add(bins, idx, 1)
```

Available atomics:
- `pgc.atomic_add(field, index, value)` — atomic addition
- `pgc.atomic_min(field, index, value)` — atomic minimum
- `pgc.atomic_max(field, index, value)` — atomic maximum

## Shared Memory

Shared memory is visible to all threads within a workgroup. Use it for
cooperative algorithms like parallel reductions:

```python
@pgc.kernel
def block_reduce(data, partial_sums, n):
    for i in range(n):
        smem = pgc.shared(pgc.f32, 256)
        tid = pgc.thread_id()
        smem[tid] = data[i]
        pgc.barrier()

        # Tree reduction within workgroup
        stride = 128
        while stride > 0:
            if tid < stride:
                smem[tid] = smem[tid] + smem[tid + stride]
            pgc.barrier()
            stride = stride // 2

        if tid == 0:
            pgc.atomic_add(partial_sums, 0, smem[0])
```

- `pgc.shared(dtype, size)` — allocate threadgroup memory
- `pgc.barrier()` — synchronize threads in the workgroup
- `pgc.thread_id()` — thread index within the workgroup

### `shared_like`

When the shared memory dtype should match a field parameter, use `pgc.shared_like`
instead of hardcoding the type:

```python
@pgc.kernel
def generic_reduce(data, partial_sums):
    for i in range(data.shape[0]):
        smem = pgc.shared_like(data, 256)   # inherits dtype from data
        tid = pgc.thread_id()
        smem[tid] = data[i]
        pgc.barrier()
        # ... reduction ...
```

This is especially useful for kernels that need to work with both `f32` and `i32`
fields without separate implementations.

## 3D Textures

Wrap a field as a 3D texture for hardware-accelerated trilinear interpolation:

```python
# Create texture from a field
data = pgc.field(dtype=pgc.f32, shape=(W * H * D,))
data.from_numpy(volume_data.ravel())
tex = pgc.texture3d(data, shape=(W, H, D))

@pgc.kernel
def sample_volume(tex, output, n):
    for i in range(n):
        u = float(i) / float(n)
        output[i] = tex.sample(u, 0.5, 0.5)  # normalized [0,1] coords
```

On Metal, this uses hardware texture units with `texture3d<float>.sample()`.
On other backends, PGC generates a software trilinear interpolation fallback.
On Level Zero (Intel GPUs), hardware `image3d_t` sampling is used when the
device supports it.

`tex.sample()` also works inside `@pgc.func` — texture metadata is
propagated through inlining automatically.

## Vectors

PGC provides a `Vector` type for multi-component fields. Vector operations
are scalarized at the IR level:

```python
v = pgc.Vector.field(3, dtype=pgc.f32, shape=(n,))

@pgc.kernel
def normalize_vectors(v, n):
    for i in range(n):
        vec = v[i]                   # loads 3 components
        length = sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)
        v[i] = vec / length          # stores 3 components
```

## Printing (Debug)

`print()` works inside kernels on CPU, CUDA, and HIP for debugging:

```python
@pgc.kernel
def debug_kernel(data, n):
    for i in range(n):
        if i < 3:
            print("val:", data[i])
```

This emits `printf` calls. On Metal, print is a no-op (Metal has no printf).
