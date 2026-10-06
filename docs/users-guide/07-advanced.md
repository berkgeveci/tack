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

For a multi-dimensional field the index is a tuple with one index per
dimension, and a vector may supply several of them. On a vector field a
vector value updates every component of the element, one atomic per
component:

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

Arithmetic works on whole vectors, component by component, and a scalar
operand is applied to every component. So do the math builtins, the casts
and conditional expressions:

```python
@tack.kernel
def step(pos, vel, grid, out, dt, n):
    for i in range(n):
        v = vel[i]
        v = min(max(v, -10.0), 10.0)        # clamp each component
        cell = int(floor(pos[i] / 0.25))    # a vector of i32 cell indices
        v = v if pos[i].y > 0.0 else -v     # one condition, whole vectors
        speed = v.norm()                    # methods are listed below
        vel[i] = v
        pos[i] += v * dt                    # augmented store to a field element
        out[i] = grid[cell] * speed         # grid[cell] is grid[cell[0], cell[1], cell[2]]
```

A list of scalars is a vector, so `tack.Vector` can be left out wherever
a vector is expected: `pos[i] = [x, y]`, `v = [0.0, 0.0]`,
`w.dot([1.0, 0.0])`, `p - [1.0, 0.0]`. A matrix still needs
`tack.Matrix([[...], [...]])`.

The vector methods are `norm()`, `norm_sqr()`, `dot(w)`, `cross(w)`
(3-vectors), `normalized()`, and the reductions over components `sum()`,
`min()` and `max()`. `norm(eps)` is `sqrt(norm_sqr() + eps)`, a length
that is never zero. `normalized(eps)` divides by `norm() + eps`, for a
vector that may be zero. `min` and `max` as functions take two or more
values, as in Python: `min(a, b, c)`.

Comparing vectors compares component by component and gives a *mask*, a
vector of `0`/`1`. `any` and `all` reduce a mask to one truth value, and
`tack.select(mask, a, b)` picks per component:

```python
@tack.kernel
def confine(pos, vel, alive, n):
    for i in range(n):
        p = pos[i]
        inside = -1.0 <= p <= 1.0                  # a mask, every link evaluated
        if not all(inside):
            alive[i] = 0
        vel[i] = tack.select(inside, vel[i], -vel[i])    # reflect the components outside
        pos[i] = tack.select(p > 1.0, 1.0, p)             # a scalar arm fills every component
        hits = (p > 0.9) and (vel[i] > 0.0)              # and/or act per component on masks
        if any(hits):
            alive[i] += hits.sum()
```

A mask is an ordinary integer vector, so `mask.sum()` counts. `if` and
`while` take a scalar condition: a mask there is rejected with the hint
to reduce it with `any` or `all`.

A device function can return several values, vectors among them, to be
unpacked at the call:

```python
@tack.func
def closest_hit(origin, direction):
    ...
    return distance, normal, color        # a scalar and two vectors

distance, normal, color = closest_hit(o, d)
```

A vector field exchanges data with NumPy flat or with one row per
vector: `v.from_numpy(a)` accepts an array of shape `(*shape, n)` or the
flat `(prod(shape) * n,)`, and `v.to_numpy(vectors=True)` returns
`(*shape, n)`. Plain `v.to_numpy()` returns the flat storage.

Two vectors in one operation must have the same number of components.
An assignment evaluates its whole right side before it stores anything,
so `v = v.cross(w)` and `pos[i] = pos[i].cross(axis[i])` read the old
components. A tuple assignment does the same and then assigns its targets
from left to right, and its targets may be field elements:
`pos[i], vel[i] = p, v`, or `a[i], b[i] = b[i], a[i]` to swap two.

A store to a field element must match the field: a vector of the field's
width, or a scalar, which sets every component (`vel[i] = 0.0`). A vector
of another width, or a vector stored into a field of scalars, is
rejected.

Components are scalars. `vec[0]` and `vec.x` read one (`x`, `y`, `z`, `w`
name the first four); `vec[1] = x`, `vec.y = x` and `vec[1] += x` write
one. The same forms work on a field element without naming the vector
first: `v[i][2]` and `v[i].z` read a component, `v[i][2] = x` and
`v[i].z += x` store one. The index may be a runtime value (`vec[k]` for a
loop variable `k`), which lowers to a chain of selects rather than a
branch. A runtime index must be in `[0, n)`: outside it, on either side,
a read gives the last component and a write does nothing, since a kernel
cannot raise. Only a literal index counts from the end (`vec[-1]`), and a
literal out of range is rejected at lowering.

## Matrices

`tack.Matrix` gives small fixed-size matrices, up to 4×4. Like vectors
they are scalarized: a matrix is its entries, and every operation expands
to scalar arithmetic at lowering.

```python
F = tack.Matrix.field(2, 2, dtype=tack.f32, shape=(n,))    # a 2x2 matrix per particle
C = tack.Matrix.field(2, 2, dtype=tack.f32, shape=(n,))

@tack.kernel
def update(F, C, x, dt, n):
    for p in range(n):
        F[p] = (tack.Matrix.identity(2) + dt * C[p]) @ F[p]   # '@' multiplies
        J = F[p].determinant()
        stress = (F[p] - F[p].inverse().transpose()) * J      # '*' is entry by entry
        x[p] += stress @ tack.Vector([0.0, -1.0]) * dt        # matrix @ vector
```

- **Building one.** `tack.Matrix([[a, b], [c, d]])` from rows of scalars,
  `tack.Matrix([u, v])` from vectors as rows, `tack.Matrix.identity(n)`,
  or `u.outer_product(v)`.
- **Products.** `A @ B` for matrices; `A @ v` and `v @ A` take a vector as
  a column and as a row and give a vector; `u @ v` is the dot product.
  Shapes must agree. `+`, `-`, `*`, `/`, the math builtins and scalar
  operands work entry by entry, as for vectors.
- **Methods.** `transpose()`; for square matrices `trace()`; for 2×2 and
  3×3 `determinant()` and `inverse()`. `inverse()` divides by the
  determinant without checking it. `norm()` and `sum()` run over all
  entries.
- **Entries.** `A[i, j]` reads one and `A[i, j] = x` or `A[i, j] += x`
  writes one, on a matrix variable or directly on a field element
  (`F[p][0, 1]`). The indices may be runtime values, which must then be
  in range.
- **Fields.** A matrix field stores each matrix in row-major order.
  `from_numpy` takes `(*shape, n, m)` and `to_numpy(vectors=True)` returns
  it. `F[p] = A`, `F[p] += A`, `F[p] *= s` and
  `tack.atomic_add(F, p, A)` work as they do for vector fields.
- **Device functions** take and return matrices, alone or among several
  values.

A matrix and a vector with the same number of entries are different
things: storing one where the other belongs, or adding them, is rejected.

The statistics in `tack.algorithms` (`dot`, `norm`, `var`, ...) accept a
vector field and reduce over all its components in storage order.

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
