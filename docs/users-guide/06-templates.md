# Templates

## @pgc.data_oriented

The template system lets you write generic kernels that work with different
data structures. Classes decorated with `@pgc.data_oriented` can be passed
as kernel arguments with `pgc.template()` type hints:

```python
@pgc.data_oriented
class AOSArray:
    def __init__(self, data):
        self.data = data       # pgc.field → becomes a buffer parameter

    @pgc.func
    def get_value(self, i):
        return self.data[i]

@pgc.data_oriented
class ConstantArray:
    def __init__(self, value):
        self.value = value     # Python scalar → compile-time constant

    @pgc.func
    def get_value(self, i):
        return self.value
```

A single kernel works with both:

```python
@pgc.kernel
def add_arrays(a: pgc.template(), b: pgc.template(), out, n):
    for i in range(n):
        out[i] = a.get_value(i) + b.get_value(i)

# Works with any combination:
add_arrays(AOSArray(field1), AOSArray(field2), out, n)
add_arrays(AOSArray(field1), ConstantArray(3.14), out, n)
```

## How It Works

When a `@pgc.data_oriented` object is passed as a `pgc.template()` argument:

- **Field attributes** (`self.data`) become kernel buffer parameters
- **Scalar attributes** (`self.value`) become compile-time constants baked
  into the generated code
- **`@pgc.func` methods** are inlined at the call site

Each unique combination of template types produces a separately compiled
kernel. `add_arrays(AOSArray, AOSArray)` and `add_arrays(AOSArray, ConstantArray)`
are two different compiled kernels with different generated code.

## Cell Set Example

Templates are ideal for topology abstractions where the same algorithm
should work with different mesh representations:

```python
@pgc.data_oriented
class CellSetStructured3D:
    def __init__(self, nx, ny, nz):
        self.nx = nx               # compile-time constant
        self.ny = ny
        self.nxy = nx * ny
        self.nx_plus1 = nx + 1
        self.nxy_plus1 = (nx + 1) * (ny + 1)
        self.points_per_cell = 8

    @pgc.func
    def get_point_id(self, cell_id, local_idx):
        # Compute connectivity from grid dimensions — zero storage
        ci = cell_id % self.nx
        cj = (cell_id // self.nx) % self.ny
        ck = cell_id // self.nxy
        base = ck * self.nxy_plus1 + cj * self.nx_plus1 + ci
        result = base
        if local_idx == 1:
            result = base + 1
        # ... (7 more vertices)
        return result

@pgc.data_oriented
class CellSetExplicit:
    def __init__(self, connectivity, points_per_cell):
        self.connectivity = connectivity    # pgc.field
        self.points_per_cell = points_per_cell  # compile-time constant

    @pgc.func
    def get_point_id(self, cell_id, local_idx):
        return self.connectivity[cell_id * self.points_per_cell + local_idx]
```

One kernel, two cell set types:

```python
@pgc.kernel
def cell_average(cs: pgc.template(), point_data, cell_data, n_cells):
    for c in range(n_cells):
        total = 0.0
        for v in range(cs.points_per_cell):    # loop count is a constant
            total = total + point_data[cs.get_point_id(c, v)]
        cell_data[c] = total / float(cs.points_per_cell)
```

The loop `range(cs.points_per_cell)` compiles to `range(8)` for hexes and
`range(4)` for tetrahedra. The `get_point_id` call inlines to computed
connectivity (structured) or a buffer load (explicit). Same algorithm,
different generated GPU code.

## Cell Type Abstraction

You can separate topology (cell set) from geometry (cell type):

```python
@pgc.data_oriented
class Hexahedron:
    def __init__(self):
        self.num_points = 8

    @pgc.func
    def center(self, dim):
        return 0.5

    @pgc.func
    def weight(self, vertex, r, s, t):
        # Trilinear shape function
        wr = r
        if vertex % 2 == 0:
            wr = 1.0 - r
        ws = s
        if (vertex // 2) % 2 == 0:
            ws = 1.0 - s
        wt = t
        if vertex // 4 == 0:
            wt = 1.0 - t
        return wr * ws * wt
```

A generic interpolation kernel takes both:

```python
@pgc.kernel
def interpolate_to_center(cs: pgc.template(), ct: pgc.template(),
                          point_data, cell_data, n_cells):
    for c in range(n_cells):
        pc0 = ct.center(0)
        pc1 = ct.center(1)
        pc2 = ct.center(2)
        val = 0.0
        for v in range(ct.num_points):
            w = ct.weight(v, pc0, pc1, pc2)
            val = val + w * point_data[cs.get_point_id(c, v)]
        cell_data[c] = val
```

This single kernel works with any cell set + cell type combination.
