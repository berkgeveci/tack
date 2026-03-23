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
        self.value = value     # instance scalar → runtime parameter

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

- **Class-level scalars** (defined on the class, not in `__init__`) → compile-time
  constants baked into generated code. Changing them triggers recompilation.
- **Instance scalars** (set in `__init__`) → runtime kernel scalar parameters.
  Changing them does **not** trigger recompilation.
- **Instance fields** (`pgc.Field` set in `__init__`) → kernel buffer parameters
- **`@pgc.func` methods** → inlined at the call site

Each unique combination of template types and **class-level constants** produces
a separately compiled kernel. Instance scalars and fields can change freely
between calls without recompilation.

## Class Constants vs Instance Scalars

Use **class variables** for structural constants that the compiler needs
(allocation sizes, topology parameters). Use **instance variables** for
runtime values (dimensions, coefficients, thresholds):

```python
@pgc.data_oriented
class ImageFilter:
    block_size = 16          # class variable → compile-time constant
    num_channels = 3         # class variable → compile-time constant

    def __init__(self, width, height, data):
        self.width = width   # instance → runtime parameter (no recompile)
        self.height = height # instance → runtime parameter (no recompile)
        self.data = data     # instance → field parameter

    @pgc.func
    def get_pixel(self, x, y):
        return self.data[y * self.width + x]
```

**Rule of thumb:** if it controls an allocation size (`pgc.local_array`,
`pgc.shared`), it **must** be a class variable. Everything else can be
an instance variable.

```python
@pgc.kernel
def process(filt: pgc.template(), out):
    for i in range(filt.width):              # runtime — OK as instance var
        buf = pgc.local_array(pgc.f32, filt.block_size)  # must be class constant
        # ...
```

Calling the same kernel with different image sizes reuses the compiled kernel:

```python
filt_small = ImageFilter(256, 256, small_data)
filt_large = ImageFilter(1024, 1024, large_data)
process(filt_small, out1)   # compiles once
process(filt_large, out2)   # reuses compiled kernel — no JIT
```

## Cell Set Example

Templates are ideal for topology abstractions where the same algorithm
should work with different mesh representations:

```python
@pgc.data_oriented
class CellSetStructured3D:
    points_per_cell = 8  # class constant — used for local_array sizes

    def __init__(self, nx, ny, nz):
        self.nx = nx               # runtime parameter
        self.ny = ny
        self.nxy = nx * ny
        self.nx_plus1 = nx + 1
        self.nxy_plus1 = (nx + 1) * (ny + 1)

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
    points_per_cell = 4  # class constant — used for local_array sizes

    def __init__(self, connectivity):
        self.connectivity = connectivity    # pgc.field

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
    num_points = 8  # class constant

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
