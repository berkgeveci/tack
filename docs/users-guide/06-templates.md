# Templates

## @tack.device_class

The template system lets you write generic kernels that work with different
data structures. Classes decorated with `@tack.device_class` can be passed
as kernel arguments. They are recognized by type, so the parameter that
takes one needs no annotation.

The short examples below are fragments showing the class protocol. The
[rasterizer](#kernels-as-methods) is a complete program with initialized input
and a checked output; [Physarum](tutorials/physarum.md) uses the same ideas in
a simulation.

```python
@tack.device_class
class AOSArray:
    def __init__(self, data):
        self.data = data       # tack.array → becomes a buffer parameter

    @tack.func
    def get_value(self, i):
        return self.data[i]

@tack.device_class
class ConstantArray:
    def __init__(self, value):
        self.value = value     # instance scalar → runtime parameter

    @tack.func
    def get_value(self, i):
        return self.value
```

A single kernel works with both:

```python
@tack.kernel
def add_arrays(a, b, out, n):
    for i in range(n):
        out[i] = a.get_value(i) + b.get_value(i)

# Works with any combination:
add_arrays(AOSArray(array1), AOSArray(array2), out, n)
add_arrays(AOSArray(array1), ConstantArray(3.14), out, n)
```

## How It Works

When a `@tack.device_class` object is passed as a kernel argument:

- **Class-level scalars** (defined on the class, not in `__init__`) → compile-time
  constants baked into generated code. Changing them triggers recompilation.
- **Instance scalars** (set in `__init__`) → runtime kernel scalar parameters.
  Changing them does **not** trigger recompilation.
- **Instance arrays** (`tack.Array` set in `__init__`) → kernel buffer parameters
- **`@tack.func` methods** → inlined at the call site

Each unique combination of template types and **class-level constants** produces
a separately compiled kernel. Instance scalar values can change without recompilation. Replacing an array
with one whose dtype, vector width or baked-in dimensions differ can create
another variant; array metadata still matters.

## Class Constants vs Instance Scalars

Use **class variables** for structural constants that the compiler needs
(allocation sizes, topology parameters). Use **instance variables** for
runtime values (dimensions, coefficients, thresholds):

```python
@tack.device_class
class ImageFilter:
    block_size = 16          # class variable → compile-time constant
    num_channels = 3         # class variable → compile-time constant

    def __init__(self, width, height, data):
        self.width = width   # instance → runtime parameter (no recompile)
        self.height = height # instance → runtime parameter (no recompile)
        self.data = data     # instance → array parameter

    @tack.func
    def get_pixel(self, x, y):
        return self.data[y * self.width + x]
```

For an object attribute used as an allocation size (`tack.local_array`,
`tack.shared`), use a class constant. Literals, `tack.constant` values and
resolved array dimensions can also supply sizes; an instance scalar is a runtime
parameter. Other changing parameters can be instance variables.

```python
@tack.kernel
def process(filt, out):
    for i in range(filt.width):              # runtime — OK as instance var
        buf = tack.local_array(tack.f32, filt.block_size)  # must be class constant
        # ...
```

Calling the same kernel with different image sizes reuses the compiled kernel:

```python
filt_small = ImageFilter(256, 256, small_data)
filt_large = ImageFilter(1024, 1024, large_data)
process(filt_small, out1)   # compiles once
process(filt_large, out2)   # reuses compiled kernel — no JIT
```

## Inheritance and Function Attributes

A device class can derive from another. The subclass has its
bases' `@tack.func` methods and class constants and may override either;
a method may call an inherited one through `self`:

```python
@tack.device_class
class Fluid:
    rho_0 = 1000.0

    @tack.func
    def pressure(self, rho):
        return self.stiffness * (rho / self.rho_0 - 1.0)

@tack.device_class
class WeaklyCompressible(Fluid):
    gamma = 7.0

    @tack.func
    def pressure(self, rho):                       # overrides Fluid.pressure
        return self.stiffness * ((rho / self.rho_0) ** self.gamma - 1.0)
```

An object can also hold a device function as an attribute and call it,
which is how one class works with several interchangeable functions:

```python
@tack.func
def cubic(r, h): ...

@tack.func
def spiky(r, h): ...

@tack.device_class
class Model:
    def __init__(self, smoothing):
        self.smoothing = smoothing                 # cubic or spiky

    @tack.func
    def density(self, r):
        return self.mass * self.smoothing(r, self.h)
```

The function is resolved when the kernel is compiled, so which function
an attribute holds is part of the kernel's specialization, like a class
constant: a kernel given `Model(cubic)` and then `Model(spiky)` compiles
once for each. Assigning another function to the attribute of an existing
object takes effect at the next call.

## Objects Holding Objects

A device-class object may hold others as attributes, to any depth, and
reach their methods and attributes through them:

```python
@tack.device_class
class Counting:
    def __init__(self, start, step):
        self.start, self.step = start, step

    @tack.func
    def get(self, k):
        return self.start + self.step * k

@tack.device_class
class Permuted:
    def __init__(self, ids, base):
        self.ids = ids                             # an array of indices
        self.base = base                           # any object with get(k)

    @tack.func
    def get(self, k):
        return self.base.get(self.ids[k])

@tack.kernel
def read(arr, out):
    for i in range(out.shape[0]):
        out[i] = arr.get(i)

read(Permuted(ids, Counting(10, 5)), out)          # out[i] = 10 + 5 * ids[i]
read(Permuted(ids, Permuted(more, Counting(0, 1))), out)
```

The tree is one template: every object's arrays and scalars become kernel
parameters, its constants are compiled in, and its methods are inlined, so
`Permuted(ids, Counting(10, 5))` compiles to `10 + 5 * ids[i]`, the code
you would write by hand. A kernel can reach in too (`arr.base.step`). Which
classes the tree holds, and where, is part of the specialization: new
values reuse the compiled kernel, another nesting compiles anew. An object
may not hold itself, directly or not.

Each level costs a few microseconds per call on the host, where the tree is
walked once to gather its arguments: about 4 us per level on the CPU
backend of an M1 Max, against 8 us for a whole call with a one-level
template. The compiled code pays nothing.

## Passing Objects to Functions

A device-class object, or one it holds, can be passed to a `@tack.func` or
to another object's method, which then uses it as it would `self`:

```python
@tack.func
def mean_of_two(arr, k):
    return 0.5 * (arr.get(k) + arr.get(k + 1))

@tack.device_class
class Blend:
    def __init__(self, weights):
        self.weights = weights

    @tack.func
    def mix(self, a, b, k):                        # a and b are objects too
        return self.weights[k] * a.get(k) + (1.0 - self.weights[k]) * b.get(k)

@tack.kernel
def smooth(arr, blend, other, out):
    for i in range(out.shape[0]):
        out[i] = mean_of_two(arr, i) + blend.mix(arr, other, i)
```

Each such call gets its own copy of the callee, with the object bound in, so
it costs nothing at run time; a method may pass on `self` (`helper(self, k)`)
or a held object (`helper(self.base, k)`) the same way. The callee cannot
keep the object in a local or return it.

## Kernels as Methods

A kernel can be a method of the class. Called on an object, it receives
that object as `self`, a template like any other argument, so the
object's arrays, scalars, constants and methods are reached through it
and nothing has to be passed in. Save this complete example in a `.py` file.
It draws one initialized triangle into a `(height, width)` RGB array:

<!-- tack-check: rasterizer -->
```python
import numpy as np
import tack

tack.set_default_device("cpu")

@tack.device_class
class Rasterizer:
    def __init__(self, n, width, height):
        self.n = n
        self.width, self.height = width, height
        self.vertices = tack.empty((n, 3), tack.vec2(tack.f32))
        self.pixels = tack.empty((height, width), tack.vec3(tack.f32))
        self.pixels.fill(0.0)              # draw writes only covered pixels

    @staticmethod
    @tack.func
    def edge(p, a, b):                     # needs nothing from the object
        return (p.x - a.x) * (b.y - a.y) - (p.y - a.y) * (b.x - a.x)

    @tack.func
    def inside(self, p, t):
        a, b, c = self.vertices[t, 0], self.vertices[t, 1], self.vertices[t, 2]
        return self.edge(p, a, b) >= 0.0 and self.edge(p, b, c) >= 0.0 \
            and self.edge(p, c, a) >= 0.0

    @tack.kernel
    def draw(self, shade):
        for row, column in tack.ndindex(self.height, self.width):
            for t in range(self.n):
                if self.inside([column + 0.5, row + 0.5], t):
                    self.pixels[row, column] = [shade, shade, shade]

raster = Rasterizer(1, 32, 32)
tack.copyto(raster.vertices, np.array([[[8, 8], [8, 24], [24, 8]]], dtype=np.float32))
raster.draw(0.8)
image = raster.pixels.numpy()
np.testing.assert_allclose(image[10, 10], [0.8, 0.8, 0.8])
np.testing.assert_array_equal(image[0, 0], [0.0, 0.0, 0.0])
assert np.count_nonzero(image[:, :, 0]) > 0
```

The kernel compiles once per class and is shared by its objects, as a
kernel taking the object as an argument would be; `Rasterizer.draw(raster,
0.8)` is the same call. A device function that uses nothing of the object
can be a `@staticmethod` (the decorator goes above `@tack.func`) and is
still called through `self`. The class must be decorated with
`@tack.device_class`.

## Iterating an Object

A class can say what a loop over it runs through, and a kernel then loops
over the object itself. `__tack_iterate__` names one to three of its scalar
attributes, fastest first:

```python
@tack.device_class
class Grid:
    __tack_iterate__ = ("nx", "ny")

    def __init__(self, nx, ny):
        self.nx = nx
        self.ny = ny

    @tack.func
    def flat(self, c):
        return c[0] + self.nx * c[1]


@tack.kernel
def label(cells, out):
    for c in cells:
        out[cells.flat(c)] = cells.flat(c) + 1
```

With one name, `for c in cells` is `for c in range(cells.n)` and `c` is an
integer. With two or three, it is a `tack.ndindex` over them, slowest
first, and `c` is the vector of indices with `c[0]` running fastest; as the
kernel's parallel loop it launches in that shape, so recovering the
indices costs no division. The object's methods receive `c` as it comes,
so the same kernel runs over a class that iterates one-dimensionally and
one that iterates over a grid. Instance attributes are runtime extents,
as any instance scalar is: a new grid size does not recompile.

## Indexable Objects

A class becomes indexable in kernels by defining Python's own indexing
methods as device functions: `__getitem__`, `__setitem__` for stores, and
`__len__`. A kernel then indexes the object as it would an array:

```python
@tack.device_class
class Counting:
    def __init__(self, n, start, step):
        self.n, self.start, self.step = n, start, step

    @tack.func
    def __getitem__(self, k):
        return self.start + self.step * k

    @tack.func
    def __len__(self):
        return self.n


@tack.kernel
def double(a, out):
    for i in range(len(a)):
        out[i] = a[i] * 2.0


double(tack.arange(5, dtype=tack.f32), out)    # a stored array
double(Counting(5, 1.0, 0.5), out)             # an indexable object
```

The two calls compile to two variants of one kernel: a direct load for the
array, the inlined `__getitem__` for the object. Nothing marks `a` as
either. `obj[i] = v` calls `__setitem__(i, v)`, and `obj[i] += v` calls
`__getitem__` and then `__setitem__`, with the index evaluated once. Several
indices, `obj[i, j]`, arrive as one vector, so `__getitem__(self, k)` reads
`k[0]` and `k[1]`. `len(obj)` may be the parallel loop's bound: it is
evaluated when the kernel launches, like an array's length. An object held
by another is indexed the same way, `self.base[k]`.

An instance tuple of numbers is one runtime scalar per entry, read with a
literal index: `self.shape = (nx, ny)` gives `obj.shape[0]` and
`obj.shape[1]`, and `len(obj.shape)` is 2. A new shape does not recompile.
A tuple on the class, `shape = (4, 4)`, is read the same way but compiled
in, as a class's numbers are: a fixed tile size the compiler can unroll.

```python
@tack.device_class
class Grid:
    def __init__(self, values, nx, ny):
        self.values = values
        self.shape = (nx, ny)

    @tack.func
    def __getitem__(self, k):
        return self.values[k[1] * self.shape[0] + k[0]]

    @tack.func
    def __setitem__(self, k, v):
        self.values[k[1] * self.shape[0] + k[0]] = v


@tack.kernel
def bump(g):
    for i, j in tack.ndindex(g.shape[0], g.shape[1]):
        g[i, j] += 1.0
```

A store into an object without `__setitem__` is refused when the kernel is
compiled, and so is an atomic on an indexable object, which has no memory
of its own to update. The methods run only in kernels: on the host, Python
calls `__len__` to truth-test an object, so test an indexable with
`is None`, not `if obj:`.

## Cell Set Example

Templates are ideal for topology abstractions where the same algorithm
should work with different mesh representations:

```python
@tack.device_class
class CellSetStructured3D:
    points_per_cell = 8  # class constant — used for local_array sizes

    def __init__(self, nx, ny, nz):
        self.nx = nx               # runtime parameter
        self.ny = ny
        self.nxy = nx * ny
        self.nx_plus1 = nx + 1
        self.nxy_plus1 = (nx + 1) * (ny + 1)

    @tack.func
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

@tack.device_class
class CellSetExplicit:
    points_per_cell = 4  # class constant — used for local_array sizes

    def __init__(self, connectivity):
        self.connectivity = connectivity    # tack.array

    @tack.func
    def get_point_id(self, cell_id, local_idx):
        return self.connectivity[cell_id * self.points_per_cell + local_idx]
```

One kernel, two cell set types:

```python
@tack.kernel
def cell_average(cs, point_data, cell_data, n_cells):
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
@tack.device_class
class Hexahedron:
    num_points = 8  # class constant

    @tack.func
    def center(self, dim):
        return 0.5

    @tack.func
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
@tack.kernel
def interpolate_to_center(cs, ct,
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

## Organize an application around state

Use a device class when several kernels share the same arrays and model
parameters. The [Physarum tutorial](tutorials/physarum.md) gathers positions and
headings into `Agents`, uses an initialization kernel as a method, and passes the
object to three simulation stages. Class constants describe the sensing model;
instance dimensions and arrays describe this particular simulation.

The cell-set sketches above show a more generic use: pass different topology
implementations to one kernel. They illustrate a protocol; `tack.dataset` is the built-in
mesh API that follows it, with a template per shape and per topology (see
[Datasets and visualization](09-visualization.md#writing-your-own-algorithm)).
