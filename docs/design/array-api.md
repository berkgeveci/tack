# Arrays, element types and dataset fields

Tack separates storage from the meaning of values on a mesh. `tack.Array`
holds typed backend storage; `tack.dataset.Field` couples coefficient values
with a function space on a topology. Lowercase `tack.array` and `tack.empty`
are factory functions that return `Array` instances.

## Shape and element type

```python
import numpy as np
import tack

tack.set_default_device("cpu")
temperature = tack.empty((100,))
velocity = tack.empty((100,), tack.vec3(tack.f32))
stress = tack.empty((100,), tack.mat3(tack.f32))
velocity.fill([1.0, 0.0, 0.0])
assert velocity.shape == (100,)
assert velocity.size == 100
assert velocity.dtype is tack.vec3(tack.f32)
assert velocity.numpy().shape == (100, 3)
```

All three arrays contain 100 elements. Their dtypes describe a scalar,
a vector, and a matrix respectively. A scalar array of shape `(100, 3)`
and a vector array of shape `(100,)` occupy the same number of bytes, but
kernel indexing reads a scalar in the first case and a vector in the second.
A matrix and a vector with the same number of components remain distinct.

`vec(n, scalar_dtype)` constructs a vector element type; `vec2`, `vec3`
and `vec4` are conveniences. `mat(rows, columns, scalar_dtype)` constructs
a matrix type up to 4x4, with square conveniences `mat2`, `mat3` and `mat4`.
Component types default to f32. Type descriptors are immutable and reusable.
They allocate no backend storage and do not define a finite-element space.

## Construction and transfer

| Operation | Values | Allocation and ownership |
|---|---|---|
| `empty(shape, dtype=f32, *, device=None)` | Uninitialized | Independent storage on `device`, by default the default device |
| `array(values, dtype=None)` | Copies input values | Independent storage; scalar dtype inferred if omitted |
| `empty_like(prototype, dtype=None)` | Does not read or initialize values | Independent storage matching metadata |
| `zeros`, `ones`, `full`, and their `_like` forms | Initialized | Same element types as `empty` |
| `array_from_ptr(ptr, shape, dtype)` | Shares existing values | Borrowed external storage, read-only by default |
| `from_dlpack(source)` | Shares existing values | Holds the source alive; preserves supported access constraints |

Without an explicit dtype, `array(np.zeros((n, 3), np.float32))` is a scalar
array of shape `(n, 3)`. It never infers vector elements from dimensions.
Specifying `dtype=vec3()` consumes a trailing dimension of size three and
gives array shape `(n,)`. Matrix dtypes similarly consume two matching
trailing dimensions. Wrong component dimensions raise `ValueError`.

`array(existing_tack_array)` copies on that array's device, or through the
host when `device=` names another. `empty_like` reads only metadata and
allocates on the prototype's device unless `device=` says otherwise. Every
array records its device (`a.device`); `a.to_device(d)` returns the array
itself when its memory is usable on `d`, else a copy. A scalar dtype
override preserves vector/matrix dimensions and changes component precision;
an element dtype override selects another element type at the same array shape.

## Views and operations

`copy` and `astype` allocate independent storage. Passing a scalar type to
`astype` changes component precision while retaining element dimensions.
An explicit element dtype must have the same component dimensions.
`reshape` shares storage and changes the number of array dimensions while
preserving the element type and count. A view preserves imported-storage
lifetime and access restrictions.

`numpy()` and its async equivalent return a host snapshot with component
axes appended to array shape: `(n, 3)` for vectors, `(n, 3, 3)` for matrices.
DLPack exports the same scalar tensor dimensions; importing that tensor
produces a scalar array with those axes because DLPack does not carry Tack's
element-type distinction. It shares values without guessing vector types.
Metal exports its host-addressable storage as a CPU tensor; importing a host
tensor into Metal requires `copy=True`, because arbitrary host pointers cannot
be wrapped as MTLBuffers without a copy.

`scalar_dtype`, `scalar_size`, and `storage_shape` expose storage metadata
for implementation and interop. Storage is contiguous and unpadded, and
interleaved unless the array is structure-of-arrays (below); matrices store
rows together. The compiler continues to lower vector
and matrix accesses to scalar component operations. Runtime type inference
and backend capability checks use the component type; vector/matrix shape
remains part of specialization keys.

Array reductions and `tack.algorithms` statistics reduce scalar components
in storage order; a statistic's explicit `n` counts scalar components.

## Implicit arrays and the indexing protocol

`tack.arrays` holds arrays that compute each value from its index rather
than store it (`CountingArray`, `ConstantArray`, `CartesianProduct`,
`UniformCoordinates`, `Function`, `Transform`, `Cast`, random arrays) and
arrays that rearrange others (`Permutation`, `Concatenate`, `View`,
`Strided`, `ExtractComponent`, `Components`). They play the part of
Viskores' fancy array handles: a kernel reads `a[i]` and the indexing
expression is inlined into it.

They are ordinary `@tack.device_class` objects, and kernels index them
through Python's own protocol, not a Tack-specific one: a device class
whose `__getitem__`, `__setitem__` and `__len__` are `@tack.func` methods is
indexed in a kernel as an array is.

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
```

Template rewriting (`template_rewrite._Indexing`) turns `obj[i]` into
`obj.__getitem__(i)`, `obj[i] = v` into `obj.__setitem__(i, v)`,
`obj[i] += v` into a get and a set sharing one evaluation of the index,
and `len(obj)` into `obj.__len__()`; the calls then inline like any method.
Several indices arrive as one vector. So the library's arrays have no
privileged path: a user's class joins them by defining the same methods,
and a kernel written for stored arrays takes any of them unchanged.

The choices this rests on:

- **Specialization, not dispatch.** Each class, and each composition of
  them (a `Permutation` of a `Transform` of a `CountingArray`), compiles
  its own variant of the kernel, and a composed array inlines to the
  expression one would write by hand. There are no type lists to declare,
  as in Viskores, and no per-element virtual call, as in VTK's generic
  arrays. The cost is a variant per composition a program uses; the
  template part of the cache key keeps them apart and the disk cache keeps
  them across processes.
- **Values that change stay runtime parameters.** Instance numbers (a
  length, a start, a step) and instance tuples of numbers (`self.shape =
  (nx, ny)`, one scalar per entry) are kernel arguments, so a new length
  does not recompile. Class attributes, tuples included, are compiled in.
- **Writability composes.** An array that wraps stored ones writes
  through: a `View` or `Strided` store lands in the wrapped array, a
  `Permutation` store is a scatter, `Concatenate` writes the part holding
  the index, `Components` writes each component to its own array. Each is
  writable exactly when what it writes into is. A computed array defines no
  `__setitem__`, and storing into it is a `TypeError` when the kernel
  compiles. Since every method of a held object is resolved, a missing
  method inside one is refused only when a kernel actually uses it
  (`__tack_refuse__`).
- **No address, no atomics.** An atomic needs a storage location, so an
  atomic on an indexable object is refused.
- **Host operations materialize.** `sum`, `min`, `max` and `mean` compute
  the values into a temporary array on the device of the arrays the
  implicit array holds (`a.device`), then reduce it as a stored array.
  `tack.algorithms` reads implicit inputs in its kernels without storing
  them.

## Structure-of-arrays storage

A vector or matrix array can keep each component in an array of its own:

```python
pos = tack.soa(x, y, z)                                  # existing arrays, no copy
vel = tack.zeros((n,), tack.vec3(), layout="soa")        # new ones
```

The motivation is data that arrives that way: a simulation's coordinate
arrays, or another library's tensors imported with `from_dlpack` or
`array_from_ptr`, on the CPU or a GPU. Copying them into an interleaved
buffer would double the memory and add a pass per exchange.

SOA is a layout of `tack.Array`, not a separate kind of object. A
`SoaArray` has the dtype, shape, device and host API of any vector array;
`a.layout` is `"soa"` (`"aos"` for interleaved and scalar arrays) and
`a.components` gives the component arrays. Kernels index it exactly as an
interleaved array, including component stores, augmented stores and
vector atomics, so one kernel serves both layouts. `tack.arrays.Components`
expresses the same layout through the indexing protocol; it differs in
taking any arrays, computed ones included, and in being an implicit
array rather than storage: it reads back with `numpy()` and reduces, but
takes no `copyto`, `fill` or atomics, and is not an `Array`.

**Lowering.** A `SoaArray` has no buffer of its own, and backends never
see one. Dispatch replaces it, in its place among the arguments, by its
component arrays (`_flatten_soa`, after template expansion, so an SOA
array a device class holds is flattened too), and lowering replaces its
parameter, in the same place, by one parameter per component. Lowering
learns the layout from the vector-array metadata (`("soa", width)`), and
every access computes each component's location: `(array, i * n + c)` for
an interleaved array, `(component c, i)` for an SOA one. Shape queries,
the strides of multi-dimensional indexing and debug-mode extents read the
first component, which has the array's shape. The consequences:

- No backend changed. Each component is an ordinary scalar array
  argument, so what holds for those holds per component: the CPU's
  disjointness check (`tack.soa(x, x)` is simply not disjoint), Metal's
  argument buffer, read-only checks (reported as "component 1 of parameter
  'a'"), atomic alignment and debug-mode bounds checks.
- The layout is part of the call signature, the template keys and the
  vector-array metadata in the IR key, so a kernel holds one variant per
  layout and never runs one layout's code on the other.
- The SOA array's own name is never a value in the IR. A kernel may index
  it, ask its shape or length, or pass it to a device function, which
  then uses the caller's component parameters; binding it to another name
  is refused.

**What is refused, and why.** A scalar atomic at a flat component index
(`atomic_add(a, 3 * i + 1, x)`) names a position in one interleaved buffer,
which an SOA array does not have; an atomic with a vector value works. For
the same reason an SOA array is not exported whole through DLPack,
`export_memory` or to VTK. Its components are, and `tack.array(a,
layout="aos")` copies it into one buffer.

**Host operations** work per component: `numpy()` stacks the components
as trailing dimensions, `copyto` and `fill` split values among them, and
reductions combine the components' results. `copyto` between layouts, and
`array`/`asarray` with another `layout`, copy element by element in a
kernel. Statistics that pair the components of two arrays (`dot`) copy an
SOA input to an interleaved buffer first, so pairs match whatever the
layouts.

## Dataset fields

```python
import tack.dataset as td

data = td.uniform_grid((2, 2, 2))
space = td.H1(data)
values = tack.empty((space.size,), tack.vec3())
values.fill([1.0, 0.0, 0.0])
data.fields["velocity"] = td.Field(space, values)
```

The array's element type controls components; the field's space controls
DOF association and interpolation on the mesh. A three-component array does
not itself mean geometry, H1 velocity, or an H(div) finite-element field.

## Compatibility

The earlier names (`field`, `field_from_numpy`, `field_from_ptr`,
`Vector.field`, `Matrix.field`, `to_numpy`/`from_numpy`) were later removed
without compatibility wrappers, along with the flat host layout of vector
and matrix arrays; host values move with `numpy()` and `tack.copyto`.
Internal `Field` imports name `Array`.

Kernel value construction (`Vector([...])`, `Matrix(...)`, and
`Matrix.identity(...)`) was retained by this change, and later replaced
without compatibility wrappers: in a kernel the element-type names build
values (`tack.vec3(x, y, z)`, `tack.mat2([a, b], [c, d])`), and
`tack.identity(n)` is the identity matrix. On the host those names remain
dtype descriptions.
