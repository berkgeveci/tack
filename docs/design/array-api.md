# Arrays, element types and dataset fields

Tack separates storage from the meaning of values on a mesh. `tack.Array`
holds typed backend storage; `tack.dataset.Field` couples coefficient values
with a function space on a topology. Lowercase `tack.array` and `tack.empty`
are factory functions that return `Array` instances.

## Shape and element type

```python
import numpy as np
import tack

tack.init(arch=tack.cpu)
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
| `empty(shape, dtype=f32)` | Uninitialized | Independent storage on the active backend |
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

`array(existing_tack_array)` copies on the device. `empty_like` reads only
metadata, including for a prototype on another backend. A scalar dtype
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
for implementation and interop. Storage remains contiguous, interleaved and
unpadded; matrices store rows together. The compiler continues to lower vector
and matrix accesses to scalar component operations. Runtime type inference
and backend capability checks use the component type; vector/matrix shape
remains part of specialization keys.

Array reductions and `tack.algorithms` statistics reduce scalar components
in storage order; a statistic's explicit `n` counts scalar components.
Rendering accepts typed coordinate, normal and RGB arrays
and obtains scalar storage views without a host transfer.

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
`Matrix.identity(...)`) is retained. The element-type factories in this change
are host-side dtype descriptions, not additional kernel value constructors.
