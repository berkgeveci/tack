# Arrays and types

Every kernel reads its inputs from arrays and writes its results into them,
so arrays are where a calculation starts. You make them on a device, move
values between them and NumPy, and choose their element types. A Python number
passed to a kernel gets a type too, and you can predict it. Some arrays compute
their values instead of storing them.

## An array from input to output

A `tack.Array` is memory on one device with a shape and an element type, much
like a NumPy array that lives on the GPU. This program copies a NumPy array in,
runs a kernel over it, and copies the result out:

<!-- tack-check: input-to-output -->
```python
import numpy as np
import tack

tack.set_default_device("cpu")

host = np.arange(6, dtype=np.float32).reshape(2, 3)
source = tack.array(host)                  # a copy of host, on the device
out = tack.empty(host.shape)               # new storage, values not set

@tack.kernel
def add_one(source, out):
    for row, column in tack.ndindex(source.shape[0], source.shape[1]):
        out[row, column] = source[row, column] + 1.0

add_one(source, out)
print(out.numpy())                         # [[1. 2. 3.]
                                           #  [4. 5. 6.]]
np.testing.assert_array_equal(out.numpy(), host + 1.0)
```

`tack.array` copies its input and keeps its type, here `f32`. `tack.empty`
allocates an array whose values are not set. Use it when a kernel writes every
value before anything reads one. `out.numpy()` returns a new NumPy array, a
copy. Make working arrays once and reuse them across the steps of a
calculation, because allocating is far slower than reusing.

## Make arrays

Every function that makes an array takes the shape first and the element type
second, as NumPy does. Without a type, a new array is `f32`. A type is a Tack
type such as `tack.i32`, or anything NumPy accepts as one, such as `np.int32`
or `"float64"`.

| Function | Values | Example |
|---|---|---|
| `tack.empty(shape, dtype)` | Not set | `tack.empty((100, 100))` |
| `tack.zeros`, `tack.ones` | Zeros, ones | `tack.zeros((256,), tack.i32)` |
| `tack.full(shape, value, dtype)` | `value` everywhere | `tack.full((100,), 3.14)` |
| `tack.arange(start, stop, step)` | A range, as NumPy's | `tack.arange(1, 64, 2)` |
| `tack.array(values, dtype)` | A copy of a NumPy array or a list | `tack.array([1.0, 2.0])` |
| `tack.empty_like(a)`, `zeros_like`, `ones_like`, `full_like` | As above, with `a`'s shape and type | `tack.empty_like(source)` |

A new array from `empty` holds whatever was in memory, on every device. Write
every value before you read it, or use `zeros`. The `_like` functions read only
the prototype's shape and type, never its values, and the prototype can be a
NumPy array or a Tack array. A `dtype=` argument overrides the type.

`tack.array` with no type keeps a NumPy array's type, so a `float64` array
becomes an `f64` array and NumPy's default `int64` becomes `i64`. A list of
Python floats becomes `f32`, and a list of Python integers `i32`, or `i64` if
a value needs it. `tack.asarray` is the same, except that it returns a Tack
array already of the type asked for as it is, without copying.

## Move values in and out

Values move between NumPy and Tack through three calls. `tack.copyto(a,
values)` writes into an existing array. `a.numpy()` reads an array into a new
NumPy array, and `np.asarray(a)` does the same. `a.fill(value)` sets every element.

<!-- tack-check: move-values -->
```python
import numpy as np
import tack

tack.set_default_device("cpu")

data = tack.empty((1024,))
tack.copyto(data, np.linspace(0.0, 1.0, 1024, dtype=np.float32))
assert data.numpy()[-1] == 1.0

tack.copyto(data, 0.5)                     # a scalar fills every element
other = tack.zeros((1024,))
tack.copyto(other, data)                   # from array to array, on the device
assert np.asarray(other).max() == 0.5

data.fill(2.0)
assert data[3] == 2.0                      # read one element, as a Python float
```

`tack.copyto` is strict, like `np.copyto`. The values must have the array's
shape or broadcast to it, and a mismatch raises `ValueError`. Values are
converted to the array's type. Arrays have no slice assignment such as
`a[...] = values`: build the values in NumPy and call `tack.copyto`, use
`fill`, or write them in a kernel.

Reading one element with `a[i]`, `a[i, j]` or a negative index is for
inspecting results and for decisions in host code. On a discrete GPU each such
read is a transfer of its own, so read many values with `numpy()`. Writing one
element from the host, `a[i] = x`, raises `TypeError`, and so does iterating
over an array.

On a GPU, these calls keep their place among the queued launches. A
`tack.copyto` does not overwrite values a queued launch still reads, and
`numpy()` waits for the launches before it. On Apple silicon the CPU and GPU
share memory, so these are plain memory copies. On NVIDIA, AMD and Intel GPUs
they are transfers between host and device memory.

A *zero-dimensional* array holds a single value, written `a[()]` as in NumPy.
It suits a value kernels compute and other kernels read, such as a counter:

```python
strength = tack.empty(())
strength.fill(2.0)

@tack.kernel
def add_strength(data, strength):
    for i in range(data.shape[0]):
        data[i] += strength[()]
```

For a value that only the host changes between calls, pass a scalar argument
instead (see [Scalar arguments](#scalar-arguments)).

## Shape counts elements, the type describes one

An array's *shape* counts its elements. Its *element type* describes one
element, which is a scalar, a small vector or a small matrix. A scalar array of
shape `(4, 3)` and a vector array of shape `(4,)` with element type
`tack.vec3()` both hold twelve numbers. A kernel reads a number from each
element of the first and a 3-vector from each element of the second.

<!-- tack-check: element-types -->
```python
import numpy as np
import tack

tack.set_default_device("cpu")

velocity = tack.empty((4,), tack.vec3())       # four 3-vectors of f32
velocity.fill([1.0, 0.0, 0.0])
assert velocity.shape == (4,) and velocity.size == 4
assert velocity.dtype is tack.vec3(tack.f32)
assert velocity.numpy().shape == (4, 3)        # components become the last axis

positions = tack.array(np.zeros((10, 3), np.float32), dtype=tack.vec3())
assert positions.shape == (10,)                # ten vectors
scalars = tack.array(np.zeros((10, 3), np.float32))
assert scalars.shape == (10, 3)                # thirty numbers
```

On the host, a vector array's components are an extra last axis: `numpy()`
returns shape `(n, 3)` for `n` 3-vectors, and `tack.copyto` expects that shape.
`tack.array` never guesses vectors from a trailing axis of three. Give the
element type, and it consumes the matching trailing axes. `tack.vec2`,
`tack.vec3`, `tack.vec4` and `tack.vec(n)` describe vector types, and
`tack.mat2`, `tack.mat3`, `tack.mat4` and `tack.mat(rows, columns)` matrix
types up to 4 by 4. Their components are `f32` unless you name another type,
as in `tack.vec3(tack.i32)`. [Vectors and matrices](vectors-and-matrices.md)
shows what kernels do with them.

The other properties count elements too:

```python
f = tack.empty((4, 3))
f.size, f.ndim, len(f), f.nbytes           # 12, 2, 4, 48

v = tack.empty((4,), tack.vec3())
v.size, v.ndim, v.nbytes                   # 4, 1, 48
```

## Copy, convert and reshape

Three methods make arrays from arrays:

- `a.copy()` makes a new array with its own storage.
- `a.astype(tack.f64)` makes a new array of another type. A scalar type
  changes a vector array's component type and keeps it a vector array.
- `a.reshape(rows, columns)` or `a.reshape(-1)` returns a view with another
  shape over the same storage. Nothing is copied, so writes through either
  are seen by both.

`tack.concat([a, b, c])` joins one-dimensional arrays of one type into a new
array. [Memory and parallel correctness](memory-and-correctness.md) explains
what to watch for when a kernel gets two views of the same storage.

## Arrays live on a device

Every array lives on one device, recorded as `a.device`. The functions above
make arrays on the default device, which `tack.set_default_device` sets. Each
also takes `device=`, as in `tack.zeros((n,), device="cuda")`, and the
`_like` functions use the prototype's device. `a.to_device("cpu")` returns the
array itself when that device can use its memory, and a copy otherwise.

A kernel runs on the device of its arrays. When they are on the default
device, it runs there. When they are all on another device, it runs on theirs.
Arrays on two devices that cannot address each other's memory raise
`ValueError`. [Choose a device](choose-a-device.md) covers devices in full.

Arrays can also share memory with other libraries instead of copying:
`tack.from_dlpack` takes a PyTorch or CuPy tensor, and `tack.array_from_ptr`
wraps a pointer a simulation hands you. [Share arrays](share-arrays.md)
explains both.

## Scalar types

Tack has ten scalar types, each matching a NumPy type:

| Tack | NumPy | | Tack | NumPy |
|---|---|---|---|---|
| `tack.i8` | `np.int8` | | `tack.u8` | `np.uint8` |
| `tack.i16` | `np.int16` | | `tack.u16` | `np.uint16` |
| `tack.i32` | `np.int32` | | `tack.u32` | `np.uint32` |
| `tack.i64` | `np.int64` | | `tack.u64` | `np.uint64` |
| `tack.f32` | `np.float32` | | `tack.f64` | `np.float64` |

Use `f32` for real numbers and `i32` for indices unless you need more. `f32`
runs on every device, and GPUs often run it much faster than `f64`. Apple GPUs
have no `f64` at all, so with Metal as the device, making an `f64` array
raises `TypeError`. Some Intel GPUs lack it too.
[Choose a device](choose-a-device.md) says which devices have it.

### The default float type

`f32` is the type a new array gets when nothing names one. A program that
works in double precision throughout can change that once:

```python
tack.set_default_float(tack.f64)       # tack.get_default_float() reads it

x = tack.zeros((n,))                   # f64
v = tack.empty((n,), tack.vec3())      # three f64 components
y = tack.array([0.1, 0.2])             # f64: Python floats follow it
z = tack.array(host_f32)               # f32: a NumPy array keeps its type
```

The setting covers `empty`, `zeros` and `ones` without a type, Python floats in
`array`, `full` and `arange`, vector and matrix types without a component
type, and the implicit arrays below. It is process-wide, like the default
device, and arrays already made keep their type. It does not change kernels:
a kernel computes in the precision of the arrays it is given.

## Scalar arguments

Kernels take Python numbers as arguments directly. A scalar argument is a
runtime value, so calling with a new value does not compile again:

```python
@tack.kernel
def saxpy(x, y, out, alpha):
    for i in range(x.shape[0]):
        out[i] = alpha * x[i] + y[i]

saxpy(x, y, out, 2.5)
saxpy(x, y, out, 3.0)                  # the same compiled kernel
```

Tack chooses a scalar's type from its value and the kernel's arrays:

- A Python `float` is `f32`, or `f64` when any array argument of the kernel is
  `f64`. So a double-precision calculation does not lose precision through
  its scalars.
- A Python `int` is `i32` when its value fits, then `i64`, then `u64` for
  values above the largest `i64`. A value outside all three raises `TypeError`.
- A NumPy scalar follows the same rules as the Python number it holds. So
  `np.float64(2.5)` is `f32` in a kernel whose arrays are all `f32`, and
  `np.int16(1)` is `i32`. To choose a scalar's type yourself, cast it in the
  kernel (see [Casts](#casts)).

A kernel may take as many scalar arguments as you like. Because the type
depends on the value, passing `5` and later `2**40` compiles two versions of
the kernel, one for `i32` and one for `i64`.

## Casts

In a kernel, `float(x)` converts to `f32` and `int(x)` to `i32`, truncating
toward zero. To convert to any other type, call the type:

```python
@tack.kernel
def precise(x, out):
    for i in range(x.shape[0]):
        wide = tack.f64(x[i])          # compute in double precision
        out[i] = tack.f32(wide * wide)

@tack.kernel
def high_byte(data, out):
    for i in range(data.shape[0]):
        bits = tack.u32(data[i])       # unsigned, for bit operations
        out[i] = tack.i32(bits >> 24)
```

Every scalar type is a cast: `tack.i8()` to `tack.u64()`, `tack.f32()` and
`tack.f64()`. Integer casts keep the low bits, so `tack.u8(-1)` is 255.
[Control flow and math](control-flow-and-math.md#how-arithmetic-behaves)
explains how types combine in arithmetic, and why `int(floor(x))` differs
from `int(x)` for negative numbers.

## Structure of arrays

A vector array stores each element's components together: x, y, z, then x, y,
z. Simulations often store them apart instead, one array per component.
`tack.soa(x, y, z)` makes a `vec3` array from three existing arrays without
copying them. `layout="soa"` makes a new one that way.

<!-- tack-check: soa-arrays -->
```python
import numpy as np
import tack

tack.set_default_device("cpu")

@tack.kernel
def advance(pos, vel, dt):
    for i in range(len(pos)):
        pos[i] += vel[i] * dt

x, y, z = (np.zeros(4, np.float32) for _ in range(3))     # someone else's arrays
pos = tack.soa(tack.from_dlpack(x), tack.from_dlpack(y), tack.from_dlpack(z))
vel = tack.full((4,), [1.0, 2.0, 3.0], tack.vec3(), layout="soa")

advance(pos, vel, 0.5)
assert np.allclose(y, 1.0)                     # written into their memory
assert pos.layout == "soa" and pos.components[1].size == 4
assert np.allclose(pos.numpy(), [[0.5, 1.0, 1.5]] * 4)
```

Kernels index a structure-of-arrays array exactly as they index an interleaved
one, so the same kernel takes either layout and compiles a version for each.
On the host it reads and writes as any vector array. `a.components` gives the
component arrays back, and `tack.array(a, layout="aos")` copies it into one
interleaved array. Two things need one buffer and so work only on the
interleaved layout: an atomic at a flat component index, and exporting the
whole array to another library. Its components export individually.

## Implicit arrays

An *implicit array* computes each value from its index instead of storing it.
`tack.arrays` has them: a counting sequence, a constant, a permutation of
another array, a view of part of one, a function of the index, and more. A
kernel indexes them as it indexes a stored array, so a kernel written for
stored arrays takes them unchanged:

<!-- tack-check: implicit-arrays -->
```python
import numpy as np
import tack
from tack import arrays

tack.set_default_device("cpu")

@tack.kernel
def scaled(a, out, s):
    for i in range(len(a)):
        out[i] = a[i] * s

out = tack.zeros((4,))
scaled(tack.array([1.0, 2.0, 3.0, 4.0]), out, 2.0)        # stored values
assert np.allclose(out.numpy(), [2, 4, 6, 8])

scaled(arrays.CountingArray(4, 1.0, 1.0), out, 2.0)       # 1, 2, 3, 4, computed
assert np.allclose(out.numpy(), [2, 4, 6, 8])

ids = tack.array([3, 0, 2, 1])
scaled(arrays.Permutation(ids, arrays.CountingArray(4, 0.0, 10.0)), out, 1.0)
assert np.allclose(out.numpy(), [30, 0, 20, 10])
```

They compose: the last call reads a permutation of a counting array. Each kind
of array, and each composition, compiles its own version of the kernel, and
the generated code is the expression you would write by hand. Nothing is
looked up per element.

| Array | Value `k` | VTK counterpart |
|---|---|---|
| `CountingArray(size, start, step)` | `start + k * step` | `vtkAffineArray` |
| `ConstantArray(value, size)` | `value` | `vtkConstantArray` |
| `Permutation(ids, base)` | `base[ids[k]]` | `vtkIndexedArray` |
| `Concatenate(a, b, ...)` | from the part that holds `k` | `vtkCompositeArray` |
| `View(base, start, size)` | `base[start + k]` | |
| `Strided(values, stride, offset)` | `values[offset + k * stride]` | `vtkStridedArray` |
| `Components(a, b, ...)` | a vector of `a[k]`, `b[k]`, ... | |
| `ExtractComponent(base, c)` | component `c` of `base[k]` | |
| `Function(func, size)` | `func(k)`, a device function | `vtkStdFunctionArray` |
| `Transform(func, base)` | `func(base[k])` | |
| `Cast(base, dtype)` | `base[k]` converted | |
| `CartesianProduct(x, y, z)`, `UniformCoordinates(...)` | grid points | the points of a rectilinear grid or an image |
| `RandomUniform`, `RandomNormal` | random numbers | |

Arrays that wrap stored arrays can be written through. A store into a `View`
or a `Strided` lands in the array it wraps, a store into a `Permutation` is a
scatter, and `Components` writes each component to its own array. An implicit
array is writable exactly when what it writes into is. A store into a computed
array, such as a `CountingArray`, raises `TypeError` when the kernel
compiles.

An implicit array is not a `tack.Array`, but it has `numpy()`, `sum()`,
`min()`, `max()` and `mean()`, which compute its values first. Your own
classes join this family by defining `__getitem__` and `__len__` as device
functions, as [Device classes](device-classes.md) shows.

## Details

**Reductions.** `a.sum()`, `a.min()`, `a.max()` and `a.mean()` reduce every
number in an array, vector components included, and return a Python `float`.
[Reductions, scans and sorts](reductions-scans-sorts.md) covers them and the
statistics in `tack.algorithms`.

**Read-only arrays.** An array that wraps memory Tack was given read-only,
such as a read-only DLPack import, refuses `tack.copyto` and `fill`. A kernel
call that passes it where the kernel may store raises `ValueError` before the
kernel runs. Kernels read it normally.

**Indexing in kernels.** A kernel indexes an array with one index per
dimension, `grid[i, j]`, or one flat index in row-major order, `grid[k]`.
[Kernels](kernels.md#index-arrays) has the details.

**Storage.** Storage is contiguous with no padding. Matrices store their
rows one after another. `a.scalar_dtype` is the component type,
`a.storage_shape` the shape with component axes, and `a.layout` is `"aos"`
for interleaved arrays and `"soa"` for structure-of-arrays ones.

## Where to go next

- [Kernels](kernels.md) shows the loops that read and write these arrays.
- [Vectors and matrices](vectors-and-matrices.md) shows what kernels can do
  with vector and matrix elements.
