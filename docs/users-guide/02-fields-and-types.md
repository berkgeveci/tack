# Arrays and Types

Arrays hold the state that kernels read and write. Copy input values from
NumPy, allocate an output, then check the downloaded result.
The introductory grid examples use `(height, width)` and `[row, column]`.

## An array from input to output

Save this complete example in a `.py` file:

```python
import numpy as np
import tack

tack.init(arch=tack.cpu)
host = np.arange(6, dtype=np.float32).reshape(2, 3)
source = tack.array(host)                            # copy input values
out = tack.empty(dtype=tack.f32, shape=host.shape)            # every output will be written

@tack.kernel
def add_one(source, out):
    for row, column in tack.ndrange(source.shape[0], source.shape[1]):
        out[row, column] = source[row, column] + 1.0

add_one(source, out)
np.testing.assert_array_equal(out.to_numpy(), host + 1.0)
print(out.to_numpy())                                 # [[1. 2. 3.], [4. 5. 6.]]
```

`array` copies its NumPy input; `to_numpy` also returns a copy. Choose
`tack.zeros` when a kernel needs an initial value of zero, and `tack.empty`
when you will write every value before reading it. Allocate arrays once and
reuse them across timesteps.

Use `tack.f32` for a portable starting point for real-valued data and `tack.i32`
for ordinary indices. The [scalar type table](#scalar-types) lists the other
widths; [precision details](#precision-and-integer-arithmetic) cover promotion
and wrapping when your calculation needs them.

## Arrays

`tack.Array` is typed storage on a backend. `tack.empty` allocates it;
`tack.array` copies existing values into it. Both are functions, hence their
lowercase names. A dataset `tack.dataset.Field` is a different object: a space
plus its coefficient values on a topology.

```python
import numpy as np
import tack

tack.init(arch=tack.cpu)
velocity = tack.empty(shape=(4,), dtype=tack.vec3(tack.f32))
velocity.fill([1.0, 0.0, 0.0])
assert isinstance(velocity, tack.Array)
assert velocity.shape == (4,) and velocity.dtype is tack.vec3(tack.f32)
np.testing.assert_array_equal(velocity.to_numpy(), np.tile([1.0, 0.0, 0.0], (4, 1)))
```

An array's **shape counts elements**. Its **dtype describes one element**:
scalar, fixed-size vector or fixed-size matrix. A scalar array of shape
`(4, 3)` and a vector array of shape `(4,)` with dtype `vec3(f32)` both
contain 12 scalars, but a kernel reads a scalar from the former and a
3-vector from the latter.

| Construction | Array shape | Element type | NumPy download shape |
|---|---|---|---|
| `tack.empty(shape=(n,), dtype=tack.f32)` | `(n,)` | f32 scalar | `(n,)` |
| `tack.empty(shape=(n,), dtype=tack.vec3())` | `(n,)` | Three f32 components | `(n, 3)` |
| `tack.empty(shape=(n,), dtype=tack.mat3())` | `(n,)` | 3x3 f32 matrix | `(n, 3, 3)` |

`vec2`, `vec3`, `vec4`, `mat2`, `mat3` and `mat4` default to f32 components.
Use `vec(n, dtype)` for another width and `mat(rows, columns, dtype)` for a
rectangular matrix (up to 4x4). These functions describe types; they do not
allocate arrays.

The same allocation functions work for every element type:

```python
# 1D array
data = tack.empty(dtype=tack.f32, shape=(1024,))

# 2D array (linearized internally)
grid = tack.empty(dtype=tack.f32, shape=(100, 100))

# Integer array
indices = tack.empty(dtype=tack.i32, shape=(256,))
```

A new array's contents are unspecified, on every backend, until a kernel,
`from_numpy` or `fill` writes them. `tack.zeros(dtype, shape)` makes a
array of zeros (`ones` and `full` likewise), filled on the device.

### Copy values or allocate without copying

`tack.array(values, dtype=None)` always creates independent storage and
copies values. With no dtype it infers a scalar type and uses every input
dimension as an array dimension. It never guesses vector or matrix elements:

```python
positions = tack.array(np.zeros((10, 3), np.float32), dtype=tack.vec3())
scalars = tack.array(np.zeros((10, 3), np.float32))
assert positions.shape == (10,)
assert scalars.shape == (10, 3)
```

An explicit vector or matrix dtype consumes matching trailing input dimensions.
A mismatch raises `ValueError`. Lists are accepted too.

`tack.empty_like(prototype)` reads only shape and type metadata, allocates
independent storage on the active backend, and leaves values uninitialized.
It preserves a Tack array's element type. A scalar dtype override changes the
component type; an element dtype override selects a new element type:

```python
scratch = tack.empty_like(positions)                 # ten vec3(f32) elements
integer_positions = tack.empty_like(positions, dtype=tack.i32)  # vec3(i32)
```

For scalar values:

```python
# Create an array from a numpy array (infers shape and dtype)
x = tack.array(np.arange(1024, dtype=np.float32))

# Override dtype
y = tack.array(some_array, dtype=tack.f32)
```

Or allocate first, then copy:

```python
data = tack.empty(dtype=tack.f32, shape=(1024,))
data.from_numpy(np.arange(1024, dtype=np.float32))

# Copy data back to numpy
result = data.to_numpy()
```

On Metal (Apple Silicon), arrays use shared memory — `from_numpy`/`to_numpy`
are simple memory copies with no DMA transfers. On CUDA, HIP and Level Zero,
these are explicit host-device copies. On every GPU backend they take their
place in order with the queued launches: `from_numpy` does not overtake a
launch that still reads the old values, and `to_numpy` waits for the
launches before it (see [When a call returns](08-backends.md#when-a-call-returns)).

### Array Access in Kernels

```python
@tack.kernel
def double(data):
    for i in range(data.shape[0]):
        val = data[i]           # 1D access
        data[i] = val * 2.0

@tack.kernel
def clear(grid):
    for idx in range(grid.shape[0] * grid.shape[1]):
        grid[idx] = 0.0         # Flat indexing into 2D array
```

A kernel has exactly one parallel loop, so independent loops go in
separate kernels.

### Indexing

An array takes one index per dimension, `img[i, j]`, or a single flat,
row-major index, `img[k]`. A kernel given an array whose number of
dimensions does not match the number of indices raises `TypeError` at the
call, with the source position: `grid[i, j]` on a three-dimensional array
would otherwise address some other element silently. A vector of integers
supplies one index per component, so `grid[cell]` is
`grid[cell[0], cell[1]]`.

### Vector and Matrix Arrays

An array can hold a small vector or matrix in each element:

```python
pos = tack.empty(dtype=tack.vec3(tack.f32), shape=(n,))        # a 3-vector per element
vel = tack.empty(dtype=tack.vec2(tack.f32), shape=(height, width))  # on a 2-D grid
F = tack.empty(dtype=tack.mat2(tack.f32), shape=(n,))       # a 2x2 matrix per element
```

In a kernel `pos[i]` is the whole vector and `F[i]` the whole matrix; see
[Vectors](14-vectors-and-matrices.md#vectors) and [Matrices](14-vectors-and-matrices.md#matrices)
for what can be done with them. The storage is one flat array with the
components of each element together (a matrix in row-major order).

Data moves to and from NumPy either flat or one row per element:

```python
pos.from_numpy(points)                   # (n, 3), or the flat (n * 3,)
points = pos.to_numpy()      # (n, 3)
flat = pos.to_numpy().reshape(-1)        # flatten the downloaded snapshot
F.from_numpy(gradients)                  # (n, 2, 2), or flat
gradients = F.to_numpy()     # (n, 2, 2)
pos.fill([0.0, 1.0, 0.0])                # every element to this vector
pos.fill(0.0)                            # every component to this scalar
```

### Reading One Element from the Host

`f[i]`, `f[i, j]` and `f[None]` (for a zero-dimensional array) read one
element in host code: a Python number from an array of scalars, a NumPy
array from a vector or matrix array. Negative indices count from the end.

```python
head = queue[None]                 # a counter a kernel maintains
comet = pos[0]                     # a 3-vector, as a NumPy array
if temperature[i, j] > melting:    # a host-side decision on one value
    ...
```

This is for inspection and for decisions on single values. On CPU and
Metal it reads unified memory directly; on the other backends every read
is a transfer of that one element, so many elements are read at once
with `to_numpy()`. Arrays
are not written by element from the host: build the values in a NumPy
array and `from_numpy` it, `fill` the array, or write them in a kernel.
Iterating an array raises for the same reason.

### A scalar shared between launches

A zero-dimensional array holds one logical element, accessed as `f[None]`:

```python
strength = tack.empty(dtype=tack.f32, shape=())
strength.fill(2.0)

@tack.kernel
def add_strength(data, strength):
    for i in range(data.shape[0]):
        data[i] += strength[None]
```

Use a scalar argument for a host-supplied value that simply changes per call.
A scalar array is useful when kernels also compute or retain that value. A
zero-dimensional vector array works similarly: `gravity = tack.empty(
dtype=tack.vec2(), shape=())` and `gravity.fill([0.0, -9.8])`.

### Reductions

`data.sum()`, `data.min()`, `data.max()` and `data.mean()` reduce every
scalar component of an array and return a Python `float`; statistics such as `var`,
`norm` and `histogram` are in `tack.algorithms`. Vector and matrix components
are reduced together, and `mean()` divides by the scalar component count.
See
[Reductions and Scans](11-reductions-and-scans.md) for where each runs,
its accumulator type and its NaN and empty-array behavior.

## Array Utilities

### Allocate matching storage without copying

Use `tack.array(host)` when you need a copy of existing values.
Use `tack.empty_like` for an output or scratch array that will be written before
it is read:

```python
source = tack.array(np.arange(8, dtype=np.float32))  # copies values
out = tack.empty_like(source)                                 # same shape and dtype
integer_out = tack.empty_like(source, dtype=tack.i32)           # override dtype
grid_out = tack.empty_like(np.empty((4, 8), dtype=np.float32))  # no upload
```

The prototype may be a NumPy array or Tack array. Storage is independent,
writable and allocated on the active backend. Vector and matrix array
prototypes retain their logical shape and component dimensions. A NumPy array
produces a scalar array of the same shape, including any trailing dimensions.
No values are copied or initialized; write every value before reading it.

The kernel helpers `local_array_like(array, size)` and `shared_like(array, size)`
also copy no values. They inherit the scalar component dtype and use the
explicitly supplied scratch size; see [Advanced Features](07-advanced.md#local_array_like).

### Constructors

```python
# Allocate filled arrays
x = tack.zeros(dtype=tack.f32, shape=(1024,))
y = tack.ones(dtype=tack.i32, shape=(256,))
z = tack.full(tack.f32, (100,), 3.14)
idx = tack.arange(64)                          # [0, 1, 2, ..., 63] as i32
idx_f = tack.arange(64, dtype=tack.f32)         # [0.0, 1.0, ..., 63.0]
```

### Copy, Convert, Reshape

```python
# Copy (new array, separate buffer)
backup = data.copy()

# Convert dtype (returns new array)
data_f64 = data.astype(tack.f64)
indices_u8 = indices.astype(tack.u8)

# Reshape (metadata only, shares buffer — no copy)
flat = grid.reshape((height * width,))
grid2d = flat.reshape((height, width))
```

### Concatenate

```python
# Concatenate 1D arrays (same dtype required)
combined = tack.concat([part_a, part_b, part_c])
```

### Size and Length

```python
f = tack.empty(dtype=tack.f32, shape=(4, 3))
f.size       # 12 (total elements)
len(f)       # 4 (first dimension)
```

## Scalar Types

Tack provides these scalar types:

| Type | Description | Numpy equivalent |
|------|-------------|-----------------|
| `tack.i8` | 8-bit signed int | `np.int8` |
| `tack.u8` | 8-bit unsigned int | `np.uint8` |
| `tack.i16` | 16-bit signed int | `np.int16` |
| `tack.u16` | 16-bit unsigned int | `np.uint16` |
| `tack.i32` | 32-bit signed int | `np.int32` |
| `tack.u32` | 32-bit unsigned int | `np.uint32` |
| `tack.i64` | 64-bit signed int | `np.int64` |
| `tack.u64` | 64-bit unsigned int | `np.uint64` |
| `tack.f32` | 32-bit float | `np.float32` |
| `tack.f64` | 64-bit float | `np.float64` |

GPU backends use `f32` by default. Python `float` scalar arguments are
automatically promoted to `f64` when any array argument uses `f64`, preventing
silent precision loss. Otherwise, float scalars default to `f32`.

## Scalar Arguments

Kernels accept Python scalars directly — no need to wrap them in arrays:

```python
@tack.kernel
def saxpy(x, y, out, alpha, n):
    for i in range(n):
        out[i] = alpha * x[i] + y[i]

saxpy(x, y, out, 2.5, 1024)  # alpha=2.5, n=1024 passed as scalars
```

Scalars are automatically packed into constant buffers on GPU backends that
have binding limits (like Metal's 31-buffer limit). You can use as many
scalar arguments as you need.

Both Python types and numpy scalar types (`np.float32`, `np.int32`) work.
Integer arguments use `i32` when their value fits, then `i64`, then `u64`
for positive values above the signed 64-bit maximum. Values below `-2^63`
or above `2^64 - 1` are rejected. Integer literals follow the same rule.

## Type Casts

Inside kernels, `int()` and `float()` cast to `i32` and `f32` respectively.
For explicit control over the target type, use the Tack type cast functions:

```python
@tack.kernel
def precise_compute(x, out):
    for i in range(x.shape[0]):
        val = tack.f64(x[i])      # promote to double precision
        out[i] = tack.f32(val)    # back to single

@tack.kernel
def bitwise_ops(data, out):
    for i in range(data.shape[0]):
        bits = tack.u32(data[i])  # unsigned for bitwise ops
        out[i] = tack.i32(bits >> 8)
```

Available casts: `tack.i8()`, `tack.u8()`, `tack.i16()`, `tack.u16()`, `tack.i32()`, `tack.u32()`,
`tack.i64()`, `tack.u64()`, `tack.f32()`, `tack.f64()`.

Note: `tack.f64()` is not supported on Metal (Apple GPUs lack double precision).

### Precision and integer arithmetic

Start with explicit input dtypes and cast where the computation needs a different
precision. An `f64` output alone does not widen integer division: write
`tack.f64(a) / tack.f64(b)` for a double-precision quotient. Integer arithmetic
wraps at its inferred width. Expand the details when these rules affect your
calculation.

??? note "Literal precision, promotion, wrapping, division and powers"

    Float literals in a kernel take the precision of what they meet, the way
    NumPy treats Python scalars. With an `f64` array `x`, `x[i] * 0.1` computes
    in `f64` with the `f64` nearest 0.1, and `tack.f64(0.1)` is that value
    exactly. A literal on its own or beside only integers is `f32`. A local
    assigned once, to literals (`a = 0.1`), behaves like the literal wherever
    it is read, so `x[i] * a` is exact too. A local assigned only literals more
    than once is `f64` in a kernel with an `f64` array, as float scalar
    arguments are. The
    [language contract](../reference/language-contract.md#floating-point-execution-policy)
    has the full rule.

    Integer arithmetic (`+`, `-`, `*`, negation, and bitwise operations) wraps at
    the expression's fixed width. Integer casts and stores keep the low bits at
    the destination width: `tack.u8(-1)` is 255, and `tack.i8(255)` is -1.
    Widening preserves the source value before wrapping; `tack.i64(tack.u8(255))`
    is 255. Each intermediate operation wraps at its own inferred width.

    Integer promotion preserves both input ranges. For example, `i8 + u16`
    uses `i32`, and `i32 + u32` uses `i64`. Combining a signed type with `u64`
    requires an explicit cast, including comparisons. To add an unsigned literal,
    write `value + tack.u64(1)`. Choose a signed cast instead if signed wrapping
    is intended. Integer `min`/`max` use exact values; `abs` retains the input
    type, so the signed minimum stays negative under the wrapping rule.

    Integer `/` produces an `f32` quotient: `7 / 2` is `3.5`. For `f64`
    precision, explicitly cast the operands, for example
    `tack.f64(a) / tack.f64(b)`; an `f64` output array alone does not change
    the intermediate precision. An evaluated integer divisor must be nonzero.
    With a floating operand, division uses the promoted floating precision.

    Integer `**` and two-argument `pow` preserve the base type and compute exact
    power modulo its width. For an `i8` base, `3 ** 5` is `-13`, regardless of
    the exponent's integer width. Signed/u64 pairs are accepted for both true
    division and power because their type rules differ from integer promotion.
    Integer exponents must be nonnegative; negative literals are rejected.
    `0 ** 0` is `1`. Cast the base to floating point for negative powers or
    floating results: `tack.f32(a) ** e`. With a floating operand, both power
    operands use the promoted floating precision.

    Shifts keep the left operand's type. Counts must be integers from zero up
    to one less than that type's width. Unsigned right shifts fill with zeros;
    signed right shifts fill with the sign bit. Float-to-integer conversions
    truncate toward zero and require finite input with a representable truncated
    value. See the [language contract](../reference/language-contract.md) for
    division constraints and numerical policies still under development.

## Device Pointer Interop

For interop with external libraries (pycuda, cupy, Catalyst/in-situ), you
can wrap an existing device pointer as a Tack array without copying:

```python
# Wrap a CUDA device pointer (read-only by default)
array = tack.array_from_ptr(cuda_device_ptr, dtype=tack.f32, shape=(n,))

# Wrap with write access
array = tack.array_from_ptr(ptr, dtype=tack.f32, shape=(n,), writable=True)
```

The array does **not** own the memory — Tack will not free it. On CPU, you
can pass a numpy array directly. On Metal, pass an `MTLBuffer` object. On
CUDA/HIP/Level Zero, pass the device address as an integer (a NumPy integer
or a `CUdeviceptr` works too); a host address is rejected with `ValueError`.
A launch that uses a wrapped pointer completes before the call returns, so
the memory's owner can read the results at once.

Read-only arrays raise an error on `from_numpy()` and `fill()`, and a kernel
call that passes one where the kernel may store to it raises `ValueError`
before the kernel runs. Kernel reads work normally.

For layout, views and simultaneous accesses, see [Memory and parallel correctness](13-memory-and-parallelism.md).
For DLPack and borrowed allocations, see [Sharing arrays with other libraries](16-interoperability.md).


## Migrating storage code

| Previous spelling | Preferred spelling |
|---|---|
| `tack.field(dtype, shape)` | `tack.empty(dtype=dtype, shape=shape)` |
| `tack.field_from_numpy(values, dtype)` | `tack.array(values, dtype=dtype)` |
| `tack.Vector.field(3, dtype, shape)` | `tack.empty(dtype=tack.vec3(dtype), shape=shape)` |
| `tack.Matrix.field(2, 3, dtype, shape)` | `tack.empty(dtype=tack.mat(2, 3, dtype), shape=shape)` |
| `tack.field_from_ptr(ptr, dtype, shape)` | `tack.array_from_ptr(ptr, dtype, shape)` |

The old allocation entry points remain compatibility names. In particular,
`Vector.field` and `Matrix.field` retain their old scalar dtype, flat host
shape, and flat default NumPy download. New typed arrays expose the element
dtype and logical shape and expand component dimensions in `to_numpy()` by
default. Kernel element access is the same. `td.Field` keeps its name and meaning.
