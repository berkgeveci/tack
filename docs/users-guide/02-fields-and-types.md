# Fields and Types

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
automatically promoted to `f64` when any field argument uses `f64`, preventing
silent precision loss. Otherwise, float scalars default to `f32`.

## Fields

A field is a device-resident array — the fundamental data container in Tack.

```python
# 1D field
data = tack.field(dtype=tack.f32, shape=(1024,))

# 2D field (linearized internally)
grid = tack.field(dtype=tack.f32, shape=(100, 100))

# Integer field
indices = tack.field(dtype=tack.i32, shape=(256,))
```

### Creating from Numpy

The easiest way to create a field with data is `tack.field_like`:

```python
# Create a field from a numpy array (infers shape and dtype)
x = tack.field_like(np.arange(1024, dtype=np.float32))

# Override dtype
y = tack.field_like(some_array, dtype=tack.f32)
```

Or allocate first, then copy:

```python
data = tack.field(dtype=tack.f32, shape=(1024,))
data.from_numpy(np.arange(1024, dtype=np.float32))

# Copy data back to numpy
result = data.to_numpy()
```

On Metal (Apple Silicon), fields use shared memory — `from_numpy`/`to_numpy`
are simple memory copies with no DMA transfers. On CUDA/HIP, these are
explicit host-device copies.

### Field Access in Kernels

```python
@tack.kernel
def example(data, grid):
    for i in range(data.shape[0]):
        val = data[i]           # 1D access
        data[i] = val * 2.0

    for idx in range(grid.shape[0] * grid.shape[1]):
        grid[idx] = 0.0         # Flat indexing into 2D field
```

### Reductions

Fields support GPU-accelerated reductions:

```python
total = data.sum()
minimum = data.min()
maximum = data.max()
average = data.mean()
```

### Statistics and Analysis

GPU-accelerated statistics are available in `tack.algorithms`:

```python
from tack.algorithms import var, std, norm, absmax, count_nonzero, dot, histogram

# Variance and standard deviation (GPU kernel-based)
v = var(data)
s = std(data)

# Norms
l2 = norm(data, ord=2)      # Euclidean
l1 = norm(data, ord=1)      # sum of absolute values
linf = norm(data, ord=float('inf'))  # max absolute value

# Other reductions
mx = absmax(data)            # max |x[i]|
nz = count_nonzero(data)    # number of non-zero elements
d = dot(field_a, field_b)   # dot product Σa[i]*b[i]

# Histogram (atomic-based GPU binning)
counts, edges = histogram(data, bins=50, range=(-3, 3))
```

All operations run entirely on the active GPU backend using atomic
operations for reductions — no host roundtrips.

## Field Utilities

### Constructors

```python
# Allocate filled fields
x = tack.zeros(dtype=tack.f32, shape=(1024,))
y = tack.ones(dtype=tack.i32, shape=(256,))
z = tack.full(tack.f32, (100,), 3.14)
idx = tack.arange(64)                          # [0, 1, 2, ..., 63] as i32
idx_f = tack.arange(64, dtype=tack.f32)         # [0.0, 1.0, ..., 63.0]
```

### Copy, Convert, Reshape

```python
# Copy (new field, separate buffer)
backup = data.copy()

# Convert dtype (returns new field)
data_f64 = data.astype(tack.f64)
indices_u8 = indices.astype(tack.u8)

# Reshape (metadata only, shares buffer — no copy)
flat = grid.reshape((width * height,))
grid2d = flat.reshape((height, width))
```

### Concatenate

```python
# Concatenate 1D fields (same dtype required)
combined = tack.concat([part_a, part_b, part_c])
```

### Size and Length

```python
f = tack.field(dtype=tack.f32, shape=(4, 3))
f.size       # 12 (total elements)
len(f)       # 4 (first dimension)
```

## Scalar Arguments

Kernels accept Python scalars directly — no need to wrap them in fields:

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

## Device Pointer Interop

For interop with external libraries (pycuda, cupy, Catalyst/in-situ), you
can wrap an existing device pointer as a Tack field without copying:

```python
# Wrap a CUDA device pointer (read-only by default)
field = tack.field_from_ptr(cuda_device_ptr, dtype=tack.f32, shape=(n,))

# Wrap with write access
field = tack.field_from_ptr(ptr, dtype=tack.f32, shape=(n,), writable=True)
```

The field does **not** own the memory — Tack will not free it. On CPU, you
can pass a numpy array directly. On Metal, pass an `MTLBuffer` object. On
CUDA/HIP/Level Zero, pass the device pointer as an integer.

Read-only fields will raise an error on `from_numpy()` and `fill()`.
Kernel reads work normally.
