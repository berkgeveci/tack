# Fields and Types

## Scalar Types

PGC provides these scalar types:

| Type | Description | Numpy equivalent |
|------|-------------|-----------------|
| `pgc.f32` | 32-bit float | `np.float32` |
| `pgc.f64` | 64-bit float | `np.float64` |
| `pgc.i32` | 32-bit signed int | `np.int32` |
| `pgc.i64` | 64-bit signed int | `np.int64` |
| `pgc.u32` | 32-bit unsigned int | `np.uint32` |
| `pgc.u64` | 64-bit unsigned int | `np.uint64` |

GPU backends use `f32` by default. Python `float` arguments are mapped to
`f32` (not `f64`) for GPU compatibility.

## Fields

A field is a device-resident array — the fundamental data container in PGC.

```python
# 1D field
data = pgc.field(dtype=pgc.f32, shape=(1024,))

# 2D field (linearized internally)
grid = pgc.field(dtype=pgc.f32, shape=(100, 100))

# Integer field
indices = pgc.field(dtype=pgc.i32, shape=(256,))
```

### Creating from Numpy

The easiest way to create a field with data is `pgc.field_like`:

```python
# Create a field from a numpy array (infers shape and dtype)
x = pgc.field_like(np.arange(1024, dtype=np.float32))

# Override dtype
y = pgc.field_like(some_array, dtype=pgc.f32)
```

Or allocate first, then copy:

```python
data = pgc.field(dtype=pgc.f32, shape=(1024,))
data.from_numpy(np.arange(1024, dtype=np.float32))

# Copy data back to numpy
result = data.to_numpy()
```

On Metal (Apple Silicon), fields use shared memory — `from_numpy`/`to_numpy`
are simple memory copies with no DMA transfers. On CUDA/HIP, these are
explicit host-device copies.

### Field Access in Kernels

```python
@pgc.kernel
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
```

## Scalar Arguments

Kernels accept Python scalars directly — no need to wrap them in fields:

```python
@pgc.kernel
def saxpy(x, y, out, alpha, n):
    for i in range(n):
        out[i] = alpha * x[i] + y[i]

saxpy(x, y, out, 2.5, 1024)  # alpha=2.5, n=1024 passed as scalars
```

Scalars are automatically packed into constant buffers on GPU backends that
have binding limits (like Metal's 31-buffer limit). You can use as many
scalar arguments as you need.

Both Python types and numpy scalar types (`np.float32`, `np.int32`) work.

## Device Pointer Interop

For interop with external libraries (pycuda, cupy, Catalyst/in-situ), you
can wrap an existing device pointer as a PGC field without copying:

```python
# Wrap a CUDA device pointer (read-only by default)
field = pgc.field_from_ptr(cuda_device_ptr, dtype=pgc.f32, shape=(n,))

# Wrap with write access
field = pgc.field_from_ptr(ptr, dtype=pgc.f32, shape=(n,), writable=True)
```

The field does **not** own the memory — PGC will not free it. On CPU, you
can pass a numpy array directly. On Metal, pass an `MTLBuffer` object. On
CUDA/HIP/Level Zero, pass the device pointer as an integer.

Read-only fields will raise an error on `from_numpy()` and `fill()`.
Kernel reads work normally.
