# Quick reference

Use this page while writing a kernel. For a first complete program, start with
[Getting Started](01-getting-started.md).

## Host code and device code

| Task | In Python, outside a kernel | In a kernel or `@tack.func` |
|---|---|---|
| Allocate storage | `tack.empty`, `tack.zeros`, `tack.full`, `tack.array` | Fixed-size scratch with `tack.local_array` |
| Choose a device | `tack.set_default_device("cuda")`, `with tack.device("cuda"):`, `device=` on a constructor, `data.to_device(d)` | Runs on its arrays' device |
| Access values | `data.numpy()` or `np.asarray(data)` (copies), or `data[i]` for one element | `data[i]`, `grid[row, column]` |
| Write values | `tack.copyto(data, host_values)` or `tack.copyto(data, other_array)`, `data.fill(value)` | `data[i] = value` |
| Repeat work | Python timestep loop calling kernels | One parallel `for` in a kernel; sequential nested loops |
| Reuse an operation | Ordinary Python function | Helper decorated with `@tack.func` |
| Change a parameter | Pass a scalar argument | Read that argument |
| Fix a model parameter | Declare `DT = tack.constant(0.01)` | Read `DT` |
| Sum a whole array | `data.sum()` or `tack.algorithms` statistics | Write contributions, or use an atomic where appropriate |
| Wait for the GPU | `tack.sync()`; host array reads also wait | `tack.barrier()` waits only within a workgroup |

Device code supports a numerical subset of Python. NumPy calls, file operations
and arbitrary Python objects belong on the host. Save decorated functions in a
`.py` file so Tack can read their source. See the [execution model](12-execution-model.md).

## Choose storage

| Need | Allocation | NumPy result |
|---|---|---|
| Uninitialized scalar array | `tack.empty((h, w))` (f32) | `numpy()` → `(h, w)` |
| Zero-filled integer array | `tack.zeros((h, w), tack.i32)` | `numpy()` → `(h, w)` |
| Allocate without copying | `tack.empty_like(prototype)` | Same array shape and element type; unspecified values |
| Copy an existing NumPy array | `tack.array(a)` | Same shape and dtype (float64 stays f64) |
| One RGB vector per pixel | `tack.empty((h, w), tack.vec3())` | `numpy()` → `(h, w, 3)` |
| One 2×2 matrix per particle | `tack.empty((n,), tack.mat2())` | `numpy()` → `(n, 2, 2)` |

Initialize every value before reading it. Shape comes first, then dtype, as in
NumPy; the dtype defaults to f32 (`tack.set_default_float(tack.f64)` changes
that). Array shape counts elements; the
dtype describes one scalar, vector or matrix element. `numpy()` appends
component dimensions, and `tack.copyto` expects them. `reshape` preserves the element type and shares storage;
`copy` allocates independent storage. See [Arrays and Types](02-arrays-and-types.md).

## Choose an execution pattern

| Calculation | Pattern | Example |
|---|---|---|
| Independent element update | One iteration owns each output | [First kernel](01-getting-started.md#your-first-kernel) |
| Neighbourhood stencil | Read an old array, write a new array, swap | [Heat diffusion](tutorials/heat.md) |
| Many inputs contribute to one output | Initialize an accumulator, then use atomics | [Particle-to-grid simulation](tutorials/mpm.md) |
| Unordered append with known capacity | Claim slots with the old value from `atomic_add` | [Kernel results](03-kernels.md#return-the-length-of-an-atomic-append) |
| Ordered or exactly sized output | Count, exclusive scan, allocate, write | [Marching squares](tutorials/contours.md) |
| Reuse data within a GPU workgroup | Shared memory and barriers, with complete groups | [Tiled N-body](tutorials/nbody.md) |

Separate globally dependent stages into separate kernel calls. The device runs
them in order. A workgroup barrier cannot synchronize the whole grid. See
[Memory and Parallel Correctness](13-memory-and-parallelism.md).

## Choose a dataset operation

| Need | Start with |
|---|---|
| Build discontinuous nodal coefficients | [Construct an L2 field](../vis-guide/03-fields-and-spaces.md#construct-an-l2-field) |
| Initialize quadratic nodal values | [Construct a quadratic H1 field](../vis-guide/03-fields-and-spaces.md#construct-a-quadratic-h1-field) |
| Continuous, per-cell or discontinuous values | [Fields and Spaces](../vis-guide/03-fields-and-spaces.md) |
| Select whole cells or cut a region | [Filter and Analyze](../vis-guide/04-filters.md) |
| Work per cell or face without hard-coding connectivity | [Write Mesh Algorithms](../vis-guide/05-mesh-kernels.md) |
| Evaluate a field at physical positions | [Probing](../vis-guide/06-pipelines.md#sample-at-physical-positions) |
| Trace a steady velocity field or render extracted geometry | [Pipelines](../vis-guide/06-pipelines.md) |
| Filter and reduce a dataset in pieces, with ghost cells | [Work with Data in Pieces](../vis-guide/07-partitioned.md) |

`tack.Array` describes storage. `tack.dataset.Field` associates values with a
topology through a space. A field's association must be correct before a filter
can interpolate or propagate it.

## Keep these numerical choices explicit

- Use `f32` for a portable starting point. Metal has no `f64`; on Level Zero,
  check `get_backend().supports_f64` after importing `get_backend` from
  `tack.runtime.dispatch`.
- `int(x)` truncates toward zero. Use `int(floor(x))` for the cell containing
  a possibly negative coordinate.
- Vector `*` is component-wise; vector `dot` and matrix `@` form products.
- Integer `/` produces `f32`. Cast both operands to `tack.f64` for a
  double-precision quotient on a backend that supports it.
- Floating parallel sums can differ in their low bits. Compare against an
  independent result with a tolerance chosen for the calculation.

See [Control Flow and Math](04-control-flow.md),
[numerical details](02-arrays-and-types.md#type-casts), and
[reductions](11-reductions-and-scans.md).

## Check, then time

Use `tack.set_debug(True)` to find invalid indices. Compare a small
case with a NumPy reference before increasing the input size. For timing, disable
debug mode, warm the kernel, reuse allocations, and call `tack.sync()` before
stopping the clock. Measure transfers separately when studying kernel execution.

[Debugging and Timing](15-debugging-and-timing.md) has complete examples and a
symptom table. [Backends](08-backends.md) explains capabilities and setup.
