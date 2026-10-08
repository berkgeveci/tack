# Changelog

All notable changes to Tack are recorded here. Rules cited by name live in
[`docs/reference/language-contract.md`](docs/reference/language-contract.md).

## Unreleased

### Added

- `tack.data.shapes` (tack-vis): VTK's ten linear cells -- vertex, line,
  triangle, pixel, quad, tetrahedron, voxel, hexahedron, wedge and
  pyramid -- as `@tack.data_oriented` classes. A kernel takes one as a
  template argument (`cell`) and calls its methods: parametric points and
  center, shape functions and their gradients, edges and faces, an inside
  test, and, on a cell's gathered points, interpolation, the Jacobian and
  Newton inversion of a world point (`world_to_parametric`). Ids, point
  order, parametric coordinates, edges, faces and shape functions are
  VTK's, and the tests compare them with VTK's. Each compiled kernel is
  specialized to one shape; a mesh of several shapes runs a kernel once
  per shape present. See *Cell shapes* in the visualization chapter.
- `tack.data` cell sets and datasets (tack-vis, prototype):
  `ExplicitCellSet(types, offsets, connectivity)` in
  `vtkUnstructuredGrid`'s layout, `SingleTypeCellSet`, `StructuredCellSet`,
  and `DataSet(points, cells, point_data, cell_data)`.
  `tack.data.for_each_shape(kernel, data, *args)` groups the cells by
  shape on the device (once, kept on the cell set) and runs the kernel
  once per shape present, passing a view that is both the shape and that
  shape's cells: `num_cells`, `point_id(c, j)`, `cell_id(c)`, and for a
  dataset `point(c, j)` and `gather_points(c, pts)`.
  `tack.interop.vtk.vtk_to_dataset` and `dataset_to_vtk` copy unstructured
  and structured grids, with their named point and cell data.
- `tack.data` filters (tack-vis, prototype), each taking a `DataSet` and
  returning one: `cell_centers`, `point_data_to_cell_data`,
  `cell_data_to_point_data` (through `point_links`, the cells using each
  point, built by sorting and kept on the cell set, so the averages are
  reproducible) and `external_faces` (faces of 3D cells that only one cell
  has, found by sorting keyed faces, in each cell's outward order). The
  tests compare them with vtkCellCenters, vtkPointDataToCellData,
  vtkCellDataToPointData and vtkGeometryFilter.
- Rectilinear grids in `tack.data` (tack-vis, prototype):
  `RectilinearCoordinates(x, y, z)` stores only the three coordinate
  arrays, and `rectilinear_grid(x, y, z)` makes a dataset of them with a
  `StructuredCellSet`. Structured cells read their points' coordinates
  from their own (i, j, k), without division; other cells over the same
  coordinates work too. `vtk_to_dataset` and `dataset_to_vtk` convert
  `vtkRectilinearGrid`, and the filters work on these grids.
- `tack.data.contour(data, values, isovalue)` (tack-vis, prototype): the
  isosurface of point data in a dataset's 3D cells, as triangles, for
  unstructured, structured and rectilinear data alike. The case tables are
  VTK's, generated from VTK's cells by `tools/make_contour_tables.py`, and
  the result matches `vtkContourGrid` triangle for triangle, orientation
  included. Points are merged by edge, so the surface is connected, and
  point data is interpolated onto them. Cells below three dimensions
  contribute nothing yet.
- A kernel can loop over a `@tack.data_oriented` object: `for c in cells:`.
  The class declares the attributes the loop runs through as
  `__tack_iterate__`, one to three names, fastest first. One makes `c` an
  integer over `range`; two or three make `c` the vector of indices over
  an `ndrange`, which as the parallel loop launches in its own shape. The
  object's methods take `c` as it comes, so one kernel runs over objects
  that iterate differently. See *Iterating an Object* in the User's Guide.

### Kernels that now compute different results

- A local assigned once, to float literals, takes the precision of what
  it meets where it is read, as the literal written there would: with an
  f64 `x`, `a = 0.1; y = x * a` uses the f64 nearest 0.1. It was an f32
  local, so f64 kernels computed with the f32 rounding of the literal.
  This includes literals a device function returns, such as a cell's
  parametric center returned as a vector of literals, which made shape
  weights f32 on f64 data. f32 kernels are unchanged. A local assigned
  more than once keeps one type, as before.

### Performance changes users may notice

- A kernel's parallel loop over a two- or three-dimensional
  `tack.ndrange` no longer divides to recover its indices. GPUs launch a
  2D or 3D grid of the loop's shape, and CPU workers walk their chunk row
  by row. On an M1 Max a 200³ stencil went from 4.7 to 0.9 ms on Metal and
  from 6.5 to 1.7 ms on CPU, and a 4000×2000 elementwise kernel from 3.3
  to 0.4 ms on Metal. Results are unchanged. Kernels using workgroup
  primitives, and `ndrange` over four or more dimensions, keep the old
  lowering; so does a GPU launch with an extent past the device's grid
  limits, decided per dispatch with no recompile.

### Fixes with no source change needed

- A vector or matrix `tack.constant` with a type, or of integers, read at
  a runtime index (`FACES[4 * f + k]`) is one load from a constant array
  instead of a chain of conditional expressions, a comparison per entry
  at every lookup. On Metal a loop that stores to a field and does many
  such lookups crashed the compiler service at pipeline creation (M1
  Max, macOS 26): "Compilation failed due to an interrupted connection".
  The result is unchanged, including an index outside the table giving
  the last value. Untyped float constants keep the chain, since their
  components take the precision of what they meet.
- An `ndrange` whose sizes are both negative (`ndrange(-3, -4)`) runs no
  iterations. Their product is positive, and the loop ran 12 iterations
  with negative indices.

## 0.3.0 — 2026-10-07

The headline: the kernel language grew the vocabulary that simulation and
graphics code reaches for, found by porting 50 Taichi, Warp and
awesome-taichi programs and checking each against a NumPy reference on
every backend. Vectors work wherever scalars do, small matrices and
masks are values, kernels can be methods and return a value, atomics
return the old value, and constants, `tack.math` and `tack.random` cover
what the ports carried themselves. The same work found and fixed wrong
results that 0.2.0 shipped with, most importantly a Metal compiler fault
that dropped field updates inside loops, and a vector assignment that
read its own partly overwritten target on every backend; see *Kernels
that now compute different results* and *Fixes*.

Validated on Linux (CPU, CUDA on an RTX 4060 Ti, HIP on an MI210 and an
MI300X, Level Zero on an Intel Max 1100) and macOS (CPU and Metal on an
M1 Max), together with the 57 example checks in tack-examples.

### Added

#### Vectors and matrices

- Vectors in the places a scalar works. The math builtins (`min`, `max`,
  `abs`, `floor`, `sqrt`, `pow`, ...) and the casts (`int`, `float`,
  `tack.f32`, ...) apply to each component, with a scalar argument
  repeated: `min(max(v, -1.0), 1.0)`, `int(floor(p))`. A conditional
  expression selects whole vectors on one condition. `x`, `y`, `z` and
  `w` name the first four components, to read (`v.x`, `vf[i].y`) and to
  assign (`v.x = a`, `v.y += a`). A component of a vector field element
  can be stored directly: `vf[i][c] = a`, `vf[i][c] += a`, `vf[i].y = a`.
  `vf[i] += vec` and `vf[i] *= scalar` update an element in place. A
  vector indexes a field one dimension per component: `grid[cell]` is
  `grid[cell[0], cell[1]]`. All of these used to fail IR verification
  with "expected expr node", or with an `AttributeError` for the
  augmented store. Vectors of different widths in one operation, a
  component name past the width, and a vector combined into a scalar
  target are rejected at lowering with the source position.
- `tack.Matrix`: small fixed-size matrices, up to 4×4, scalarized like
  vectors. `tack.Matrix.field(n, m, dtype, shape)` allocates a field of
  them. In kernels, `tack.Matrix([[a, b], [c, d]])`, rows given as
  vectors, and `tack.Matrix.identity(n)` build one; `@` multiplies
  matrices and vectors, with `+`, `-`, `*`, `/` entry by entry;
  `transpose()`, `trace()`, and for 2×2 and 3×3 `determinant()` and
  `inverse()`; `A[i, j]` reads and writes entries; `u.outer_product(v)`
  makes one from two vectors. Matrices pass through device functions,
  tuple assignment, conditional expressions, field stores and atomics as
  vectors do. See *Matrices* in the User's Guide.
- A list of scalars in a kernel or device function is a vector:
  `pos[i] = [x, y]` is `pos[i] = tack.Vector([x, y])`, wherever a vector
  is accepted. It was rejected as "unsupported List". An empty list, a
  list of lists (write `tack.Matrix`), a list as an assignment target and
  a loop over a list are still rejected.
- Vector components by runtime index: `vec[k]` reads and `vec[k] = x` /
  `vec[k] += x` write the component a loop variable or field value
  selects, through a chain of selects; past the last component a read
  gives the last component and a write does nothing. `vf[i][c]` and
  `f(x)[c]` index a vector-valued expression without naming it first.
  These used to fail IR verification with "expected expr node". A literal
  index out of range, and a subscript of a scalar value, are rejected at
  lowering with the source position.
- Elementwise comparison: `v > 0.0`, `lo <= v <= hi`, `a == b` on vectors
  and matrices give a mask, a vector of `0`/`1` per component; `and`,
  `or` and `not` act per component on masks; `any(mask)` and `all(mask)`
  reduce one; `tack.select(mask, a, b)` picks per component, with scalar
  masks and scalar arms broadcast. A mask as an `if` or `while` condition
  is rejected with the hint to reduce it.
- `min` and `max` take two or more values, as in Python
  (`min(a, b, c)`), and vectors have the reductions `v.sum()`, `v.min()`
  and `v.max()` over their components. `v.normalized(eps)` divides by
  `norm() + eps`.
- `v.norm(eps)` is `sqrt(v.norm_sqr() + eps)`, for a length that must not
  be zero. It was "norm() takes no arguments".
- A device function can return vectors among several values:
  `return distance, normal, color`, unpacked with
  `d, n, c = closest_hit(...)`. Only scalars could be returned together;
  a vector slot was never bound and lowering failed on its name.
- Tuple assignment to field elements and vector components:
  `x[i], v[i] = p, q` (whole vectors included), a swap such as
  `a[i], b[i] = b[i], a[i]`, and `lo[i], hi[i] = f(...)` for a device
  function that returns two values. The whole right side is evaluated
  first and the targets are then assigned from left to right, so
  `a[i], i = x, j` stores at the old `i`. Only plain names were accepted
  as targets.
- Vector fields exchange data with NumPy one row per vector:
  `from_numpy` accepts `(*shape, n)` as well as the flat storage shape,
  and `to_numpy(vectors=True)` returns `(*shape, n)`. Plain `to_numpy()`
  still returns the flat array.
- `fill` takes one element's value for a vector or matrix field:
  `colors.fill([1.0, 1.0, 1.0])`. A scalar still sets every component.

#### Constants, math and random numbers

- `tack.constant(value, dtype=None)`: a named constant that kernels and
  device functions may read from the scope that defines them.
  `DT = tack.constant(0.01)` at module level lets a kernel write `DT`; it
  reads as the literal, and stays an ordinary Python number for host
  code. Kernels still capture nothing else: a plain module-level value
  raises `NameError` as before, and the message now names
  `tack.constant`. With a dtype the constant is typed, so
  `tack.constant(747796405, tack.u32)` multiplies in wrapping u32
  arithmetic (a plain literal there promotes to i64) and
  `tack.constant(0.1, tack.f64)` is the exact double. `math.pi`, `math.e`
  and `math.tau` can be written in kernels.
- Vector and matrix constants: `tack.constant((0.5, 0.5, 0.0))` is the
  vector with those components in a kernel, `tack.constant(((1, 0), (0, 1)))`
  the matrix with those rows, typed per component as scalar constants
  are. On the host they index, iterate and convert to NumPy arrays.
- `sinh`, `cosh` and `tanh` as math builtins, on every backend.
- `tack.math`: `fract`, `mix`, `clamp`, `saturate`, `smoothstep`, `step`,
  `sign`, `length`, `distance` and `normalize` as device functions with
  GLSL's definitions, on scalars and vectors alike.
- `tack.random`: random numbers in kernels from a counter-based generator
  with explicit state. `seed(index, stream)`, then `u, state =
  uniform(state)`, `normal`, `direction2`, `direction3`; NumPy mirrors
  (`np_uniform`, ...) draw exactly the same values, and `uniform` and the
  states are bit-identical on every backend. The example ports' `rng`
  module, adopted.

#### Kernels, loops and atomics

- A kernel may end in `return expr`, with a return annotation naming the
  type (`-> tack.f32`, `-> float`, `-> int`); the call returns that value
  as a Python number. It is computed after the launch by a one-thread
  epilogue kernel that stores it into a hidden field, so it may read
  fields, arguments, constants and locals bound outside the loop, not a
  local the loop assigns. A return anywhere else is still rejected.
- An atomic is an expression as well as a statement: `slot =
  tack.atomic_add(counter, 0, 1)` is the element's value just before the
  update, on every backend and for every supported type, which is how
  threads claim unique slots and append to a shared list. On a vector
  field with a vector value it is the vector of the components' old
  values. Barriers remain statement-only.
- Atomics take an index per dimension: `tack.atomic_add(grid, (i, j), v)`,
  where a vector supplies one index per component
  (`tack.atomic_add(grid, cell, v)`). On a vector field a vector value
  updates every component of the element. Multi-dimensional and vector
  targets previously needed hand-linearized indices, and a tuple index
  failed IR verification.
- `tack.ndrange` takes `(start, end)` pairs as well as sizes:
  `for i, j in tack.ndrange((1, n - 1), (1, m - 1))` visits the interior
  of a grid. An empty or reversed pair runs no iterations.
- One element of a field can be read from host code: `f[i]`, `f[i, j]`,
  `f[None]`, giving a Python number or, for a vector or matrix field, a
  NumPy array. Writing an element, slicing and iterating are refused with
  a message pointing at `from_numpy`, `fill` and `to_numpy`. On CPU and
  Metal the read is a slice of unified memory; on CUDA, HIP and Level
  Zero it is a copy of just the element.

#### Data-oriented classes

- A kernel can be a method of a `@tack.data_oriented` class:
  `@tack.kernel def step(self, dt)` is called as `grid.step(0.1)`, and the
  object is a template like any other argument. It failed with "expects 1
  arguments, got 0". `tack.inspect` takes the bound form too.
- A `@tack.func` under `@staticmethod` in a data-oriented class is called
  through `self` like the other methods. It was not found.
- A `@tack.data_oriented` object can hold a `@tack.func` as an instance
  attribute and call it from its methods or from a kernel
  (`self.smoothing = cubic`, then `self.smoothing(r, h)`). Which function
  the attribute holds is part of the kernel's specialization.

#### Algorithms

- `tack.algorithms.argsort`, `sort_by_key`, `gather`, `unique` and
  `reduce_by_key`: a stable radix sort for `i32`/`u32`/`i64`/`u64` keys
  and segmented reductions over runs of equal keys, built from ordinary
  kernels and the scan so they run on every backend. See *Sorting and
  segmented reductions* in the User's Guide.
- The `tack.algorithms` statistics (`var`, `std`, `norm`, `absmax`,
  `count_nonzero`, `dot`, `histogram`) accept vector fields and reduce
  over all components in storage order; they used to fail IR
  verification.

### Kernels that now compute different results

- A vector assignment whose right side reads its own target now reads
  the old components. Vectors are assigned one component at a time, and
  the right side was evaluated in between, so later components saw
  earlier ones already overwritten: `v = v.normalized()` (the result
  was not a unit vector), `v = v.cross(w)`,
  `v = tack.Vector([v[1], v[2], v[0]])`, `v += v.cross(w)` and
  `a[i] = a[i].cross(b[i])` all computed wrong vectors, on every backend
  and without an error. Components that could observe an earlier
  component's assignment are now evaluated first, as Python does. For a
  store to a field element that means any component that loads from a
  field, since fields may share storage. Assignments whose components
  only read their own component (`v = v * 2.0`, `a[i] = a[i] + b[i]`)
  computed the right result before and still do.

- A scalar stored to a vector field element sets every component:
  `vf[i] = 0.0` names element `i`, as a load of `vf[i]` does, and as
  `vf[i] *= 2.0` does. It wrote the single component at flat index `i`.
- A vector field's element index is computed in 64 bits. It was narrowed
  to i32 before being scaled by the vector width, so a field of more than
  2^31 components was indexed wrongly.

### Code that is now rejected

- A multi-dimensional field indexed with the wrong number of indices.
  `grid[i, j]` on a three-dimensional field linearized with the sizes it
  had and silently addressed another element; so did a vector index of
  the wrong width and an atomic's index. The dispatch that binds such a
  field now raises `TypeError` with the source position. A single index
  is still a flat, row-major index.
- A vector or tuple anywhere one value is required, such as a comparison
  (`vf[i] < 1.0`), a condition, a loop bound or `print`. Operations that
  do not map over a vector's components used to fail IR verification
  with "expected expr node" and a path into the lowered tree; every such
  statement now raises `UnsupportedSyntaxError` naming the kernel or
  device function and the source position.
- A vector stored where it does not fit: `s[i] = vec` into a field of
  scalars or a local array, and `vf[i] = vec` into a field of vectors of
  another width. Both wrote components at offsets computed from the
  value's width, over whatever was there.
- Reading a `for` loop's variable after its loop, before the name is
  assigned again, raises `NameError` at lowering with the read's position,
  on every backend. CPU raised a `NameError` from codegen and the GPU
  backends failed to compile when the name had no other binding; when it
  did (`d = 100` before `for d in range(4)`), every backend silently read
  the outer value where Python gives the last value iterated. Copy the
  value to another name inside the loop. Sibling loops may still share a
  variable, and a loop variable may still be assigned as an ordinary local
  afterwards.

### Fixes with no source change needed

- The size of a local or shared array may be arithmetic on constants and
  dimensions: `tack.local_array(tack.f32, MAX_HITS * 4)` is folded to one
  constant when dimensions are resolved. Only a literal was, and Metal
  refused the kernel with "array size is not a constant expression".
- Reading a zero-dimensional field to the host (`f.to_numpy()` for
  `shape=()`) crashed the process on CUDA: cuda-python does not take a
  zero-dimensional NumPy array as a host buffer. The copy now goes through
  a flat view.
- A parallel loop may start at a negated argument: `range(-n, n + 1)` and
  `tack.ndrange((-n, n + 1), ...)` failed with "Cannot resolve loop range
  expression: IRUnaryOp", because the host evaluates the launch size and
  had no case for a unary minus.
- Unary plus on an integer (`+n`) compiles on Metal, CUDA, HIP and Level
  Zero. The generators emitted a one-argument call to the two-argument
  add helper, which the device compilers rejected.
- **Metal computed wrong results, silently, for a field element updated
  in a loop.** `for k in range(n): total[0] += x[k]; counter[0] += 1` left
  `total` at `start + x[n - 1]`: Apple's compiler read `total[0]` once,
  before the loop. It happened when the element's address did not depend
  on the thread, the loop also stored to another field of the same type,
  and the trip count was a runtime value; `while` loops and f32 fields
  were affected alike. Present since field pointers moved into one
  argument buffer, so in 0.2.0. A loop that stores to a field is now
  compiled in a function of its own, which costs under 3% for the kernels
  that have one and nothing for the others.
- A zero-dimensional scalar field, `tack.field(dtype, shape=())`, can be
  allocated on Metal. Its buffer zeroed the new storage with `[:]`, which
  a zero-dimensional array rejects with "too many indices". CPU was not
  affected, nor zero-dimensional vector fields.
- A subclass of a `@tack.data_oriented` class inherits its bases'
  `@tack.func` methods and class constants and may override them. Only
  the class's own were found, so a method calling an inherited one failed
  with "self.weight ... is neither a class constant ... nor a @tack.func
  method". A subclass without the decorator is handled the same way.
- CPU threading on two-thread machines: one worker the scheduler took
  off its core could make a cheap kernel look expensive. The fan-out's
  per-worker median picked the slower of two workers, so a single stalled
  worker set a floor under the serial cost estimate and raised the
  kernel's threading threshold with it, until a later recheck corrected
  it. The median over an even number of workers is now the lower middle
  value. Machines with three or more threads were not affected.
- A field of zero elements can be allocated on Metal and CUDA, as it
  already could on CPU and Level Zero. A filter that selects nothing now
  returns an empty field instead of failing to allocate it.
- A kernel that used a name as a sequential loop's variable and later
  assigned it as a plain local (`for d in range(4): ...` then `d = ...`)
  failed to compile on Metal, CUDA, HIP and Level Zero with an undeclared
  identifier; CPU accepted it. The C-family generators now treat the loop
  header's declaration as ending with its block, so the later assignment
  declares a new local, as on CPU.
- A field can be passed down through any number of nested device
  functions. Three or more levels failed to compile on every backend
  ("Cannot coerce float* to i32" on CPU, a pointer-to-integer error from
  the GPU compilers): copy propagation resolved one link of the chain of
  parameter copies per pass and left a local holding the field.
- A `@tack.data_oriented` template's vector field attributes are now
  detected as vector fields: `self.vel[i, j]` in a template method, or
  `obj.vel[i, j]` in the kernel, lowered as a scalar field access and
  failed IR verification.
- A `@tack.func` given a vector field now loads whole vectors from it:
  `return vf[i]` inside the function lowered to a single scalar load, so
  a kernel doing `out[i] = f(vf, i)` wrote one component and left the
  rest. The inliner propagated vector-variable and texture metadata to a
  function's parameters but not vector-field metadata.
- A device function's locals bound by tuple unpacking (`t, u = f()`, or
  `for i, j in ...`) are now renamed when the function is inlined. They
  escaped renaming before, so the inlined body overwrote the caller's
  variables of the same names.

### Tooling

- `test_a_stale_parallel_rate_gets_re_measured` failed about one run in
  thirteen on two-thread machines and once on a CI runner: it asserted
  that two threads beat one on a bandwidth-bound kernel, which they need
  not. The scenario now uses a compute-bound kernel, so the assertion
  tests the recovery path it was written for.
- The CPU threading tests no longer fail on two-thread machines: the
  per-worker median above was the cause, and a test that compared against
  the estimate before its own fan-out now compares against the one after.

### Known limitations

- **ROCm 7.0 and 7.1 miscompile some integer code on HIP.** Their device
  compiler (AMD clang 20, inside hipRTC) can evaluate a 64-bit signed
  comparison wrongly when it sits inside a long integer expression; the
  kernel returns a wrong value with no error. Tack's generated source is
  correct, and right on CUDA, as host C++ and under ROCm's clang 23. Use
  a newer ROCm where possible; on 7.0 and 7.1, check integer-heavy kernels
  against the CPU backend.
- **HIP on gfx90a (MI200) samples 3D textures in software.** The device
  reports image support but refuses linear 3D filtering, which is all
  `Texture3D` offers; Tack detects this at init and falls back. MI300
  parts have no texture hardware and always sample in software.
- **Metal compiles a kernel whose sequential loop stores to a field in a
  function of its own**, to avoid an Apple GPU compiler fault (see
  *Fixes*). The fault's mechanism is not known, so the rule is wider than
  the trigger; it costs under 3% on the kernels that take it.
- **Not in this release:** struct types (use one field per member), a
  3×3 SVD (2×2 only, in the examples), swizzles such as `v.xy` or `v.yzw`
  (write `tack.Vector([v.y, v.z, v.w])`), and automatic differentiation.
- **Windows has not been revalidated.**
- **HIP leaks one hipRTC program object per compiled variant.**
  `hiprtcDestroyProgram` segfaults in hip-python, so Tack does not call it;
  the variant cache keeps compilations rare.
- **Level Zero relies on two workarounds for Intel driver defects:** f64
  `floor`/`ceil` restore the sign of zero with `copysign`, and narrow signed
  negation and `abs` are emitted as non-inlined helpers.
- **Raster depth bias:** in mixed scenes, path-traced surfaces are pushed
  back by a fixed 0.5% relative depth bias so a wireframe lying on its own
  surface is drawn; it can drop wire pixels at grazing angles.

## 0.2.0 — 2026-10-05

The headline: Tack now has a written language contract, and the compiler
enforces it on every backend. Most of the changes below are cases where a
kernel used to compute a **different answer on different backends**, or
silently did something other than what its source said. Code that relied
on one backend's old behavior can change results or start raising.

### Kernels that now compute different results

Each was a correctness fix: the old result disagreed with Python, with the
contract, or with another backend.

| Change | Before | Now | Contract section |
|---|---|---|---|
| Integer `/` | Truncated on CUDA/HIP/Metal/OpenCL; on CPU narrowed back to the integer type | True division, f32 result: `7 / 2 == 3.5`. Use `//` for floor division, or `int(a / b)` within range for truncation | *Required: true division* |
| Integer `//` and `%` with negative operands | Truncated toward zero on GPUs | Python floor semantics: `-7 // 3 == -3`, `-7 % 3 == 2`, `7 % -3 == -2` | *Required: integer floor division and remainder* |
| Integer `**` and `pow` | Went through floating `pow`; large results lost precision; `pow` and `**` could disagree | Exact power modulo 2^N at the **base's** type: i8 `3 ** 5 == -13`. Cast the base to float for floating power | *Required: integer power* |
| Floating `//` and `%` | CPU `%` took the dividend's sign; GPU `//` narrowed through an integer and overflowed | Python semantics in the promoted float type, correct near rounded boundaries; `//` returns an integer-valued float | *Required: floating floor division and remainder* |
| Fixed-width integer overflow | Varied by backend; GPU code relied on C promotion and signed-overflow undefined behavior | Every `+`, `-`, `*`, unary `-`, `~`, `<<`, `&`, `\|` and `^` wraps at its annotated width before the next operation uses it: i8 `127 + 1 == -128` | *Required: fixed-width arithmetic and integer conversion* |
| Mixed-signedness promotion | Inconsistent; unsigned high bits could compare as negative | Narrowest type holding both ranges: i8 with u16 → i32, i32 with u32 → i64 | same |
| Integer casts and stores | Widening followed the destination's signedness on CPU | Reduce mod 2^N, widen by the **source** signedness: i8 `-1` → u64 is 2^64 − 1 | same |
| Right shift of unsigned values, unsigned comparisons, `min`/`max` | Signed on CPU in places; integer `min`/`max` could go through floating point | Logical right shift for unsigned; exact integer comparison | same |
| CUDA and Metal floating math | `--use_fast_math` on CUDA and fast math on Metal for every kernel | Safe math everywhere: NaN, infinity and signed zero preserved; `(a+b)+c` keeps its grouping; fused multiply-add still allowed. CUDA/Metal images move closer to the CPU reference | *Floating-point execution policy* |
| `sqrt`, trig, `exp`, `log` on integer or mixed arguments | Argument precision varied; CPU libm calls could fail to compile | Arguments convert to the annotated result precision first: f32 unless an argument is f64 | same |
| Comparisons and `and`/`or` | CPU could yield −1 for true | Normalized i32 `0`/`1`; `and`/`or` return Booleans, not an operand | *Types and numerical behavior* |
| Field `min()`/`max()` | Device paths mishandled values beyond ±1e38, same-sign infinities and NaNs | Full normal range, infinities, NaN propagation, order-independent zero ties; empty `min`/`max` raise `ValueError`, empty `sum` is +0, empty `mean` is NaN | *Field and parallel reductions* |
| Top-level `range(start, end)` | Ignored `start` on some paths; empty ranges could run | Runs exactly `[start, end)`; nothing when empty or reversed | *Execution and ordering* |
| Evaluation order | Not defined | Left to right; `and`/`or` short-circuit; conditional expressions evaluate one arm; augmented assignment evaluates its index once | same |
| Overlapping field arguments | Could read stale values on Metal and through `restrict` on other backends | Program order preserved when fields alias, including the same field twice and overlapping views | *Memory and aliasing* |
| Raster points and wireframes | Coincident primitives could produce mixed colour channels | Depth winner and its colour selected together, deterministically: lowest primitive index at equal depth | Rendering |
| Mixed solid and raster scenes | Wireframe and point actors were path traced as surfaces; volume plus wireframe dropped the volume | Raster actors composite over the path-traced or ray-cast image with depth | Rendering |
| Float literals in f64 code | Always f32: `x_f64 * 0.1` multiplied by 0.10000000149…, and `tack.f64(0.1)` widened that rounded value | A literal takes the precision of the operand it meets (NumPy NEP 50 "weak" scalars), so f64 expressions use the exact double and `tack.f64(0.1)` is exact. Comparisons with a literal change for values between its f32 and f64 roundings. f32 kernels are unchanged. Edge cases: a local assigned only literals (`a = 0.1`) is still f32, so write `tack.f64(0.1)`; `i * 0.1` is f32; write `1.0 / 3.0`, not `1 / 3` | *Required: float literals are weakly typed* |
| Textures after their field changes | CPU sampled the field live; GPU hardware paths kept the data from first use (and could serve a stale texture at a reused address) | Every backend samples a copy taken by `tack.texture3d()`; call `tex.update()` after changing the field. `render_volume()` refreshes its volume each call | *Memory and aliasing* → Textures |
| Locals set before the parallel loop and reassigned inside it | On CPU the value carried from one iteration to the next within a worker's chunk; GPU threads each started fresh | Every iteration starts from the value set before the loop, on every backend | *Execution and ordering* |
| CUDA/HIP/Level Zero launches of 2^32 or more iterations | The thread index wrapped silently (on Level Zero because Intel's `get_global_id` wraps at 2^32); reductions over 2^32 elements on Level Zero summed wrongly | Every iteration runs; launches beyond a backend's grid limit raise `ValueError` | *Execution and ordering* |
| Level Zero: wrapped negation and `abs` of the signed minimum, widened | Intel's IGC 2.7.11 produced `-(-32768)` as 32768 in an i32 result, and `abs(INT_MIN)` as 2^31 in i64 | The contract's wrapped value, through a workaround in the generated code | *Required: fixed-width arithmetic and integer conversion* |

### Code that is now rejected

These raise a diagnostic naming the kernel or device function and its
source line, instead of being silently dropped or miscompiled.

- `assert`, `try`, `with`, generators, imports, nested definitions,
  comprehensions, annotated assignments, keyword or starred arguments.
- `break` out of the top-level parallel loop, and kernel `return`.
- Reading a name the kernel never binds (`NameError`); capturing a
  numeric value from the enclosing Python scope.
- Binding a second name to a local or shared array (`view = tmp`).
- A negative integer literal exponent; a literal zero or negative `range`
  step.
- Mixing any signed integer with `u64` without an explicit cast,
  including in comparisons, conditional arms and joined assignments.
- Calling an ordinary Python function, or a runtime function value, from a
  kernel; recursive device-function calls. Device calls now resolve through
  the defining module, so importing another module with a same-named
  function can no longer change a kernel.
- **On CPU:** kernels using `shared`, `shared_like`, `barrier`,
  `thread_id`, `block_sum`, `block_min` or `block_max`. CPU does not
  emulate workgroups; use `local_array` / `local_array_like` for private
  scratch. `examples/09_shared_memory.py` now requires a GPU `--arch`.
- **On GPUs:** collectives in a launch whose iteration count is not a
  multiple of 256, and barriers or block reductions inside branches the
  compiler cannot prove uniform.
- Atomics outside the declared domain (for example 64-bit atomics on Metal
  or Level Zero, 8/16-bit atomics on GPUs), atomics on private or shared
  arrays, and unaligned imported atomic targets.
- Block reductions on non-f32 arguments without an explicit `tack.f32(...)`.
- Field stores, atomics, barriers, block reductions and `print` outside
  the parallel loop, before or after it, including through inlined device
  functions. They used to run a backend-dependent number of times (once
  per CPU chunk or probe, once per GPU thread). Plain local assignments
  stay allowed. A kernel with no parallel loop, two of them, or one inside
  a branch now gets a source diagnostic too.
- `tack.texture3d()` on an f64 field, with `interp` other than `'linear'`
  (`'nearest'` was accepted but silently linear), or with a shape whose
  W·H·D differs from the field's size. A `Volume` built from an f64 field
  now fails at construction.
- Kernels that may store to a read-only field: `field_from_ptr` without
  `writable=True`, or a DLPack import flagged read-only.
- DLPack import of a Metal (`kDLMetal`) tensor, which never worked, now
  fails up front with a clear message.
- `tack.inspect(..., mode="optimized")` on a non-CPU backend, which used to
  return the source text. Inspection also applies dispatch's dtype check.
- An unrecognized `TACK_CPU_POLICY` value (it silently selected v1).

### Fixes with no source change needed

- CUDA: a shared context can outlive the backend that created it, and a
  field whose context is gone reports it instead of faulting.
- CPU: floating `atomic_min`/`atomic_max` are now real atomics (they were a
  load-compare-store race).
- CPU: field loads and stores no longer claim four-byte alignment, which
  was wrong for byte fields and unaligned imported buffers.
- Level Zero: integer `min`/`max`, float ternaries and nested local arrays
  compile on the real device compiler; signed zero survives f64
  `floor`/`ceil` despite an Intel driver defect.
- Python-legal names that are keywords in CUDA, OpenCL or MSL (`default`,
  `half`, `kernel`, …) now work as parameter and local names.
- NumPy 1.x works again with the CPU backend.
- `tack.algorithms` statistics and scans: `var` and `std` with an explicit
  `n` smaller than the field used the whole field's mean (`var([1, -2, 3,
  -4], n=2)` was 2.5, not 2.25), and `histogram` without a range used the
  whole field's range; both now cover the first `n` elements. Every
  statistics and scan function now raises `ValueError` for an `n` outside
  a field instead of reading, or for the scans writing, past it. A scan of
  zero elements returns 0 instead of reading index −1, and `var`/`std` of
  zero elements return NaN.
- `exclusive_scan` scanned in `i32` whatever the field dtypes, truncating
  floats and wrapping 64-bit integers, and both scans returned their total
  through `i32`. Both now scan in the output field's dtype and return the
  total in it, as a Python `int` or `float`.
- A process that exits while another library (VTK's DLPack support, for
  example) still holds an unconsumed Tack DLPack capsule no longer crashes
  at interpreter shutdown.
- Compiled-variant caching: vector width and alias relationships are part
  of the key; template classes release their cached code when collected.
- GPU backends: dispatching one kernel from several threads at once is
  serialized per compiled variant (per backend on Level Zero). It used to
  share the scalar argument buffer, and on CUDA 603 of 1200 concurrent
  dispatches got another thread's scalars. CUDA dispatch from a thread
  other than the one that called `tack.init` still needs Tack's context
  made current.
- `field.sum()`/`min()`/`max()` over 2^32 or more f32 elements work on
  CUDA, HIP and Level Zero, and use NumPy on Metal, instead of failing with
  `struct.error`.
- `from_numpy` on a reshaped view works on CPU and Metal; `field_from_ptr`
  checks the memory space of NumPy-integer and `CUdeviceptr` pointers, and
  on HIP a field wrapped from a NumPy-integer address now works (its copies
  failed with `hipErrorInvalidValue`).
- `tack.init(arch=tack.cpu, num_threads=N)` is accepted; `TACK_NO_REINIT=0`
  (and `false`, `no`, `off`) now means off; kernel errors name the kernel
  once; unreadable device-function source raises `RuntimeError` like a
  kernel's; textures no longer leak a GPU texture object per compiled
  kernel.

### Performance changes users may notice

Measured on the project's test machines.

| Change | Effect | Why |
|---|---|---|
| CPU threading decisions | Large CPU dispatches no longer stall on whole-frame serial rechecks; the CPU path tracer and cheap-kernel dispatches improved on the measured hosts | Threading-policy repairs |
| CPU disjoint-field specialization | Up to −70% on store-accumulator and −22% on stencil kernels on an older Xeon; ~3.5 µs extra per CPU dispatch | Proven non-overlapping fields compile with `noalias` |
| Cold compile | First call of a large kernel is slower (CUDA path tracer roughly +50% on the oldest test host) | IR verification at every pass boundary; partly offset by faster IR cloning |
| CUDA path tracer, warm | ~6–7% slower | Safe floating-point math |
| CPU path tracer, warm | ~4% slower on the oldest test host | Wrapped i32 index arithmetic; a renderer-side fix is planned |
| Textures | One extra full copy of the data per texture, on CPU as well | Textures copy their field |
| `render_volume()` | One device copy of the volume per call | Keeps the ray caster and the path tracer showing the same data |

### Tooling

- `docs/reference/language-contract.md`: the language contract.
- `validate_all.py --arch <backend>` checks every call and fails when the
  requested backend is unavailable.
- Host-side compiler checks honor `TACK_CLANG` and fail under
  `TACK_REQUIRE_CLANG=1` instead of skipping.
- `TACK_EXAMPLES_ARCH` selects the backend for the example sweep.
- New diagnostic probes in `benchmarks/`: threading decisions, LLVM
  path-tracer comparison, volume cold start, IR cloning, raster differential.

### Known limitations

- **ROCm 7.0.2 miscompiles some integer code on HIP.** Its device compiler
  (AMD clang 20, inside hipRTC) can evaluate a 64-bit signed comparison
  wrongly when it sits inside a long integer expression. The kernel then
  returns a wrong value with no error. Tack's generated source is correct;
  the same kernel is right on CUDA, as host C++ and under ROCm's clang 23.
  It depends on the surrounding expression, so Tack can't avoid it. Use a
  ROCm release newer than 7.0 where possible; on 7.0.2, check
  integer-heavy kernels against the CPU backend.
- **Windows has not been revalidated** for this release; the supported
  backends were validated on Linux (CPU, CUDA, HIP, Level Zero) and macOS
  (CPU, Metal on an M1 Max and an M3).
- **HIP leaks one hipRTC program object per compiled variant.**
  `hiprtcDestroyProgram` segfaults in hip-python, so Tack does not call it;
  the variant cache keeps compilations rare.
- **Level Zero relies on two workarounds for Intel driver defects:** f64
  `floor`/`ceil` restore the sign of zero with `copysign`, and narrow signed
  negation and `abs` are emitted as non-inlined helpers. Results follow the
  contract; both defects will be reported upstream.
- **Raster depth bias:** in mixed scenes, path-traced surfaces are pushed
  back by a fixed 0.5% relative depth bias (not slope-scaled) so a
  wireframe lying on its own surface is drawn. It can drop wire pixels at
  grazing angles or let very close geometry show through.
