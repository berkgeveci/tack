# Vectors and matrices

Positions, velocities, colours and deformation gradients all have several
components. This page shows how to write their arithmetic in a kernel as
whole values, `p + dt * v` or `F @ G`, instead of one scalar name per
component. It is for anyone who has written a first kernel and now has
data with more than one number per element. After it you can store
vectors and matrices in arrays, compute with them in kernels, and read
them back into NumPy.

<!-- tack-check: vectors-first -->
```python
import numpy as np
import tack

tack.set_default_device("cpu")

positions = tack.array(np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32),
                       dtype=tack.vec2())
velocities = tack.empty((3,), tack.vec2())
velocities.fill([1.0, 0.5])

@tack.kernel
def advance(positions, velocities, dt):
    for i in range(positions.shape[0]):
        positions[i] += dt * velocities[i]

advance(positions, velocities, 0.5)
print(positions.shape)      # (3,): three elements
print(positions.numpy())    # [[0.5 0.25] [1.5 0.25] [0.5 1.25]]
np.testing.assert_allclose(positions.numpy(), [[0.5, 0.25], [1.5, 0.25], [0.5, 1.25]])
```

Each iteration loads one two-component velocity, scales it by the scalar
`dt`, and adds it to its own position. The array has three elements, and
each element is a vector of two `f32` components.

## Values and arrays

Every vector in Tack has two sides, and the same names serve both.

- On the host, `tack.vec2()`, `tack.vec3()`, `tack.vec4()` and
  `tack.vec(n)` are **element types**. You pass them to the array
  constructors, as you would pass `tack.f32`.
- In a kernel, `tack.vec3(x, y, z)` builds a **vector value**. A value
  lives inside one iteration, in registers. It is not memory.

A vector array stores one vector value per element. Matrices work the same
way. `tack.mat2()`, `tack.mat3()`, `tack.mat4()` and `tack.mat(rows,
columns)` are element types, and `tack.mat2(...)` in a kernel is a value.

Vectors and matrices have a fixed size for each compiled kernel. Tack
expands every operation on them into scalar arithmetic when it compiles the
kernel, so they cost the same as writing the components out by hand. There
are no vectors whose length is decided at run time.

## Store vectors in arrays

The component type defaults to `f32`, or to the
[default float type](arrays-and-types.md) if you changed it. Give it
explicitly for another type: `tack.vec3(tack.f64)` or `tack.vec2(tack.i32)`.

An array's `shape` counts elements, not components. A vector array of
shape `(n,)` and element type `tack.vec3()` has `size` equal to `n` and
`scalar_size` equal to `3 * n`. Its `dtype` is the vector type and its
`scalar_dtype` is the component type. The component dimensions appear only
when you cross to NumPy.

| Allocation | `shape` | `numpy()` gives |
|---|---|---|
| `tack.empty((n,), tack.vec3())` | `(n,)` | `(n, 3)` |
| `tack.empty((h, w), tack.vec3())` | `(h, w)` | `(h, w, 3)` |
| `tack.empty((n,), tack.mat2())` | `(n,)` | `(n, 2, 2)` |
| `tack.empty((n,), tack.mat(3, 2))` | `(n,)` | `(n, 3, 2)` |

`tack.array(values, dtype=tack.vec2())` and `tack.copyto(array, values)`
take NumPy data with the component dimensions last. Values that broadcast to
that shape work too, so `velocities.fill([1.0, 0.5])` sets every element.
Copies and reshapes keep the element type. A reshape changes only the
element dimensions.

In memory each element's components are stored together, one element after
another. Data that arrives as separate arrays, one per component, can stay
that way. `tack.soa(x, y, z)` makes a vector array whose components are the
three given arrays, without copying them. Kernels read and write it as they
do any other vector array.

## Compute with vectors in a kernel

### Build a vector

`tack.vec3(x, y, z)` builds a vector from three scalars. `tack.vec3(s)`
repeats one scalar in every component, and `tack.vec3(v)` copies a
3-vector. A list of scalars is also a vector, of any width. It can stand
wherever a vector is expected: `positions[i] = [x, y]`,
`d = p - [1.0, 0.0]`, or `v.dot([0.0, 0.0, 1.0])`.

### Arithmetic and functions

Operators work component by component, and a scalar operand applies to
every component. So `*` between two vectors multiplies matching components.
It is not a dot product; use `dot` or `@` for that. The math builtins, the
casts and conditional expressions also work per component. An array
indexed with an integer vector takes one index per component, so with a
2-vector `cell`, `grid[cell]` is `grid[cell[0], cell[1]]`.

<!-- tack-check: vectors-componentwise -->
```python
import numpy as np
import tack

tack.set_default_device("cpu")

positions = tack.array(np.array([[0.1, 0.6], [0.9, 0.3]], dtype=np.float32), dtype=tack.vec2())
velocities = tack.array(np.array([[20.0, -1.0], [2.0, 3.0]], dtype=np.float32),
                        dtype=tack.vec2())
grid = tack.array(np.arange(16, dtype=np.float32).reshape(4, 4))
sampled = tack.empty((2,))

@tack.kernel
def step(positions, velocities, grid, sampled):
    for i in range(positions.shape[0]):
        p = positions[i]
        v = min(max(velocities[i], -10.0), 10.0)    # clamp each component
        v = v if p.y > 0.5 else -v                  # one condition, whole vectors
        cell = int(floor(p / 0.25))                 # a vector of i32 cell indices
        sampled[i] = grid[cell] * v.norm()          # grid[cell[0], cell[1]]
        velocities[i] = v

step(positions, velocities, grid, sampled)
np.testing.assert_allclose(velocities.numpy(), [[10.0, -1.0], [-2.0, -3.0]])
np.testing.assert_allclose(sampled.numpy(), [2 * np.hypot(10, 1), 13 * np.hypot(2, 3)],
                           rtol=1e-6)
```

### Methods

| Method | Result |
|---|---|
| `v.dot(w)` | the dot product |
| `v.cross(w)` | the cross product, for 3-vectors |
| `v.norm()` | the length |
| `v.norm_sqr()` | the squared length |
| `v.normalized()` | `v` divided by its length |
| `v.sum()`, `v.min()`, `v.max()` | the sum, least and greatest component |
| `v.outer_product(w)` | the matrix of products `v[i] * w[j]` |

Two methods take an optional `eps` for vectors that may be zero.
`v.norm(eps)` is `sqrt(v.norm_sqr() + eps)`, a length that is never zero.
`v.normalized(eps)` divides by `v.norm() + eps`. The functions `min` and
`max` take two or more values, as in Python: `min(a, b, c)`.

### Read and write components

`v[0]` and `v.x` read one component. The names `x`, `y`, `z` and `w` name
the first four. `v[1] = s`, `v.y = s` and `v.y += s` write one. The same
forms work directly on an array element. `positions[i].x` reads a
component of element `i`, and `positions[i].x += s` updates it in place.

A local variable holds a copy. After `p = positions[i]`, changing `p.x`
changes only the local; write `positions[i] = p` to store it back. There
are no swizzles such as `v.xy` or `tack.vec4(v, 1.0)`. Build the vector
from its components instead.

The index may be a run-time value, such as a loop variable. It must then
lie between 0 and the width minus one, because a kernel cannot raise an
error. Details says what happens outside that range.

### Compare vectors with masks

Comparing vectors compares each component and gives a **mask**, a vector
of `0` and `1`. Three functions use masks. `any(mask)` and `all(mask)`
reduce a mask to one truth value. `tack.select(mask, a, b)` picks `a` or
`b` per component.

<!-- tack-check: vectors-masks -->
```python
import numpy as np
import tack

tack.set_default_device("cpu")

positions = tack.array(np.array([[0.5, 0.2], [1.5, 0.2], [0.5, -1.5]], dtype=np.float32),
                       dtype=tack.vec2())
velocities = tack.array(np.array([[1, 1], [1, 1], [1, -1]], dtype=np.float32),
                        dtype=tack.vec2())
alive = tack.ones((3,), tack.i32)

@tack.kernel
def confine(positions, velocities, alive):
    for i in range(positions.shape[0]):
        p = positions[i]
        inside = -1.0 <= p <= 1.0                     # a mask: one comparison per component
        if not all(inside):
            alive[i] = 0
        velocities[i] = tack.select(inside, velocities[i], -velocities[i])
        positions[i] = min(max(p, -1.0), 1.0)

confine(positions, velocities, alive)
np.testing.assert_array_equal(alive.numpy(), [1, 0, 0])
np.testing.assert_allclose(velocities.numpy(), [[1, 1], [-1, 1], [1, 1]])
np.testing.assert_allclose(positions.numpy(), [[0.5, 0.2], [1.0, 0.2], [0.5, -1.0]])
```

A scalar arm of `select` fills every component, as in
`tack.select(p > 1.0, 1.0, p)`. Over masks, `and`, `or` and `not` act per
component. A mask is an ordinary integer vector, so `mask.sum()` counts its
true components.

An `if` or `while` needs one truth value. A mask there is refused when the
kernel compiles, with a message that suggests `any(...)`, `all(...)` or
`tack.select`.

### Return vectors from device functions

A [device function](device-functions.md) takes and returns vectors and
matrices. It can return several values, vectors among them, which the
caller unpacks:

<!-- tack-check: vectors-functions -->
```python
import numpy as np
import tack

tack.set_default_device("cpu")

@tack.func
def reflect(d, n):
    dn = d.dot(n)
    return d - 2.0 * dn * n, dn            # a vector and a scalar

directions = tack.array(np.array([[1, -1, 0], [0, -2, 1]], dtype=np.float32), dtype=tack.vec3())
reflected = tack.empty((2,), tack.vec3())
cosines = tack.empty((2,))

@tack.kernel
def bounce(directions, reflected, cosines):
    for i in range(directions.shape[0]):
        r, cosine = reflect(directions[i].normalized(), [0.0, 1.0, 0.0])
        reflected[i] = r
        cosines[i] = cosine

bounce(directions, reflected, cosines)
expected = np.array([[1, 1, 0], [0, 2, 1]]) / np.array([[np.sqrt(2)], [np.sqrt(5)]])
np.testing.assert_allclose(reflected.numpy(), expected, rtol=1e-6)
np.testing.assert_allclose(cosines.numpy(), [-1 / np.sqrt(2), -2 / np.sqrt(5)], rtol=1e-6)
```

## Compute with matrices

Matrices are small and fixed in size, up to 4×4. Like a vector, a matrix
is its entries, and every operation expands to scalar arithmetic. This
kernel updates a deformation gradient `F` per particle, and the example
checks it against NumPy:

<!-- tack-check: matrices-update -->
```python
import numpy as np
import tack

tack.set_default_device("cpu")

rng = np.random.default_rng(1)
n = 4
F0 = (np.eye(2) + 0.1 * rng.standard_normal((n, 2, 2))).astype(np.float32)
C0 = (0.1 * rng.standard_normal((n, 2, 2))).astype(np.float32)
F = tack.array(F0, dtype=tack.mat2())          # one 2x2 matrix per particle
C = tack.array(C0, dtype=tack.mat2())
J = tack.empty((n,))
force = tack.empty((n,), tack.vec2())

@tack.kernel
def update(F, C, J, force, dt):
    for p in range(F.shape[0]):
        F[p] = (tack.identity(2) + dt * C[p]) @ F[p]            # '@' multiplies matrices
        J[p] = F[p].determinant()
        stress = (F[p] - F[p].inverse().transpose()) * J[p]     # '*' is entry by entry
        force[p] = stress @ [0.0, -1.0]                         # matrix times vector

update(F, C, J, force, 0.5)

F_expected = (np.eye(2) + 0.5 * C0) @ F0
J_expected = np.linalg.det(F_expected)
stress = (F_expected - np.linalg.inv(F_expected).transpose(0, 2, 1)) * J_expected[:, None, None]
np.testing.assert_allclose(F.numpy(), F_expected, rtol=1e-5)
np.testing.assert_allclose(J.numpy(), J_expected, rtol=1e-5)
np.testing.assert_allclose(force.numpy(), stress @ [0.0, -1.0], rtol=1e-4, atol=1e-6)
```

### Build a matrix

- `tack.mat2([a, b], [c, d])` builds a matrix from rows of scalars, and
  `tack.mat2(u, v)` from vectors as rows.
- `tack.mat2(a, b, c, d)` takes four scalars in row-major order. `mat3` and
  `mat4` work the same way.
- `tack.mat(row, ...)` builds any shape up to 4×4, such as three rows of
  two.
- `tack.identity(n)` is the identity matrix, as `np.identity` is. `n` must
  be a literal number.
- `u.outer_product(v)` is the matrix of products of two vectors.

### Multiply and combine

`A @ B` multiplies matrices. `A @ v` treats the vector as a column, and
`v @ A` treats it as a row; both give a vector. `u @ v` between two vectors
is their dot product. The shapes must agree.

The operators `+`, `-`, `*` and `/`, the math builtins and scalar operands
work entry by entry, as they do for vectors. The difference between `*`
and `@` matters. `A * B` multiplies matching entries, and `A @ B` is the
matrix product.

### Methods and entries

| Method | Result | Shapes |
|---|---|---|
| `A.transpose()` | the transpose | any |
| `A.trace()` | the sum of the diagonal | square |
| `A.determinant()` | the determinant | 2×2 and 3×3 |
| `A.inverse()` | the inverse | 2×2 and 3×3 |
| `A.norm()`, `A.norm_sqr()`, `A.sum()` | over all entries | any |

`inverse()` divides by the determinant without checking it, so a singular
matrix gives infinities or NaNs. Test the determinant first when that can
happen.

`A[i, j]` reads one entry, and `A[i, j] = x` or `A[i, j] += x` writes one.
This works on a matrix variable and directly on an array element, as in
`F[p][0, 1]`. Run-time indices are allowed, and must be in range.

### Matrix arrays

A matrix array stores each matrix in row-major order. `tack.copyto` takes
NumPy data of shape `(*shape, rows, columns)`, and `numpy()` returns it.
`F[p] = A`, `F[p] += A`, `F[p] *= s` and `tack.atomic_add(F, p, A)` work as
they do for vector arrays.

A matrix and a vector with the same number of entries are different things.
Storing one where the other belongs, or adding them, is refused.

## Use vectors and matrices in real programs

The [MPM tutorial](../users-guide/tutorials/mpm.md) stores a 2×2 affine
velocity matrix per particle. A matrix-vector product predicts a particle's
velocity at a nearby grid node, and an outer product rebuilds the matrix
during interpolation. The tutorial shows why `*` and `@` mean different
things. The [dye tutorial](../users-guide/tutorials/dye.md) reuses one
interpolation function on RGB vectors, and the
[n-body tutorial](../users-guide/tutorials/nbody.md) accumulates
three-component directions.

## Details

**Assignment order.** An assignment evaluates its whole right side before
it stores anything. So `v = v.cross(w)` and
`positions[i] = positions[i].cross(axis[i])` read the old components. A
tuple assignment does the same and then assigns its targets from left to
right. Its targets may be array elements, so `a[i], b[i] = b[i], a[i]`
swaps two elements.

**Widths must match.** Two vectors in one operation must have the same
number of components. A store to an array element must be a vector of the
array's width or a scalar. A scalar sets every component, as in
`velocities[i] = 0.0`. A vector stored into an array of scalars is refused.

**Run-time component indices.** A run-time index selects a component
through a chain of comparisons, not a branch. Outside the range from 0 to
the width minus one, a read gives the last component and a write does
nothing. Only a literal index may count from the end, as in `v[-1]`. A
literal index out of range is refused when the kernel compiles.

**Atomics on vectors.** `tack.atomic_add` with a vector or matrix value
updates every component, one atomic per component. The element is not
updated as one indivisible whole. See
[atomics](workgroups-atomics-textures.md).

**Reductions over vector arrays.** `Array.sum()` and the statistics in
`tack.algorithms`, such as `dot` and `norm`, treat a vector array as all
its components in storage order. See
[Reductions, scans and sorts](reductions-scans-sorts.md).

The [language contract](../reference/language-contract.md#vectors-and-matrices)
states these rules precisely.

## Where to go next

- [Device classes](device-classes.md) group arrays, constants and methods
  into objects that kernels take, the next step in organizing a program.
- [Reductions, scans and sorts](reductions-scans-sorts.md) combine many
  values, vectors included, into a few.
- The [MPM tutorial](../users-guide/tutorials/mpm.md) puts matrices to work
  in a full simulation.
