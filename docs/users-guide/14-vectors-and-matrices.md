# Vectors and matrices

Particles, colours, directions and local deformation all have several components.
Tack lets you express their arithmetic as small values instead of maintaining one
scalar name per component. A value lives inside an iteration; a vector or matrix
**field** stores one such value per element across iterations and launches.

## Start with positions and velocities

```python
import numpy as np
import tack

tack.init(arch=tack.cpu)
pos = tack.Vector.field(2, dtype=tack.f32, shape=(3,))
vel = tack.Vector.field(2, dtype=tack.f32, shape=(3,))
pos.from_numpy(np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32))
vel.fill([1.0, 0.0])

@tack.kernel
def advance(pos, vel, dt):
    for i in range(pos.shape[0]):
        pos[i] += dt * vel[i]

advance(pos, vel, 0.5)
print(pos.to_numpy(vectors=True))    # [[0.5, 0.0], [1.5, 0.0], [0.5, 1.0]]
```

Each iteration loads a two-component velocity and updates its own position.
`dt` broadcasts to both components. Multiplying two vectors with `*` multiplies
corresponding components; use `dot` or `@` for a dot product.

A local assignment `p = pos[i]` loads a value. Changing `p.x` changes that local;
write `pos[i] = p` to store it back. Writing `pos[i].x` targets the field directly.
Vectors have a fixed width for each compiled specialization. There are no
runtime-length vector values or swizzle constructors such as `v.rgb`/`vec4(v, 1)`;
construct the components explicitly.

## Vectors

Tack provides a `Vector` type for multi-component fields. Vector operations
are scalarized at the IR level:

```python
v = tack.Vector.field(3, dtype=tack.f32, shape=(n,))

@tack.kernel
def normalize_vectors(v, n):
    for i in range(n):
        vec = v[i]                   # loads 3 components
        length = sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)
        v[i] = vec / length          # stores 3 components
```

Arithmetic works on whole vectors, component by component, and a scalar
operand is applied to every component. So do the math builtins, the casts
and conditional expressions:

```python
@tack.kernel
def step(pos, vel, grid, out, dt, n):
    for i in range(n):
        v = vel[i]
        v = min(max(v, -10.0), 10.0)        # clamp each component
        cell = int(floor(pos[i] / 0.25))    # a vector of i32 cell indices
        v = v if pos[i].y > 0.0 else -v     # one condition, whole vectors
        speed = v.norm()                    # methods are listed below
        vel[i] = v
        pos[i] += v * dt                    # augmented store to a field element
        out[i] = grid[cell] * speed         # grid[cell] is grid[cell[0], cell[1], cell[2]]
```

A list of scalars is a vector, so `tack.Vector` can be left out wherever
a vector is expected: `pos[i] = [x, y]`, `v = [0.0, 0.0]`,
`w.dot([1.0, 0.0])`, `p - [1.0, 0.0]`. A matrix still needs
`tack.Matrix([[...], [...]])`.

The vector methods are `norm()`, `norm_sqr()`, `dot(w)`, `cross(w)`
(3-vectors), `normalized()`, and the reductions over components `sum()`,
`min()` and `max()`. `norm(eps)` is `sqrt(norm_sqr() + eps)`, a length
that is never zero. `normalized(eps)` divides by `norm() + eps`, for a
vector that may be zero. `min` and `max` as functions take two or more
values, as in Python: `min(a, b, c)`.

Comparing vectors compares component by component and gives a *mask*, a
vector of `0`/`1`. `any` and `all` reduce a mask to one truth value, and
`tack.select(mask, a, b)` picks per component:

```python
@tack.kernel
def confine(pos, vel, alive, n):
    for i in range(n):
        p = pos[i]
        inside = -1.0 <= p <= 1.0                  # a mask, every link evaluated
        if not all(inside):
            alive[i] = 0
        vel[i] = tack.select(inside, vel[i], -vel[i])    # reflect the components outside
        pos[i] = tack.select(p > 1.0, 1.0, p)             # a scalar arm fills every component
        hits = (p > 0.9) and (vel[i] > 0.0)              # and/or act per component on masks
        if any(hits):
            alive[i] += hits.sum()
```

A mask is an ordinary integer vector, so `mask.sum()` counts. `if` and
`while` take a scalar condition: a mask there is rejected with the hint
to reduce it with `any` or `all`.

A device function can return several values, vectors among them, to be
unpacked at the call:

```python
@tack.func
def closest_hit(origin, direction):
    ...
    return distance, normal, color        # a scalar and two vectors

distance, normal, color = closest_hit(o, d)
```

A vector field exchanges data with NumPy flat or with one row per
vector: `v.from_numpy(a)` accepts an array of shape `(*shape, n)` or the
flat `(prod(shape) * n,)`, and `v.to_numpy(vectors=True)` returns
`(*shape, n)`. Plain `v.to_numpy()` returns the flat storage.

Two vectors in one operation must have the same number of components.
An assignment evaluates its whole right side before it stores anything,
so `v = v.cross(w)` and `pos[i] = pos[i].cross(axis[i])` read the old
components. A tuple assignment does the same and then assigns its targets
from left to right, and its targets may be field elements:
`pos[i], vel[i] = p, v`, or `a[i], b[i] = b[i], a[i]` to swap two.

A store to a field element must match the field: a vector of the field's
width, or a scalar, which sets every component (`vel[i] = 0.0`). A vector
of another width, or a vector stored into a field of scalars, is
rejected.

Components are scalars. `vec[0]` and `vec.x` read one (`x`, `y`, `z`, `w`
name the first four); `vec[1] = x`, `vec.y = x` and `vec[1] += x` write
one. The same forms work on a field element without naming the vector
first: `v[i][2]` and `v[i].z` read a component, `v[i][2] = x` and
`v[i].z += x` store one. The index may be a runtime value (`vec[k]` for a
loop variable `k`), which lowers to a chain of selects rather than a
branch; a typed or integer `tack.constant` read that way is instead one
load from a constant array. A runtime index must be in `[0, n)`: outside it, on either side,
a read gives the last component and a write does nothing, since a kernel
cannot raise. Only a literal index counts from the end (`vec[-1]`), and a
literal out of range is rejected at lowering.

## Matrices

`tack.Matrix` gives small fixed-size matrices, up to 4×4. Like vectors
they are scalarized: a matrix is its entries, and every operation expands
to scalar arithmetic at lowering.

```python
F = tack.Matrix.field(2, 2, dtype=tack.f32, shape=(n,))    # a 2x2 matrix per particle
C = tack.Matrix.field(2, 2, dtype=tack.f32, shape=(n,))

@tack.kernel
def update(F, C, x, dt, n):
    for p in range(n):
        F[p] = (tack.Matrix.identity(2) + dt * C[p]) @ F[p]   # '@' multiplies
        J = F[p].determinant()
        stress = (F[p] - F[p].inverse().transpose()) * J      # '*' is entry by entry
        x[p] += stress @ tack.Vector([0.0, -1.0]) * dt        # matrix @ vector
```

- **Building one.** `tack.Matrix([[a, b], [c, d]])` from rows of scalars,
  `tack.Matrix([u, v])` from vectors as rows, `tack.Matrix.identity(n)`,
  or `u.outer_product(v)`.
- **Products.** `A @ B` for matrices; `A @ v` and `v @ A` take a vector as
  a column and as a row and give a vector; `u @ v` is the dot product.
  Shapes must agree. `+`, `-`, `*`, `/`, the math builtins and scalar
  operands work entry by entry, as for vectors.
- **Methods.** `transpose()`; for square matrices `trace()`; for 2×2 and
  3×3 `determinant()` and `inverse()`. `inverse()` divides by the
  determinant without checking it. `norm()` and `sum()` run over all
  entries.
- **Entries.** `A[i, j]` reads one and `A[i, j] = x` or `A[i, j] += x`
  writes one, on a matrix variable or directly on a field element
  (`F[p][0, 1]`). The indices may be runtime values, which must then be
  in range.
- **Fields.** A matrix field stores each matrix in row-major order.
  `from_numpy` takes `(*shape, n, m)` and `to_numpy(vectors=True)` returns
  it. `F[p] = A`, `F[p] += A`, `F[p] *= s` and
  `tack.atomic_add(F, p, A)` work as they do for vector fields.
- **Device functions** take and return matrices, alone or among several
  values.

A matrix and a vector with the same number of entries are different
things: storing one where the other belongs, or adding them, is rejected.

The statistics in `tack.algorithms` (`dot`, `norm`, `var`, ...) accept a
vector field and reduce over all its components in storage order.


## Matrices in a real computation

In [MPM](tutorials/mpm.md), each particle stores a 2×2 affine velocity matrix.
A matrix-vector product predicts a velocity contribution at a nearby grid node;
an outer product reconstructs the affine matrix during interpolation. The
example shows why `*` and `@` mean different operations, and why a matrix field
is clearer than a four-component vector used as a matrix by hand.

The [dye tutorial](tutorials/dye.md) reuses one interpolation function on RGB
vectors. [N-body](tutorials/nbody.md) accumulates three-component directions.
Use those examples after the small position update above.
