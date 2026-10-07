# A particle-to-grid fluid simulation

Material point methods keep material state on particles and use a temporary
grid to exchange momentum. This fluid-block example is a compact application
of matrix fields, weighted interpolation and multidimensional atomic scatter.

![An MPM block falling under gravity](../../assets/tutorials/mpm.png)

**Origin:** adapted from [Taichi's mpm88](https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/simulation/mpm88.py),
the MLS-MPM example by Yuanming Hu, Copyright © The Taichi Authors, Apache-2.0.
The scheme is retained; this version uses explicit Tack atomics, four separate
kernels, a seeded initial block, a smaller grid and property checks. For the
method and its history, see the authors' [Taichi MPM project](https://github.com/yuanming-hu/taichi_mpm).

## State on particles, scratch on a grid

| Field | Element | Meaning |
|---|---|---|
| `x` | 2-vector | Particle position |
| `v` | 2-vector | Particle velocity |
| `C` | 2×2 matrix | Affine variation of velocity near the particle |
| `J` | scalar | Local relative volume |
| `momentum` | 2-vector per grid node | Accumulated momentum, then grid velocity |
| `mass` | scalar per grid node | Accumulated particle mass |

All particles in this example have the same mass. The fixed grid size, spacing,
timestep and material stiffness are `tack.constant` values. Particle count is
a runtime kernel argument. The constitutive stress is a simple volume-based
fluid model; this is not the multi-material snow/plasticity model in `mpm99`.

## Scatter particle mass and momentum

A particle influences a 3×3 stencil. Its two coordinates produce quadratic
B-spline weights, stored as a 3×2 local matrix. A stencil weight is the product
of one weight from each axis:

```python
--8<-- "docs/examples/mpm.py:scatter"
```

`affine @ dpos` is a matrix-vector product. `weight * (...)` scales the vector.
Many particles visit the same node, so both grid updates require atomics. The
momentum vector is updated component by component atomically; the next launch
waits for all components to finish before reading it.

The stencil must fit inside the grid. Initial positions and boundary handling
keep this small example in its supported range; Tack does not generally check
the indices. If extending the model to emit particles or change boundaries,
check that invariant explicitly.

## Update the grid, then gather

The grid-update kernel divides momentum by mass at occupied nodes, adds gravity
and prevents outward velocities near the box edges. The field now holds grid
velocity. A second interpolation transfers it back to particles:

```python
--8<-- "docs/examples/mpm.py:gather"
```

The outer product `gv.outer_product(dpos)` reconstructs `C`; `trace()` updates
the relative volume. Each gather iteration owns one particle, so its stores do
not require atomics. It reads a completed grid.

## Four ordered stages

```python
--8<-- "docs/examples/mpm.py:loop"
```

Clearing the grid is essential: its fields contain temporary contributions for
one substep. Those fields are reused instead of reallocated. Each stage has one
parallel loop, with nested stencil loops sequential within an iteration.

## Check and explore

```bash
uv run python docs/examples/mpm.py --check
uv run --with matplotlib python docs/examples/mpm.py --output mpm.png
```

The small check verifies total scattered mass and finite, in-domain positions.
Before boundary contact it also checks the expected uniform gravitational
acceleration and unchanged horizontal positions. These tests exercise numerical
invariants; they are not a full reference validation of long fluid dynamics.

The figure shows steps 0, 300, 600 and 900. Increase stiffness cautiously:
explicit time integration needs a correspondingly smaller timestep. Compare
early trajectories and conserved mass rather than expecting identical late
particle positions after contact on every backend.

## Complete program

[Download mpm.py](../../examples/mpm.py). The adapted code is
[Apache-2.0 licensed](../../examples/LICENSE-APACHE-2.0.txt).

```python
--8<-- "docs/examples/mpm.py"
```
