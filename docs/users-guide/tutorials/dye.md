# Advecting coloured dye

A blob of RGB dye rotates in a prescribed velocity field. Every output cell
traces backward to find where its colour came from, then interpolates the old
dye there. This introduces vector fields and reusable device functions without
requiring a fluid pressure solver.

![Dye advected around the centre](../../assets/tutorials/dye.png)

**Origin:** a teaching adaptation of [Taichi's stable-fluid example](https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/simulation/stable_fluid.py),
Copyright © The Taichi Authors, Apache-2.0. The interpolation/advection pattern
is retained; this version uses an analytic rotating velocity, a first-order
backtrace and a seeded initial blob. It omits pressure projection, forces and
vorticity confinement. It is an advection example, rather than a complete
incompressible-fluid solver.

## One vector per cell

The dye uses `tack.Vector.field(3, dtype=tack.f32, shape=(n, n))`. Each element is
an RGB vector. The storage convention is `[x, y]`; its NumPy form has shape
`(n, n, 3)`. The plotting function transposes the first two axes for an image
library expecting `[row, column, RGB]`.

We use integer coordinates for the sample locations. The rotational velocity
at a point relative to the centre is `omega * [-y, x]`. Its backward departure
point over a timestep is `p - dt * velocity(p)`.

## Interpolate four samples

A departure point usually falls between grid samples. `floor` finds the lower
corner; the fractional part supplies the weights. The same expression works on
RGB vectors as on scalar values:

```python
--8<-- "docs/examples/dye.py:sample"
```

Use `floor` before `int`: `int(-0.3)` truncates to zero, whereas `floor(-0.3)`
identifies the interval starting at −1. Clamping each load keeps the sampler
within its field. Here clamped coordinates extend the boundary colour outward;
they do not represent a periodic boundary.

## One output pixel per iteration

```python
--8<-- "docs/examples/dye.py:advect"
```

All four reads come from `old`, and each iteration writes just `new[i, j]`.
The host swaps the two fields after the call. Updating dye in place would let
other iterations interpolate partially updated state.

Repeated interpolation smooths the dye. A backward first-order trajectory also
introduces integration error. Semi-Lagrangian advection is useful for robust
visual simulations, but it is not exactly mass-conserving; avoid treating a
plausible image as evidence of a conservative transport method.

## Run and experiment

```bash
uv run python docs/examples/dye.py --arch cpu --check
uv run --with vtk python docs/examples/dye.py --output dye.png
```

The check compares several steps with a separate NumPy bilinear implementation
and checks finite, bounded colour values. The normal run shows the initial blob
and steps 30, 60 and 90.

Try a smaller timestep with proportionally more steps for the same elapsed time.
Replace the rotational velocity with a constant translation and compare clamped
versus periodic sampling. After that, a reusable velocity-field sampler and a
higher-order backtrace are natural extensions.

## Complete program

[Download dye.py](../../examples/dye.py). The adapted code is
[Apache-2.0 licensed](../../examples/LICENSE-APACHE-2.0.txt).

```python
--8<-- "docs/examples/dye.py"
```
