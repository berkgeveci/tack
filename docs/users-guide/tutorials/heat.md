# Heat diffusion on a plate

Two hot spots on a cold square spread and fade as heat diffuses. This is a good
first simulation: each timestep needs one stencil kernel, two fields and a swap
of Python references.

![Two hot spots diffusing](../../assets/tutorials/heat.png)

**Origin:** adapted from [Tack's heat-equation example](https://github.com/berkgeveci/tack/blob/main/packages/tack-core/examples/17_heat_equation.py),
Copyright © 2026 Kitware, Inc., BSD-3-Clause. This version swaps buffers instead
of copying them each step and includes an independent NumPy check.

## The equation and its storage

The heat equation is `du/dt = alpha * laplacian(u)`. On a square grid with spacing
`dx`, an explicit step uses the four adjacent cells:

```text
new[i,j] = old[i,j] + r * (old[i-1,j] + old[i+1,j]
                         + old[i,j-1] + old[i,j+1] - 4 * old[i,j])
r = alpha * dt / dx²
```

Choose `r <= 1/4` for this two-dimensional scheme; the example uses `0.2`.
The boundary is held at zero temperature. Heat can therefore leave the plate;
the total heat does not remain constant.

Each scalar field has shape `(n, n)`. Here the first index is the plotted row
and the second is the column. Each iteration owns one output cell. It reads
neighbours from a separate old field:

```python
--8<-- "docs/examples/heat.py:step"
```

The boundary branch writes zero on every step. It also prevents invalid
neighbour loads at the edge. Those loads are inside the conditional, so edge
iterations do not evaluate them.

## Repeat the timestep in Python

Allocate the fields once and swap their roles after each completed launch:

```python
--8<-- "docs/examples/heat.py:loop"
```

The swap copies no field data. The next launch reads the just-computed state.
Updating `old` in place would give other iterations a changing stencil input,
so it would not implement this scheme. Readbacks are only for the figure's
checkpoints; remove them when timing the timestep loop.

## Run and check

```bash
uv run python docs/examples/heat.py --arch cpu --check
uv run --with matplotlib python docs/examples/heat.py --arch metal --output heat.png
```

The first command compares forty small-grid timesteps against a NumPy slice
implementation and checks the zero boundary and the temperature bound. For the
figure, run without `--check`: it shows steps 0, 200, 400 and 600 using the same
colour scale. The falling peak is a physical feature of diffusion.

Try changing `r` from `0.2` to `0.05` while keeping the number of steps fixed.
Then double the grid resolution and choose `dt` four times smaller to keep `r`
fixed for the same `alpha` and physical domain. Compare at the same physical
time, rather than at the same number of steps. A periodic boundary would require
wrapped indexing and would conserve total heat instead of losing it at the edges.

## Complete program

[Download heat.py](../../examples/heat.py).

```python
--8<-- "docs/examples/heat.py"
```
