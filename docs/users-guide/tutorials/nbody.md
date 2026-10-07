# Tiling N-body forces

Each body accumulates the gravitational contribution of every other body. The
plain implementation is already parallel over bodies. A GPU version can reuse
positions within a workgroup by loading them into shared memory in tiles.

![Bodies and gravitational accelerations](../../assets/tutorials/nbody.png)

**Origin:** adapted from [NVIDIA Warp's tile N-body example](https://github.com/NVIDIA/warp/blob/main/warp/examples/tile/example_tile_nbody.py),
Copyright © NVIDIA CORPORATION & AFFILIATES, Apache-2.0. The algorithm follows
Lars Nyland, Mark Harris and Jan Prins, [Fast N-Body Simulation with CUDA, GPU Gems 3 chapter 31](https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-31-fast-n-body-simulation-cuda).
This teaching version calculates one acceleration field, uses seeded normal
positions, and implements both ordinary and shared-memory kernels rather than
Warp's tile API. It does not integrate trajectories.

## Begin with the portable kernel

The pair interaction is `r / (dot(r,r) + epsilon²)^(3/2)` for unit masses.
Softening keeps the denominator nonzero. The self contribution is zero because
its displacement is zero.

```python
--8<-- "docs/examples/nbody.py:plain"
```

Iteration `i` owns `acceleration[i]`. The inner loop is sequential. This costs
O(n²) interactions and works on every backend.

## Cooperate on each tile

Each 256-lane workgroup processes a tile of 256 positions. Every lane loads one
position; every lane then reads all 256 positions from shared storage:

```python
--8<-- "docs/examples/nbody.py:tiled"
```

The first barrier ensures the tile is filled before any lane reads it. The
second ensures every lane has finished reading before the next tile overwrites
it. Both are outside lane-dependent conditions. Keep every lane participating;
an early exit around either barrier violates the collective contract.

The shared array packs three coordinates per position. `TILE * 3` is a constant
expression, so Tack resolves it to 768 scalars before generating GPU code. It
is workgroup storage, unlike a `local_array`, which would give each lane a
private array and no reuse. `TILE = 256` matches Tack's default 256-lane
workgroup; changing the constant alone does not change that launch configuration.

## Choose a capability, not a backend name

```python
--8<-- "docs/examples/nbody.py:choose"
```

CPU has no workgroup execution model and runs the plain kernel. The GPU path
requires a body count divisible by 256. This example rejects partial tiles;
it does not silently drop remaining bodies or introduce padding. A general
padded implementation must keep all lanes at both barriers and give padded
bodies zero physical contribution.

## Validate before timing

```bash
uv run python docs/examples/nbody.py --arch cpu --check
uv run python docs/examples/nbody.py --arch metal --check
uv run --with vtk python docs/examples/nbody.py --output nbody.png
```

The check compares the acceleration with a vectorized NumPy `float64` reference.
On a GPU it also compares the tiled result with the plain kernel. A CPU check
alone does not execute shared memory or barriers. The VTK view shows an XY projection
of 3-D positions and a subset of projected acceleration arrows.

For performance work, compile both kernels first, reuse their fields, time warm
calls at the same `n`, and record the selected device. Shared memory reduces
repeated position loads within a workgroup, but it also adds barriers and uses
resources. Measure the net effect; this tutorial promises no universal speedup.
The arithmetic remains O(n²).

## Complete program

[Download nbody.py](../../examples/nbody.py). The adapted code is
[Apache-2.0 licensed](../../examples/LICENSE-APACHE-2.0.txt).

```python
--8<-- "docs/examples/nbody.py"
```
