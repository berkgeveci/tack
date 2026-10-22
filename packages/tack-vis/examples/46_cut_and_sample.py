"""46 -- Cutting and sampling: implicit functions, clip, extract, slice and probe.

VTK's wavelet on a uniform grid, cut and sampled the common ways, each result
checked:

- clip by a sphere: the part inside it, cells cut to its surface, whose
  volume approaches the ball's;
- extract_geometry by a box: the whole cells inside it, as many as the box
  holds;
- extract_points by a sphere: the points inside it, as vertex cells;
- slice by a plane: triangles lying in the plane;
- probe: the field at points along a line, exact where they are grid points
  and interpolated between; a CellLocator built once serves several probes.

Usage:
  uv run python packages/tack-vis/examples/46_cut_and_sample.py
  uv run python packages/tack-vis/examples/46_cut_and_sample.py --arch metal --extent 40
"""

import argparse

import numpy as np

import tack
import tack.dataset as td

_parser = argparse.ArgumentParser()
_parser.add_argument('--arch', default='cpu',
                     choices=['cpu', 'metal', 'cuda', 'hip', 'level_zero'])
_parser.add_argument('--extent', type=int, default=20,
                     help='wavelet extent: (2*extent+1)^3 points of spacing 1 (default 20)')
_args = _parser.parse_args()
tack.set_default_device(_args.arch)

e = _args.extent
data = td.wavelet(e)
x = data.positions()
values = data.fields["RTData"].values.numpy()
print(f"wavelet: {data.num_points} points, {data.num_cells} cells over [-{e}, {e}]^3")


# ================================================================
# CLIP BY A SPHERE
# ================================================================

radius = 0.6 * e
ball = td.clip(data, td.Sphere((0.0, 0.0, 0.0), radius), inside=True)
inside = ball.positions()
volume = td.cell_geometry(ball).measure.values.numpy().sum()
exact = 4.0 / 3.0 * np.pi * radius ** 3
print(f"clip inside a sphere of radius {radius:g}: {ball.num_cells} cells, volume "
      f"{volume:.1f} against the ball's {exact:.1f}")
assert np.all(np.linalg.norm(inside, axis=1) <= radius * (1 + 1e-5))
# Cut cells follow the sphere with straight edges, inscribing it, so the volume
# falls short by about (spacing / radius)^2: 0.5% here, 3% at --extent 8.
assert 0 <= (exact - volume) / exact < 1.5 / radius ** 2
assert "RTData" in ball.fields                     # interpolated onto the cut points


# ================================================================
# EXTRACT WHOLE CELLS IN A BOX, AND POINTS IN A SPHERE
# ================================================================

lower, upper = (-4.0, -2.0, 0.0), (6.0, 3.0, 5.0)
box = td.extract_geometry(data, td.Box(lower, upper))
cells_in_box = int(np.prod(np.subtract(upper, lower)))  # unit cells, box on grid lines
print(f"extract_geometry in a box: {box.num_cells} whole cells (the box holds "
      f"{cells_in_box})")
assert box.num_cells == cells_in_box

near = td.extract_points(data, td.Sphere((0.0, 0.0, 0.0), radius))
expected = int((np.linalg.norm(x, axis=1) <= radius).sum())
print(f"extract_points in the sphere: {near.num_cells} of {data.num_points} points")
assert near.num_cells == expected


# ================================================================
# SLICE BY A PLANE
# ================================================================

origin, normal = np.array([0.0, 0.0, 2.5]), np.array([0.0, 1.0, 2.0]) / np.sqrt(5.0)
cut = td.slice(data, td.Plane(origin, normal))
distance = (cut.positions() - origin) @ normal
print(f"slice by a tilted plane: {cut.num_cells} triangles, every point within "
      f"{np.abs(distance).max():.1e} of the plane")
assert cut.num_cells > 0 and np.allclose(distance, 0.0, atol=1e-4)


# ================================================================
# PROBE ALONG A LINE
# ================================================================

locator = td.CellLocator(data)                     # built once, used by both probes
on_grid = np.stack([np.arange(-e, e + 1, dtype=float), np.full(2 * e + 1, 3.0),
                    np.full(2 * e + 1, -7.0)], axis=1)
sampled = td.probe(data, on_grid, locator=locator)
grid_values = values[[int(np.flatnonzero((x == p).all(axis=1))[0]) for p in on_grid]]
print(f"probe at {len(on_grid)} grid points on a line: valid everywhere, and the "
      f"grid's own values")
assert np.all(sampled.fields["valid"].values.numpy() == 1)
assert np.allclose(sampled.fields["RTData"].values.numpy(), grid_values, rtol=1e-6)

# Halfway between grid points along x, the field is the mean of its two
# neighbours (point data is interpolated linearly along a cell's edge).
between = on_grid[:-1] + np.array([0.5, 0.0, 0.0])
halfway = td.probe(data, between, locator=locator).fields["RTData"].values.numpy()
print(f"probe halfway between them: the mean of the neighbours, to "
      f"{np.abs(halfway - 0.5 * (grid_values[:-1] + grid_values[1:])).max():.1e}")
assert np.allclose(halfway, 0.5 * (grid_values[:-1] + grid_values[1:]), rtol=1e-5)

outside = td.probe(data, [[2.0 * e, 0.0, 0.0]], locator=locator)
assert outside.fields["valid"].values.numpy()[0] == 0  # no cell holds it
