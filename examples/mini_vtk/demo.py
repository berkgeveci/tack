"""mini_vtk demo -- end-to-end: create dataset, run filters, inspect results.

Usage:
  uv run python -m examples.mini_vtk.demo
  uv run python -m examples.mini_vtk.demo --arch metal
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import time
import pgc

import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument('--arch', default='cpu',
                     choices=['cpu', 'metal', 'cuda', 'hip', 'vulkan', 'level_zero'])
_parser.add_argument('--size', type=int, default=50, help='Grid cells per dimension')
_args = _parser.parse_args()
pgc.init(arch=getattr(pgc, _args.arch))

from examples.mini_vtk import (
    make_rectilinear_dataset, make_explicit_hex_dataset, filters,
    AOSArray, CellSetExplicit, Hexahedron, Tetrahedron, Wedge,
    AOSTupleArray,
)
from examples.mini_vtk.cellsets.explicit import from_structured


N = _args.size
print(f"mini_vtk demo -- {N}x{N}x{N} rectilinear grid on {_args.arch}")
print("=" * 60)

# ================================================================
# 1. Create a rectilinear dataset
# ================================================================
x = np.linspace(0.0, 1.0, N + 1, dtype=np.float32)
y = np.linspace(0.0, 1.0, N + 1, dtype=np.float32)
z = np.linspace(0.0, 1.0, N + 1, dtype=np.float32)

t0 = time.perf_counter()
ds = make_rectilinear_dataset(x, y, z)
t1 = time.perf_counter()
print(f"\n1. Created dataset: {ds.num_points:,} points, {ds.num_cells:,} cells ({t1-t0:.3f}s)")

# ================================================================
# 2. Point elevation filter
# ================================================================
t0 = time.perf_counter()
filters.point_elevation(ds, direction=(0.0, 0.0, 1.0), name="elevation")
t1 = time.perf_counter()

elev = ds.get_point_array("elevation")
elev_np = elev.data.to_numpy()
print(f"2. Point elevation: range [{elev_np.min():.3f}, {elev_np.max():.3f}] ({t1-t0:.4f}s)")

# Add two more point scalars for the multi-field demo
filters.point_elevation(ds, direction=(1.0, 0.0, 0.0), name="elev_x")
filters.point_elevation(ds, direction=(0.0, 1.0, 0.0), name="elev_y")

# ================================================================
# 3. Cell average filter (simple averaging)
# ================================================================
t0 = time.perf_counter()
filters.cell_average(ds, "elevation", "cell_elevation")
t1 = time.perf_counter()

cell_elev = ds.get_cell_array("cell_elevation")
cell_np = cell_elev.data.to_numpy()
print(f"3. Cell average: range [{cell_np.min():.3f}, {cell_np.max():.3f}] ({t1-t0:.4f}s)")

# ================================================================
# 4. Parametric center filter (shape-function interpolation, hex)
# ================================================================
t0 = time.perf_counter()
filters.parametric_center(ds, Hexahedron(), "elevation", "elevation_pcenter")
t1 = time.perf_counter()

pc_np = ds.get_cell_array("elevation_pcenter").data.to_numpy()
print(f"4. Parametric center (hex): range [{pc_np.min():.3f}, {pc_np.max():.3f}] ({t1-t0:.4f}s)")

# For a linear field on a regular grid, cell average == parametric center
np.testing.assert_allclose(pc_np, cell_np, rtol=1e-4)
print("   Matches cell average for linear field -- OK")

# ================================================================
# 5. Parametric center for coordinates -> cell center positions
# ================================================================
t0 = time.perf_counter()
filters.parametric_center(ds, Hexahedron(), output_name="center")
t1 = time.perf_counter()

cx_np = ds.get_cell_array("center_x").data.to_numpy()
cy_np = ds.get_cell_array("center_y").data.to_numpy()
cz_np = ds.get_cell_array("center_z").data.to_numpy()
print(f"5. Cell centers (hex, local_array): x=[{cx_np.min():.3f},{cx_np.max():.3f}], "
      f"y=[{cy_np.min():.3f},{cy_np.max():.3f}], "
      f"z=[{cz_np.min():.3f},{cz_np.max():.3f}] ({t1-t0:.4f}s)")

# ================================================================
# 5b. Multi-field parametric center with cached weights (local_array)
# ================================================================
t0 = time.perf_counter()
filters.parametric_center_multi(
    ds, Hexahedron(),
    ["elevation", "elev_x", "elev_y"],
    ["mc_elev", "mc_elev_x", "mc_elev_y"])
t1 = time.perf_counter()

mc_z = ds.get_cell_array("mc_elev").data.to_numpy()
mc_x = ds.get_cell_array("mc_elev_x").data.to_numpy()
mc_y = ds.get_cell_array("mc_elev_y").data.to_numpy()
# Should match single-field parametric center results
np.testing.assert_allclose(mc_z, pc_np, rtol=1e-4)
np.testing.assert_allclose(mc_x, cx_np, rtol=1e-4)
np.testing.assert_allclose(mc_y, cy_np, rtol=1e-4)
print(f"5b. Multi-field center (local_array, 3 fields): ({t1-t0:.4f}s) -- matches single-field OK")

# ================================================================
# 6. Threshold filter -- extract cells in the middle third
# ================================================================
lo, hi = 0.33, 0.67
t0 = time.perf_counter()
ds_thresh = filters.threshold(ds, "cell_elevation", lo, hi)
t1 = time.perf_counter()
print(f"6. Threshold [{lo}, {hi}]: {ds_thresh.num_cells:,} / {ds.num_cells:,} cells ({t1-t0:.4f}s)")

# ================================================================
# 7. Parametric center on thresholded dataset (explicit cell set, same kernel)
# ================================================================
t0 = time.perf_counter()
filters.parametric_center(ds_thresh, Hexahedron(), "elevation", "thresh_pcenter")
t1 = time.perf_counter()

tp_np = ds_thresh.get_cell_array("thresh_pcenter").data.to_numpy()[:ds_thresh.num_cells]
print(f"7. Parametric center (thresholded): range [{tp_np.min():.3f}, {tp_np.max():.3f}] ({t1-t0:.4f}s)")

# ================================================================
# 8. Demonstrate cell type polymorphism: same filter, tet cell type
# ================================================================
# Build a small tet mesh: one tet from 4 points
tet_points = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
], dtype=np.float32)
tet_conn = np.array([0, 1, 2, 3], dtype=np.int32)

coord_field = pgc.field(dtype=pgc.f32, shape=(12,))
coord_field.from_numpy(tet_points.ravel())
conn_field = pgc.field(dtype=pgc.i32, shape=(4,))
conn_field.from_numpy(tet_conn)

from examples.mini_vtk.dataset import Dataset
tet_ds = Dataset(
    AOSTupleArray(coord_field, 4, 3),
    CellSetExplicit(conn_field, 4),
    num_points=4, num_cells=1,
)

# Add a scalar: value = x + y + z at each point
scalar_np = np.array([0.0, 1.0, 1.0, 1.0], dtype=np.float32)  # sum of coords
scalar_field = pgc.field(dtype=pgc.f32, shape=(4,))
scalar_field.from_numpy(scalar_np)
tet_ds.add_point_array("scalar", AOSArray(scalar_field))

# Parametric center with Tetrahedron
filters.parametric_center(tet_ds, Tetrahedron(), "scalar", "tet_center")
tet_center = tet_ds.get_cell_array("tet_center").data.to_numpy()[0]
# Center of tet at (0.25, 0.25, 0.25): expected = 0.25*1 + 0.25*1 + 0.25*1 = 0.75
expected_tet = 0.25 * 1.0 + 0.25 * 1.0 + 0.25 * 1.0
assert abs(tet_center - expected_tet) < 1e-5, f"tet center: got {tet_center}, expected {expected_tet}"
print(f"8. Tet parametric center: {tet_center:.4f} (expected {expected_tet:.4f}) -- OK")

# ================================================================
# 9. Build explicit connectivity from structured (for comparison)
# ================================================================
t0 = time.perf_counter()
explicit_cs = from_structured(ds.cell_set, ds.num_cells)
t1 = time.perf_counter()
print(f"9. Explicit from structured: {ds.num_cells:,} cells ({t1-t0:.4f}s)")

ds_explicit = make_rectilinear_dataset(x, y, z)
ds_explicit.cell_set = explicit_cs
filters.point_elevation(ds_explicit, direction=(0.0, 0.0, 1.0), name="elevation")
filters.cell_average(ds_explicit, "elevation", "cell_elevation")
expl_np = ds_explicit.get_cell_array("cell_elevation").data.to_numpy()
np.testing.assert_allclose(expl_np, cell_np, rtol=1e-5)
print("   Explicit matches structured -- OK")

print("\n" + "=" * 60)
print("All checks passed!")
