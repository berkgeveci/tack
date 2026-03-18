"""mini_vtk demo — end-to-end: create dataset, run filters, inspect results.

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
    make_rectilinear_dataset, filters, AOSArray, ConstantArray
)
from examples.mini_vtk.cellsets.explicit import from_structured


N = _args.size
print(f"mini_vtk demo — {N}x{N}x{N} rectilinear grid on {_args.arch}")
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

# ================================================================
# 3. Cell average filter
# ================================================================
t0 = time.perf_counter()
filters.cell_average(ds, "elevation", "cell_elevation")
t1 = time.perf_counter()

cell_elev = ds.get_cell_array("cell_elevation")
cell_np = cell_elev.data.to_numpy()
print(f"3. Cell average: range [{cell_np.min():.3f}, {cell_np.max():.3f}] ({t1-t0:.4f}s)")

# ================================================================
# 4. Threshold filter — extract cells in the middle third
# ================================================================
lo, hi = 0.33, 0.67
t0 = time.perf_counter()
ds_thresh = filters.threshold(ds, "cell_elevation", lo, hi)
t1 = time.perf_counter()

print(f"4. Threshold [{lo}, {hi}]: {ds_thresh.num_cells:,} / {ds.num_cells:,} cells ({t1-t0:.4f}s)")

# Verify: thresholded dataset still has point arrays
assert "elevation" in ds_thresh.point_data

# ================================================================
# 5. Run cell average on the thresholded dataset (explicit cell set)
# ================================================================
t0 = time.perf_counter()
filters.cell_average(ds_thresh, "elevation", "cell_elevation")
t1 = time.perf_counter()

cell_thresh = ds_thresh.get_cell_array("cell_elevation")
cell_thresh_np = cell_thresh.data.to_numpy()[:ds_thresh.num_cells]
print(f"5. Cell average (thresholded): range [{cell_thresh_np.min():.3f}, {cell_thresh_np.max():.3f}] ({t1-t0:.4f}s)")

# Verify the thresholded range
assert cell_thresh_np.min() >= lo - 0.05, f"min {cell_thresh_np.min()} < {lo}"
assert cell_thresh_np.max() <= hi + 0.05, f"max {cell_thresh_np.max()} > {hi}"

# ================================================================
# 6. Build explicit connectivity from structured (for comparison)
# ================================================================
t0 = time.perf_counter()
explicit_cs = from_structured(ds.cell_set, ds.num_cells)
t1 = time.perf_counter()
print(f"6. Explicit from structured: {ds.num_cells:,} cells ({t1-t0:.4f}s)")

# Run cell average with explicit cell set to verify same results
ds_explicit = make_rectilinear_dataset(x, y, z)
ds_explicit.cell_set = explicit_cs
filters.point_elevation(ds_explicit, direction=(0.0, 0.0, 1.0), name="elevation")
filters.cell_average(ds_explicit, "elevation", "cell_elevation")
expl_np = ds_explicit.get_cell_array("cell_elevation").data.to_numpy()
np.testing.assert_allclose(expl_np, cell_np, rtol=1e-5)
print("   Explicit matches structured — OK")

print("\n" + "=" * 60)
print("All checks passed!")
