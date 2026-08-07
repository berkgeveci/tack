"""25 -- Point-to-cell averaging: structured vs explicit connectivity.

Demonstrates the cell set abstraction for 3D hexahedral meshes.  The same
averaging kernel works with both connectivity representations:

1. CellSetStructured3D -- connectivity computed on the fly from grid
   dimensions using div/mod.  Zero connectivity storage.
2. CellSetExplicitHex -- connectivity stored in an explicit array of
   8 point IDs per cell.

Both cell set types expose get_cell_points(cell_id) -> (p0, ..., p7)
via multi-return @tack.func.  The structured version computes the 8 corner
point IDs from (ci, cj, ck) decomposition; the explicit version reads
them from a flat connectivity field.

Usage:
  uv run python examples/25_point_to_cell.py
  uv run python examples/25_point_to_cell.py --arch metal
  uv run python examples/25_point_to_cell.py --arch metal --size 200
"""

import argparse
import time

import numpy as np

import tack

_parser = argparse.ArgumentParser()
_parser.add_argument('--arch', default='cpu', choices=['cpu', 'metal', 'cuda', 'hip', 'level_zero'])
_parser.add_argument('--size', type=int, default=100,
                     help='Grid cells per dimension (default 100)')
_parser.add_argument('--warmup', type=int, default=2)
_parser.add_argument('--trials', type=int, default=5)
_args = _parser.parse_args()
_arch = getattr(tack, _args.arch)
tack.init(arch=_arch)


# ================================================================
# CELL SET TYPES
# ================================================================

@tack.data_oriented
class CellSetStructured3D:
    """Structured hex mesh: connectivity from grid dimensions, zero storage.

    Grid of nx*ny*nz cells with (nx+1)*(ny+1)*(nz+1) points.
    Point ordering is x-fastest: point(i,j,k) = k*(nx+1)*(ny+1) + j*(nx+1) + i.
    """

    def __init__(self, nx, ny, nz):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.nx_plus1 = nx + 1
        self.nxy = nx * ny
        self.nxy_plus1 = (nx + 1) * (ny + 1)

    @tack.func
    def get_cell_points(self, cell_id):
        ci = cell_id % self.nx
        cj = (cell_id // self.nx) % self.ny
        ck = cell_id // self.nxy

        base = ck * self.nxy_plus1 + cj * self.nx_plus1 + ci

        p0 = base
        p1 = base + 1
        p2 = base + self.nx_plus1
        p3 = base + self.nx_plus1 + 1
        p4 = base + self.nxy_plus1
        p5 = base + self.nxy_plus1 + 1
        p6 = base + self.nxy_plus1 + self.nx_plus1
        p7 = base + self.nxy_plus1 + self.nx_plus1 + 1
        return p0, p1, p2, p3, p4, p5, p6, p7


@tack.data_oriented
class CellSetExplicitHex:
    """Explicit hex mesh: connectivity stored as flat field [c0p0..c0p7, c1p0..c1p7, ...]."""

    def __init__(self, connectivity):
        self.connectivity = connectivity

    @tack.func
    def get_cell_points(self, cell_id):
        base = cell_id * 8
        p0 = self.connectivity[base]
        p1 = self.connectivity[base + 1]
        p2 = self.connectivity[base + 2]
        p3 = self.connectivity[base + 3]
        p4 = self.connectivity[base + 4]
        p5 = self.connectivity[base + 5]
        p6 = self.connectivity[base + 6]
        p7 = self.connectivity[base + 7]
        return p0, p1, p2, p3, p4, p5, p6, p7


# ================================================================
# AVERAGING KERNEL -- works with any cell set type
# ================================================================

@tack.kernel
def point_to_cell_average(cell_set: tack.template(), point_data, cell_data, n_cells):
    """Average point data to cells: cell_data[c] = mean of 8 corner values."""
    for c in range(n_cells):
        p0, p1, p2, p3, p4, p5, p6, p7 = cell_set.get_cell_points(c)
        avg = (point_data[p0] + point_data[p1] + point_data[p2] + point_data[p3]
             + point_data[p4] + point_data[p5] + point_data[p6] + point_data[p7]) * 0.125
        cell_data[c] = avg


# ================================================================
# BUILD EXPLICIT CONNECTIVITY FROM STRUCTURED GRID
# ================================================================

@tack.kernel
def build_explicit_connectivity(struct: tack.template(), conn, n_cells):
    """Expand structured connectivity into explicit array."""
    for c in range(n_cells):
        p0, p1, p2, p3, p4, p5, p6, p7 = struct.get_cell_points(c)
        base = c * 8
        conn[base] = p0
        conn[base + 1] = p1
        conn[base + 2] = p2
        conn[base + 3] = p3
        conn[base + 4] = p4
        conn[base + 5] = p5
        conn[base + 6] = p6
        conn[base + 7] = p7


# ================================================================
# TEST & BENCHMARK
# ================================================================

N = _args.size
nx, ny, nz = N, N, N
n_points = (nx + 1) * (ny + 1) * (nz + 1)
n_cells = nx * ny * nz
warmup = _args.warmup
trials = _args.trials

print(f"Grid: {nx}x{ny}x{nz} = {n_cells:,} cells, {n_points:,} points")
print(f"Backend: {_args.arch}")
print(f"Warmup: {warmup}, Trials: {trials}")
print()

# Point data: some interesting scalar field
point_data = tack.field(dtype=tack.f32, shape=(n_points,))
pd = np.arange(n_points, dtype=np.float32) * 0.01
point_data.from_numpy(pd)

cell_data = tack.field(dtype=tack.f32, shape=(n_cells,))

results = {}


# --- Structured ---
print("Structured cell set...")
struct = CellSetStructured3D(nx, ny, nz)
print("  Connectivity memory: 0 bytes (computed on the fly)")

for i in range(warmup):
    point_to_cell_average(struct, point_data, cell_data, n_cells)

times = []
for i in range(trials):
    t0 = time.perf_counter()
    point_to_cell_average(struct, point_data, cell_data, n_cells)
    t1 = time.perf_counter()
    times.append(t1 - t0)

struct_time = min(times)
struct_result = cell_data.to_numpy().copy()
results["Structured"] = struct_time
print(f"  Best of {trials}: {struct_time:.4f}s")


# --- Explicit (built from structured) ---
print("\nExplicit cell set...")
conn_field = tack.field(dtype=tack.i32, shape=(n_cells * 8,))
build_explicit_connectivity(struct, conn_field, n_cells)

explicit = CellSetExplicitHex(conn_field)
conn_mem = n_cells * 8 * 4
print(f"  Connectivity memory: {conn_mem:,} bytes ({conn_mem/1024/1024:.1f} MB)")

for i in range(warmup):
    point_to_cell_average(explicit, point_data, cell_data, n_cells)

times = []
for i in range(trials):
    t0 = time.perf_counter()
    point_to_cell_average(explicit, point_data, cell_data, n_cells)
    t1 = time.perf_counter()
    times.append(t1 - t0)

explicit_time = min(times)
explicit_result = cell_data.to_numpy().copy()
results["Explicit"] = explicit_time
print(f"  Best of {trials}: {explicit_time:.4f}s")


# --- numpy reference ---
print("\nnumpy reference...")

# Build connectivity array on host
conn_np = conn_field.to_numpy().reshape(n_cells, 8)
pd_np = pd  # already have the numpy array

for i in range(warmup):
    corners = pd_np[conn_np]  # (n_cells, 8)
    np_result = corners.mean(axis=1)

times = []
for i in range(trials):
    t0 = time.perf_counter()
    corners = pd_np[conn_np]  # fancy indexing
    np_result = corners.mean(axis=1)
    t1 = time.perf_counter()
    times.append(t1 - t0)

numpy_time = min(times)
results["numpy"] = numpy_time
print(f"  Best of {trials}: {numpy_time:.4f}s")


# --- VTK (optional) ---
vtk_result = None
try:
    from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
    from vtkmodules.vtkCommonDataModel import vtkImageData
    from vtkmodules.vtkFiltersCore import vtkPointDataToCellData

    print("\nVTK vtkPointDataToCellData...")

    img = vtkImageData()
    img.SetDimensions(nx + 1, ny + 1, nz + 1)

    vtk_arr = numpy_to_vtk(pd_np, deep=True)
    vtk_arr.SetName("scalar")
    img.GetPointData().AddArray(vtk_arr)
    img.GetPointData().SetActiveScalars("scalar")

    p2c = vtkPointDataToCellData()
    p2c.SetInputData(img)
    p2c.ProcessAllArraysOn()

    for i in range(warmup):
        p2c.Modified()
        p2c.Update()

    times = []
    for i in range(trials):
        p2c.Modified()
        t0 = time.perf_counter()
        p2c.Update()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    vtk_time = min(times)
    results["VTK"] = vtk_time
    vtk_out = p2c.GetOutput()
    vtk_result = vtk_to_numpy(vtk_out.GetCellData().GetArray("scalar")).copy()
    print(f"  Best of {trials}: {vtk_time:.4f}s")

    del p2c, img, vtk_out

except ImportError:
    print("\nVTK not installed -- skipping VTK comparison.")
    print("  Install with: uv pip install vtk")


# --- Validation ---
print("\n--- Validation ---")
assert np.allclose(struct_result, explicit_result, atol=1e-4), "Structured vs Explicit mismatch"
assert np.allclose(struct_result, np_result, atol=1e-4), "Structured vs numpy mismatch"
if vtk_result is not None:
    assert np.allclose(struct_result, vtk_result, atol=1e-4), "Structured vs VTK mismatch"
print("  Structured == Explicit == numpy" + (" == VTK" if vtk_result is not None else "") + ": OK")

# Spot-check cell 0
nxp1, nxyp1 = nx + 1, (nx + 1) * (ny + 1)
pts_0 = [0, 1, nxp1, nxp1+1, nxyp1, nxyp1+1, nxyp1+nxp1, nxyp1+nxp1+1]
expected_0 = np.mean([pd[p] for p in pts_0])
assert abs(struct_result[0] - expected_0) < 1e-4, f"Cell 0 mismatch: {struct_result[0]} vs {expected_0}"
print(f"  Cell 0 spot-check: OK (avg of points {pts_0} = {expected_0:.4f})")


# --- Summary ---
print("\n" + "=" * 55)
print(f"  {'Type':<12} {'Time':>8}  {'Speedup':>8}  {'Conn Memory':>14}")
print("-" * 55)
baseline = results.get("VTK", numpy_time)
row_names = ["Structured", "Explicit", "numpy"]
if "VTK" in results:
    row_names.append("VTK")
for name in row_names:
    t = results[name]
    speedup = baseline / t
    if name == "Structured":
        mem = "0 B"
    else:
        mem = f"{n_cells*8*4/1024/1024:.0f} MB"
    print(f"  {name:<12} {t:>7.4f}s  {speedup:>7.2f}x  {mem:>14}")
print("=" * 55)

print("""
Key insight: the same point_to_cell_average kernel works with both
CellSetStructured3D (zero connectivity memory) and CellSetExplicitHex
(explicit connectivity array).  Multi-return @tack.func allows
get_cell_points() to return all 8 corner IDs at once, avoiding redundant
div/mod recomputation that per-corner access would cause.
""")


# ================================================================
# PASS 2: MULTI-COMPONENT (3-component tuple arrays)
# ================================================================

print("=" * 60)
print("PASS 2: 3-component point-to-cell averaging")
print("=" * 60)
print()

# --- Tuple array types (from example 23) ---

@tack.data_oriented
class AOSTupleArray:
    """AOS layout: [x0,y0,z0, x1,y1,z1, ...]."""

    def __init__(self, data, num_tuples, num_components):
        self.data = data
        self.num_tuples = num_tuples
        self.num_components = num_components

    @tack.func
    def get_value(self, i, c):
        return self.data[i * self.num_components + c]

    @tack.func
    def set_value(self, i, c, val):
        self.data[i * self.num_components + c] = val


@tack.data_oriented
class SOATupleArray3:
    """SOA layout for 3 components: separate fields c0, c1, c2."""

    def __init__(self, c0, c1, c2):
        self.c0 = c0
        self.c1 = c1
        self.c2 = c2

    @tack.func
    def get_value(self, i, c):
        result = self.c0[i]
        if c == 1:
            result = self.c1[i]
        if c == 2:
            result = self.c2[i]
        return result

    @tack.func
    def set_value(self, i, c, val):
        if c == 0:
            self.c0[i] = val
        if c == 1:
            self.c1[i] = val
        if c == 2:
            self.c2[i] = val


# --- Multi-component averaging kernel ---

@tack.kernel
def point_to_cell_average_mc(cell_set: tack.template(),
                             point_data: tack.template(),
                             cell_data: tack.template(),
                             n_cells, nc):
    """Average multi-component point data to cells.

    Point IDs computed once per cell, then inner loop over components.
    """
    for c in range(n_cells):
        p0, p1, p2, p3, p4, p5, p6, p7 = cell_set.get_cell_points(c)
        for comp in range(nc):
            avg = (point_data.get_value(p0, comp)
                 + point_data.get_value(p1, comp)
                 + point_data.get_value(p2, comp)
                 + point_data.get_value(p3, comp)
                 + point_data.get_value(p4, comp)
                 + point_data.get_value(p5, comp)
                 + point_data.get_value(p6, comp)
                 + point_data.get_value(p7, comp)) * 0.125
            cell_data.set_value(c, comp, avg)


nc = 3

# Build 3-component point data (some wavelet-ish values)
np.random.seed(42)
pd_mc = np.random.randn(n_points, nc).astype(np.float32)

results2 = {}


# --- Structured + AOS ---
print("Structured + AOS (3 components)...")
aos_pt_field = tack.field(dtype=tack.f32, shape=(n_points * nc,))
aos_pt_field.from_numpy(pd_mc.ravel())
aos_pt = AOSTupleArray(aos_pt_field, n_points, nc)

aos_cell_field = tack.field(dtype=tack.f32, shape=(n_cells * nc,))
aos_cell = AOSTupleArray(aos_cell_field, n_cells, nc)

for i in range(warmup):
    point_to_cell_average_mc(struct, aos_pt, aos_cell, n_cells, nc)

times = []
for i in range(trials):
    t0 = time.perf_counter()
    point_to_cell_average_mc(struct, aos_pt, aos_cell, n_cells, nc)
    t1 = time.perf_counter()
    times.append(t1 - t0)

struct_aos_time = min(times)
struct_aos_result = aos_cell_field.to_numpy().reshape(n_cells, nc).copy()
results2["Struct+AOS"] = struct_aos_time
print(f"  Best of {trials}: {struct_aos_time:.4f}s")


# --- Structured + SOA ---
print("\nStructured + SOA (3 components)...")
soa_pt_fields = [tack.field(dtype=tack.f32, shape=(n_points,)) for _ in range(nc)]
for c in range(nc):
    soa_pt_fields[c].from_numpy(pd_mc[:, c].copy())
soa_pt = SOATupleArray3(*soa_pt_fields)

soa_cell_fields = [tack.field(dtype=tack.f32, shape=(n_cells,)) for _ in range(nc)]
soa_cell = SOATupleArray3(*soa_cell_fields)

for i in range(warmup):
    point_to_cell_average_mc(struct, soa_pt, soa_cell, n_cells, nc)

times = []
for i in range(trials):
    t0 = time.perf_counter()
    point_to_cell_average_mc(struct, soa_pt, soa_cell, n_cells, nc)
    t1 = time.perf_counter()
    times.append(t1 - t0)

struct_soa_time = min(times)
struct_soa_result = np.column_stack([f.to_numpy() for f in soa_cell_fields]).copy()
results2["Struct+SOA"] = struct_soa_time
print(f"  Best of {trials}: {struct_soa_time:.4f}s")


# --- Explicit + AOS ---
print("\nExplicit + AOS (3 components)...")
aos_cell_field2 = tack.field(dtype=tack.f32, shape=(n_cells * nc,))
aos_cell2 = AOSTupleArray(aos_cell_field2, n_cells, nc)

for i in range(warmup):
    point_to_cell_average_mc(explicit, aos_pt, aos_cell2, n_cells, nc)

times = []
for i in range(trials):
    t0 = time.perf_counter()
    point_to_cell_average_mc(explicit, aos_pt, aos_cell2, n_cells, nc)
    t1 = time.perf_counter()
    times.append(t1 - t0)

explicit_aos_time = min(times)
results2["Explicit+AOS"] = explicit_aos_time
print(f"  Best of {trials}: {explicit_aos_time:.4f}s")


# --- numpy reference ---
print("\nnumpy (3 components)...")

for i in range(warmup):
    corners_mc = pd_mc[conn_np]  # (n_cells, 8, 3)
    np_mc_result = corners_mc.mean(axis=1)

times = []
for i in range(trials):
    t0 = time.perf_counter()
    corners_mc = pd_mc[conn_np]  # (n_cells, 8, 3)
    np_mc_result = corners_mc.mean(axis=1)
    t1 = time.perf_counter()
    times.append(t1 - t0)

numpy_mc_time = min(times)
results2["numpy"] = numpy_mc_time
print(f"  Best of {trials}: {numpy_mc_time:.4f}s")


# --- VTK (3-component) ---
vtk_mc_result = None
try:
    from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
    from vtkmodules.vtkCommonDataModel import vtkImageData
    from vtkmodules.vtkFiltersCore import vtkPointDataToCellData

    print("\nVTK (3 components)...")

    img = vtkImageData()
    img.SetDimensions(nx + 1, ny + 1, nz + 1)

    vtk_arr = numpy_to_vtk(pd_mc, deep=True)
    vtk_arr.SetName("vectors")
    img.GetPointData().AddArray(vtk_arr)

    p2c = vtkPointDataToCellData()
    p2c.SetInputData(img)
    p2c.ProcessAllArraysOn()

    for i in range(warmup):
        p2c.Modified()
        p2c.Update()

    times = []
    for i in range(trials):
        p2c.Modified()
        t0 = time.perf_counter()
        p2c.Update()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    vtk_mc_time = min(times)
    results2["VTK"] = vtk_mc_time
    vtk_out = p2c.GetOutput()
    vtk_mc_result = vtk_to_numpy(vtk_out.GetCellData().GetArray("vectors")).copy()
    print(f"  Best of {trials}: {vtk_mc_time:.4f}s")

    del p2c, img, vtk_out

except ImportError:
    print("\nVTK not installed -- skipping VTK comparison.")


# --- Validation (multi-component) ---
print("\n--- Validation (3-component) ---")
assert np.allclose(struct_aos_result, struct_soa_result, atol=1e-4), "Struct AOS vs SOA mismatch"
assert np.allclose(struct_aos_result, np_mc_result, atol=1e-4), "Struct AOS vs numpy mismatch"
if vtk_mc_result is not None:
    assert np.allclose(struct_aos_result, vtk_mc_result, atol=1e-4), "Struct AOS vs VTK mismatch"
print("  Struct+AOS == Struct+SOA == numpy" + (" == VTK" if vtk_mc_result is not None else "") + ": OK")


# --- Summary (multi-component) ---
print("\n" + "=" * 60)
print(f"  {'Type':<16} {'Time':>8}  {'Speedup':>8}")
print("-" * 60)
baseline2 = results2.get("VTK", numpy_mc_time)
row_names2 = ["Struct+AOS", "Struct+SOA", "Explicit+AOS", "numpy"]
if "VTK" in results2:
    row_names2.append("VTK")
for name in row_names2:
    t = results2[name]
    speedup = baseline2 / t
    print(f"  {name:<16} {t:>7.4f}s  {speedup:>7.2f}x")
print("=" * 60)
