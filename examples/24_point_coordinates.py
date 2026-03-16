"""24 — Point coordinates: Product vs AOS vs SOA performance benchmark.

Compares three ways to store 3D point coordinates when evaluating an
expensive scalar function (wavelet) over a large rectilinear grid:

1. ProductPoints — three 1D arrays, coordinates computed on the fly.
   Storage: O(nx+ny+nz).  No coordinate memory allocated.
2. SOAPoints — three full-size arrays (x, y, z).  Storage: O(3N).
3. AOSPoints — one interleaved array.  Storage: O(3N).
4. numpy — CPU reference using broadcasting.

The wavelet function:  f(x,y,z) = sin(kx*x) * cos(ky*y) * sin(kz*z + x*y)

Usage:
  uv run python examples/24_point_coordinates.py
  uv run python examples/24_point_coordinates.py --arch metal
  uv run python examples/24_point_coordinates.py --arch metal --size 500
"""

import gc
import time
import numpy as np
import pgc

import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument('--arch', default='cpu', choices=['cpu', 'metal', 'cuda', 'hip', 'vulkan', 'level_zero'])
_parser.add_argument('--size', type=int, default=500,
                     help='Grid size per dimension (default 500 → 125M points)')
_parser.add_argument('--warmup', type=int, default=2)
_parser.add_argument('--trials', type=int, default=5)
_args = _parser.parse_args()
_arch = getattr(pgc, _args.arch)
pgc.init(arch=_arch)


# ================================================================
# POINT COORDINATE TYPES
# ================================================================

@pgc.data_oriented
class AOSPoints:
    """Interleaved: [x0,y0,z0, x1,y1,z1, ...]."""

    def __init__(self, data, num_points):
        self.data = data
        self.num_points = num_points

    @pgc.func
    def get_x(self, id):
        return self.data[id * 3 + 0]

    @pgc.func
    def get_y(self, id):
        return self.data[id * 3 + 1]

    @pgc.func
    def get_z(self, id):
        return self.data[id * 3 + 2]


@pgc.data_oriented
class SOAPoints:
    """Separate arrays: x=[...], y=[...], z=[...]."""

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    @pgc.func
    def get_x(self, id):
        return self.x[id]

    @pgc.func
    def get_y(self, id):
        return self.y[id]

    @pgc.func
    def get_z(self, id):
        return self.z[id]


@pgc.data_oriented
class ProductPoints:
    """Rectilinear cross product of 3 arrays. Storage: O(nx+ny+nz)."""

    def __init__(self, xc, yc, zc):
        self.xc = xc
        self.yc = yc
        self.zc = zc
        self.nx = xc.shape[0]
        self.ny = yc.shape[0]
        self.nx_by_ny = xc.shape[0] * yc.shape[0]

    @pgc.func
    def get_x(self, id):
        return self.xc[id % self.nx]

    @pgc.func
    def get_y(self, id):
        return self.yc[(id // self.nx) % self.ny]

    @pgc.func
    def get_z(self, id):
        return self.zc[id // self.nx_by_ny]


# ================================================================
# WAVELET KERNEL — same kernel for all coordinate types
# ================================================================

@pgc.kernel
def compute_wavelet(pts: pgc.template(), output, kx, ky, kz, n):
    """f(x,y,z) = sin(kx*x) * cos(ky*y) * sin(kz*z + x*y)"""
    for id in range(n):
        x = pts.get_x(id)
        y = pts.get_y(id)
        z = pts.get_z(id)
        output[id] = sin(kx * x) * cos(ky * y) * sin(kz * z + x * y)


# ================================================================
# BENCHMARK
# ================================================================

N = _args.size
num_points = N * N * N
warmup = _args.warmup
trials = _args.trials

kx, ky, kz = 3.7, 2.3, 5.1

print(f"Grid: {N}x{N}x{N} = {num_points:,} points")
print(f"Backend: {_args.arch}")
print(f"Warmup: {warmup}, Trials: {trials}")
print()

# 1D coordinate arrays (shared across all representations)
x1d = np.linspace(-np.pi, np.pi, N, dtype=np.float32)
y1d = np.linspace(-np.pi, np.pi, N, dtype=np.float32) * 1.3
z1d = np.linspace(-np.pi, np.pi, N, dtype=np.float32) * 0.7

results = {}


# --- Product Points ---
print("Product Points...")
xc = pgc.field(dtype=pgc.f32, shape=(N,))
yc = pgc.field(dtype=pgc.f32, shape=(N,))
zc = pgc.field(dtype=pgc.f32, shape=(N,))
xc.from_numpy(x1d)
yc.from_numpy(y1d)
zc.from_numpy(z1d)
product = ProductPoints(xc, yc, zc)
output = pgc.field(dtype=pgc.f32, shape=(num_points,))

coord_mem = (N + N + N) * 4
print(f"  Coordinate memory: {coord_mem:,} bytes ({coord_mem/1024:.1f} KB)")

for i in range(warmup):
    compute_wavelet(product, output, kx, ky, kz, num_points)

times = []
for i in range(trials):
    t0 = time.perf_counter()
    compute_wavelet(product, output, kx, ky, kz, num_points)
    t1 = time.perf_counter()
    times.append(t1 - t0)

product_time = min(times)
product_result = output.to_numpy()[:8].copy()
results["Product"] = product_time
print(f"  Best of {trials}: {product_time:.4f}s")

# Keep output, release product-specific fields
del product, xc, yc, zc
gc.collect()


# --- SOA Points ---
print("\nSOA Points...")
sx = pgc.field(dtype=pgc.f32, shape=(num_points,))
sy = pgc.field(dtype=pgc.f32, shape=(num_points,))
sz = pgc.field(dtype=pgc.f32, shape=(num_points,))

# Build SOA coordinates using a Product → SOA expansion kernel
xc = pgc.field(dtype=pgc.f32, shape=(N,))
yc = pgc.field(dtype=pgc.f32, shape=(N,))
zc = pgc.field(dtype=pgc.f32, shape=(N,))
xc.from_numpy(x1d)
yc.from_numpy(y1d)
zc.from_numpy(z1d)
tmp_product = ProductPoints(xc, yc, zc)

@pgc.data_oriented
class _WritableSOA:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
    @pgc.func
    def set_x(self, id, val):
        self.x[id] = val
    @pgc.func
    def set_y(self, id, val):
        self.y[id] = val
    @pgc.func
    def set_z(self, id, val):
        self.z[id] = val
    @pgc.func
    def get_x(self, id):
        return self.x[id]
    @pgc.func
    def get_y(self, id):
        return self.y[id]
    @pgc.func
    def get_z(self, id):
        return self.z[id]

@pgc.kernel
def _expand(src: pgc.template(), dst: pgc.template(), n):
    for id in range(n):
        dst.set_x(id, src.get_x(id))
        dst.set_y(id, src.get_y(id))
        dst.set_z(id, src.get_z(id))

wsoa = _WritableSOA(sx, sy, sz)
_expand(tmp_product, wsoa, num_points)
del tmp_product, xc, yc, zc
gc.collect()

soa = SOAPoints(sx, sy, sz)
coord_mem = num_points * 3 * 4
print(f"  Coordinate memory: {coord_mem:,} bytes ({coord_mem/1024/1024:.1f} MB)")

for i in range(warmup):
    compute_wavelet(soa, output, kx, ky, kz, num_points)

times = []
for i in range(trials):
    t0 = time.perf_counter()
    compute_wavelet(soa, output, kx, ky, kz, num_points)
    t1 = time.perf_counter()
    times.append(t1 - t0)

soa_time = min(times)
soa_result = output.to_numpy()[:8].copy()
results["SOA"] = soa_time
print(f"  Best of {trials}: {soa_time:.4f}s")

del soa, sx, sy, sz
gc.collect()


# --- AOS Points ---
print("\nAOS Points...")
aos_field = pgc.field(dtype=pgc.f32, shape=(num_points * 3,))

# Build AOS coordinates via Product → AOS expansion
xc = pgc.field(dtype=pgc.f32, shape=(N,))
yc = pgc.field(dtype=pgc.f32, shape=(N,))
zc = pgc.field(dtype=pgc.f32, shape=(N,))
xc.from_numpy(x1d)
yc.from_numpy(y1d)
zc.from_numpy(z1d)
tmp_product = ProductPoints(xc, yc, zc)

@pgc.data_oriented
class _WritableAOS:
    def __init__(self, data, num_points):
        self.data = data
        self.num_points = num_points
    @pgc.func
    def set_x(self, id, val):
        self.data[id * 3 + 0] = val
    @pgc.func
    def set_y(self, id, val):
        self.data[id * 3 + 1] = val
    @pgc.func
    def set_z(self, id, val):
        self.data[id * 3 + 2] = val
    @pgc.func
    def get_x(self, id):
        return self.data[id * 3 + 0]
    @pgc.func
    def get_y(self, id):
        return self.data[id * 3 + 1]
    @pgc.func
    def get_z(self, id):
        return self.data[id * 3 + 2]

waos = _WritableAOS(aos_field, num_points)
_expand(tmp_product, waos, num_points)
del tmp_product, xc, yc, zc
gc.collect()

aos = AOSPoints(aos_field, num_points)
coord_mem = num_points * 3 * 4
print(f"  Coordinate memory: {coord_mem:,} bytes ({coord_mem/1024/1024:.1f} MB)")

for i in range(warmup):
    compute_wavelet(aos, output, kx, ky, kz, num_points)

times = []
for i in range(trials):
    t0 = time.perf_counter()
    compute_wavelet(aos, output, kx, ky, kz, num_points)
    t1 = time.perf_counter()
    times.append(t1 - t0)

aos_time = min(times)
aos_result = output.to_numpy()[:8].copy()
results["AOS"] = aos_time
print(f"  Best of {trials}: {aos_time:.4f}s")

del aos, aos_field, output
gc.collect()


# --- numpy (explicit coordinates) ---
print("\nnumpy (explicit coordinates)...")

# Build full (N*N*N, 3) coordinate array
# Use z,y,x nesting order so ravel gives x-fastest, matching ProductPoints
print("  Building coordinate array...")
zz, yy, xx = np.meshgrid(z1d, y1d, x1d, indexing='ij')
coords = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)  # (num_points, 3)
del xx, yy, zz
gc.collect()

coord_mem = coords.nbytes
print(f"  Coordinate memory: {coord_mem:,} bytes ({coord_mem/1024/1024:.0f} MB)")

cx = coords[:, 0]
cy = coords[:, 1]
cz = coords[:, 2]

# Warm up
for i in range(warmup):
    out_np = np.sin(kx * cx) * np.cos(ky * cy) * np.sin(kz * cz + cx * cy)

times = []
for i in range(trials):
    t0 = time.perf_counter()
    out_np = np.sin(kx * cx) * np.cos(ky * cy) * np.sin(kz * cz + cx * cy)
    t1 = time.perf_counter()
    times.append(t1 - t0)

numpy_time = min(times)
# coords is in x-fastest order (indexing='ij'), same as our Product layout
numpy_result = out_np[:8].copy()
results["numpy"] = numpy_time
print(f"  Best of {trials}: {numpy_time:.4f}s")

del coords, cx, cy, cz, out_np
gc.collect()


# --- Validation ---
print("\n--- Validation ---")
assert np.allclose(product_result, soa_result, atol=1e-4), "Product vs SOA mismatch"
assert np.allclose(product_result, aos_result, atol=1e-4), "Product vs AOS mismatch"
assert np.allclose(product_result, numpy_result, atol=1e-4), "Product vs numpy mismatch"
print("  Product == SOA == AOS == numpy: OK")


# --- Summary ---
print("\n" + "=" * 50)
print(f"  {'Type':<12} {'Time':>8}  {'Speedup':>8}  {'Coord Memory':>14}")
print("-" * 50)
baseline = numpy_time
for name in ["Product", "SOA", "AOS", "numpy"]:
    t = results[name]
    speedup = baseline / t
    if name == "Product":
        mem = f"{(N*3)*4:,} B"
    else:
        mem = f"{num_points*3*4/1024/1024:.0f} MB"
    print(f"  {name:<12} {t:>7.4f}s  {speedup:>7.2f}x  {mem:>14}")
print("=" * 50)
