"""10 -- Vector operations in kernels.

Tack has a built-in Vector type for 3D math.  Construct vectors from
scalar values, then use methods like .dot(), .cross(), .normalized(),
.norm(), and .norm_sqr().

Vectors are temporary values inside kernels -- component data lives in
separate scalar fields (one field per component).

Usage:
  uv run python examples/10_vectors.py
"""

import numpy as np
import tack

import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument('--arch', default='cpu', choices=['cpu', 'metal', 'cuda', 'hip', 'level_zero'])
_arch = getattr(tack, _parser.parse_args().arch)
tack.init(arch=_arch)

n = 100

# Separate fields for each component
ax = tack.field(dtype=tack.f32, shape=(n,))
ay = tack.field(dtype=tack.f32, shape=(n,))
az = tack.field(dtype=tack.f32, shape=(n,))
nx_out = tack.field(dtype=tack.f32, shape=(n,))
ny_out = tack.field(dtype=tack.f32, shape=(n,))
nz_out = tack.field(dtype=tack.f32, shape=(n,))
dot_out = tack.field(dtype=tack.f32, shape=(n,))

# Fill with points on a spiral
theta = np.linspace(0, 4 * np.pi, n, dtype=np.float32)
ax.from_numpy(np.cos(theta))
ay.from_numpy(np.sin(theta))
az.from_numpy(np.linspace(0, 1, n, dtype=np.float32))


@tack.kernel
def normalize_vectors(ax, ay, az, nx_out, ny_out, nz_out):
    """Normalize each vector to unit length."""
    for i in range(ax.shape[0]):
        v = tack.Vector([ax[i], ay[i], az[i]])
        n = v.normalized()
        nx_out[i] = n[0]
        ny_out[i] = n[1]
        nz_out[i] = n[2]


normalize_vectors(ax, ay, az, nx_out, ny_out, nz_out)

# Verify unit length
rx = nx_out.to_numpy()
ry = ny_out.to_numpy()
rz = nz_out.to_numpy()
lengths = np.sqrt(rx**2 + ry**2 + rz**2)
assert np.allclose(lengths, 1.0, atol=1e-5)
print("1. Normalization: all vectors have unit length")


# --- Dot product ---

bx = tack.field(dtype=tack.f32, shape=(n,))
by = tack.field(dtype=tack.f32, shape=(n,))
bz = tack.field(dtype=tack.f32, shape=(n,))

# b = (1, 0, 0) everywhere
bx.from_numpy(np.ones(n, dtype=np.float32))
by.from_numpy(np.zeros(n, dtype=np.float32))
bz.from_numpy(np.zeros(n, dtype=np.float32))


@tack.kernel
def compute_dot(ax, ay, az, bx, by, bz, dot_out):
    """Compute dot product of two vector fields."""
    for i in range(ax.shape[0]):
        a = tack.Vector([ax[i], ay[i], az[i]])
        b = tack.Vector([bx[i], by[i], bz[i]])
        dot_out[i] = a.dot(b)


compute_dot(ax, ay, az, bx, by, bz, dot_out)
dots = dot_out.to_numpy()
expected = np.cos(theta)  # dot with (1,0,0) = x component
assert np.allclose(dots, expected, atol=1e-5)
print("2. Dot product: matches expected cos(theta)")


# --- Cross product ---

cx_out = tack.field(dtype=tack.f32, shape=(n,))
cy_out = tack.field(dtype=tack.f32, shape=(n,))
cz_out = tack.field(dtype=tack.f32, shape=(n,))


@tack.kernel
def compute_cross(ax, ay, az, bx, by, bz, cx, cy, cz):
    """Cross product of two vector fields."""
    for i in range(ax.shape[0]):
        a = tack.Vector([ax[i], ay[i], az[i]])
        b = tack.Vector([bx[i], by[i], bz[i]])
        c = a.cross(b)
        cx[i] = c[0]
        cy[i] = c[1]
        cz[i] = c[2]


compute_cross(ax, ay, az, bx, by, bz, cx_out, cy_out, cz_out)
print("3. Cross product: OK")

# Verify a . (a x b) = 0 (cross product is perpendicular to both inputs)
cr_x = cx_out.to_numpy()
cr_y = cy_out.to_numpy()
cr_z = cz_out.to_numpy()
a_x = ax.to_numpy()
a_y = ay.to_numpy()
a_z = az.to_numpy()
perp = a_x * cr_x + a_y * cr_y + a_z * cr_z
assert np.allclose(perp, 0.0, atol=1e-5)
print("   Verified: a . (a x b) = 0 (perpendicular)")

print("\nVector operations: OK")
