"""10 -- Vector operations in kernels.

PGC has a built-in Vector type for 3D math.  Construct vectors from
scalar values, then use methods like .dot(), .cross(), .normalized(),
.norm(), and .norm_sqr().

Vectors are temporary values inside kernels -- component data lives in
separate scalar fields (one field per component).

Usage:
  uv run python examples/10_vectors.py
"""

import numpy as np
import pgc

import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument('--arch', default='cpu', choices=['cpu', 'metal', 'cuda', 'hip', 'level_zero', 'wgpu'])
_arch = getattr(pgc, _parser.parse_args().arch)
pgc.init(arch=_arch)

n = 100

# Separate fields for each component
ax = pgc.field(dtype=pgc.f32, shape=(n,))
ay = pgc.field(dtype=pgc.f32, shape=(n,))
az = pgc.field(dtype=pgc.f32, shape=(n,))
nx_out = pgc.field(dtype=pgc.f32, shape=(n,))
ny_out = pgc.field(dtype=pgc.f32, shape=(n,))
nz_out = pgc.field(dtype=pgc.f32, shape=(n,))
dot_out = pgc.field(dtype=pgc.f32, shape=(n,))

# Fill with points on a spiral
theta = np.linspace(0, 4 * np.pi, n, dtype=np.float32)
ax.from_numpy(np.cos(theta))
ay.from_numpy(np.sin(theta))
az.from_numpy(np.linspace(0, 1, n, dtype=np.float32))


@pgc.kernel
def normalize_vectors(ax, ay, az, nx_out, ny_out, nz_out):
    """Normalize each vector to unit length."""
    for i in range(ax.shape[0]):
        v = pgc.Vector([ax[i], ay[i], az[i]])
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

bx = pgc.field(dtype=pgc.f32, shape=(n,))
by = pgc.field(dtype=pgc.f32, shape=(n,))
bz = pgc.field(dtype=pgc.f32, shape=(n,))

# b = (1, 0, 0) everywhere
bx.from_numpy(np.ones(n, dtype=np.float32))
by.from_numpy(np.zeros(n, dtype=np.float32))
bz.from_numpy(np.zeros(n, dtype=np.float32))


@pgc.kernel
def compute_dot(ax, ay, az, bx, by, bz, dot_out):
    """Compute dot product of two vector fields."""
    for i in range(ax.shape[0]):
        a = pgc.Vector([ax[i], ay[i], az[i]])
        b = pgc.Vector([bx[i], by[i], bz[i]])
        dot_out[i] = a.dot(b)


compute_dot(ax, ay, az, bx, by, bz, dot_out)
dots = dot_out.to_numpy()
expected = np.cos(theta)  # dot with (1,0,0) = x component
assert np.allclose(dots, expected, atol=1e-5)
print("2. Dot product: matches expected cos(theta)")


# --- Cross product ---

cx_out = pgc.field(dtype=pgc.f32, shape=(n,))
cy_out = pgc.field(dtype=pgc.f32, shape=(n,))
cz_out = pgc.field(dtype=pgc.f32, shape=(n,))


@pgc.kernel
def compute_cross(ax, ay, az, bx, by, bz, cx, cy, cz):
    """Cross product of two vector fields."""
    for i in range(ax.shape[0]):
        a = pgc.Vector([ax[i], ay[i], az[i]])
        b = pgc.Vector([bx[i], by[i], bz[i]])
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
