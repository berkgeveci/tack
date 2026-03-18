"""29 -- Volume rendering of a rectilinear grid.

Ray-cast volume renderer using front-to-back compositing with a simple
transfer function.  The scalar field is the same gyroid used in example 27,
evaluated on a cosine-spaced rectilinear grid.

Each pixel casts one ray through the volume.  At each step the ray samples
the scalar field via trilinear interpolation (binary search to locate the
cell on each axis, then lerp the eight corners).  The sample is mapped to
RGBA through a piecewise-linear transfer function and composited.

Usage:
  uv run python examples/29_volume_render.py
  uv run python examples/29_volume_render.py --arch metal
  uv run python examples/29_volume_render.py --arch metal --size 200
"""

import time
import numpy as np
import pgc

import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument('--arch', default='cpu',
                     choices=['cpu', 'metal', 'cuda', 'hip', 'level_zero'])
_parser.add_argument('--size', type=int, default=100,
                     help='Grid cells per dimension (default 100)')
_parser.add_argument('--width', type=int, default=800)
_parser.add_argument('--height', type=int, default=800)
import os as _os
_parser.add_argument('--save', default=_os.path.join(_os.path.dirname(__file__), '..', 'results', 'volrender_rect.png'))
_parser.add_argument('--warmup', type=int, default=2)
_parser.add_argument('--trials', type=int, default=5)
_args = _parser.parse_args()
_arch = getattr(pgc, _args.arch)
pgc.init(arch=_arch)

WIDTH = _args.width
HEIGHT = _args.height
MAX_STEPS = 800
TF_SIZE = 256


# ================================================================
# TRANSFER FUNCTION TABLE (built on CPU, uploaded as field)
# ================================================================

def build_transfer_function(vmin, vmax, n_layers=10):
    """Gaussian-layered transfer function with transparency.

    Near-zero values (the gyroid zero-crossing surface) are mostly
    transparent so you can see through to the interior structure.
    Extreme positive/negative values are more opaque and colorful.
    """
    tf = np.zeros((TF_SIZE, 4), dtype=np.float32)
    t = np.linspace(vmin, vmax, TF_SIZE)
    vmid = (vmin + vmax) / 2.0
    half_range = (vmax - vmin) / 2.0

    centers = np.linspace(vmin + 0.08 * (vmax - vmin),
                          vmax - 0.08 * (vmax - vmin), n_layers)
    width = ((vmax - vmin) / n_layers) ** 2 * 0.5

    for k in range(n_layers):
        frac = k / max(n_layers - 1, 1)
        # Cool-to-warm: blue -> white -> red
        r = np.clip(2.0 * frac, 0, 1)
        g = np.clip(1.0 - 2.5 * abs(frac - 0.5), 0, 1)
        b = np.clip(2.0 * (1.0 - frac), 0, 1)
        # Opacity peaks at extremes, dips near center -> transparency
        dist_from_center = abs(frac - 0.5) * 2.0  # 0 at center, 1 at edges
        alpha = 0.005 + 0.06 * dist_from_center ** 1.5
        gauss = np.exp(-(t - centers[k]) ** 2 / width)
        tf[:, 0] += gauss * r * alpha
        tf[:, 1] += gauss * g * alpha
        tf[:, 2] += gauss * b * alpha
        tf[:, 3] += gauss * alpha

    mask = tf[:, 3] > 0
    tf[mask, 0] /= tf[mask, 3]
    tf[mask, 1] /= tf[mask, 3]
    tf[mask, 2] /= tf[mask, 3]
    tf[:, :3] = np.clip(tf[:, :3], 0, 1)
    return tf


# ================================================================
# GPU FUNCTIONS
# ================================================================

@pgc.func
def find_cell(inv_table, coords, n_cells, inv_size, pos, cmin, inv_stride):
    """O(1) cell lookup via inverse table + +/-1 correction.

    inv_table: precomputed table mapping uniform bins to cell indices.
    coords:    the 1-D coordinate array (n_cells + 1 entries).
    n_cells:   number of cells on this axis.
    inv_size:  number of entries in inv_table.
    pos:       world coordinate to locate.
    cmin:      minimum coordinate value (coords[0]).
    inv_stride: (cmax - cmin) / inv_size.
    """
    # Uniform bin index -> approximate cell index
    k = int((pos - cmin) / inv_stride)
    if k < 0:
        k = 0
    if k >= inv_size:
        k = inv_size - 1
    ix = inv_table[k]
    # +/-1 correction: the bin may round to the wrong cell at boundaries
    if ix > 0 and pos < coords[ix]:
        ix = ix - 1
    if ix < n_cells - 1 and pos >= coords[ix + 1]:
        ix = ix + 1
    # Final clamp
    if ix < 0:
        ix = 0
    if ix >= n_cells:
        ix = n_cells - 1
    return ix


@pgc.func
def sample_tex(tex, xcoords, ycoords, zcoords,
               inv_x, inv_y, inv_z,
               inv_size, cmin_x, cmin_y, cmin_z,
               inv_sx, inv_sy, inv_sz,
               nx, ny, nz,
               px, py, pz):
    """Sample scalar via texture3d with O(1) cell lookup for coord mapping."""
    ix = find_cell(inv_x, xcoords, nx, inv_size, px, cmin_x, inv_sx)
    iy = find_cell(inv_y, ycoords, ny, inv_size, py, cmin_y, inv_sy)
    iz = find_cell(inv_z, zcoords, nz, inv_size, pz, cmin_z, inv_sz)

    # Fractional position within cell -> normalized texture coordinate
    x0 = xcoords[ix]
    x1 = xcoords[ix + 1]
    y0 = ycoords[iy]
    y1 = ycoords[iy + 1]
    z0 = zcoords[iz]
    z1 = zcoords[iz + 1]

    fx = (px - x0) / (x1 - x0 + 1.0e-20)
    fy = (py - y0) / (y1 - y0 + 1.0e-20)
    fz = (pz - z0) / (z1 - z0 + 1.0e-20)
    fx = max(0.0, min(1.0, fx))
    fy = max(0.0, min(1.0, fy))
    fz = max(0.0, min(1.0, fz))

    # Convert index+frac to normalized [0,1] for texture sampling
    tu = (float(ix) + fx) / float(nx)
    tv = (float(iy) + fy) / float(ny)
    tw = (float(iz) + fz) / float(nz)

    return tex.sample(tu, tv, tw)


@pgc.func
def apply_tf(tf_r, tf_g, tf_b, tf_a, val, vmin, vrange):
    """Look up transfer function RGBA for a scalar value."""
    t = (val - vmin) / vrange
    t = max(0.0, min(1.0, t))
    idx = int(t * 255.0)
    if idx < 0:
        idx = 0
    if idx > 255:
        idx = 255
    return tf_r[idx], tf_g[idx], tf_b[idx], tf_a[idx]


@pgc.kernel
def render(img_r, img_g, img_b,
           tex, xcoords, ycoords, zcoords,
           inv_x, inv_y, inv_z,
           tf_r, tf_g, tf_b, tf_a,
           cam_x, cam_y, cam_z,
           fwd_x, fwd_y, fwd_z,
           right_x, right_y, right_z,
           up_x, up_y, up_z,
           bmin_x, bmin_y, bmin_z,
           bmax_x, bmax_y, bmax_z,
           fov_half, step_size, opacity_scale,
           vmin, vrange,
           cmin_x, cmin_y, cmin_z,
           inv_sx, inv_sy, inv_sz,
           nx_p1, ny_p1, nxy_p1,
           width, height, inv_size,
           n_pixels):
    """Cast one ray per pixel, front-to-back compositing."""
    for pixel in range(n_pixels):
        i = pixel % width
        j = pixel // width
        aspect = float(width) / float(height)
        u = (2.0 * (float(i) + 0.5) / float(width) - 1.0) * fov_half * aspect
        v = (2.0 * (float(j) + 0.5) / float(height) - 1.0) * fov_half

        # Ray direction
        rd_x = fwd_x + u * right_x + v * up_x
        rd_y = fwd_y + u * right_y + v * up_y
        rd_z = fwd_z + u * right_z + v * up_z
        rd_len = sqrt(rd_x * rd_x + rd_y * rd_y + rd_z * rd_z)
        rd_x = rd_x / rd_len
        rd_y = rd_y / rd_len
        rd_z = rd_z / rd_len

        # Ray-box intersection (AABB)
        t_near = -3.4e38
        t_far = 3.4e38

        # X
        if abs(rd_x) < 1.0e-10:
            if cam_x < bmin_x or cam_x > bmax_x:
                t_near = 3.4e38
                t_far = -3.4e38
        else:
            t1 = (bmin_x - cam_x) / rd_x
            t2 = (bmax_x - cam_x) / rd_x
            if t1 > t2:
                t1, t2 = t2, t1
            t_near = max(t_near, t1)
            t_far = min(t_far, t2)

        # Y
        if abs(rd_y) < 1.0e-10:
            if cam_y < bmin_y or cam_y > bmax_y:
                t_near = 3.4e38
                t_far = -3.4e38
        else:
            t1 = (bmin_y - cam_y) / rd_y
            t2 = (bmax_y - cam_y) / rd_y
            if t1 > t2:
                t1, t2 = t2, t1
            t_near = max(t_near, t1)
            t_far = min(t_far, t2)

        # Z
        if abs(rd_z) < 1.0e-10:
            if cam_z < bmin_z or cam_z > bmax_z:
                t_near = 3.4e38
                t_far = -3.4e38
        else:
            t1 = (bmin_z - cam_z) / rd_z
            t2 = (bmax_z - cam_z) / rd_z
            if t1 > t2:
                t1, t2 = t2, t1
            t_near = max(t_near, t1)
            t_far = min(t_far, t2)

        t_near = max(t_near, 0.0)

        # March and composite
        cr = 0.0
        cg = 0.0
        cb = 0.0
        alpha = 0.0

        if t_near < t_far:
            t = t_near
            for _ in range(800):
                if t >= t_far or alpha > 0.99:
                    break
                sx = cam_x + t * rd_x
                sy = cam_y + t * rd_y
                sz = cam_z + t * rd_z

                val = sample_tex(tex, xcoords, ycoords, zcoords,
                                 inv_x, inv_y, inv_z,
                                 inv_size, cmin_x, cmin_y, cmin_z,
                                 inv_sx, inv_sy, inv_sz,
                                 nx_p1 - 1, ny_p1 - 1,
                                 nxy_p1 // nx_p1 - 1,
                                 sx, sy, sz)

                sr, sg, sb, sa = apply_tf(tf_r, tf_g, tf_b, tf_a,
                                          val, vmin, vrange)
                sa = 1.0 - exp(0.0 - sa * opacity_scale * step_size)
                if sa > 1.0:
                    sa = 1.0

                cr = cr + (1.0 - alpha) * sa * sr
                cg = cg + (1.0 - alpha) * sa * sg
                cb = cb + (1.0 - alpha) * sa * sb
                alpha = alpha + (1.0 - alpha) * sa

                t = t + step_size

        # Background: dark gray gradient
        bg = 0.15 + 0.1 * float(j) / float(height)
        img_r[pixel] = cr + (1.0 - alpha) * bg
        img_g[pixel] = cg + (1.0 - alpha) * bg
        img_b[pixel] = cb + (1.0 - alpha) * bg


@pgc.kernel
def compute_gyroid(scalar, xcoords, ycoords, zcoords, nx_p1, ny_p1):
    for i in range(scalar.shape[0]):
        ix = i % nx_p1
        iy = (i // nx_p1) % ny_p1
        iz = i // (nx_p1 * ny_p1)
        x = xcoords[ix]
        y = ycoords[iy]
        z = zcoords[iz]
        scalar[i] = sin(x) * cos(y) + sin(y) * cos(z) + sin(z) * cos(x)


# ================================================================
# BUILD GRID AND SCALAR FIELD
# ================================================================

N = _args.size
nx, ny, nz = N, N, N
n_points = (nx + 1) * (ny + 1) * (nz + 1)

print(f"Grid: {nx}x{ny}x{nz} = {nx*ny*nz:,} cells, {n_points:,} points")
print(f"Image: {WIDTH}x{HEIGHT}")
print(f"Backend: {_args.arch}")

# Cosine-spaced rectilinear coordinates (same as example 27)
t_param = np.linspace(0, np.pi, nx + 1, dtype=np.float32)
xc_np = np.float32(-np.pi) + np.float32(2.0 * np.pi) * (1.0 - np.cos(t_param)) / 2.0
t_param = np.linspace(0, np.pi, ny + 1, dtype=np.float32)
yc_np = np.float32(-np.pi) + np.float32(2.0 * np.pi) * (1.0 - np.cos(t_param)) / 2.0
t_param = np.linspace(0, np.pi, nz + 1, dtype=np.float32)
zc_np = np.float32(-np.pi) + np.float32(2.0 * np.pi) * (1.0 - np.cos(t_param)) / 2.0

xcoords = pgc.field(dtype=pgc.f32, shape=(nx + 1,))
ycoords = pgc.field(dtype=pgc.f32, shape=(ny + 1,))
zcoords = pgc.field(dtype=pgc.f32, shape=(nz + 1,))
xcoords.from_numpy(xc_np)
ycoords.from_numpy(yc_np)
zcoords.from_numpy(zc_np)

# Compute gyroid scalar field on GPU
print("Computing scalar field...")
scalar = pgc.field(dtype=pgc.f32, shape=(n_points,))
compute_gyroid(scalar, xcoords, ycoords, zcoords, nx + 1, ny + 1)

vmin = float(scalar.min())
vmax = float(scalar.max())
vrange = vmax - vmin
print(f"Scalar range: [{vmin:.3f}, {vmax:.3f}]")

# Wrap scalar field as a 3D texture for hardware-accelerated sampling
tex = pgc.texture3d(scalar, shape=(nx + 1, ny + 1, nz + 1))


# ================================================================
# TRANSFER FUNCTION
# ================================================================

tf_np = build_transfer_function(vmin, vmax)
tf_r = pgc.field(dtype=pgc.f32, shape=(TF_SIZE,))
tf_g = pgc.field(dtype=pgc.f32, shape=(TF_SIZE,))
tf_b = pgc.field(dtype=pgc.f32, shape=(TF_SIZE,))
tf_a = pgc.field(dtype=pgc.f32, shape=(TF_SIZE,))
tf_r.from_numpy(tf_np[:, 0].copy())
tf_g.from_numpy(tf_np[:, 1].copy())
tf_b.from_numpy(tf_np[:, 2].copy())
tf_a.from_numpy(tf_np[:, 3].copy())


# ================================================================
# CAMERA
# ================================================================

# Domain bounds
bmin = np.array([xc_np[0], yc_np[0], zc_np[0]], dtype=np.float32)
bmax = np.array([xc_np[-1], yc_np[-1], zc_np[-1]], dtype=np.float32)
center = (bmin + bmax) / 2.0
diag = float(np.linalg.norm(bmax - bmin))

# Spherical camera
theta, phi = 0.6, 0.35
radius = 1.1 * diag
cam_pos = center + radius * np.array([
    np.cos(phi) * np.cos(theta),
    np.cos(phi) * np.sin(theta),
    np.sin(phi),
], dtype=np.float32)

forward = center - cam_pos
forward = forward / np.linalg.norm(forward)
world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
right = np.cross(forward, world_up)
right = right / np.linalg.norm(right)
up = np.cross(right, forward)

fov_half = float(np.tan(30.0 * np.pi / 180.0))

# Step size: fraction of smallest cell diagonal
min_dx = min(np.diff(xc_np).min(), np.diff(yc_np).min(), np.diff(zc_np).min())
step_size = float(min_dx) * 0.5
# Don't let step count exceed MAX_STEPS
min_step = diag / MAX_STEPS
step_size = max(step_size, min_step)
opacity_scale = 8.0

print(f"Step size: {step_size:.5f}, opacity scale: {opacity_scale}")


# ================================================================
# RENDER + BENCHMARK
# ================================================================

n_pixels = WIDTH * HEIGHT
img_r = pgc.field(dtype=pgc.f32, shape=(n_pixels,))
img_g = pgc.field(dtype=pgc.f32, shape=(n_pixels,))
img_b = pgc.field(dtype=pgc.f32, shape=(n_pixels,))

# Build inverse lookup tables for O(1) cell location
INV_TABLE_SIZE = max(nx, ny, nz) * 2  # 2x oversampling for accuracy


def build_inv_table(coords_np, n_cells, table_size):
    """Build a uniform-binned inverse lookup: world coord -> cell index."""
    cmin = float(coords_np[0])
    cmax = float(coords_np[-1])
    stride = (cmax - cmin) / table_size
    table = np.zeros(table_size, dtype=np.int32)
    cell = 0
    for k in range(table_size):
        pos = cmin + (k + 0.5) * stride
        while cell < n_cells - 1 and coords_np[cell + 1] <= pos:
            cell += 1
        table[k] = cell
    return table, cmin, stride


inv_x_np, cmin_x, inv_sx = build_inv_table(xc_np, nx, INV_TABLE_SIZE)
inv_y_np, cmin_y, inv_sy = build_inv_table(yc_np, ny, INV_TABLE_SIZE)
inv_z_np, cmin_z, inv_sz = build_inv_table(zc_np, nz, INV_TABLE_SIZE)

inv_x = pgc.field(dtype=pgc.i32, shape=(INV_TABLE_SIZE,))
inv_y = pgc.field(dtype=pgc.i32, shape=(INV_TABLE_SIZE,))
inv_z = pgc.field(dtype=pgc.i32, shape=(INV_TABLE_SIZE,))
inv_x.from_numpy(inv_x_np)
inv_y.from_numpy(inv_y_np)
inv_z.from_numpy(inv_z_np)

# Scalar parameters are passed directly -- the backend automatically packs
# them into constant buffers to stay within GPU binding limits.
render_args = (img_r, img_g, img_b,
               tex, xcoords, ycoords, zcoords,
               inv_x, inv_y, inv_z,
               tf_r, tf_g, tf_b, tf_a,
               cam_pos[0], cam_pos[1], cam_pos[2],
               forward[0], forward[1], forward[2],
               right[0], right[1], right[2],
               up[0], up[1], up[2],
               bmin[0], bmin[1], bmin[2],
               bmax[0], bmax[1], bmax[2],
               fov_half, step_size, opacity_scale,
               vmin, vrange,
               cmin_x, cmin_y, cmin_z,
               inv_sx, inv_sy, inv_sz,
               nx + 1, ny + 1, (nx + 1) * (ny + 1),
               WIDTH, HEIGHT, INV_TABLE_SIZE,
               n_pixels)


def do_render():
    render(*render_args)


# Warmup (includes JIT on first call)
print(f"Warmup ({_args.warmup} runs)...")
for w in range(_args.warmup):
    t0 = time.perf_counter()
    do_render()
    t1 = time.perf_counter()
    if w == 0:
        print(f"  First render (incl. JIT): {t1 - t0:.3f}s")

# Benchmark
print(f"Benchmark ({_args.trials} trials)...")
times = []
for t_i in range(_args.trials):
    t0 = time.perf_counter()
    do_render()
    t1 = time.perf_counter()
    times.append(t1 - t0)
    print(f"  Trial {t_i+1}: {t1 - t0:.4f}s")

print()
print(f"  Min:  {min(times):.4f}s")
print(f"  Max:  {max(times):.4f}s")
print(f"  Mean: {sum(times)/len(times):.4f}s")
print(f"  Mrays/s: {n_pixels / min(times) / 1e6:.1f}")


# ================================================================
# SAVE IMAGE
# ================================================================

r_np = img_r.to_numpy().reshape(HEIGHT, WIDTH)
g_np = img_g.to_numpy().reshape(HEIGHT, WIDTH)
b_np = img_b.to_numpy().reshape(HEIGHT, WIDTH)
img = np.stack([r_np, g_np, b_np], axis=-1)
img = np.clip(img, 0.0, 1.0)
img = img[::-1]  # flip vertically (row 0 = bottom)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img)
    ax.set_axis_off()
    plt.tight_layout()
    fig.savefig(_args.save, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Saved: {_args.save}")
except ImportError:
    print("matplotlib not available -- skipping image save")
    # Fallback: save raw as .npy
    np.save(_args.save.replace('.png', '.npy'), img)
    print(f"Saved raw array: {_args.save.replace('.png', '.npy')}")
