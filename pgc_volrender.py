#!/usr/bin/env python3
"""
pgc_volrender.py - PGC volume rendering of AMR data.

Port of taichi_volrender.py to PGC (Portable GPU Compute).

Usage:
  python pgc_volrender.py [dataset_path] [--arch cpu|metal]
"""

import sys
import time
import numpy as np
import pgc

# ── Constants ────────────────────────────────────────────────────────────
WIDTH, HEIGHT = 800, 800
MAX_BLOCKS = 4096
MAX_CELLS = 50_000_000
LG = 256
TF_SIZE = 256
MAX_STEPS = 600


# ── GPU functions ────────────────────────────────────────────────────────

@pgc.func
def fetch_cell(b, ix, iy, iz):
    """Fetch cell value from block b at integer indices, clamped."""
    dims = block_cell_dims[b]
    dx = float(dims[0])
    dy = float(dims[1])
    dz = float(dims[2])
    ix_c = max(0.0, min(ix, dx - 1.0))
    iy_c = max(0.0, min(iy, dy - 1.0))
    iz_c = max(0.0, min(iz, dz - 1.0))
    offset = float(block_data_offsets[b])
    idx = offset + ix_c + iy_c * dx + iz_c * dx * dy
    return field_data[idx]


@pgc.func
def sample_at(pos_x, pos_y, pos_z):
    """Sample field value at world position using trilinear interpolation."""
    dm = domain_min_f[None]
    ls = lookup_spacing_f[None]
    gi = int((pos_x - dm[0]) / ls[0])
    gj = int((pos_y - dm[1]) / ls[1])
    gk = int((pos_z - dm[2]) / ls[2])
    gi = max(0.0, min(float(gi), float(LG_CONST[0]) - 1.0))
    gj = max(0.0, min(float(gj), float(LG_CONST[0]) - 1.0))
    gk = max(0.0, min(float(gk), float(LG_CONST[0]) - 1.0))

    # Lookup grid is flat 1D: index = (gi * LG + gj) * LG + gk
    lg = LG_CONST[0]
    b_i = lookup_grid[(gi * lg + gj) * lg + gk]
    b = float(b_i)
    result = 0.0
    if b_i >= 0:
        orig = block_origins[b]
        spac = block_spacings[b]

        lx = (pos_x - orig[0]) / spac[0] - 0.5
        ly = (pos_y - orig[1]) / spac[1] - 0.5
        lz = (pos_z - orig[2]) / spac[2] - 0.5

        ix = float(int(floor(lx)))
        iy = float(int(floor(ly)))
        iz = float(int(floor(lz)))
        fx = lx - ix
        fy = ly - iy
        fz = lz - iz

        c000 = fetch_cell(b, ix,       iy,       iz)
        c100 = fetch_cell(b, ix + 1.0, iy,       iz)
        c010 = fetch_cell(b, ix,       iy + 1.0, iz)
        c110 = fetch_cell(b, ix + 1.0, iy + 1.0, iz)
        c001 = fetch_cell(b, ix,       iy,       iz + 1.0)
        c101 = fetch_cell(b, ix + 1.0, iy,       iz + 1.0)
        c011 = fetch_cell(b, ix,       iy + 1.0, iz + 1.0)
        c111 = fetch_cell(b, ix + 1.0, iy + 1.0, iz + 1.0)

        c00 = c000 * (1.0 - fx) + c100 * fx
        c10 = c010 * (1.0 - fx) + c110 * fx
        c01 = c001 * (1.0 - fx) + c101 * fx
        c11 = c011 * (1.0 - fx) + c111 * fx

        c0 = c00 * (1.0 - fy) + c10 * fy
        c1 = c01 * (1.0 - fy) + c11 * fy

        result = c0 * (1.0 - fz) + c1 * fz
    return result


@pgc.func
def apply_tf(value):
    """Map field value to RGBA via log-scale transfer function."""
    log_val = log(max(value, 1e-30))
    t = (log_val - tf_log_min_f[0]) / tf_log_range_f[0]
    t = max(0.0, min(1.0, t))
    idx = max(0.0, min(t * 255.0, 255.0))
    return tf_table[idx]


@pgc.func
def ray_box_intersect(ro_x, ro_y, ro_z, rd_x, rd_y, rd_z,
                      bmin_x, bmin_y, bmin_z, bmax_x, bmax_y, bmax_z):
    """Ray-AABB intersection, returns (t_near, t_far).

    Manually unrolled from ti.static(range(3)).
    """
    t_near = -3.4e38
    t_far = 3.4e38

    # X axis
    if abs(rd_x) < 1e-10:
        if ro_x < bmin_x or ro_x > bmax_x:
            t_near = 3.4e38
            t_far = -3.4e38
    else:
        t1 = (bmin_x - ro_x) / rd_x
        t2 = (bmax_x - ro_x) / rd_x
        if t1 > t2:
            t1, t2 = t2, t1
        t_near = max(t_near, t1)
        t_far = min(t_far, t2)

    # Y axis
    if abs(rd_y) < 1e-10:
        if ro_y < bmin_y or ro_y > bmax_y:
            t_near = 3.4e38
            t_far = -3.4e38
    else:
        t1 = (bmin_y - ro_y) / rd_y
        t2 = (bmax_y - ro_y) / rd_y
        if t1 > t2:
            t1, t2 = t2, t1
        t_near = max(t_near, t1)
        t_far = min(t_far, t2)

    # Z axis
    if abs(rd_z) < 1e-10:
        if ro_z < bmin_z or ro_z > bmax_z:
            t_near = 3.4e38
            t_far = -3.4e38
    else:
        t1 = (bmin_z - ro_z) / rd_z
        t2 = (bmax_z - ro_z) / rd_z
        if t1 > t2:
            t1, t2 = t2, t1
        t_near = max(t_near, t1)
        t_far = min(t_far, t2)

    return t_near, t_far


@pgc.kernel
def render(pixels, pixel_alpha,
           cam_pos, cam_focus, cam_up,
           domain_min_f, domain_max_f,
           step_size_f, opacity_scale_f, fov_half_tan_f,
           tf_log_min_f, tf_log_range_f, tf_table,
           block_origins, block_spacings, block_cell_dims,
           block_data_offsets, field_data,
           lookup_grid, lookup_spacing_f, LG_CONST):
    """Cast rays through AMR volume, accumulate color via transfer function."""
    pos = cam_pos[None]
    focus = cam_focus[None]
    up_vec = cam_up[None]
    dm_min = domain_min_f[None]
    dm_max = domain_max_f[None]
    dt = step_size_f[0]
    o_scale = opacity_scale_f[0]

    forward = (focus - pos).normalized()
    right = forward.cross(up_vec).normalized()
    actual_up = right.cross(forward)

    fov_half = fov_half_tan_f[0]
    aspect = 800.0 / 800.0

    for i, j in pgc.ndrange(800, 800):
        u = (2.0 * (float(i) + 0.5) / 800.0 - 1.0) * fov_half * aspect
        v = (2.0 * (float(j) + 0.5) / 800.0 - 1.0) * fov_half

        ray_d = (forward + u * right + v * actual_up).normalized()

        t_near, t_far = ray_box_intersect(
            pos[0], pos[1], pos[2],
            ray_d[0], ray_d[1], ray_d[2],
            dm_min[0], dm_min[1], dm_min[2],
            dm_max[0], dm_max[1], dm_max[2])
        t_near = max(t_near, 0.0)

        color = pgc.Vector([0.0, 0.0, 0.0])
        alpha = 0.0

        if t_near < t_far:
            t = t_near
            for _ in range(600):
                if t >= t_far or alpha > 0.99:
                    break

                sample_x = pos[0] + t * ray_d[0]
                sample_y = pos[1] + t * ray_d[1]
                sample_z = pos[2] + t * ray_d[2]
                value = sample_at(sample_x, sample_y, sample_z)

                if value > 0.0:
                    rgba = apply_tf(value)
                    sa = 1.0 - exp(0.0 - rgba[3] * o_scale * dt)
                    sa = min(sa, 1.0)

                    color += (1.0 - alpha) * sa * pgc.Vector([rgba[0], rgba[1], rgba[2]])
                    alpha += (1.0 - alpha) * sa

                t += dt

        pixels[i, j] = color
        pixel_alpha[i, j] = alpha


# ── Python helpers ───────────────────────────────────────────────────────

def build_transfer_function(vmin, vmax, n_layers=10, colormap_name='inferno'):
    """Build Gaussian transfer function matching yt's add_layers() style."""
    import matplotlib

    log_min = float(np.log(max(vmin, 1e-30)))
    log_max = float(np.log(max(vmax, 1e-30)))
    log_range = log_max - log_min

    mi = log_min + log_range / (10.0 * n_layers)
    ma = log_max - log_range / (10.0 * n_layers)
    centers = np.linspace(mi, ma, n_layers)

    d = (ma - mi) / max(n_layers - 1, 1)
    w = d * d * 0.3

    alphas = np.logspace(-2, 0, n_layers)

    cmap = matplotlib.colormaps[colormap_name]
    layer_colors = np.zeros((n_layers, 3))
    for k in range(n_layers):
        rel = np.clip((centers[k] - log_min) / log_range, 0, 1)
        r, g, b, _ = cmap(rel)
        layer_colors[k] = [r, g, b]

    tf = np.zeros((TF_SIZE, 4), dtype=np.float32)
    x = np.linspace(log_min, log_max, TF_SIZE)

    for k in range(n_layers):
        gauss = alphas[k] * np.exp(-(x - centers[k])**2 / w)
        tf[:, 0] += gauss * layer_colors[k, 0]
        tf[:, 1] += gauss * layer_colors[k, 1]
        tf[:, 2] += gauss * layer_colors[k, 2]
        tf[:, 3] += gauss

    mask = tf[:, 3] > 0
    tf[mask, 0] /= tf[mask, 3]
    tf[mask, 1] /= tf[mask, 3]
    tf[mask, 2] /= tf[mask, 3]
    tf[:, :3] = np.clip(tf[:, :3], 0, 1)

    return tf, log_min, log_range


def load_amr_data(dataset_path):
    """Load AMR data via yt, return blocks + domain info."""
    import yt

    ds = yt.load(dataset_path)
    idx = ds.index

    field = None
    for ftype, fname in ds.field_list:
        if fname.lower() in ('density', 'dens'):
            field = (ftype, fname)
            break
    if field is None:
        raise ValueError(f"No density field in {[f[1] for f in ds.field_list]}")

    print(f"Dataset: {ds}")
    print(f"Field: {field}")
    print(f"Domain: {ds.domain_left_edge.d} -> {ds.domain_right_edge.d}")
    print(f"Levels: {ds.max_level + 1}, Grids: {len(idx.grids)}")

    origin = ds.domain_left_edge.d.astype(np.float32)
    domain_width = (ds.domain_right_edge - ds.domain_left_edge).d.astype(np.float32)
    h0 = domain_width / ds.domain_dimensions
    refine_by = int(ds.refine_by)
    refines = ds.domain_dimensions > 1

    grids_by_level = [[] for _ in range(ds.max_level + 1)]
    for g in idx.grids:
        grids_by_level[g.Level].append(g)

    blocks = []
    total_cells = 0

    for lvl in range(ds.max_level + 1):
        h = h0.copy()
        h[refines] = h0[refines] / (refine_by ** lvl)

        for g in grids_by_level[lvl]:
            go = g.LeftEdge.d.astype(np.float32)
            dims = g.ActiveDimensions.astype(int)
            data = g[field].d.astype(np.float32)
            flat = data.flatten(order='F')

            blocks.append({
                'origin': go,
                'spacing': h.astype(np.float32),
                'cell_dims': dims,
                'level': lvl,
                'data': flat,
                'data_offset': total_cells,
            })
            total_cells += flat.size

    print(f"Total blocks: {len(blocks)}, Total cells: {total_cells:,}")
    return blocks, origin, origin + domain_width, ds


def spherical_camera(center, theta, phi, radius):
    """Compute camera position from spherical coordinates."""
    x = radius * np.cos(phi) * np.cos(theta)
    y = radius * np.cos(phi) * np.sin(theta)
    z = radius * np.sin(phi)
    return center + np.array([x, y, z], dtype=np.float32)


def save_image(pixels_field, alpha_field, path):
    """Save rendered pixels to PNG."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    data = pixels_field.to_numpy()
    # Reshape from flat interleaved (W*H*3,) to (W, H, 3)
    img = data.reshape(WIDTH, HEIGHT, 3)
    img = np.clip(img, 0, 1)
    # Layout: [i=x, j=y], transpose for imshow
    img = np.transpose(img, (1, 0, 2))[::-1]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img)
    ax.set_axis_off()
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Saved: {path}")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    # Parse arguments
    dataset = '/Users/berk.geveci/Work/yt/data/enzo_tiny_cosmology/DD0046/DD0046'
    save_path = '/Users/berk.geveci/Work/yt/data/pgc_volrender.png'
    arch = pgc.cpu

    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    if args:
        dataset = args[0]
    for a in sys.argv[1:]:
        if a.startswith('--save='):
            save_path = a.split('=', 1)[1]
        if a == '--arch=metal':
            arch = pgc.metal
        if a == '--arch=cpu':
            arch = pgc.cpu
        if a == '--arch=cuda':
            arch = pgc.cuda
        if a == '--arch=hip':
            arch = pgc.hip

    pgc.init(arch=arch)
    print(f"Backend: {arch}")

    # Allocate fields
    pixels = pgc.Vector.field(3, dtype=pgc.f32, shape=(WIDTH, HEIGHT))
    pixel_alpha = pgc.field(dtype=pgc.f32, shape=(WIDTH, HEIGHT))

    block_origins = pgc.Vector.field(3, dtype=pgc.f32, shape=(MAX_BLOCKS,))
    block_spacings = pgc.Vector.field(3, dtype=pgc.f32, shape=(MAX_BLOCKS,))
    block_cell_dims = pgc.Vector.field(3, dtype=pgc.i32, shape=(MAX_BLOCKS,))
    block_data_offsets = pgc.field(dtype=pgc.i32, shape=(MAX_BLOCKS,))

    field_data = pgc.field(dtype=pgc.f32, shape=(MAX_CELLS,))

    lookup_grid = pgc.field(dtype=pgc.i32, shape=(LG * LG * LG,))

    tf_table = pgc.Vector.field(4, dtype=pgc.f32, shape=(TF_SIZE,))

    cam_pos = pgc.Vector.field(3, dtype=pgc.f32, shape=())
    cam_focus = pgc.Vector.field(3, dtype=pgc.f32, shape=())
    cam_up = pgc.Vector.field(3, dtype=pgc.f32, shape=())
    domain_min_f = pgc.Vector.field(3, dtype=pgc.f32, shape=())
    domain_max_f = pgc.Vector.field(3, dtype=pgc.f32, shape=())
    lookup_spacing_f = pgc.Vector.field(3, dtype=pgc.f32, shape=())

    step_size_f = pgc.field(dtype=pgc.f32, shape=(1,))
    tf_log_min_f = pgc.field(dtype=pgc.f32, shape=(1,))
    tf_log_range_f = pgc.field(dtype=pgc.f32, shape=(1,))
    opacity_scale_f = pgc.field(dtype=pgc.f32, shape=(1,))
    fov_half_tan_f = pgc.field(dtype=pgc.f32, shape=(1,))

    # LG constant field (to pass LG into kernel)
    LG_CONST = pgc.field(dtype=pgc.f32, shape=(1,))
    LG_CONST.from_numpy(np.array([float(LG)], dtype=np.float32))

    # Load data
    blocks, dom_min, dom_max, ds = load_amr_data(dataset)

    # Upload block metadata
    origins_np = np.zeros((MAX_BLOCKS, 3), dtype=np.float32)
    spacings_np = np.zeros((MAX_BLOCKS, 3), dtype=np.float32)
    dims_np = np.zeros((MAX_BLOCKS, 3), dtype=np.int32)
    offsets_np = np.zeros(MAX_BLOCKS, dtype=np.int32)
    all_data_list = []

    for i, b in enumerate(blocks):
        origins_np[i] = b['origin']
        spacings_np[i] = b['spacing']
        dims_np[i] = b['cell_dims'].astype(np.int32)
        offsets_np[i] = int(b['data_offset'])
        all_data_list.append(b['data'])

    block_origins.from_numpy(origins_np.flatten())
    block_spacings.from_numpy(spacings_np.flatten())
    block_cell_dims.from_numpy(dims_np.flatten())
    block_data_offsets.from_numpy(offsets_np)

    all_data_flat = np.concatenate(all_data_list)
    padded = np.zeros(MAX_CELLS, dtype=np.float32)
    padded[:all_data_flat.size] = all_data_flat
    field_data.from_numpy(padded)

    # Build spatial lookup grid
    dom_width = dom_max - dom_min
    lg_spacing = dom_width / LG

    lookup = np.full(LG * LG * LG, -1, dtype=np.int32)  # i32, -1 for empty
    sorted_blocks = sorted(enumerate(blocks), key=lambda x: x[1]['level'])

    for bi, b in sorted_blocks:
        lo = np.floor((b['origin'] - dom_min) / lg_spacing).astype(int)
        hi_corner = b['origin'] + b['cell_dims'] * b['spacing']
        hi = np.ceil((hi_corner - dom_min) / lg_spacing).astype(int)
        lo = np.clip(lo, 0, LG)
        hi = np.clip(hi, 0, LG)
        # Fill lookup grid (flattened)
        for ii in range(lo[0], hi[0]):
            for jj in range(lo[1], hi[1]):
                for kk in range(lo[2], hi[2]):
                    lookup[(ii * LG + jj) * LG + kk] = bi

    lookup_grid.from_numpy(lookup)

    # Upload domain info
    domain_min_f.from_numpy(dom_min.astype(np.float32))
    domain_max_f.from_numpy(dom_max.astype(np.float32))
    lookup_spacing_f.from_numpy(lg_spacing.astype(np.float32))

    # Camera
    center = (dom_min + dom_max) / 2.0
    diag = float(np.linalg.norm(dom_max - dom_min))
    theta, phi, radius = 0.64, 0.38, 1.08 * diag
    cam_position = spherical_camera(center, theta, phi, radius)

    cam_pos.from_numpy(cam_position.astype(np.float32))
    cam_focus.from_numpy(center.astype(np.float32))
    cam_up.from_numpy(np.array([0.0, 0.0, 1.0], dtype=np.float32))

    # Transfer function
    tf_data, log_min, log_range = build_transfer_function(
        all_data_flat[all_data_flat > 0].min(), all_data_flat.max())
    tf_table.from_numpy(tf_data.flatten())
    tf_log_min_f.from_numpy(np.array([log_min], dtype=np.float32))
    tf_log_range_f.from_numpy(np.array([log_range], dtype=np.float32))

    # Rendering parameters
    opacity_scale_f.from_numpy(np.array([200.0], dtype=np.float32))

    finest_h = (dom_max - dom_min) / ds.domain_dimensions
    refines = ds.domain_dimensions > 1
    finest_h[refines] /= (ds.refine_by ** ds.max_level)
    finest_step = float(finest_h[refines].min()) * 0.5
    min_step = diag / MAX_STEPS * 1.5
    step_size_f.from_numpy(np.array([max(finest_step, min_step)], dtype=np.float32))

    fov_half_tan_f.from_numpy(np.array([np.tan(30.0 * np.pi / 180.0)], dtype=np.float32))

    print(f"Uploaded {all_data_flat.size:,} cells, lookup grid {LG}^3")
    print("Rendering...")

    def do_render():
        render(pixels, pixel_alpha,
               cam_pos, cam_focus, cam_up,
               domain_min_f, domain_max_f,
               step_size_f, opacity_scale_f, fov_half_tan_f,
               tf_log_min_f, tf_log_range_f, tf_table,
               block_origins, block_spacings, block_cell_dims,
               block_data_offsets, field_data,
               lookup_grid, lookup_spacing_f, LG_CONST)

    t0 = time.time()
    do_render()
    t1 = time.time()
    print(f"Render time (incl. JIT): {t1 - t0:.3f}s")

    # Benchmark mode: re-render N times after warmup
    bench_n = 0
    for a in sys.argv[1:]:
        if a.startswith('--bench='):
            bench_n = int(a.split('=', 1)[1])
    if bench_n > 0:
        times = []
        for i in range(bench_n):
            t0 = time.time()
            do_render()
            t = time.time() - t0
            times.append(t)
        avg = sum(times) / len(times)
        print(f"Bench ({bench_n} runs): min={min(times):.4f}s avg={avg:.4f}s")

    save_image(pixels, pixel_alpha, save_path)


if __name__ == '__main__':
    main()
