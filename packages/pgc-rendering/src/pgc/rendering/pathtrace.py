"""Path tracing kernels and render entry point.

Provides a single-kernel path tracer with BVH acceleration.  Each sample
launches one kernel where every pixel traces a full path (primary ray +
bounces).  Multiple samples are accumulated and resolved to the canvas.
"""

import numpy as np
import pgc

from pgc.rendering.bvh import BVH, STACK_DEPTH


# ================================================================
# RENDER CONFIGURATION (template)
# ================================================================

@pgc.data_oriented
class _RenderConfig:
    """Compile-time rendering parameters."""

    def __init__(self, bg_color, max_bounces,
                 vol_bounds=None, vol_scalar_range=None,
                 vol_step=0.0, vol_opacity=0.0,
                 vol_tf_size=0, vol_max_steps=0):
        self.bg_r = float(bg_color[0])
        self.bg_g = float(bg_color[1])
        self.bg_b = float(bg_color[2])
        self.max_bounces = int(max_bounces)
        # Volume params (zeros when no volume)
        if vol_bounds is not None:
            self.vol_bmin_x = float(vol_bounds[0][0])
            self.vol_bmin_y = float(vol_bounds[0][1])
            self.vol_bmin_z = float(vol_bounds[0][2])
            self.vol_bmax_x = float(vol_bounds[1][0])
            self.vol_bmax_y = float(vol_bounds[1][1])
            self.vol_bmax_z = float(vol_bounds[1][2])
            self.vol_ext_x = self.vol_bmax_x - self.vol_bmin_x
            self.vol_ext_y = self.vol_bmax_y - self.vol_bmin_y
            self.vol_ext_z = self.vol_bmax_z - self.vol_bmin_z
        else:
            self.vol_bmin_x = 0.0
            self.vol_bmin_y = 0.0
            self.vol_bmin_z = 0.0
            self.vol_bmax_x = 0.0
            self.vol_bmax_y = 0.0
            self.vol_bmax_z = 0.0
            self.vol_ext_x = 0.0
            self.vol_ext_y = 0.0
            self.vol_ext_z = 0.0
        self.vol_vmin = float(vol_scalar_range[0]) if vol_scalar_range else 0.0
        self.vol_vrange = float(vol_scalar_range[1] - vol_scalar_range[0]) if vol_scalar_range else 1.0
        self.vol_step = float(vol_step)
        self.vol_opacity = float(vol_opacity)
        self.vol_tf_size = int(vol_tf_size)
        self.vol_max_steps = int(vol_max_steps)
        self.vol_nx = 0
        self.vol_ny = 0
        self.vol_nz = 0


# ================================================================
# DEVICE HELPERS
# ================================================================

@pgc.func
def _ray_tri(ox, oy, oz, dx, dy, dz,
             v0x, v0y, v0z, v1x, v1y, v1z, v2x, v2y, v2z):
    """Moller-Trumbore ray-triangle intersection.

    Returns (t, u, v).  t < 0 means no hit.
    """
    e1x = v1x - v0x
    e1y = v1y - v0y
    e1z = v1z - v0z
    e2x = v2x - v0x
    e2y = v2y - v0y
    e2z = v2z - v0z

    hx = dy * e2z - dz * e2y
    hy = dz * e2x - dx * e2z
    hz = dx * e2y - dy * e2x

    det = e1x * hx + e1y * hy + e1z * hz

    t_out = -1.0
    u_out = 0.0
    v_out = 0.0

    if det < -1.0e-10 or det > 1.0e-10:
        inv_det = 1.0 / det
        sx = ox - v0x
        sy = oy - v0y
        sz = oz - v0z
        u = inv_det * (sx * hx + sy * hy + sz * hz)
        if u >= 0.0 and u <= 1.0:
            qx = sy * e1z - sz * e1y
            qy = sz * e1x - sx * e1z
            qz = sx * e1y - sy * e1x
            v = inv_det * (dx * qx + dy * qy + dz * qz)
            if v >= 0.0 and u + v <= 1.0:
                t = inv_det * (e2x * qx + e2y * qy + e2z * qz)
                if t > 0.001:
                    t_out = t
                    u_out = u
                    v_out = v

    return t_out, u_out, v_out


@pgc.func
def _ray_aabb(ox, oy, oz, inv_dx, inv_dy, inv_dz,
              bxmin, bymin, bzmin, bxmax, bymax, bzmax):
    """Slab-method ray-AABB intersection.

    Takes inverse ray direction for efficiency.
    Returns (t_near, t_far).  Hit if t_near <= t_far and t_far >= 0.
    """
    t1x = (bxmin - ox) * inv_dx
    t2x = (bxmax - ox) * inv_dx
    tminx = min(t1x, t2x)
    tmaxx = max(t1x, t2x)

    t1y = (bymin - oy) * inv_dy
    t2y = (bymax - oy) * inv_dy
    tminy = min(t1y, t2y)
    tmaxy = max(t1y, t2y)

    t1z = (bzmin - oz) * inv_dz
    t2z = (bzmax - oz) * inv_dz
    tminz = min(t1z, t2z)
    tmaxz = max(t1z, t2z)

    t_near = max(tminx, max(tminy, tminz))
    t_far = min(tmaxx, min(tmaxy, tmaxz))

    return t_near, t_far


@pgc.func
def _hash(x):
    """Integer hash to decorrelate pixel seeds (Wang hash variant)."""
    x = (x ^ 61) ^ (x >> 16)
    x = x + (x << 3)
    x = x ^ (x >> 4)
    x = x * 668265261
    x = x ^ (x >> 15)
    # Map to positive range
    if x < 0:
        x = -x
    return x


@pgc.func
def _vol_tf_lookup(vol_tf, val, vmin, vrange, tf_size):
    """Look up RGBA from volume transfer function."""
    t = (val - vmin) / vrange
    if t < 0.0:
        t = 0.0
    if t > 1.0:
        t = 1.0
    idx = int(t * float(tf_size - 1))
    if idx < 0:
        idx = 0
    if idx >= tf_size:
        idx = tf_size - 1
    base = idx * 4
    return vol_tf[base], vol_tf[base + 1], vol_tf[base + 2], vol_tf[base + 3]


@pgc.func
def _vol_sample(vol_data, u, v, w, nx, ny, nz):
    """Trilinear interpolation into a flat 3D scalar field.

    u, v, w in [0, 1].  nx, ny, nz are the grid dimensions.
    Data layout: x varies fastest (index = ix + iy*nx + iz*nx*ny).
    """
    # Map [0,1] to grid coordinates
    fx = u * float(nx - 1)
    fy = v * float(ny - 1)
    fz = w * float(nz - 1)
    # Clamp
    if fx < 0.0:
        fx = 0.0
    if fx > float(nx - 2):
        fx = float(nx - 2)
    if fy < 0.0:
        fy = 0.0
    if fy > float(ny - 2):
        fy = float(ny - 2)
    if fz < 0.0:
        fz = 0.0
    if fz > float(nz - 2):
        fz = float(nz - 2)

    ix = int(fx)
    iy = int(fy)
    iz = int(fz)
    dx = fx - float(ix)
    dy = fy - float(iy)
    dz = fz - float(iz)

    # 8 corners
    stride_y = nx
    stride_z = nx * ny
    i000 = ix + iy * stride_y + iz * stride_z
    i100 = i000 + 1
    i010 = i000 + stride_y
    i110 = i000 + stride_y + 1
    i001 = i000 + stride_z
    i101 = i001 + 1
    i011 = i001 + stride_y
    i111 = i001 + stride_y + 1

    # Trilinear interpolation
    c00 = vol_data[i000] * (1.0 - dx) + vol_data[i100] * dx
    c10 = vol_data[i010] * (1.0 - dx) + vol_data[i110] * dx
    c01 = vol_data[i001] * (1.0 - dx) + vol_data[i101] * dx
    c11 = vol_data[i011] * (1.0 - dx) + vol_data[i111] * dx
    c0 = c00 * (1.0 - dy) + c10 * dy
    c1 = c01 * (1.0 - dy) + c11 * dy
    return c0 * (1.0 - dz) + c1 * dz


@pgc.func
def _halton2(index):
    """Halton sequence base 2."""
    result = 0.0
    f = 0.5
    i = index + 1
    while i > 0:
        r = i - (i // 2) * 2
        if r == 1:
            result = result + f
        i = i // 2
        f = f * 0.5
    return result


@pgc.func
def _halton3(index):
    """Halton sequence base 3."""
    result = 0.0
    f = 1.0 / 3.0
    i = index + 1
    while i > 0:
        result = result + float(i - (i // 3) * 3) * f
        i = i // 3
        f = f / 3.0
    return result


# ================================================================
# PATH TRACE KERNEL
# ================================================================

@pgc.kernel
def _pathtrace(fb_r, fb_g, fb_b,
               points, conn, tri_colors, point_colors, normals,
               node_aabb, node_children, tri_ids,
               light_data,
               vol_tf, vol_data,
               mat_ids, mat_table,
               stack,
               camera: pgc.template(),
               config: pgc.template(),
               width, height, n_inner, n_tris, n_samples,
               has_normals, has_point_colors, n_lights,
               has_volume, n_pixels):
    """Trace all samples for each pixel and accumulate into fb_r/g/b."""

    for pid in range(n_pixels):
        px = pid % width
        py = pid // width
        stack_base = pid * 24
        total_r = 0.0
        total_g = 0.0
        total_b = 0.0

        for samp in range(n_samples):
         # ---- camera ray with per-pixel sub-pixel jitter ----
         seed = _hash(pid) + samp
         jx = _halton2(seed) - 0.5
         jy = _halton3(seed) - 0.5
         ppx = float(px) + 0.5 + jx
         ppy = float(py) + 0.5 + jy
         rdx = camera.corner_x + camera.dx_x * ppx + camera.dy_x * ppy
         rdy = camera.corner_y + camera.dx_y * ppx + camera.dy_y * ppy
         rdz = camera.corner_z + camera.dx_z * ppx + camera.dy_z * ppy
         rd_len = sqrt(rdx * rdx + rdy * rdy + rdz * rdz)
         rdx = rdx / rd_len
         rdy = rdy / rd_len
         rdz = rdz / rd_len
         rox = camera.pos_x + camera.odx_x * ppx + camera.ody_x * ppy
         roy = camera.pos_y + camera.odx_y * ppx + camera.ody_y * ppy
         roz = camera.pos_z + camera.odx_z * ppx + camera.ody_z * ppy

         # ---- path trace state ----
         acc_r = 0.0
         acc_g = 0.0
         acc_b = 0.0
         thr_r = 1.0
         thr_g = 1.0
         thr_b = 1.0
         alive = 1

         for bounce in range(config.max_bounces + 1):
          if alive == 1:

            # ---- inverse direction for AABB tests ----
            inv_dx = 1.0 / (rdx + 1.0e-20)
            inv_dy = 1.0 / (rdy + 1.0e-20)
            inv_dz = 1.0 / (rdz + 1.0e-20)

            # ---- BVH traversal: closest hit ----
            hit_t = 1.0e30
            hit_tri = -1
            hit_u = 0.0
            hit_v = 0.0

            stack[stack_base] = 0
            sp = 1

            while sp > 0:
                sp = sp - 1
                node = stack[stack_base + sp]

                # Test node AABB
                ab = node * 6
                tn, tf = _ray_aabb(rox, roy, roz, inv_dx, inv_dy, inv_dz,
                                   node_aabb[ab], node_aabb[ab + 1],
                                   node_aabb[ab + 2], node_aabb[ab + 3],
                                   node_aabb[ab + 4], node_aabb[ab + 5])
                if tn <= tf and tf >= 0.0 and tn < hit_t:
                    if node >= n_inner:
                        # Leaf — intersect triangle
                        tri = tri_ids[node - n_inner]
                        i0 = conn[tri * 3]
                        i1 = conn[tri * 3 + 1]
                        i2 = conn[tri * 3 + 2]
                        t, u, v = _ray_tri(
                            rox, roy, roz, rdx, rdy, rdz,
                            points[i0 * 3], points[i0 * 3 + 1], points[i0 * 3 + 2],
                            points[i1 * 3], points[i1 * 3 + 1], points[i1 * 3 + 2],
                            points[i2 * 3], points[i2 * 3 + 1], points[i2 * 3 + 2])
                        if t > 0.0 and t < hit_t:
                            hit_t = t
                            hit_tri = tri
                            hit_u = u
                            hit_v = v
                    else:
                        # Inner — push children
                        left = node_children[node * 2]
                        right = node_children[node * 2 + 1]
                        if sp < 23:
                            stack[stack_base + sp] = left
                            sp = sp + 1
                        if sp < 23:
                            stack[stack_base + sp] = right
                            sp = sp + 1

            # ---- volume march (primary ray only) ----
            if has_volume == 1 and bounce == 0:
                # Ray-AABB for volume bounds
                vn = -3.4e38
                vf = 3.4e38
                if abs(rdx) < 1.0e-10:
                    if rox < config.vol_bmin_x or rox > config.vol_bmax_x:
                        vn = 3.4e38
                        vf = -3.4e38
                else:
                    vt1 = (config.vol_bmin_x - rox) * inv_dx
                    vt2 = (config.vol_bmax_x - rox) * inv_dx
                    if vt1 > vt2:
                        vt1, vt2 = vt2, vt1
                    vn = max(vn, vt1)
                    vf = min(vf, vt2)
                if abs(rdy) < 1.0e-10:
                    if roy < config.vol_bmin_y or roy > config.vol_bmax_y:
                        vn = 3.4e38
                        vf = -3.4e38
                else:
                    vt1 = (config.vol_bmin_y - roy) * inv_dy
                    vt2 = (config.vol_bmax_y - roy) * inv_dy
                    if vt1 > vt2:
                        vt1, vt2 = vt2, vt1
                    vn = max(vn, vt1)
                    vf = min(vf, vt2)
                if abs(rdz) < 1.0e-10:
                    if roz < config.vol_bmin_z or roz > config.vol_bmax_z:
                        vn = 3.4e38
                        vf = -3.4e38
                else:
                    vt1 = (config.vol_bmin_z - roz) * inv_dz
                    vt2 = (config.vol_bmax_z - roz) * inv_dz
                    if vt1 > vt2:
                        vt1, vt2 = vt2, vt1
                    vn = max(vn, vt1)
                    vf = min(vf, vt2)

                if vn < 0.0:
                    vn = 0.0
                # Stop at surface hit
                if hit_tri >= 0 and hit_t < vf:
                    vf = hit_t

                if vn < vf:
                    vt = vn
                    for _vs in range(config.vol_max_steps):
                        if vt >= vf:
                            break
                        if thr_r < 0.01 and thr_g < 0.01 and thr_b < 0.01:
                            break
                        vsx = rox + vt * rdx
                        vsy = roy + vt * rdy
                        vsz = roz + vt * rdz
                        vtu = (vsx - config.vol_bmin_x) / (config.vol_ext_x + 1.0e-20)
                        vtv = (vsy - config.vol_bmin_y) / (config.vol_ext_y + 1.0e-20)
                        vtw = (vsz - config.vol_bmin_z) / (config.vol_ext_z + 1.0e-20)
                        vval = _vol_sample(vol_data, vtu, vtv, vtw,
                                           config.vol_nx, config.vol_ny,
                                           config.vol_nz)
                        vsr, vsg, vsb, vsa = _vol_tf_lookup(
                            vol_tf, vval, config.vol_vmin,
                            config.vol_vrange, config.vol_tf_size)
                        vsa = 1.0 - exp(0.0 - vsa * config.vol_opacity * config.vol_step)
                        if vsa > 1.0:
                            vsa = 1.0
                        # Add volume contribution and reduce throughput
                        acc_r = acc_r + thr_r * vsa * vsr
                        acc_g = acc_g + thr_g * vsa * vsg
                        acc_b = acc_b + thr_b * vsa * vsb
                        thr_r = thr_r * (1.0 - vsa)
                        thr_g = thr_g * (1.0 - vsa)
                        thr_b = thr_b * (1.0 - vsa)
                        vt = vt + config.vol_step

            # ---- miss: background ----
            if hit_tri < 0:
                acc_r = acc_r + thr_r * config.bg_r
                acc_g = acc_g + thr_g * config.bg_g
                acc_b = acc_b + thr_b * config.bg_b
                alive = 0

            # ---- hit: shade + bounce ----
            if hit_tri >= 0:
                hx = rox + hit_t * rdx
                hy = roy + hit_t * rdy
                hz = roz + hit_t * rdz

                i0 = conn[hit_tri * 3]
                i1 = conn[hit_tri * 3 + 1]
                i2 = conn[hit_tri * 3 + 2]

                # Compute face normal (always needed as fallback)
                e1x = points[i1 * 3]     - points[i0 * 3]
                e1y = points[i1 * 3 + 1] - points[i0 * 3 + 1]
                e1z = points[i1 * 3 + 2] - points[i0 * 3 + 2]
                e2x = points[i2 * 3]     - points[i0 * 3]
                e2y = points[i2 * 3 + 1] - points[i0 * 3 + 1]
                e2z = points[i2 * 3 + 2] - points[i0 * 3 + 2]
                nx = e1y * e2z - e1z * e2y
                ny = e1z * e2x - e1x * e2z
                nz = e1x * e2y - e1y * e2x
                n_len = sqrt(nx * nx + ny * ny + nz * nz) + 1.0e-20
                nx = nx / n_len
                ny = ny / n_len
                nz = nz / n_len

                if has_normals == 1:
                    # Try interpolating vertex normals
                    w0 = 1.0 - hit_u - hit_v
                    snx = w0 * normals[i0 * 3]     + hit_u * normals[i1 * 3]     + hit_v * normals[i2 * 3]
                    sny = w0 * normals[i0 * 3 + 1] + hit_u * normals[i1 * 3 + 1] + hit_v * normals[i2 * 3 + 1]
                    snz = w0 * normals[i0 * 3 + 2] + hit_u * normals[i1 * 3 + 2] + hit_v * normals[i2 * 3 + 2]
                    sn_len = sqrt(snx * snx + sny * sny + snz * snz)
                    if sn_len > 0.01:
                        nx = snx / sn_len
                        ny = sny / sn_len
                        nz = snz / sn_len

                # Save dot product before flip (needed for glass)
                d_dot_n = nx * rdx + ny * rdy + nz * rdz

                # Flip normal to face the ray
                if d_dot_n > 0.0:
                    nx = -nx
                    ny = -ny
                    nz = -nz

                # ---- albedo ----
                alb_r = tri_colors[hit_tri * 3]
                alb_g = tri_colors[hit_tri * 3 + 1]
                alb_b = tri_colors[hit_tri * 3 + 2]
                if has_point_colors == 1:
                    pc_w0 = 1.0 - hit_u - hit_v
                    alb_r = pc_w0 * point_colors[i0 * 3]     + hit_u * point_colors[i1 * 3]     + hit_v * point_colors[i2 * 3]
                    alb_g = pc_w0 * point_colors[i0 * 3 + 1] + hit_u * point_colors[i1 * 3 + 1] + hit_v * point_colors[i2 * 3 + 1]
                    alb_b = pc_w0 * point_colors[i0 * 3 + 2] + hit_u * point_colors[i1 * 3 + 2] + hit_v * point_colors[i2 * 3 + 2]

                # ---- material lookup ----
                mb = mat_ids[hit_tri] * 4
                mat_type = int(mat_table[mb])
                mat_ior = mat_table[mb + 1]

                # ---- direct lighting (matte only) ----
                if mat_type == 0:
                    s_ox = hx + nx * 0.005
                    s_oy = hy + ny * 0.005
                    s_oz = hz + nz * 0.005

                    for li in range(n_lights):
                        lb = li * 7
                        lx = light_data[lb] - hx
                        ly = light_data[lb + 1] - hy
                        lz = light_data[lb + 2] - hz
                        l_dist = sqrt(lx * lx + ly * ly + lz * lz) + 1.0e-20
                        lx = lx / l_dist
                        ly = ly / l_dist
                        lz = lz / l_dist
                        n_dot_l = nx * lx + ny * ly + nz * lz
                        if n_dot_l < 0.0:
                            n_dot_l = 0.0

                        # Shadow ray (any-hit BVH traversal)
                        in_shadow = 0
                        s_inv_dx = 1.0 / (lx + 1.0e-20)
                        s_inv_dy = 1.0 / (ly + 1.0e-20)
                        s_inv_dz = 1.0 / (lz + 1.0e-20)

                        stack[stack_base] = 0
                        sp = 1
                        while sp > 0:
                          if in_shadow == 0:
                            sp = sp - 1
                            s_node = stack[stack_base + sp]
                            s_ab = s_node * 6
                            s_tn, s_tf = _ray_aabb(s_ox, s_oy, s_oz,
                                                   s_inv_dx, s_inv_dy, s_inv_dz,
                                                   node_aabb[s_ab], node_aabb[s_ab + 1],
                                                   node_aabb[s_ab + 2], node_aabb[s_ab + 3],
                                                   node_aabb[s_ab + 4], node_aabb[s_ab + 5])
                            if s_tn <= s_tf and s_tf >= 0.0 and s_tn < l_dist:
                                if s_node >= n_inner:
                                    s_tri = tri_ids[s_node - n_inner]
                                    si0 = conn[s_tri * 3]
                                    si1 = conn[s_tri * 3 + 1]
                                    si2 = conn[s_tri * 3 + 2]
                                    st, su, sv = _ray_tri(
                                        s_ox, s_oy, s_oz, lx, ly, lz,
                                        points[si0 * 3], points[si0 * 3 + 1], points[si0 * 3 + 2],
                                        points[si1 * 3], points[si1 * 3 + 1], points[si1 * 3 + 2],
                                        points[si2 * 3], points[si2 * 3 + 1], points[si2 * 3 + 2])
                                    if st > 0.0 and st < l_dist:
                                        in_shadow = 1
                                else:
                                    s_left = node_children[s_node * 2]
                                    s_right = node_children[s_node * 2 + 1]
                                    if sp < 23:
                                        stack[stack_base + sp] = s_left
                                        sp = sp + 1
                                    if sp < 23:
                                        stack[stack_base + sp] = s_right
                                        sp = sp + 1
                          if in_shadow == 1:
                            sp = 0

                        if in_shadow == 0:
                            l_int = light_data[lb + 3]
                            l_cr = light_data[lb + 4]
                            l_cg = light_data[lb + 5]
                            l_cb = light_data[lb + 6]
                            light_c = l_int * n_dot_l / (l_dist * l_dist)
                            acc_r = acc_r + thr_r * alb_r * light_c * l_cr
                            acc_g = acc_g + thr_g * alb_g * light_c * l_cg
                            acc_b = acc_b + thr_b * alb_b * light_c * l_cb

                # ---- prepare bounce ray ----
                seq = _hash(seed * (config.max_bounces + 1) + bounce)
                u1 = _halton2(seq)

                if mat_type == 0:
                    # ---- Matte: cosine-weighted hemisphere ----
                    u2 = _halton3(seq)
                    r_h = sqrt(u1)
                    theta = 6.2831853 * u2
                    sx_h = r_h * cos(theta)
                    sy_h = r_h * sin(theta)
                    sz_h = sqrt(1.0 - u1)
                    if sz_h < 0.0:
                        sz_h = 0.0
                    ref_x = 0.0
                    ref_y = 1.0
                    ref_z = 0.0
                    if abs(ny) > 0.9:
                        ref_x = 1.0
                        ref_y = 0.0
                    tx = ny * ref_z - nz * ref_y
                    ty = nz * ref_x - nx * ref_z
                    tz = nx * ref_y - ny * ref_x
                    t_len = sqrt(tx * tx + ty * ty + tz * tz) + 1.0e-20
                    tx = tx / t_len
                    ty = ty / t_len
                    tz = tz / t_len
                    bx = ny * tz - nz * ty
                    by = nz * tx - nx * tz
                    bz = nx * ty - ny * tx
                    rdx = sx_h * tx + sy_h * bx + sz_h * nx
                    rdy = sx_h * ty + sy_h * by + sz_h * ny
                    rdz = sx_h * tz + sy_h * bz + sz_h * nz
                    rox = hx + nx * 0.005
                    roy = hy + ny * 0.005
                    roz = hz + nz * 0.005
                    thr_r = thr_r * alb_r
                    thr_g = thr_g * alb_g
                    thr_b = thr_b * alb_b

                if mat_type == 1:
                    # ---- Specular: perfect reflection ----
                    refl_dot = rdx * nx + rdy * ny + rdz * nz
                    rdx = rdx - 2.0 * refl_dot * nx
                    rdy = rdy - 2.0 * refl_dot * ny
                    rdz = rdz - 2.0 * refl_dot * nz
                    rox = hx + nx * 0.005
                    roy = hy + ny * 0.005
                    roz = hz + nz * 0.005
                    thr_r = thr_r * alb_r
                    thr_g = thr_g * alb_g
                    thr_b = thr_b * alb_b

                if mat_type == 2:
                    # ---- Transparent: Snell's law + Fresnel ----
                    # Determine entering or exiting
                    entering = 1
                    if d_dot_n > 0.0:
                        entering = 0

                    if entering == 1:
                        eta = 1.0 / mat_ior
                        nn_x = nx
                        nn_y = ny
                        nn_z = nz
                    else:
                        eta = mat_ior
                        nn_x = -nx
                        nn_y = -ny
                        nn_z = -nz

                    cos_i = -(rdx * nn_x + rdy * nn_y + rdz * nn_z)
                    if cos_i < 0.0:
                        cos_i = 0.0
                    sin2_t = eta * eta * (1.0 - cos_i * cos_i)

                    # Schlick Fresnel approximation
                    r0 = (1.0 - mat_ior) / (1.0 + mat_ior)
                    r0 = r0 * r0
                    fresnel = r0 + (1.0 - r0) * pow(1.0 - cos_i, 5.0)

                    if sin2_t > 1.0:
                        # Total internal reflection
                        refl_dot = rdx * nn_x + rdy * nn_y + rdz * nn_z
                        rdx = rdx - 2.0 * refl_dot * nn_x
                        rdy = rdy - 2.0 * refl_dot * nn_y
                        rdz = rdz - 2.0 * refl_dot * nn_z
                        rox = hx + nn_x * 0.005
                        roy = hy + nn_y * 0.005
                        roz = hz + nn_z * 0.005
                    else:
                        if u1 < fresnel:
                            # Reflect
                            refl_dot = rdx * nn_x + rdy * nn_y + rdz * nn_z
                            rdx = rdx - 2.0 * refl_dot * nn_x
                            rdy = rdy - 2.0 * refl_dot * nn_y
                            rdz = rdz - 2.0 * refl_dot * nn_z
                            rox = hx + nn_x * 0.005
                            roy = hy + nn_y * 0.005
                            roz = hz + nn_z * 0.005
                        else:
                            # Refract (Snell's law)
                            cos_t = sqrt(1.0 - sin2_t)
                            rdx = eta * rdx + (eta * cos_i - cos_t) * nn_x
                            rdy = eta * rdy + (eta * cos_i - cos_t) * nn_y
                            rdz = eta * rdz + (eta * cos_i - cos_t) * nn_z
                            # Offset origin against normal (into the surface)
                            rox = hx - nn_x * 0.005
                            roy = hy - nn_y * 0.005
                            roz = hz - nn_z * 0.005
                    # Glass is clear — no throughput loss (or *= albedo for tint)
                    thr_r = thr_r * alb_r
                    thr_g = thr_g * alb_g
                    thr_b = thr_b * alb_b

         # ---- accumulate sample ----
         total_r = total_r + acc_r
         total_g = total_g + acc_g
         total_b = total_b + acc_b

        # ---- write pixel ----
        fb_r[pid] = total_r
        fb_g[pid] = total_g
        fb_b[pid] = total_b


# ================================================================
# RESOLVE KERNEL
# ================================================================

@pgc.kernel
def _resolve(canvas_r, canvas_g, canvas_b,
             fb_r, fb_g, fb_b, inv_samples, n_pixels):
    """Divide accumulated samples and apply gamma correction."""
    for i in range(n_pixels):
        r = fb_r[i] * inv_samples
        g = fb_g[i] * inv_samples
        b = fb_b[i] * inv_samples
        # Gamma 2.2
        canvas_r[i] = pow(min(r, 1.0), 0.4545)
        canvas_g[i] = pow(min(g, 1.0), 0.4545)
        canvas_b[i] = pow(min(b, 1.0), 0.4545)


# ================================================================
# PUBLIC API
# ================================================================

def render(canvas, scene, camera, samples=1, max_bounces=3,
           light_position=None, light_intensity=100.0,
           background=(0.05, 0.05, 0.1)):
    """Path trace the scene into the canvas.

    All lights in the scene are used for direct illumination.  Each light
    contributes independently with its own shadow ray.  If the scene
    contains a volume, it is ray-marched along each primary ray with
    correct depth compositing against surfaces.

    Args:
        canvas: Canvas to render into.
        scene: Scene containing actors, volumes, and lights.
        camera: PerspectiveCamera.
        samples: Number of samples per pixel (quality knob).
        max_bounces: Maximum path depth (0 = direct only).
        light_position: (x, y, z) override for a single light.  When set,
            scene lights are ignored and a single white light is used.
        light_intensity: Scalar brightness for the override light.
        background: RGB background color in [0, 1].
    """
    import time as _time

    # Use cached geometry + BVH if scene hasn't changed
    if scene._cache_version == scene._version:
        geom = scene._cached_geom
        bvh = scene._cached_bvh
        n_tris = geom['n_tris']
        _t_prepare = 0.0
        _t_bvh = 0.0
    else:
        # Merge geometry
        _t0 = _time.perf_counter()
        geom = scene._prepare()
        n_tris = geom['n_tris']
        if n_tris == 0:
            return
        _t_prepare = _time.perf_counter() - _t0

        # BVH
        _t0 = _time.perf_counter()
        bvh = BVH()
        bvh.build(geom['points'], geom['conn'], n_tris)
        _t_bvh = _time.perf_counter() - _t0

        # Cache
        scene._cached_geom = geom
        scene._cached_bvh = bvh
        scene._cache_version = scene._version

    # Pack lights into a flat field: 7 floats per light (x,y,z, intensity, r,g,b)
    if light_position is not None:
        # Override: single white light
        lights_list = [(*light_position, light_intensity, 1.0, 1.0, 1.0)]
    elif scene.lights:
        lights_list = [
            (*lt.position, lt.intensity, *lt.color) for lt in scene.lights
        ]
    else:
        # Default fallback
        lights_list = [(10.0, 10.0, 10.0, light_intensity, 1.0, 1.0, 1.0)]

    n_lights = len(lights_list)
    light_np = np.array(
        [v for lt in lights_list for v in lt], dtype=np.float32)
    light_data = canvas.get_work_buffer('light_data', pgc.f32,
                                        (light_np.shape[0],))
    light_data.from_numpy(light_np)

    # Volume data (first volume in scene, or dummy)
    has_volume = 0
    vol_tf = canvas.get_work_buffer('vol_tf_dummy', pgc.f32, (4,))
    vol_data = canvas.get_work_buffer('vol_data_dummy', pgc.f32, (1,))
    if scene.volumes:
        vol = scene.volumes[0]
        tf = vol.transfer_function
        sr = tf.range if tf.range is not None else vol.scalar_range
        config = _RenderConfig(
            background, max_bounces,
            vol_bounds=(vol.bounds_min, vol.bounds_max),
            vol_scalar_range=sr,
            vol_step=vol.step_size,
            vol_opacity=vol.opacity_scale,
            vol_tf_size=tf.n_samples,
            vol_max_steps=vol.max_steps)
        config.vol_nx = vol.dims[0]
        config.vol_ny = vol.dims[1]
        config.vol_nz = vol.dims[2]
        vol_tf = tf._get_lut_field()
        vol_data = vol.scalar_field
        has_volume = 1
    else:
        config = _RenderConfig(background, max_bounces)

    width = camera.width
    height = camera.height
    n_pixels = width * height

    # Accumulation buffers (cached on canvas to avoid per-frame allocation)
    fb_r = canvas.get_work_buffer('fb_r', pgc.f32, (n_pixels,))
    fb_g = canvas.get_work_buffer('fb_g', pgc.f32, (n_pixels,))
    fb_b = canvas.get_work_buffer('fb_b', pgc.f32, (n_pixels,))

    # BVH traversal stack (cached)
    stack = canvas.get_work_buffer('stack', pgc.i32,
                                   (n_pixels * STACK_DEPTH,))

    # Render all samples in a single kernel launch
    _t0 = _time.perf_counter()
    _pathtrace(fb_r, fb_g, fb_b,
               geom['points'], geom['conn'], geom['tri_colors'],
               geom['point_colors'], geom['normals'],
               bvh.node_aabb, bvh.node_children, bvh.tri_ids,
               light_data,
               vol_tf, vol_data,
               geom['mat_ids'], geom['mat_table'],
               stack,
               camera, config,
               width, height, bvh.n_inner, n_tris, samples,
               geom['has_normals'], geom['has_point_colors'],
               n_lights, has_volume, n_pixels)
    _t_trace = _time.perf_counter() - _t0

    # Resolve to canvas
    inv_samples = 1.0 / float(samples)
    _resolve(canvas.color_r, canvas.color_g, canvas.color_b,
             fb_r, fb_g, fb_b, inv_samples, n_pixels)

    print(f"  [breakdown] prepare={_t_prepare:.3f}s  bvh={_t_bvh:.3f}s  trace={_t_trace:.3f}s")
