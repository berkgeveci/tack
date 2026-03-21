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

    def __init__(self, light_pos, light_intensity, bg_color, max_bounces):
        self.light_x = float(light_pos[0])
        self.light_y = float(light_pos[1])
        self.light_z = float(light_pos[2])
        self.light_intensity = float(light_intensity)
        self.bg_r = float(bg_color[0])
        self.bg_g = float(bg_color[1])
        self.bg_b = float(bg_color[2])
        self.max_bounces = int(max_bounces)


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

    if det < -0.00001 or det > 0.00001:
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
               points, conn, tri_colors, normals,
               node_aabb, node_children, tri_ids,
               stack,
               camera: pgc.template(),
               config: pgc.template(),
               width, height, n_inner, n_tris, n_samples, has_normals,
               n_pixels):
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
         rdx = camera.corner_x + camera.dx_x * (float(px) + 0.5 + jx) + camera.dy_x * (float(py) + 0.5 + jy)
         rdy = camera.corner_y + camera.dx_y * (float(px) + 0.5 + jx) + camera.dy_y * (float(py) + 0.5 + jy)
         rdz = camera.corner_z + camera.dx_z * (float(px) + 0.5 + jx) + camera.dy_z * (float(py) + 0.5 + jy)
         rd_len = sqrt(rdx * rdx + rdy * rdy + rdz * rdz)
         rdx = rdx / rd_len
         rdy = rdy / rd_len
         rdz = rdz / rd_len
         rox = camera.pos_x
         roy = camera.pos_y
         roz = camera.pos_z

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

                # Flip normal to face the ray
                if nx * rdx + ny * rdy + nz * rdz > 0.0:
                    nx = -nx
                    ny = -ny
                    nz = -nz

                # ---- albedo ----
                alb_r = tri_colors[hit_tri * 3]
                alb_g = tri_colors[hit_tri * 3 + 1]
                alb_b = tri_colors[hit_tri * 3 + 2]

                # ---- direct lighting ----
                lx = config.light_x - hx
                ly = config.light_y - hy
                lz = config.light_z - hz
                l_dist = sqrt(lx * lx + ly * ly + lz * lz) + 1.0e-20
                lx = lx / l_dist
                ly = ly / l_dist
                lz = lz / l_dist
                n_dot_l = nx * lx + ny * ly + nz * lz
                if n_dot_l < 0.0:
                    n_dot_l = 0.0

                # Shadow ray (any-hit BVH traversal)
                in_shadow = 0
                s_ox = hx + nx * 0.005
                s_oy = hy + ny * 0.005
                s_oz = hz + nz * 0.005
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
                    light_c = config.light_intensity * n_dot_l / (l_dist * l_dist)
                    acc_r = acc_r + thr_r * alb_r * light_c
                    acc_g = acc_g + thr_g * alb_g * light_c
                    acc_b = acc_b + thr_b * alb_b * light_c

                # ---- prepare bounce ray ----
                seq = _hash(seed * (config.max_bounces + 1) + bounce)
                u1 = _halton2(seq)
                u2 = _halton3(seq)
                r_h = sqrt(u1)
                theta = 6.2831853 * u2
                sx_h = r_h * cos(theta)
                sy_h = r_h * sin(theta)
                sz_h = sqrt(1.0 - u1)
                if sz_h < 0.0:
                    sz_h = 0.0

                # Tangent frame from normal
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

                # New direction in world space
                rdx = sx_h * tx + sy_h * bx + sz_h * nx
                rdy = sx_h * ty + sy_h * by + sz_h * ny
                rdz = sx_h * tz + sy_h * bz + sz_h * nz

                # New origin offset from surface
                rox = hx + nx * 0.005
                roy = hy + ny * 0.005
                roz = hz + nz * 0.005

                # Update throughput (Lambertian: throughput *= albedo)
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

    Args:
        canvas: Canvas to render into.
        scene: Scene containing actors and lights.
        camera: PerspectiveCamera.
        samples: Number of samples per pixel (quality knob).
        max_bounces: Maximum path depth (0 = direct only).
        light_position: (x, y, z) override; defaults to first scene light.
        light_intensity: Scalar brightness override.
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

    # Light
    if light_position is None:
        if scene.lights:
            lp = scene.lights[0].position
            li = scene.lights[0].intensity
        else:
            lp = (10.0, 10.0, 10.0)
            li = light_intensity
    else:
        lp = light_position
        li = light_intensity

    config = _RenderConfig(lp, li, background, max_bounces)

    width = camera.width
    height = camera.height
    n_pixels = width * height

    # Accumulation buffers
    fb_r = pgc.field(dtype=pgc.f32, shape=(n_pixels,))
    fb_g = pgc.field(dtype=pgc.f32, shape=(n_pixels,))
    fb_b = pgc.field(dtype=pgc.f32, shape=(n_pixels,))

    # BVH traversal stack
    stack = pgc.field(dtype=pgc.i32, shape=(n_pixels * STACK_DEPTH,))

    # Render all samples in a single kernel launch
    _t0 = _time.perf_counter()
    _pathtrace(fb_r, fb_g, fb_b,
               geom['points'], geom['conn'], geom['tri_colors'],
               geom['normals'],
               bvh.node_aabb, bvh.node_children, bvh.tri_ids,
               stack,
               camera, config,
               width, height, bvh.n_inner, n_tris, samples,
               geom['has_normals'], n_pixels)
    _t_trace = _time.perf_counter() - _t0

    # Resolve to canvas
    inv_samples = 1.0 / float(samples)
    _resolve(canvas.color_r, canvas.color_g, canvas.color_b,
             fb_r, fb_g, fb_b, inv_samples, n_pixels)

    print(f"  [breakdown] prepare={_t_prepare:.3f}s  bvh={_t_bvh:.3f}s  trace={_t_trace:.3f}s")
