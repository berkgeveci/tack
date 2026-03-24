"""Wireframe and point rasterization rendering.

GPU kernels that project geometry to screen space and rasterize edges
(wireframe) or discs (points) with depth testing.
"""

import math
import numpy as np
import pgc


# ================================================================
# PROJECTION CONFIG (template — carries the MVP matrix)
# ================================================================

@pgc.data_oriented
class _ProjConfig:
    """View-projection matrix + screen params as scalar attributes."""

    def __init__(self, mvp, width, height, bg_color, point_size):
        # MVP matrix (row-major, 4x4)
        self.m00 = float(mvp[0, 0])
        self.m01 = float(mvp[0, 1])
        self.m02 = float(mvp[0, 2])
        self.m03 = float(mvp[0, 3])
        self.m10 = float(mvp[1, 0])
        self.m11 = float(mvp[1, 1])
        self.m12 = float(mvp[1, 2])
        self.m13 = float(mvp[1, 3])
        self.m20 = float(mvp[2, 0])
        self.m21 = float(mvp[2, 1])
        self.m22 = float(mvp[2, 2])
        self.m23 = float(mvp[2, 3])
        self.m30 = float(mvp[3, 0])
        self.m31 = float(mvp[3, 1])
        self.m32 = float(mvp[3, 2])
        self.m33 = float(mvp[3, 3])
        self.width = int(width)
        self.height = int(height)
        self.bg_r = float(bg_color[0])
        self.bg_g = float(bg_color[1])
        self.bg_b = float(bg_color[2])
        self.point_size = float(point_size)


def _build_mvp(camera):
    """Build a model-view-projection matrix from a camera."""
    pos = np.array([camera.pos_x, camera.pos_y, camera.pos_z])
    fwd = camera._forward
    right = camera._right
    up = camera._up

    # View matrix (world → camera)
    view = np.eye(4, dtype=np.float64)
    view[0, :3] = right
    view[1, :3] = up
    view[2, :3] = -fwd
    view[0, 3] = -np.dot(right, pos)
    view[1, 3] = -np.dot(up, pos)
    view[2, 3] = np.dot(fwd, pos)

    # For orthographic cameras, pos is the top-left corner origin.
    # Recompute pos as center of the view plane.
    from pgc.rendering.camera import OrthographicCamera
    if isinstance(camera, OrthographicCamera):
        # Reconstruct center from the stored corner position + half extents
        hw = camera.odx_x * camera.width * 0.5 + camera.ody_x * camera.height * 0.5
        hh = camera.odx_y * camera.width * 0.5 + camera.ody_y * camera.height * 0.5
        hz = camera.odx_z * camera.width * 0.5 + camera.ody_z * camera.height * 0.5
        center = np.array([camera.pos_x + hw,
                           camera.pos_y + hh,
                           camera.pos_z + hz])
        view[0, 3] = -np.dot(right, center)
        view[1, 3] = -np.dot(up, center)
        view[2, 3] = np.dot(fwd, center)

    # Projection matrix
    near = 0.01
    far = 1000.0

    if isinstance(camera, OrthographicCamera):
        # Ortho extents from odx/ody
        half_w = np.linalg.norm(
            np.array([camera.odx_x, camera.odx_y, camera.odx_z])
        ) * camera.width * 0.5
        half_h = np.linalg.norm(
            np.array([camera.ody_x, camera.ody_y, camera.ody_z])
        ) * camera.height * 0.5
        proj = np.zeros((4, 4), dtype=np.float64)
        proj[0, 0] = 1.0 / half_w
        proj[1, 1] = 1.0 / half_h
        proj[2, 2] = -2.0 / (far - near)
        proj[2, 3] = -(far + near) / (far - near)
        proj[3, 3] = 1.0
    else:
        # Perspective from camera attributes
        # Recover half_h from corner and forward: corner = forward - right*half_w + up*half_h
        # The corner length along up gives half_h
        corner = np.array([camera.corner_x, camera.corner_y, camera.corner_z])
        # half_h = dot(corner, up) (since corner = fwd - right*hw + up*hh)
        half_h = np.dot(corner, up)
        half_w = -np.dot(corner, right)  # negative because corner = fwd - right*hw
        # But these are tangent values, not actual lengths in clip space
        proj = np.zeros((4, 4), dtype=np.float64)
        proj[0, 0] = 1.0 / half_w
        proj[1, 1] = 1.0 / half_h
        proj[2, 2] = -(far + near) / (far - near)
        proj[2, 3] = -2.0 * far * near / (far - near)
        proj[3, 2] = -1.0

    return (proj @ view).astype(np.float64)


# ================================================================
# GPU HELPERS
# ================================================================

@pgc.func
def _project_vertex(px, py, pz,
                    m00, m01, m02, m03, m10, m11, m12, m13,
                    m20, m21, m22, m23, m30, m31, m32, m33,
                    scr_w, scr_h):
    """Project a 3D point to screen coordinates. Returns (sx, sy, depth, visible)."""
    cx = m00 * px + m01 * py + m02 * pz + m03
    cy = m10 * px + m11 * py + m12 * pz + m13
    cz = m20 * px + m21 * py + m22 * pz + m23
    cw = m30 * px + m31 * py + m32 * pz + m33

    visible = 1
    if cw < 0.001:
        visible = 0
        cw = 0.001

    ndcx = cx / cw
    ndcy = cy / cw
    depth = cz / cw

    sx = (ndcx * 0.5 + 0.5) * float(scr_w)
    sy = (0.5 - ndcy * 0.5) * float(scr_h)

    return sx, sy, depth, visible


# ================================================================
# RASTERIZATION KERNELS
# ================================================================

@pgc.kernel
def _clear_fb(canvas_r, canvas_g, canvas_b, depth_buf,
              bg_r, bg_g, bg_b, n_pixels):
    """Clear framebuffer with background color and reset depth."""
    for i in range(n_pixels):
        canvas_r[i] = bg_r
        canvas_g[i] = bg_g
        canvas_b[i] = bg_b
        depth_buf[i] = 1.0e30


@pgc.kernel
def _rasterize_wireframe(canvas_r, canvas_g, canvas_b, depth_buf,
                         points, conn, tri_colors,
                         mvp_data,
                         scr_w, scr_h, n_tris, max_edge_len):
    """Rasterize triangle edges as wireframe with depth test."""
    for t in range(n_tris):
        cr = tri_colors[t * 3]
        cg = tri_colors[t * 3 + 1]
        cb = tri_colors[t * 3 + 2]

        i0 = conn[t * 3]
        i1 = conn[t * 3 + 1]
        i2 = conn[t * 3 + 2]

        sx0, sy0, d0, v0 = _project_vertex(
            points[i0 * 3], points[i0 * 3 + 1], points[i0 * 3 + 2],
            mvp_data[0], mvp_data[1], mvp_data[2], mvp_data[3],
            mvp_data[4], mvp_data[5], mvp_data[6], mvp_data[7],
            mvp_data[8], mvp_data[9], mvp_data[10], mvp_data[11],
            mvp_data[12], mvp_data[13], mvp_data[14], mvp_data[15],
            scr_w, scr_h)
        sx1, sy1, d1, v1 = _project_vertex(
            points[i1 * 3], points[i1 * 3 + 1], points[i1 * 3 + 2],
            mvp_data[0], mvp_data[1], mvp_data[2], mvp_data[3],
            mvp_data[4], mvp_data[5], mvp_data[6], mvp_data[7],
            mvp_data[8], mvp_data[9], mvp_data[10], mvp_data[11],
            mvp_data[12], mvp_data[13], mvp_data[14], mvp_data[15],
            scr_w, scr_h)
        sx2, sy2, d2, v2 = _project_vertex(
            points[i2 * 3], points[i2 * 3 + 1], points[i2 * 3 + 2],
            mvp_data[0], mvp_data[1], mvp_data[2], mvp_data[3],
            mvp_data[4], mvp_data[5], mvp_data[6], mvp_data[7],
            mvp_data[8], mvp_data[9], mvp_data[10], mvp_data[11],
            mvp_data[12], mvp_data[13], mvp_data[14], mvp_data[15],
            scr_w, scr_h)

        # Edge 0-1
        if v0 == 1 and v1 == 1:
            _draw_edge(canvas_r, canvas_g, canvas_b, depth_buf,
                       sx0, sy0, d0, sx1, sy1, d1,
                       cr, cg, cb, scr_w, scr_h, max_edge_len)
        # Edge 1-2
        if v1 == 1 and v2 == 1:
            _draw_edge(canvas_r, canvas_g, canvas_b, depth_buf,
                       sx1, sy1, d1, sx2, sy2, d2,
                       cr, cg, cb, scr_w, scr_h, max_edge_len)
        # Edge 2-0
        if v2 == 1 and v0 == 1:
            _draw_edge(canvas_r, canvas_g, canvas_b, depth_buf,
                       sx2, sy2, d2, sx0, sy0, d0,
                       cr, cg, cb, scr_w, scr_h, max_edge_len)


@pgc.func
def _draw_edge(canvas_r, canvas_g, canvas_b, depth_buf,
               x0f, y0f, d0, x1f, y1f, d1,
               cr, cg, cb, width, height, max_steps):
    """Bresenham line with depth-interpolated depth test."""
    ix0 = int(x0f)
    iy0 = int(y0f)
    ix1 = int(x1f)
    iy1 = int(y1f)

    dx = ix1 - ix0
    dy = iy1 - iy0
    adx = dx
    if adx < 0:
        adx = -adx
    ady = dy
    if ady < 0:
        ady = -ady

    steps = adx
    if ady > adx:
        steps = ady
    if steps == 0:
        steps = 1

    sx = 1
    if dx < 0:
        sx = -1
    sy = 1
    if dy < 0:
        sy = -1

    err = adx - ady
    cx = ix0
    cy = iy0

    for step in range(max_steps):
        if step > steps:
            break
        if 0 <= cx and cx < width and 0 <= cy and cy < height:
            t = float(step) / float(steps)
            d = d0 + t * (d1 - d0)
            pid = cy * width + cx
            pgc.atomic_min(depth_buf, pid, d)
            if depth_buf[pid] >= d:
                canvas_r[pid] = cr
                canvas_g[pid] = cg
                canvas_b[pid] = cb

        if cx == ix1 and cy == iy1:
            break
        e2 = 2 * err
        if e2 > -ady:
            err = err - ady
            cx = cx + sx
        if e2 < adx:
            err = err + adx
            cy = cy + sy


@pgc.kernel
def _rasterize_points(canvas_r, canvas_g, canvas_b, depth_buf,
                      points, colors, has_colors,
                      mvp_data,
                      scr_w, scr_h, pt_size,
                      n_verts, max_radius):
    """Rasterize vertices as discs with depth test."""
    for v in range(n_verts):
        px = points[v * 3]
        py = points[v * 3 + 1]
        pz = points[v * 3 + 2]

        sx, sy, depth, visible = _project_vertex(
            px, py, pz,
            mvp_data[0], mvp_data[1], mvp_data[2], mvp_data[3],
            mvp_data[4], mvp_data[5], mvp_data[6], mvp_data[7],
            mvp_data[8], mvp_data[9], mvp_data[10], mvp_data[11],
            mvp_data[12], mvp_data[13], mvp_data[14], mvp_data[15],
            scr_w, scr_h)

        if visible == 1:
            cr = 0.8
            cg = 0.8
            cb = 0.8
            if has_colors == 1:
                cr = colors[v * 3]
                cg = colors[v * 3 + 1]
                cb = colors[v * 3 + 2]

            rad = pt_size
            isx = int(sx)
            isy = int(sy)

            for dy in range(max_radius * 2 + 1):
                oy = dy - max_radius
                for dx in range(max_radius * 2 + 1):
                    ox = dx - max_radius
                    if ox * ox + oy * oy <= rad * rad:
                        fx = isx + ox
                        fy = isy + oy
                        if 0 <= fx and fx < scr_w and 0 <= fy and fy < scr_h:
                            pid = fy * scr_w + fx
                            pgc.atomic_min(depth_buf, pid, depth)
                            if depth_buf[pid] >= depth:
                                canvas_r[pid] = cr
                                canvas_g[pid] = cg
                                canvas_b[pid] = cb


@pgc.kernel
def _apply_gamma(canvas_r, canvas_g, canvas_b, n_pixels):
    """Apply gamma correction in-place."""
    for i in range(n_pixels):
        canvas_r[i] = pow(min(canvas_r[i], 1.0), 0.4545)
        canvas_g[i] = pow(min(canvas_g[i], 1.0), 0.4545)
        canvas_b[i] = pow(min(canvas_b[i], 1.0), 0.4545)


# ================================================================
# PUBLIC API
# ================================================================

def render_raster(canvas, scene, camera, background=(0.05, 0.05, 0.1),
                  point_size=3.0):
    """Rasterize wireframe and point actors into the canvas.

    Args:
        canvas: Canvas to render into.
        scene: Scene containing wireframe/point actors.
        camera: PerspectiveCamera or OrthographicCamera.
        background: RGB background color in [0, 1].
        point_size: Radius in pixels for point rendering.
    """
    import time as _time

    width = camera.width
    height = camera.height
    n_pixels = width * height

    mvp = _build_mvp(camera)
    mvp_flat = mvp.astype(np.float32).ravel()
    mvp_field = pgc.field(dtype=pgc.f32, shape=(16,))
    mvp_field.from_numpy(mvp_flat)

    # Clear
    _clear_fb(canvas.color_r, canvas.color_g, canvas.color_b,
              canvas.depth,
              float(background[0]), float(background[1]),
              float(background[2]), n_pixels)

    max_edge_len = width + height
    pt_sz = max(1, int(point_size))

    _t0 = _time.perf_counter()
    for actor in scene.actors:
        if actor.render_mode == "wireframe":
            colors_field = pgc.field(dtype=pgc.f32, shape=(actor.n_tris * 3,))
            from pgc.rendering.scene import _fill_color
            _fill_color(colors_field, 0,
                        actor.color[0], actor.color[1], actor.color[2],
                        actor.n_tris)
            _rasterize_wireframe(
                canvas.color_r, canvas.color_g, canvas.color_b, canvas.depth,
                actor.points, actor.connectivity, colors_field,
                mvp_field,
                width, height, actor.n_tris, max_edge_len)

        elif actor.render_mode == "points":
            has_colors = 0
            colors_field = pgc.field(dtype=pgc.f32, shape=(3,))
            if actor.point_colors is not None:
                colors_field = actor.point_colors
                has_colors = 1
            elif actor.scalars is not None and actor.color_table is not None:
                colors_field = actor.color_table.map_scalars(
                    actor.scalars, actor.n_verts)
                has_colors = 1
            else:
                from pgc.rendering.scene import _fill_color
                colors_field = pgc.field(dtype=pgc.f32,
                                         shape=(actor.n_verts * 3,))
                _fill_color(colors_field, 0,
                            actor.color[0], actor.color[1], actor.color[2],
                            actor.n_verts)
                has_colors = 1

            _rasterize_points(
                canvas.color_r, canvas.color_g, canvas.color_b, canvas.depth,
                actor.points, colors_field, has_colors,
                mvp_field,
                width, height, pt_sz, actor.n_verts, pt_sz)

    _t_raster = _time.perf_counter() - _t0

    # Gamma correction
    _apply_gamma(canvas.color_r, canvas.color_g, canvas.color_b, n_pixels)

    print(f"  [rasterize] {_t_raster:.3f}s")
