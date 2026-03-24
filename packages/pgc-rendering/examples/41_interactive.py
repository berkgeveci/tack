"""41 -- Interactive renderer using imgui-bundle.

Controls:
  Left-drag on image   Orbit camera
  Scroll on image      Zoom in/out
  Right panel          Render settings, lights, colormap
"""

import math
import sys
import numpy as np
import pgc
from pgc.rendering import (
    PerspectiveCamera, Canvas, Scene, Actor, PointLight, ColorTable,
    Volume, TransferFunction, render, render_volume,
)

import argparse
_p = argparse.ArgumentParser()
_p.add_argument('--arch', default='cpu',
                choices=['cpu', 'metal', 'cuda', 'hip', 'level_zero'])
_p.add_argument('--width', type=int, default=512)
_p.add_argument('--height', type=int, default=512)
_args = _p.parse_args()
pgc.init(arch=getattr(pgc, _args.arch))


# ================================================================
# Geometry helpers
# ================================================================

def make_sphere(center=(0, 0, 0), radius=1.0, subdivisions=32):
    cx, cy, cz = center
    n_lat, n_lon = subdivisions, subdivisions * 2
    verts = [[cx, cy + radius, cz]]
    for i in range(1, n_lat):
        phi = np.pi * i / n_lat
        for j in range(n_lon):
            theta = 2.0 * np.pi * j / n_lon
            verts.append([cx + radius * np.sin(phi) * np.cos(theta),
                          cy + radius * np.cos(phi),
                          cz + radius * np.sin(phi) * np.sin(theta)])
    verts.append([cx, cy - radius, cz])
    verts = np.array(verts, dtype=np.float32)
    tris = []
    for j in range(n_lon):
        tris.append([0, 1 + j, 1 + (j + 1) % n_lon])
    for i in range(n_lat - 2):
        row = 1 + i * n_lon
        for j in range(n_lon):
            jn = (j + 1) % n_lon
            tris.append([row + j, row + n_lon + j, row + n_lon + jn])
            tris.append([row + j, row + n_lon + jn, row + jn])
    bot = len(verts) - 1
    lr = 1 + (n_lat - 2) * n_lon
    for j in range(n_lon):
        tris.append([bot, lr + (j + 1) % n_lon, lr + j])
    return verts, np.array(tris, dtype=np.int32)


def upload(v, t):
    p = pgc.field(dtype=pgc.f32, shape=(v.size,))
    p.from_numpy(v.reshape(-1))
    c = pgc.field(dtype=pgc.i32, shape=(t.size,))
    c.from_numpy(t.reshape(-1))
    return p, c


# ================================================================
# OpenGL texture helper
# ================================================================

class GLTexture:
    """Manage an OpenGL texture from a numpy RGBA image."""

    def __init__(self):
        self.tex_id = None

    def update(self, image_rgba):
        from OpenGL import GL
        h, w = image_rgba.shape[:2]
        if self.tex_id is None:
            self.tex_id = int(GL.glGenTextures(1))
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.tex_id)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER,
                           GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER,
                           GL.GL_LINEAR)
        data = np.ascontiguousarray(image_rgba)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, w, h, 0,
                        GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, data)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)


# ================================================================
# Application state
# ================================================================

PANEL_WIDTH = 320

class App:
    def __init__(self, width, height):
        self.w = width
        self.h = height

        # Camera orbit
        self.yaw = 0.0
        self.pitch = 0.3
        self.distance = 5.0
        self.target = np.array([0.0, 0.0, 0.0])
        self.fov = 45.0
        self.distance_surface = 5.0
        self.distance_volume = 12.0

        # Render settings
        self.samples = 1
        self.max_bounces = 1
        self.bg_color = [0.05, 0.05, 0.1]
        self.use_denoise = False
        self.denoise_ms = 0.0
        try:
            import oidn  # noqa: F401
            self.has_oidn = True
        except ImportError:
            self.has_oidn = False

        # Lights
        self.lights = [
            {"pos": [5.0, 8.0, 5.0], "intensity": 100.0,
             "color": [1.0, 1.0, 1.0], "enabled": True},
            {"pos": [-5.0, 6.0, -3.0], "intensity": 60.0,
             "color": [0.3, 0.5, 1.0], "enabled": False},
            {"pos": [0.0, 4.0, -6.0], "intensity": 60.0,
             "color": [1.0, 0.4, 0.3], "enabled": False},
        ]

        # Render mode: 0 = Surface, 1 = Volume
        self.render_mode = 0
        self.render_modes = ["Surface", "Volume", "Surface + Volume"]

        # Color table / surface coloring
        self.presets = ColorTable.available_presets()
        self.preset_idx = self.presets.index('viridis')
        self.use_scalar_coloring = True
        self.sphere_color = [0.8, 0.8, 0.8]

        # Volume settings
        self.vol_preset_idx = self.presets.index('inferno')
        self.vol_opacity_scale = 8.0
        self.vol_grid_size = 48

        # Build surface geometry
        self.sv, self.st = make_sphere((0, 0, 0), 1.0, 32)
        self.sp, self.sc = upload(self.sv, self.st)
        self.height_scalars = self.sv[:, 1].copy()

        pv = np.array([[-3, -1, -3], [3, -1, -3], [3, -1, 3], [-3, -1, 3]],
                       dtype=np.float32)
        pt = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        self.pp, self.pc = upload(pv, pt)

        # Build volume data (gyroid)
        self._volume = None
        self._vol_dirty = True

        # Rendered image + texture
        self.canvas = Canvas(width, height)
        self.image = np.zeros((height, width, 4), dtype=np.uint8)
        self.texture = GLTexture()
        self.needs_render = True
        self.scene_dirty = True
        self.render_ms = 0.0
        self._scene = None

    def camera_position(self):
        x = self.target[0] + self.distance * math.cos(self.pitch) * math.sin(self.yaw)
        y = self.target[1] + self.distance * math.sin(self.pitch)
        z = self.target[2] + self.distance * math.cos(self.pitch) * math.cos(self.yaw)
        return (x, y, z)

    def invalidate_scene(self):
        self.scene_dirty = True
        self.needs_render = True

    def _rebuild_scene(self):
        scene = Scene()
        if self.use_scalar_coloring:
            ct = ColorTable(self.presets[self.preset_idx])
            scene.add(Actor(self.sp, self.sc,
                            scalars=self.height_scalars,
                            color_table=ct, smooth=True))
        else:
            scene.add(Actor(self.sp, self.sc,
                            color=tuple(self.sphere_color), smooth=True))
        scene.add(Actor(self.pp, self.pc, color=(0.7, 0.7, 0.7)))

        for lt in self.lights:
            if lt["enabled"]:
                scene.add(PointLight(
                    position=tuple(lt["pos"]),
                    intensity=lt["intensity"],
                    color=tuple(lt["color"])))
        if not any(lt["enabled"] for lt in self.lights):
            scene.add(PointLight(position=(5, 8, 5), intensity=100.0))

        self._scene = scene
        self.scene_dirty = False

    def _rebuild_volume(self):
        N = self.vol_grid_size
        x = np.linspace(-np.pi, np.pi, N, dtype=np.float32)
        xx, yy, zz = np.meshgrid(x, x, x, indexing='ij')
        scalars = (np.sin(xx) * np.cos(yy) + np.sin(yy) * np.cos(zz)
                   + np.sin(zz) * np.cos(xx)).astype(np.float32)
        vmin, vmax = float(scalars.min()), float(scalars.max())
        spacing = 2 * np.pi / (N - 1)

        def opacity(t):
            # Transparent near center (gyroid zero-crossing),
            # opaque at extreme positive/negative values
            dist = abs(t - 0.5) * 2.0
            return 0.005 + 0.06 * dist ** 1.5

        tf = TransferFunction(self.presets[self.vol_preset_idx],
                              opacity_func=opacity,
                              range=(vmin, vmax))
        self._volume = Volume(
            scalars.ravel(), dims=(N, N, N),
            origin=(-np.pi, -np.pi, -np.pi),
            spacing=(spacing, spacing, spacing),
            transfer_function=tf,
            opacity_scale=self.vol_opacity_scale)
        self._vol_dirty = False

    def invalidate_volume(self):
        self._vol_dirty = True
        self.needs_render = True

    def _denoise(self):
        """Run OIDN on canvas linear HDR float buffers."""
        import oidn
        w, h = self.canvas.width, self.canvas.height
        r = self.canvas.color_r.to_numpy().reshape(h, w)
        g = self.canvas.color_g.to_numpy().reshape(h, w)
        b = self.canvas.color_b.to_numpy().reshape(h, w)
        color = np.stack([r, g, b], axis=-1).astype(np.float32)
        output = np.zeros_like(color)

        device = oidn.NewDevice(oidn.DEVICE_TYPE_DEFAULT)
        oidn.CommitDevice(device)
        filt = oidn.NewFilter(device, 'RT')
        oidn.SetSharedFilterImage(filt, 'color', color,
                                  oidn.FORMAT_FLOAT3, w, h)
        oidn.SetSharedFilterImage(filt, 'output', output,
                                  oidn.FORMAT_FLOAT3, w, h)
        oidn.CommitFilter(filt)
        oidn.ExecuteFilter(filt)
        oidn.ReleaseFilter(filt)
        oidn.ReleaseDevice(device)

        img = np.zeros((h, w, 4), dtype=np.uint8)
        img[:, :, 0] = np.clip(output[:, :, 0] * 255, 0, 255).astype(np.uint8)
        img[:, :, 1] = np.clip(output[:, :, 1] * 255, 0, 255).astype(np.uint8)
        img[:, :, 2] = np.clip(output[:, :, 2] * 255, 0, 255).astype(np.uint8)
        img[:, :, 3] = 255
        return img

    def do_render(self):
        import time

        # Rebuild scene contents as needed
        if self.render_mode == 1:
            # Volume only
            if self._vol_dirty or self._volume is None:
                self._rebuild_volume()
            scene = Scene()
            scene.add(self._volume)
        elif self.render_mode == 2:
            # Surface + Volume (integrated ray tracing)
            # Use only the sphere (no ground plane — it fills the volume
            # and cuts the march short, hiding the volume structure)
            if self._vol_dirty or self._volume is None:
                self._rebuild_volume()
            scene = Scene()
            if self.use_scalar_coloring:
                ct = ColorTable(self.presets[self.preset_idx])
                scene.add(Actor(self.sp, self.sc,
                                scalars=self.height_scalars,
                                color_table=ct, smooth=True))
            else:
                scene.add(Actor(self.sp, self.sc,
                                color=tuple(self.sphere_color), smooth=True))
            for lt_cfg in self.lights:
                if lt_cfg["enabled"]:
                    scene.add(PointLight(
                        position=tuple(lt_cfg["pos"]),
                        intensity=lt_cfg["intensity"],
                        color=tuple(lt_cfg["color"])))
            if not any(lt_cfg["enabled"] for lt_cfg in self.lights):
                scene.add(PointLight(position=(5, 8, 5), intensity=100.0))
            scene.add(self._volume)
        else:
            # Surface only
            if self.scene_dirty:
                self._rebuild_scene()
            scene = self._scene

        camera = PerspectiveCamera(
            position=self.camera_position(),
            look_at=tuple(self.target),
            fov=self.fov,
            width=self.w, height=self.h)

        # Unified render dispatch
        t0 = time.perf_counter()
        render(self.canvas, scene, camera,
               samples=self.samples,
               max_bounces=self.max_bounces,
               background=tuple(self.bg_color))
        self.render_ms = (time.perf_counter() - t0) * 1000.0

        if self.render_mode == 0 and self.use_denoise and self.has_oidn:
            t0 = time.perf_counter()
            self.image = self._denoise()
            self.denoise_ms = (time.perf_counter() - t0) * 1000.0
        else:
            self.image = self.canvas.to_numpy()
            self.denoise_ms = 0.0

        self.texture.update(self.image)
        self.needs_render = False


# ================================================================
# GUI
# ================================================================

def run_gui(app):
    from imgui_bundle import imgui, hello_imgui

    first_frame = [True]

    def gui():
        if first_frame[0]:
            app.do_render()
            first_frame[0] = False

        io = imgui.get_io()
        dw = io.display_size.x
        dh = io.display_size.y

        # Single fullscreen window — no title bar, no decorations
        imgui.set_next_window_pos(imgui.ImVec2(0, 0))
        imgui.set_next_window_size(imgui.ImVec2(dw, dh))
        wflags = (imgui.WindowFlags_.no_title_bar
                  | imgui.WindowFlags_.no_resize
                  | imgui.WindowFlags_.no_move
                  | imgui.WindowFlags_.no_collapse
                  | imgui.WindowFlags_.no_scrollbar
                  | imgui.WindowFlags_.no_scroll_with_mouse)

        imgui.begin("##main", None, wflags)

        panel_w = min(PANEL_WIDTH, dw * 0.4)
        img_region_w = dw - panel_w - imgui.get_style().item_spacing.x

        # ============================================================
        # Left child: rendered image
        # ============================================================
        imgui.begin_child("##image_region", imgui.ImVec2(img_region_w, 0),
                          imgui.ChildFlags_.none,
                          imgui.WindowFlags_.no_scrollbar)

        if app.texture.tex_id is not None:
            avail = imgui.get_content_region_avail()
            avail_h = avail.y - 24  # room for status text
            scale = min(avail.x / app.w, avail_h / app.h)
            draw_w = app.w * scale
            draw_h = app.h * scale
            # Center
            pad_x = (avail.x - draw_w) * 0.5
            if pad_x > 0:
                imgui.set_cursor_pos_x(
                    imgui.get_cursor_pos_x() + pad_x)

            tex_ref = imgui.ImTextureRef(app.texture.tex_id)
            imgui.image(tex_ref, imgui.ImVec2(draw_w, draw_h))

            # Mouse interaction on the image item
            if imgui.is_item_hovered():
                if imgui.is_mouse_dragging(imgui.MouseButton_.left):
                    app.yaw -= io.mouse_delta.x * 0.005
                    app.pitch += io.mouse_delta.y * 0.005
                    app.pitch = max(-math.pi / 2 + 0.01,
                                    min(math.pi / 2 - 0.01, app.pitch))
                    app.needs_render = True

                if io.mouse_wheel != 0.0:
                    app.distance -= io.mouse_wheel * 0.3
                    app.distance = max(1.0, min(20.0, app.distance))
                    app.needs_render = True

        status = f"Trace: {app.render_ms:.1f} ms"
        if app.denoise_ms > 0:
            status += f"  Denoise: {app.denoise_ms:.0f} ms"
        status += f"  |  {app.w}x{app.h}  |  Drag=orbit  Scroll=zoom"
        imgui.text(status)
        imgui.end_child()

        # ============================================================
        # Right child: controls panel
        # ============================================================
        imgui.same_line()
        imgui.begin_child("##controls_region", imgui.ImVec2(0, 0),
                          imgui.ChildFlags_.borders)

        imgui.text("PGC Renderer")
        imgui.separator()

        # -- Render mode --
        changed, app.render_mode = imgui.combo(
            "Mode", app.render_mode, app.render_modes)
        if changed:
            app.distance = (app.distance_volume if app.render_mode >= 1
                            else app.distance_surface)
            app.needs_render = True
        imgui.separator()

        # -- Common settings --
        if imgui.collapsing_header("Render Settings",
                                   imgui.TreeNodeFlags_.default_open):
            changed, app.bg_color = imgui.color_edit3("Background",
                                                      app.bg_color)
            if changed:
                app.needs_render = True
            changed, app.fov = imgui.slider_float(
                "FOV", app.fov, 10.0, 120.0)
            if changed:
                app.needs_render = True

            has_surf = app.render_mode in (0, 2)
            has_vol = app.render_mode in (1, 2)

            if has_surf:
                changed, app.samples = imgui.slider_int(
                    "Samples", app.samples, 1, 16)
                if changed:
                    app.needs_render = True
                changed, app.max_bounces = imgui.slider_int(
                    "Bounces", app.max_bounces, 0, 8)
                if changed:
                    app.needs_render = True
                if app.render_mode == 0:
                    if app.has_oidn:
                        changed, app.use_denoise = imgui.checkbox(
                            "Denoise (OIDN)", app.use_denoise)
                        if changed:
                            app.needs_render = True
                    else:
                        imgui.begin_disabled()
                        imgui.checkbox("Denoise (OIDN not found)", False)
                        imgui.end_disabled()

            if has_vol:
                changed, app.vol_opacity_scale = imgui.slider_float(
                    "Opacity Scale", app.vol_opacity_scale, 0.1, 30.0)
                if changed:
                    app.invalidate_volume()
                changed, app.vol_preset_idx = imgui.combo(
                    "Transfer Func", app.vol_preset_idx, app.presets)
                if changed:
                    app.invalidate_volume()

        if app.render_mode in (0, 2):
            # -- Surface coloring --
            if imgui.collapsing_header("Coloring",
                                       imgui.TreeNodeFlags_.default_open):
                changed, app.use_scalar_coloring = imgui.checkbox(
                    "Scalar Coloring", app.use_scalar_coloring)
                if changed:
                    app.invalidate_scene()

                if app.use_scalar_coloring:
                    changed, app.preset_idx = imgui.combo(
                        "Preset", app.preset_idx, app.presets)
                    if changed:
                        app.invalidate_scene()
                else:
                    changed, app.sphere_color = imgui.color_edit3(
                        "Sphere Color", app.sphere_color)
                    if changed:
                        app.invalidate_scene()

            # -- Lights --
            if imgui.collapsing_header("Lights",
                                       imgui.TreeNodeFlags_.default_open):
                for i, lt in enumerate(app.lights):
                    imgui.push_id(str(i))
                    changed_e, lt["enabled"] = imgui.checkbox(
                        f"Light {i + 1}", lt["enabled"])

                    if lt["enabled"]:
                        imgui.indent()
                        c1, lt["pos"][0] = imgui.slider_float(
                            "X", lt["pos"][0], -10.0, 10.0)
                        c2, lt["pos"][1] = imgui.slider_float(
                            "Y", lt["pos"][1], 0.0, 15.0)
                        c3, lt["pos"][2] = imgui.slider_float(
                            "Z", lt["pos"][2], -10.0, 10.0)
                        c4, lt["intensity"] = imgui.slider_float(
                            "Intensity", lt["intensity"], 0.0, 300.0)
                        c5, lt["color"] = imgui.color_edit3(
                            "Color", lt["color"])
                        if any([changed_e, c1, c2, c3, c4, c5]):
                            app.invalidate_scene()
                        imgui.unindent()
                    elif changed_e:
                        app.invalidate_scene()

                    imgui.pop_id()

        imgui.separator()
        if imgui.button("Re-render"):
            if app.render_mode in (1, 2):
                app.invalidate_volume()
            if app.render_mode in (0, 2):
                app.invalidate_scene()

        imgui.end_child()
        imgui.end()

        # ============================================================
        # Deferred render (after UI so changes are captured)
        # ============================================================
        if app.needs_render:
            app.do_render()

    params = hello_imgui.RunnerParams()
    params.app_window_params.window_title = "PGC Interactive Renderer"
    params.app_window_params.window_geometry.size = (1200, 800)
    params.fps_idling.fps_idle = 30.0
    params.callbacks.show_gui = gui

    hello_imgui.run(params)


# ================================================================
# Main
# ================================================================

if __name__ == "__main__":
    app = App(_args.width, _args.height)
    run_gui(app)
