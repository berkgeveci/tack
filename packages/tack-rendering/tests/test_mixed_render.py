"""Solid, volume and rasterized actors sharing one image.

The dispatcher path traces solids (and their volumes), then rasterizes
wireframe and point actors over the result.  These tests check that both
representations survive and that the surfaces occlude what is behind them.
"""

import numpy as np

import tack
from tack.rendering import (
    Actor,
    Canvas,
    OrthographicCamera,
    PerspectiveCamera,
    PointLight,
    Scene,
    TransferFunction,
    Volume,
    render,
)

SIZE = 32


def _triangle(z=0.0, scale=1.0, dx=0.0):
    pts_np = np.array([-scale + dx, -scale, z,
                       scale + dx, -scale, z,
                       dx, scale, z], dtype=np.float32)
    conn_np = np.array([0, 1, 2], dtype=np.int32)
    pts = tack.field(dtype=tack.f32, shape=(pts_np.size,))
    pts.from_numpy(pts_np)
    conn = tack.field(dtype=tack.i32, shape=(conn_np.size,))
    conn.from_numpy(conn_np)
    return pts, conn


def _solid(**kw):
    return Actor(*_triangle(**kw), color=(1.0, 0.0, 0.0))


def _raster(mode="wireframe", **kw):
    return Actor(*_triangle(**kw), color=(0.0, 1.0, 0.0), render_mode=mode)


def _scene(*actors):
    scene = Scene()
    for a in actors:
        scene.add(a)
    scene.add(PointLight(position=(0, 5, 5), intensity=50.0))
    return scene


def _perspective():
    return PerspectiveCamera(position=(0, 0, 4), look_at=(0, 0, 0), fov=60,
                             width=SIZE, height=SIZE)


def _render(scene, camera=None, **kw):
    camera = camera or _perspective()
    canvas = Canvas(SIZE, SIZE)
    render(canvas, scene, camera, samples=1, max_bounces=0,
           background=(0.0, 0.0, 0.0), **kw)
    return canvas


def _image(scene, camera=None, **kw):
    return _render(scene, camera, **kw).to_numpy()


def _red(img):
    return img[:, :, 0] > 0


def _green(img):
    return img[:, :, 1] > 0


class TestSolidWithRaster:
    def test_coincident_wireframe_keeps_both(self, backend):
        """The CX2 reproduction: a wireframe drawn on its own surface."""
        solid = _image(_scene(_solid()))
        wire = _image(_scene(_raster()))
        mixed = _image(_scene(_solid(), _raster()))

        assert _red(solid).sum() > 0 and _green(wire).sum() > 0
        # Every wire pixel is drawn, and nowhere else changes.
        np.testing.assert_array_equal(_green(mixed), _green(wire))
        np.testing.assert_array_equal(mixed[~_green(wire)],
                                      solid[~_green(wire)])
        np.testing.assert_array_equal(mixed[_green(wire)],
                                      wire[_green(wire)])
        assert (_red(mixed) & ~_green(mixed)).sum() > 0

    def test_points_keep_the_surface(self, backend):
        solid = _image(_scene(_solid()))
        mixed = _image(_scene(_solid(), _raster("points")), point_size=2.0)
        assert _green(mixed).sum() > 0
        np.testing.assert_array_equal(mixed[~_green(mixed)],
                                      solid[~_green(mixed)])
        assert (_red(mixed) & ~_green(mixed)).sum() > 0

    def test_surface_hides_wireframe_behind_it(self, backend):
        solid = _image(_scene(_solid(scale=1.5)))
        mixed = _image(_scene(_solid(scale=1.5), _raster(z=-1.0, scale=0.5)))
        np.testing.assert_array_equal(mixed, solid)

    def test_wireframe_in_front_is_drawn(self, backend):
        wire = _image(_scene(_raster(z=1.0, scale=0.5)))
        mixed = _image(_scene(_solid(scale=1.5), _raster(z=1.0, scale=0.5)))
        np.testing.assert_array_equal(_green(mixed), _green(wire))

    def test_wireframe_partly_outside_the_surface(self, backend):
        """Behind the surface where they overlap, visible where they do not."""
        wire = _image(_scene(_raster(z=-1.0, dx=1.0)))
        solid = _image(_scene(_solid()))
        mixed = _image(_scene(_solid(), _raster(z=-1.0, dx=1.0)))
        np.testing.assert_array_equal(_green(mixed),
                                      _green(wire) & ~_red(solid))
        assert _green(mixed).sum() > 0
        assert (_green(wire) & _red(solid)).sum() > 0

    def test_raster_actors_are_not_path_traced(self, backend):
        """A wireframe's interior shows the background, not a filled face."""
        mixed = _image(_scene(_solid(dx=-1.0, scale=0.5),
                              _raster(dx=1.0, scale=0.5)))
        wire = _image(_scene(_raster(dx=1.0, scale=0.5)))
        np.testing.assert_array_equal(_green(mixed), _green(wire))

    def test_orthographic_occlusion(self, backend):
        def cam():
            return OrthographicCamera(position=(0, 0, 4), look_at=(0, 0, 0),
                                      view_height=4.0, width=SIZE, height=SIZE)
        solid = _image(_scene(_solid(scale=1.5)), cam())
        behind = _image(_scene(_solid(scale=1.5), _raster(z=-1.0, scale=0.5)),
                        cam())
        np.testing.assert_array_equal(behind, solid)
        wire = _image(_scene(_raster(z=1.0, scale=0.5)), cam())
        front = _image(_scene(_solid(scale=1.5), _raster(z=1.0, scale=0.5)),
                       cam())
        assert _green(wire).sum() > 0
        np.testing.assert_array_equal(_green(front), _green(wire))

    def test_depth_buffer_stays_in_ray_distances(self, backend):
        solid = _render(_scene(_solid())).depth_to_numpy()
        mixed = _render(_scene(_solid(), _raster())).depth_to_numpy()
        np.testing.assert_array_equal(mixed, solid)

    def test_repeated_frames_reuse_the_solid_scene(self, backend):
        scene = _scene(_solid(), _raster())
        first = _image(scene)
        sub = scene._solid_subscene[1]
        second = _image(scene)
        assert scene._solid_subscene[1] is sub
        np.testing.assert_array_equal(first, second)
        # A new actor invalidates it.
        scene.add(_solid(z=-2.0, scale=3.0))
        third = _image(scene)
        assert scene._solid_subscene[1] is not sub
        assert _red(third).sum() > _red(first).sum()


class TestVolumeWithRaster:
    def _volume(self):
        n = 8
        data = np.ones(n**3, dtype=np.float32) * 0.5
        tf = TransferFunction('grayscale', opacity_func=lambda t: 0.5,
                              range=(0.0, 1.0))
        return Volume(data, dims=(n, n, n), origin=(-1, -1, -1),
                      spacing=(2.0 / (n - 1),) * 3,
                      transfer_function=tf, opacity_scale=10.0)

    def test_wireframe_is_drawn_over_the_volume(self, backend):
        vol_scene = Scene()
        vol_scene.add(self._volume())
        vol = _image(vol_scene)

        scene = Scene()
        scene.add(self._volume())
        scene.add(_raster(scale=1.5))
        mixed = _image(scene)
        wire = _image(_scene(_raster(scale=1.5)))

        assert vol[:, :, 0].max() > 0
        np.testing.assert_array_equal(mixed[~_green(wire)], vol[~_green(wire)])
        np.testing.assert_array_equal(mixed[_green(wire)], wire[_green(wire)])
