"""CX8: contended depth selection must keep the winning primitive's RGB."""

import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import tack
from tack.rendering import Actor, Canvas, OrthographicCamera, Scene
from tack.rendering.rasterize import (
    _clear_fb,
    _rasterize_points,
    _rasterize_wireframe,
    _resolve_raster,
    render_raster,
)
from tack.runtime.dispatch import get_backend
from tack.runtime.kernel_utils import resolve_variant

SIZE = 32
COUNT = 4097  # Multiple GPU workgroups plus a partial final group.
SENTINEL = np.iinfo(np.int32).max


def _field(values, dtype=tack.f32):
    values = np.asarray(values, dtype=dtype.numpy_dtype).ravel()
    field = tack.field(dtype, values.shape)
    field.from_numpy(values)
    return field


def _camera():
    return OrthographicCamera(position=(0, 0, 5), look_at=(0, 0, 0),
                              view_height=4, width=SIZE, height=SIZE)


def _point_actor(z, colors):
    points = np.zeros((len(z), 3), np.float32)
    points[:, 2] = z
    return Actor(_field(points), _field([0, 0, 0], tack.i32),
                 point_colors=np.asarray(colors, np.float32), render_mode='points')


def _channels(canvas):
    return np.column_stack([canvas.color_r.to_numpy(), canvas.color_g.to_numpy(),
                            canvas.color_b.to_numpy()])


@pytest.mark.parametrize('nearest', [0, 2048, COUNT - 1])
def test_overlapping_points_keep_nearest_color(backend, nearest):
    z = np.full(COUNT, -1, np.float32)
    z[nearest] = 0
    colors = np.tile([1, 0, 0], (COUNT, 1)).astype(np.float32)
    colors[nearest] = [0, 1, 0]
    scene = Scene()
    scene.add(_point_actor(z, colors))
    canvas = Canvas(SIZE, SIZE)
    for _ in range(4):
        render_raster(canvas, scene, _camera(), background=(0, 0, 0))
        rgb = _channels(canvas)
        lit = rgb.max(axis=1) > 0
        assert lit.sum() == 29
        np.testing.assert_array_equal(rgb[lit], np.tile([0, 1, 0], (29, 1)))
        # Frame scratch is reused but selection must be rebuilt each time.
        winners = canvas.get_work_buffer('raster_winners', tack.i32, (SIZE * SIZE,))
        np.testing.assert_array_equal(winners.to_numpy(), SENTINEL)


def test_point_ties_and_changed_inputs_select_one_color(backend):
    colors = np.tile([1, 0, 0], (COUNT, 1)).astype(np.float32)
    colors[0] = [0, 1, 0]
    actor = _point_actor(np.zeros(COUNT), colors)
    scene = Scene()
    scene.add(actor)
    canvas = Canvas(SIZE, SIZE)
    for frame in range(4):
        expected = [0, 1, 0] if frame % 2 == 0 else [0, 0, 1]
        colors[0] = expected
        actor.point_colors.from_numpy(colors.ravel())
        render_raster(canvas, scene, _camera(), background=(0, 0, 0))
        rgb = _channels(canvas)
        lit = rgb.max(axis=1) > 0
        assert lit.sum() == 29
        np.testing.assert_array_equal(rgb[lit], np.tile(expected, (29, 1)))


def _check_primitive_selection(mode, tie, threaded, gradient=False):
    """Read depth only after all writers finish, and color only after ID selection."""
    count = COUNT
    expected_id = 0 if tie else 2048
    depths = np.full(count, 0.5, np.float32)
    if not tie:
        depths[expected_id] = -0.25
    # Identity projection gives exact known screen coordinates/depths.
    if mode == 'points':
        points = np.zeros((count, 3), np.float32)
        points[:, 2] = depths
        kernel = _rasterize_points
    else:
        points = np.tile([[-0.5, 0, 0], [0.5, 0, 0], [0, 0.5, 0]], (count, 1))
        points = points.astype(np.float32)
        points[:, 2] = np.repeat(depths, 3)
        if gradient:
            points[:, 2] += np.tile([0.0625, 0.125, 0.1875], count)
        conn = _field(np.arange(count * 3), tack.i32)
        kernel = _rasterize_wireframe
    points = _field(points)
    mvp = _field(np.eye(4, dtype=np.float32))
    colors = np.tile([1, 0, 0], (count, 1)).astype(np.float32)
    colors[expected_id] = [0, 1, 0]
    colors = _field(colors)
    depth = _field(np.full(SIZE * SIZE, 1e30, np.float32))
    winners = _field(np.full(SIZE * SIZE, SENTINEL), tack.i32)
    rgb = [_field(np.full(SIZE * SIZE, -1, np.float32)) for _ in range(3)]
    if mode == 'points':
        args = (depth, winners, points, mvp, SIZE, SIZE, 3, count, 3)
    else:
        args = (depth, winners, points, conn, mvp, SIZE, SIZE, count, SIZE * 2)

    if gradient:
        reference = _field(np.full(SIZE * SIZE, 1e30, np.float32))
        one_triangle = _field(points.to_numpy()[expected_id * 9:(expected_id + 1) * 9])
        _rasterize_wireframe(reference, winners, one_triangle, _field([0, 1, 2], tack.i32),
                             mvp, SIZE, SIZE, 1, SIZE * 2, 0)
        expected_depth = reference.to_numpy()

    def run_phase(phase):
        if not threaded:
            kernel(*args, phase)
            return
        be = get_backend()
        variant, effective = resolve_variant(be, kernel, (*args, phase), {},
                                             build=be._build_variant)
        compiled = variant.payload
        prefix = compiled.bind(effective)
        workers = 8
        gate = threading.Barrier(workers)

        def run(slot):
            gate.wait()
            compiled.call_range(prefix, slot * count // workers, (slot + 1) * count // workers)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for future in [pool.submit(run, slot) for slot in range(workers)]:
                future.result()

    for _ in range(4):
        _clear_fb(*rgb, depth, winners, -1.0, -1.0, -1.0, SIZE * SIZE)
        run_phase(0)
        reduced = depth.to_numpy()
        covered = reduced < 1e30
        assert covered.sum() > 20
        if gradient:
            np.testing.assert_array_equal(reduced, expected_depth)
            assert np.unique(reduced[covered]).size > 3
        else:
            np.testing.assert_array_equal(reduced[covered], depths[expected_id])
        np.testing.assert_array_equal(winners.to_numpy(), SENTINEL)
        run_phase(1)
        np.testing.assert_array_equal(depth.to_numpy(), reduced)
        np.testing.assert_array_equal(winners.to_numpy()[covered], expected_id)
        np.testing.assert_array_equal(winners.to_numpy()[~covered], SENTINEL)
        for channel in rgb:
            np.testing.assert_array_equal(channel.to_numpy(), -1)
        _resolve_raster(*rgb, winners, colors, 1, SIZE * SIZE)
        result = np.column_stack([c.to_numpy() for c in rgb])
        np.testing.assert_array_equal(result[covered], np.tile([0, 1, 0], (covered.sum(), 1)))
        np.testing.assert_array_equal(result[~covered], -1)
        np.testing.assert_array_equal(winners.to_numpy(), SENTINEL)


@pytest.mark.parametrize('mode', ['points', 'wireframe'])
@pytest.mark.parametrize('tie', [False, True])
def test_primitive_selection_then_pixel_owned_resolve(backend, mode, tie):
    _check_primitive_selection(mode, tie, threaded=False)


@pytest.mark.parametrize('mode', ['points', 'wireframe'])
@pytest.mark.parametrize('tie', [False, True])
def test_cpu_workers_select_then_resolve(mode, tie):
    tack.init(arch=tack.cpu)
    _check_primitive_selection(mode, tie, threaded=True)


def test_overlapping_wireframes_with_interpolated_depth(backend):
    _check_primitive_selection('wireframe', tie=False, threaded=False, gradient=True)


def test_cached_geometry_changes_the_point_winner(backend):
    scene = Scene()
    actor = _point_actor([0, -1], [[0, 1, 0], [1, 0, 0]])
    scene.add(actor)
    canvas = Canvas(SIZE, SIZE)
    for frame in range(4):
        points = np.array([[0, 0, -1], [0, 0, -1]], np.float32)
        winner = frame % 2
        points[winner, 2] = 0
        actor.points.from_numpy(points.ravel())
        render_raster(canvas, scene, _camera(), background=(0, 0, 0))
        rgb = _channels(canvas)
        covered = rgb.max(axis=1) > 0
        expected = [0, 1, 0] if winner == 0 else [1, 0, 0]
        assert covered.sum() == 29
        np.testing.assert_array_equal(rgb[covered], np.tile(expected, (29, 1)))


@pytest.mark.parametrize('mode', ['points', 'wireframe'])
@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('tie', [False, True])
def test_raster_actors_keep_depth_and_scene_order(backend, mode, reverse, tie):
    def actor(z, color):
        points = _field([[-0.5, 0, z], [0.5, 0, z], [0, 0.5, z]])
        return Actor(points, _field([0, 1, 2], tack.i32), color=color, render_mode=mode)

    front = actor(0, (0, 1, 0))
    back = actor(0 if tie else -1, (1, 0, 0))
    actors = [front, back] if not reverse else [back, front]
    scene = Scene()
    for item in actors:
        scene.add(item)
    canvas = Canvas(SIZE, SIZE)
    render_raster(canvas, scene, _camera(), background=(0, 0, 0))
    rgb = _channels(canvas)
    lit = rgb.max(axis=1) > 0
    if mode == 'wireframe':
        assert lit.sum() == 16  # Bresenham coverage of this projected triangle.
    else:
        assert lit.sum() > 20
    expected = (1, 0, 0) if tie and not reverse else (0, 1, 0)
    np.testing.assert_array_equal(rgb[lit], np.tile(expected, (lit.sum(), 1)))


@pytest.mark.parametrize('surface_depth', [False, True])
def test_composite_preserves_uncovered_color_and_ray_depth(backend, surface_depth):
    scene = Scene()
    scene.add(_point_actor(np.zeros(COUNT), np.tile([0, 1, 0], (COUNT, 1))))
    canvas = Canvas(SIZE, SIZE)
    canvas.color_r.fill(0.25)
    canvas.color_g.fill(0.5)
    canvas.color_b.fill(0.75)
    canvas.depth.fill(-1)  # Path-traced miss: raster remains visible.
    render_raster(canvas, scene, _camera(), composite=True, surface_depth=surface_depth)
    rgb = _channels(canvas)
    covered = rgb[:, 1] == 1
    assert covered.sum() == 29
    np.testing.assert_array_equal(rgb[covered], np.tile([0, 1, 0], (29, 1)))
    np.testing.assert_array_equal(rgb[~covered], np.tile([0.25, 0.5, 0.75], ((~covered).sum(), 1)))
    np.testing.assert_array_equal(canvas.depth.to_numpy(), -1)


def test_new_actor_and_empty_frame_do_not_reuse_old_winner(backend):
    scene = Scene()
    front = _point_actor([0], [[0, 1, 0]])
    behind = _point_actor([-1], [[1, 0, 0]])
    scene.add(front)
    scene.add(behind)
    canvas = Canvas(SIZE, SIZE)
    render_raster(canvas, scene, _camera(), background=(0, 0, 0))
    rgb = _channels(canvas)
    lit = rgb.max(axis=1) > 0
    np.testing.assert_array_equal(rgb[lit], np.tile([0, 1, 0], (lit.sum(), 1)))
    scratch = canvas.get_work_buffer('raster_winners', tack.i32, (SIZE * SIZE,))
    np.testing.assert_array_equal(scratch.to_numpy(), SENTINEL)  # Behind actor selected nothing.
    front.points.from_numpy(np.array([10, 10, 0], np.float32))
    behind.points.from_numpy(np.array([10, 10, -1], np.float32))
    render_raster(canvas, scene, _camera(), background=(0, 0, 0))
    np.testing.assert_array_equal(_channels(canvas), 0)


@pytest.mark.parametrize('zero', [0.0, -0.0])
def test_zero_clip_depth_is_a_valid_fragment(backend, zero):
    points = _field([0, 0, zero])
    mvp = _field(np.eye(4, dtype=np.float32))
    depth = _field(np.full(SIZE * SIZE, 1e30, np.float32))
    winners = _field(np.full(SIZE * SIZE, SENTINEL), tack.i32)
    for phase in (0, 1):
        _rasterize_points(depth, winners, points, mvp, SIZE, SIZE, 3, 1, 3, phase)
    covered = winners.to_numpy() == 0
    assert covered.sum() == 29
    np.testing.assert_array_equal(depth.to_numpy()[covered], 0)


@pytest.mark.parametrize('invalid', [np.nan, np.inf, -np.inf, 1e30, -1e30])
def test_invalid_fragment_depth_is_discarded(backend, invalid):
    # Keep projected x/y finite, so this specifically tests fragment depth.
    points = _field([0, 0, 1])
    matrix = np.eye(4, dtype=np.float32)
    matrix[2, 2] = invalid
    depth = _field(np.full(SIZE * SIZE, 1e30, np.float32))
    winners = _field(np.full(SIZE * SIZE, SENTINEL), tack.i32)
    for phase in (0, 1):
        _rasterize_points(depth, winners, points, _field(matrix), SIZE, SIZE, 3, 1, 3, phase)
    np.testing.assert_array_equal(depth.to_numpy(), np.float32(1e30))
    np.testing.assert_array_equal(winners.to_numpy(), SENTINEL)
