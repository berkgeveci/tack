"""Tests for texture3d sampling through @pgc.func inlining."""

import numpy as np
import pytest
import pgc

_backends = []
for _arch in ["cpu", "metal"]:
    try:
        pgc.init(arch=getattr(pgc, _arch))
        _backends.append(_arch)
    except (ImportError, RuntimeError, OSError):
        pass


@pytest.fixture(params=_backends)
def backend(request):
    pgc.init(arch=getattr(pgc, request.param))
    return request.param


def test_texture_sample_direct(backend):
    """tex.sample() called directly in a kernel works."""
    W, H, D = 4, 4, 4
    n = W * H * D
    data = pgc.field(dtype=pgc.f32, shape=(n,))
    out = pgc.field(dtype=pgc.f32, shape=(1,))

    # Fill with a known pattern: value = x + y*4 + z*16
    arr = np.arange(n, dtype=np.float32)
    data.from_numpy(arr)

    tex = pgc.texture3d(data, shape=(W, H, D))

    @pgc.kernel
    def sample_center(out, tex, count):
        for i in range(count):
            # Sample at center of texel (0,0,0) → normalized (0, 0, 0)
            out[i] = tex.sample(0.0, 0.0, 0.0)

    sample_center(out, tex, 1)
    result = out.to_numpy()[0]
    # At (0,0,0) the software trilinear should return the corner value
    assert abs(result - 0.0) < 0.1


def test_texture_sample_in_func(backend):
    """tex.sample() inside a @pgc.func should work after inlining."""
    W, H, D = 4, 4, 4
    n = W * H * D
    data = pgc.field(dtype=pgc.f32, shape=(n,))
    out = pgc.field(dtype=pgc.f32, shape=(1,))

    # Fill with constant 7.0
    data.from_numpy(np.full(n, 7.0, dtype=np.float32))
    tex = pgc.texture3d(data, shape=(W, H, D))

    @pgc.func
    def do_sample(t, u, v, w):
        return t.sample(u, v, w)

    @pgc.kernel
    def sample_via_func(out, tex, count):
        for i in range(count):
            out[i] = do_sample(tex, 0.5, 0.5, 0.5)

    sample_via_func(out, tex, 1)
    result = out.to_numpy()[0]
    # Uniform field → sample anywhere gives 7.0
    np.testing.assert_allclose(result, 7.0, rtol=1e-5)


def test_texture_sample_in_nested_func(backend):
    """tex.sample() in a @pgc.func called by another @pgc.func."""
    W, H, D = 4, 4, 4
    n = W * H * D
    data = pgc.field(dtype=pgc.f32, shape=(n,))
    out = pgc.field(dtype=pgc.f32, shape=(1,))

    data.from_numpy(np.full(n, 3.0, dtype=np.float32))
    tex = pgc.texture3d(data, shape=(W, H, D))

    @pgc.func
    def inner_sample(t, u, v, w):
        return t.sample(u, v, w)

    @pgc.func
    def outer_sample(t, u, v, w):
        return inner_sample(t, u, v, w) * 2.0

    @pgc.kernel
    def nested(out, tex, count):
        for i in range(count):
            out[i] = outer_sample(tex, 0.5, 0.5, 0.5)

    nested(out, tex, 1)
    result = out.to_numpy()[0]
    np.testing.assert_allclose(result, 6.0, rtol=1e-5)


def test_texture_with_other_field_args(backend):
    """tex.sample() in @pgc.func alongside regular field parameters."""
    W, H, D = 4, 4, 4
    n = W * H * D
    data = pgc.field(dtype=pgc.f32, shape=(n,))
    coords = pgc.field(dtype=pgc.f32, shape=(3,))
    out = pgc.field(dtype=pgc.f32, shape=(1,))

    data.from_numpy(np.full(n, 5.0, dtype=np.float32))
    coords.from_numpy(np.array([0.5, 0.5, 0.5], dtype=np.float32))
    tex = pgc.texture3d(data, shape=(W, H, D))

    @pgc.func
    def sample_at(t, c):
        return t.sample(c[0], c[1], c[2])

    @pgc.kernel
    def sample_coords(out, tex, coords, count):
        for i in range(count):
            out[i] = sample_at(tex, coords)

    sample_coords(out, tex, coords, 1)
    result = out.to_numpy()[0]
    np.testing.assert_allclose(result, 5.0, rtol=1e-5)
