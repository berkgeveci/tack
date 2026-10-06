"""``tack.random``: a counter-based generator with explicit state, mirrored in NumPy."""

import numpy as np

import tack
from tack import random


@tack.kernel
def draw(out, dirs, n, frame):
    for i in range(n):
        state = random.seed(i, frame)
        u, state = random.uniform(state)
        z, state = random.normal(state)
        d, state = random.direction3(state)
        e, state = random.direction2(state)
        out[i, 0] = u
        out[i, 1] = z
        out[i, 2] = tack.f32(state)
        out[i, 3] = e.x
        dirs[i] = d


def _draw(backend_n, frame):
    out = tack.field(dtype=tack.f32, shape=(backend_n, 4))
    dirs = tack.Vector.field(3, dtype=tack.f32, shape=(backend_n,))
    draw(out, dirs, backend_n, frame)
    return out.to_numpy(), dirs.to_numpy(vectors=True)


def test_kernels_draw_what_the_numpy_mirrors_draw(backend):
    """``uniform`` and the states are bit-exact on every backend: integer
    arithmetic and a power-of-two scaling. ``normal`` and the directions
    go through log, sqrt, cos and sin, so they agree to those functions'
    rounding (3.6e-7 seen on Metal)."""
    n = 20000
    got, got_dirs = _draw(n, 7)
    state = random.np_seed(np.arange(n, dtype=np.uint32), 7)
    u, state = random.np_uniform(state)
    z, state = random.np_normal(state)
    d, state = random.np_direction3(state)
    e, state = random.np_direction2(state)
    np.testing.assert_array_equal(got[:, 0], u)
    np.testing.assert_array_equal(got[:, 2], state.astype(np.float32))
    np.testing.assert_allclose(got[:, 1], z, atol=2e-6)
    np.testing.assert_allclose(got_dirs, d, atol=2e-6)
    np.testing.assert_allclose(got[:, 3], e[:, 0], atol=2e-6)


def test_the_draws_are_distributed_as_named(backend):
    n = 100000
    got, dirs = _draw(n, 3)
    u, z = got[:, 0].astype(np.float64), got[:, 1].astype(np.float64)
    assert 0.495 < u.mean() < 0.505 and 0.080 < u.var() < 0.087       # 1/12 is 0.0833
    assert abs(z.mean()) < 0.02 and 0.97 < z.var() < 1.03
    assert np.all((u >= 0) & (u < 1))
    np.testing.assert_allclose(np.linalg.norm(dirs, axis=1), 1.0, atol=1e-6)
    assert abs(dirs[:, 2].mean()) < 0.02                              # uniform on the sphere
    assert abs(got[:, 3].astype(np.float64).mean()) < 0.02


def test_streams_and_elements_differ_and_repeat(backend):
    a, _ = _draw(1000, 1)
    b, _ = _draw(1000, 2)
    again, _ = _draw(1000, 1)
    np.testing.assert_array_equal(a, again)                           # a pure function
    assert (a[:, 0] != b[:, 0]).mean() > 0.99                         # streams differ
    assert len(np.unique(a[:, 0])) > 990                              # elements differ
