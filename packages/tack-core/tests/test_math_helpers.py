"""``tack.math``: GLSL-style helpers as device functions, on scalars and vectors."""

import numpy as np
import pytest

import tack
from tack import math as tm
from tack.math import clamp, mix


@tack.kernel
def scalars(x, out, n):
    for i in range(n):
        v = x[i]
        out[i, 0] = tm.fract(v)
        out[i, 1] = tm.mix(2.0, 4.0, v)
        out[i, 2] = clamp(v, 0.25, 0.75)
        out[i, 3] = tm.saturate(v * 2.0 - 0.5)
        out[i, 4] = tm.smoothstep(0.2, 0.8, v)
        out[i, 5] = tm.smoothstep(0.8, 0.2, v)             # reversed edges reverse the step
        out[i, 6] = tm.step(0.5, v) + 10.0 * tm.sign(v)


@tack.kernel
def vectors(p, out, n):
    for i in range(n):
        v = p[i]
        w = [0.5, 0.5, 0.5]
        out[i, 0] = tm.fract(v * 3.0).sum()
        out[i, 1] = mix(v, w, 0.25).sum() + mix(v, w, [0.0, 0.5, 1.0]).sum()
        out[i, 2] = tm.clamp(v, 0.2, 0.4).sum()
        out[i, 3] = tm.smoothstep(0.0, 1.0, v).sum()
        out[i, 4] = tm.length(v) + tm.distance(v, w)
        out[i, 5] = tm.normalize(v).norm()
        out[i, 6] = tm.step(0.25, v).sum() + 10.0 * tm.sign(v - 0.25).sum() + tm.step(v, w).sum()


def _smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def test_helpers_on_scalars(backend):
    values = np.array([-1.3, 0.0, 0.1, 0.5, 0.9, 2.75], np.float32)
    x = tack.field(dtype=tack.f32, shape=(6,))
    x.from_numpy(values)
    out = tack.field(dtype=tack.f32, shape=(6, 7))
    scalars(x, out, 6)
    v = values.astype(np.float64)
    want = np.stack([v - np.floor(v), 2 * (1 - v) + 4 * v, np.clip(v, 0.25, 0.75),
                     np.clip(v * 2 - 0.5, 0, 1), _smoothstep(0.2, 0.8, v), _smoothstep(0.8, 0.2, v),
                     (v >= 0.5) + 10.0 * np.sign(v)], axis=1)
    np.testing.assert_allclose(out.to_numpy(), want, rtol=1e-6, atol=1e-6)


def test_helpers_on_vectors(backend):
    points = np.array([[0.1, 0.2, 0.3], [1.0, -2.0, 2.0]], np.float32)
    p = tack.Vector.field(3, dtype=tack.f32, shape=(2,))
    p.from_numpy(points)
    out = tack.field(dtype=tack.f32, shape=(2, 7))
    vectors(p, out, 2)
    w = np.full(3, 0.5)
    want = []
    for v in points.astype(np.float64):
        want.append([(v * 3 - np.floor(v * 3)).sum(),
                     (v * 0.75 + w * 0.25).sum() + (v * (1 - np.array([0, 0.5, 1])) + w * np.array([0, 0.5, 1])).sum(),
                     np.clip(v, 0.2, 0.4).sum(), _smoothstep(0.0, 1.0, v).sum(),
                     np.linalg.norm(v) + np.linalg.norm(v - w), 1.0,
                     (v >= 0.25).sum() + 10.0 * np.sign(v - 0.25).sum() + (w >= v).sum()])
    np.testing.assert_allclose(out.to_numpy(), want, rtol=1e-5, atol=1e-6)


def test_helpers_are_device_functions_only():
    with pytest.raises(RuntimeError, match="cannot be called from Python"):
        tm.fract(1.5)
