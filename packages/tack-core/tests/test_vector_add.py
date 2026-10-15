"""Test vector addition — simplest kernel, validates basic pipeline."""

import numpy as np

import tack


def test_vector_add(backend):
    n = 1024
    x = tack.empty((n,))
    y = tack.empty((n,))
    out = tack.empty((n,))

    np_x = np.arange(n, dtype=np.float32)
    np_y = np.ones(n, dtype=np.float32) * 2.0

    tack.copyto(x, np_x)
    tack.copyto(y, np_y)

    @tack.kernel
    def vector_add(x, y, out):
        for i in range(len(x)):
            out[i] = x[i] + y[i]

    vector_add(x, y, out)

    result = out.numpy()
    expected = np_x + np_y
    np.testing.assert_allclose(result, expected)


def test_field_basics(backend):
    f = tack.empty((10,))
    assert f.shape == (10,)
    assert f.dtype == tack.f32

    f.fill(3.14)
    arr = f.numpy()
    np.testing.assert_allclose(arr, 3.14, rtol=1e-6)

    new_data = np.arange(10, dtype=np.float32)
    tack.copyto(f, new_data)
    np.testing.assert_array_equal(f.numpy(), new_data)
