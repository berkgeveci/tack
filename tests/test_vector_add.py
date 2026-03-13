"""Test vector addition — simplest kernel, validates basic pipeline."""

import numpy as np
import pgc


def test_vector_add():
    pgc.init(arch=pgc.cpu)

    n = 1024
    x = pgc.field(dtype=pgc.f32, shape=(n,))
    y = pgc.field(dtype=pgc.f32, shape=(n,))
    out = pgc.field(dtype=pgc.f32, shape=(n,))

    np_x = np.arange(n, dtype=np.float32)
    np_y = np.ones(n, dtype=np.float32) * 2.0

    x.from_numpy(np_x)
    y.from_numpy(np_y)

    @pgc.kernel
    def vector_add(x, y, out):
        for i in range(len(x)):
            out[i] = x[i] + y[i]

    vector_add(x, y, out)

    result = out.to_numpy()
    expected = np_x + np_y
    np.testing.assert_allclose(result, expected)


def test_field_basics():
    f = pgc.field(dtype=pgc.f32, shape=(10,))
    assert f.shape == (10,)
    assert f.dtype == pgc.f32

    f.fill(3.14)
    arr = f.to_numpy()
    np.testing.assert_allclose(arr, 3.14, rtol=1e-6)

    new_data = np.arange(10, dtype=np.float32)
    f.from_numpy(new_data)
    np.testing.assert_array_equal(f.to_numpy(), new_data)
