"""Tests for pgc.local_array — per-thread private arrays."""

import numpy as np
import pytest
import pgc


# Test on all available backends
def _available_backends():
    backends = []
    for arch in ["cpu", "metal", "cuda", "hip", "level_zero"]:
        try:
            pgc.init(arch=arch)
            backends.append(arch)
        except (ImportError, RuntimeError, OSError):
            pass
    if backends:
        pgc.init(arch=backends[0])
    return backends

_backends = _available_backends()


@pytest.fixture(params=_backends)
def backend(request):
    try:
        pgc.init(arch=request.param)
    except (ImportError, RuntimeError, OSError) as e:
        pytest.skip(f"{request.param} not available: {e}")
    return request.param


def test_local_array_store_load(backend):
    """Write to local array, read back, write to output."""
    n = 64
    out = pgc.field(dtype=pgc.f32, shape=(n,))

    @pgc.kernel
    def use_local(out, n):
        for i in range(n):
            arr = pgc.local_array(pgc.f32, 4)
            arr[0] = 1.0
            arr[1] = 2.0
            arr[2] = 3.0
            arr[3] = 4.0
            out[i] = arr[0] + arr[1] + arr[2] + arr[3]

    use_local(out, n)
    result = out.to_numpy()
    np.testing.assert_allclose(result, 10.0)


def test_local_array_loop_fill(backend):
    """Fill local array in a loop, sum it."""
    n = 32
    out = pgc.field(dtype=pgc.f32, shape=(n,))

    @pgc.kernel
    def loop_fill(out, n):
        for i in range(n):
            buf = pgc.local_array(pgc.f32, 8)
            for k in range(8):
                buf[k] = float(k) * 0.5
            total = 0.0
            for k in range(8):
                total = total + buf[k]
            out[i] = total

    loop_fill(out, n)
    expected = sum(k * 0.5 for k in range(8))
    np.testing.assert_allclose(out.to_numpy(), expected, rtol=1e-5)


def test_local_array_int(backend):
    """Local array with integer type."""
    n = 16
    out = pgc.field(dtype=pgc.i32, shape=(n,))

    @pgc.kernel
    def int_local(out, n):
        for i in range(n):
            idx = pgc.local_array(pgc.i32, 3)
            idx[0] = 10
            idx[1] = 20
            idx[2] = 30
            out[i] = idx[0] + idx[1] + idx[2]

    int_local(out, n)
    np.testing.assert_array_equal(out.to_numpy(), 60)


def test_local_array_with_template_size(backend):
    """Local array size from a template parameter (compile-time constant)."""
    n = 16
    out = pgc.field(dtype=pgc.f32, shape=(n,))

    @pgc.data_oriented
    class Config:
        size = 5  # class variable → compile-time constant (needed for local_array size)

    @pgc.kernel
    def tmpl_local(cfg: pgc.template(), out, n):
        for i in range(n):
            buf = pgc.local_array(pgc.f32, cfg.size)
            for k in range(cfg.size):
                buf[k] = float(k)
            total = 0.0
            for k in range(cfg.size):
                total = total + buf[k]
            out[i] = total

    cfg = Config()
    tmpl_local(cfg, out, n)
    expected = sum(range(5))
    np.testing.assert_allclose(out.to_numpy(), float(expected), rtol=1e-5)


def test_local_array_passed_to_func(backend):
    """Pass a local array to a @pgc.func that fills it."""
    n = 16
    out = pgc.field(dtype=pgc.f32, shape=(n,))

    @pgc.func
    def fill_buf(arr, size):
        for k in range(size):
            arr[k] = float(k) * 10.0

    @pgc.kernel
    def use_func(out, n):
        for i in range(n):
            buf = pgc.local_array(pgc.f32, 4)
            fill_buf(buf, 4)
            out[i] = buf[0] + buf[1] + buf[2] + buf[3]

    use_func(out, n)
    np.testing.assert_allclose(out.to_numpy(), 0 + 10 + 20 + 30)


def test_local_array_in_template_method(backend):
    """Pass a local array to a @pgc.data_oriented method."""
    n = 2
    out = pgc.field(dtype=pgc.f32, shape=(n,))

    @pgc.data_oriented
    class CellSet:
        points_per_cell = 4  # class variable → compile-time constant (needed for local_array size)

        def __init__(self, conn):
            self.connectivity = conn

        @pgc.func
        def get_cell_points(self, cell_id, pts):
            for v in range(self.points_per_cell):
                pts[v] = self.connectivity[cell_id * self.points_per_cell + v]

    conn_np = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32)
    data_np = np.array([10, 20, 30, 40, 50, 60, 70, 80], dtype=np.float32)

    conn = pgc.field(dtype=pgc.i32, shape=(8,))
    data = pgc.field(dtype=pgc.f32, shape=(8,))
    conn.from_numpy(conn_np)
    data.from_numpy(data_np)

    @pgc.kernel
    def avg(cs: pgc.template(), data, out, n_cells):
        for c in range(n_cells):
            pts = pgc.local_array(pgc.i32, cs.points_per_cell)
            cs.get_cell_points(c, pts)
            total = 0.0
            for v in range(cs.points_per_cell):
                total = total + data[pts[v]]
            out[c] = total / float(cs.points_per_cell)

    cs = CellSet(conn)
    avg(cs, data, out, 2)
    # cell 0: (10+20+30+40)/4 = 25, cell 1: (50+60+70+80)/4 = 65
    np.testing.assert_allclose(out.to_numpy(), [25.0, 65.0])


# --- local_array_like: inherit dtype from field ---

def test_local_array_like_f32(backend):
    """local_array_like inherits f32 from the field."""
    n = 10
    data = pgc.field(dtype=pgc.f32, shape=(n,))
    out = pgc.field(dtype=pgc.f32, shape=(n,))
    data.from_numpy(np.arange(n, dtype=np.float32))

    @pgc.kernel
    def kern(data, out):
        for i in range(data.shape[0]):
            buf = pgc.local_array_like(data, 4)
            buf[0] = data[i] * 2.0
            out[i] = buf[0]

    kern(data, out)
    np.testing.assert_allclose(out.to_numpy(), np.arange(n, dtype=np.float32) * 2.0)


def test_local_array_like_i32(backend):
    """local_array_like inherits i32 from the field."""
    n = 10
    data = pgc.field(dtype=pgc.i32, shape=(n,))
    out = pgc.field(dtype=pgc.i32, shape=(n,))
    data.from_numpy(np.arange(n, dtype=np.int32))

    @pgc.kernel
    def kern(data, out):
        for i in range(data.shape[0]):
            buf = pgc.local_array_like(data, 4)
            buf[0] = data[i] * 2
            out[i] = buf[0]

    kern(data, out)
    np.testing.assert_array_equal(out.to_numpy(), np.arange(n, dtype=np.int32) * 2)


def test_local_array_like_in_func(backend):
    """local_array_like inside @pgc.func resolves mangled field names."""
    n = 10
    data = pgc.field(dtype=pgc.f32, shape=(n,))
    out = pgc.field(dtype=pgc.f32, shape=(n,))
    data.from_numpy(np.arange(n, dtype=np.float32))

    @pgc.func
    def process(data, out, idx):
        buf = pgc.local_array_like(data, 4)
        buf[0] = data[idx] * 3.0
        out[idx] = buf[0]

    @pgc.kernel
    def kern(data, out):
        for i in range(data.shape[0]):
            process(data, out, i)

    kern(data, out)
    np.testing.assert_allclose(out.to_numpy(), np.arange(n, dtype=np.float32) * 3.0)
