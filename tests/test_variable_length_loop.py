"""Tests for sequential for-loops with runtime bounds from field loads.

This validates the pattern used by Viskores-style variable-length cell
connectivity: for i in range(offsets[c], offsets[c+1]).
"""

import numpy as np
import pytest
import pgc


def _available_backends():
    backends = ["cpu"]
    for arch in ["metal", "cuda", "hip", "vulkan", "level_zero"]:
        try:
            pgc.init(arch=arch)
            backends.append(arch)
        except (ImportError, RuntimeError, OSError):
            pass
    pgc.init(arch="cpu")
    return backends

_backends = _available_backends()


@pytest.fixture(params=_backends)
def backend(request):
    try:
        pgc.init(arch=request.param)
    except (ImportError, RuntimeError, OSError) as e:
        pytest.skip(f"{request.param} not available: {e}")
    return request.param


def test_sum_variable_length(backend):
    """Sum variable-length segments using offsets from a field."""

    @pgc.kernel
    def sum_segments(offsets, data, output, n):
        for c in range(n):
            start = offsets[c]
            end = offsets[c + 1]
            total = 0.0
            for i in range(start, end):
                total = total + data[i]
            output[c] = total

    # 3 segments: 2 elements, 3 elements, 4 elements
    offsets_np = np.array([0, 2, 5, 9], dtype=np.int32)
    data_np = np.array([1, 2, 10, 20, 30, 100, 200, 300, 400], dtype=np.float32)

    offsets = pgc.field(dtype=pgc.i32, shape=(4,))
    data = pgc.field(dtype=pgc.f32, shape=(9,))
    output = pgc.field(dtype=pgc.f32, shape=(3,))
    offsets.from_numpy(offsets_np)
    data.from_numpy(data_np)

    sum_segments(offsets, data, output, 3)

    expected = np.array([3.0, 60.0, 1000.0], dtype=np.float32)
    np.testing.assert_allclose(output.to_numpy(), expected)


def test_range_from_single_field_load(backend):
    """for i in range(count[c]) where count comes from a field."""

    @pgc.kernel
    def fill_counted(counts, output, n):
        for c in range(n):
            total = 0.0
            for i in range(counts[c]):
                total = total + 1.0
            output[c] = total

    counts_np = np.array([0, 5, 3, 10], dtype=np.int32)
    counts = pgc.field(dtype=pgc.i32, shape=(4,))
    output = pgc.field(dtype=pgc.f32, shape=(4,))
    counts.from_numpy(counts_np)

    fill_counted(counts, output, 4)

    np.testing.assert_allclose(output.to_numpy(), counts_np.astype(np.float32))


def test_indirect_gather(backend):
    """Gather with indirection: read connectivity[offset+i] then data[conn_id]."""

    @pgc.kernel
    def gather_avg(offsets, connectivity, point_data, cell_data, n):
        for c in range(n):
            start = offsets[c]
            end = offsets[c + 1]
            count = end - start
            total = 0.0
            for i in range(start, end):
                pid = connectivity[i]
                total = total + point_data[pid]
            cell_data[c] = total / float(count)

    # 2 cells: cell 0 uses points [0,1,2], cell 1 uses points [1,2,3,4]
    offsets_np = np.array([0, 3, 7], dtype=np.int32)
    conn_np = np.array([0, 1, 2, 1, 2, 3, 4], dtype=np.int32)
    point_np = np.array([10, 20, 30, 40, 50], dtype=np.float32)

    offsets = pgc.field(dtype=pgc.i32, shape=(3,))
    conn = pgc.field(dtype=pgc.i32, shape=(7,))
    point_data = pgc.field(dtype=pgc.f32, shape=(5,))
    cell_data = pgc.field(dtype=pgc.f32, shape=(2,))
    offsets.from_numpy(offsets_np)
    conn.from_numpy(conn_np)
    point_data.from_numpy(point_np)

    gather_avg(offsets, conn, point_data, cell_data, 2)

    # cell 0: (10+20+30)/3 = 20, cell 1: (20+30+40+50)/4 = 35
    expected = np.array([20.0, 35.0], dtype=np.float32)
    np.testing.assert_allclose(cell_data.to_numpy(), expected)
