"""Vector and matrix constants read at a runtime index: one table load.

``FACES[4 * f + k]`` used to lower to a chain of conditional expressions,
one comparison per entry, inlined at every lookup. A kernel whose loop
stores to a field and does many such lookups killed Metal's compiler
service at pipeline creation (M1 Max, macOS 26). Now a typed or integer
constant read at a runtime index is an ``IRTableLoad``: one constant
array per table, one load per lookup, on every backend. An index outside
the table gives the last value, as the chain did. Untyped float
constants keep the chain: their components are weak literals that take
the precision of what they meet, which one array type cannot do.
"""

import subprocess

import numpy as np
import pytest
from compiler_tools import require_clang

import tack
from tack.codegen.cuda_gen import generate_cuda_source
from tack.codegen.hip_gen import generate_hip_source
from tack.codegen.msl_gen import generate_msl_source
from tack.codegen.opencl_gen import generate_opencl_source
from tack.lang.inspect_kernel import _prepare_ir
from tack.lang.ir_type_annotate import annotate_types

FACES = tack.constant((0, 4, 7, 3, 1, 2, 6, 5, 0, 1, 5, 4, 3, 7, 6, 2, 0, 3, 2, 1, 4, 5, 6, 7),
                      tack.i32)
KINDS = tack.constant((9, 9, 9, 9, 5, 5), tack.i32)
PRIMES = tack.constant((2, 3, 5, 7, 11, 13))                     # untyped: an i32 table
BIG = tack.constant((1, 2**40, -(2**35), 7))                      # untyped: an i64 table
UNSIGNED = tack.constant((1, 2**31, 4000000000, 5), tack.u32)
WEIGHTS = tack.constant((0.25, 0.5, 1.5, -2.0), tack.f32)
EXACT = tack.constant((0.1, 0.2, 0.3), tack.f64)
WEAK = tack.constant((0.1, 0.2, 0.3))                              # untyped floats: the chain
ROTATE = tack.constant(((0, 1, 0), (-1, 0, 0), (0, 0, 1)), tack.i32)


@tack.kernel
def _faces(conn, rows, kinds, owners):
    for c in range(conn.shape[0]):
        for f in range(6):
            k = c * 6 + f
            p0 = conn[c, FACES[4 * f]]
            p1 = conn[c, FACES[4 * f + 1]]
            p2 = conn[c, FACES[4 * f + 2]]
            p3 = conn[c, FACES[4 * f + 3]] if KINDS[f] == 9 else -1
            rows[k] = [p0, p1, p2, p3]
            kinds[k] = KINDS[f]
            owners[k] = c


def test_a_storing_loop_with_many_lookups(backend):
    """The kernel that crashed Metal's compiler service."""
    conn = tack.field(tack.i32, shape=(8, 8))
    conn.from_numpy(np.arange(64, dtype=np.int32).reshape(8, 8))
    rows = tack.Vector.field(4, tack.i32, shape=(48,))
    kinds = tack.field(tack.i32, shape=(48,))
    owners = tack.field(tack.i32, shape=(48,))
    _faces(conn, rows, kinds, owners)
    table = np.array(FACES).reshape(6, 4)
    quads = np.array(KINDS) == 9
    want = np.where(quads[None, :, None] | (np.arange(4) < 3),
                    np.arange(64).reshape(8, 8)[:, table], -1)
    np.testing.assert_array_equal(rows.to_numpy(vectors=True).reshape(8, 6, 4), want)
    np.testing.assert_array_equal(kinds.to_numpy().reshape(8, 6), np.tile(KINDS, (8, 1)))
    np.testing.assert_array_equal(owners.to_numpy(), np.repeat(np.arange(8), 6))


@tack.kernel
def _lookups(index, primes, big, unsigned, weights):
    for i in range(index.shape[0]):
        k = index[i]
        primes[i] = PRIMES[k]
        big[i] = BIG[k]
        unsigned[i] = UNSIGNED[k]
        weights[i] = WEIGHTS[k]


def _expected(table, indices):
    """The chain's rule: an index outside the table gives the last value."""
    values = np.array(table)
    return np.array([values[k] if 0 <= k < len(values) else values[-1] for k in indices])


def test_every_table_type_and_out_of_range_indices(backend):
    indices = np.array([0, 1, 2, 3, 5, 6, 40, -1, -7, 2**31 - 1], np.int32)
    index = tack.field(tack.i32, shape=indices.shape)
    index.from_numpy(indices)
    primes = tack.field(tack.i32, shape=indices.shape)
    big = tack.field(tack.i64, shape=indices.shape)
    unsigned = tack.field(tack.u32, shape=indices.shape)
    weights = tack.field(tack.f32, shape=indices.shape)
    _lookups(index, primes, big, unsigned, weights)
    np.testing.assert_array_equal(primes.to_numpy(), _expected(PRIMES, indices))
    np.testing.assert_array_equal(big.to_numpy(), _expected(BIG, indices))
    np.testing.assert_array_equal(unsigned.to_numpy(), _expected(UNSIGNED, indices))
    np.testing.assert_array_equal(weights.to_numpy(), _expected(WEIGHTS, indices).astype(np.float32))


@tack.kernel
def _f64_lookups(index, exact, weak):
    for i in range(index.shape[0]):
        exact[i] = EXACT[index[i]]
        weak[i] = WEAK[index[i]] + tack.f64(0.0)


def test_f64_tables_and_weak_float_constants(f64_backend):
    indices = np.array([0, 1, 2, 9], np.int32)
    index = tack.field(tack.i32, shape=(4,))
    index.from_numpy(indices)
    exact = tack.field(tack.f64, shape=(4,))
    weak = tack.field(tack.f64, shape=(4,))
    _f64_lookups(index, exact, weak)
    np.testing.assert_array_equal(exact.to_numpy(), [0.1, 0.2, 0.3, 0.3])
    # Weak literals met an f64: they are the exact doubles, as before.
    np.testing.assert_array_equal(weak.to_numpy(), [0.1, 0.2, 0.3, 0.3])


@tack.kernel
def _matrix(rows, cols, out):
    for i in range(rows.shape[0]):
        out[i] = ROTATE[rows[i], cols[i]]


def test_matrix_entries_at_a_runtime_row_and_column(backend):
    r = np.array([0, 0, 1, 1, 2, 2, 1], np.int32)
    c = np.array([0, 1, 0, 2, 2, 1, 1], np.int32)
    rows = tack.field(tack.i32, shape=r.shape)
    cols = tack.field(tack.i32, shape=c.shape)
    rows.from_numpy(r)
    cols.from_numpy(c)
    out = tack.field(tack.i32, shape=r.shape)
    _matrix(rows, cols, out)
    np.testing.assert_array_equal(out.to_numpy(), np.array(ROTATE)[r, c])


# ── The IR and the generated source ─────────────────────────────────

def _prepared(kernel, *args):
    """The kernel's IR for these arguments, through inspection's pipeline, annotated."""
    func, _ = _prepare_ir(kernel, args)
    annotate_types(func)
    return func


def _faces_args():
    return (tack.field(tack.i32, shape=(8, 8)), tack.Vector.field(4, tack.i32, shape=(48,)),
            tack.field(tack.i32, shape=(48,)), tack.field(tack.i32, shape=(48,)))


def test_which_lookups_become_tables():
    tack.init(arch=tack.cpu)
    text = tack.inspect(_faces, *_faces_args(), mode="ir")
    assert "Table[i32]" in text and "IfExp((__eval" not in text
    index = tack.field(tack.i32, shape=(4,))
    text = tack.inspect(_f64_lookups, index, tack.field(tack.f64, shape=(4,)),
                        tack.field(tack.f64, shape=(4,)), mode="ir")
    assert text.count("Table[f64]") == 1      # EXACT; WEAK keeps its chain
    assert "IfExp(" in text


@pytest.mark.parametrize("generate, qualifier", [
    (generate_cuda_source, "__device__ __constant__ int"),
    (generate_hip_source, "__device__ __constant__ int"),
    (generate_opencl_source, "__constant int"),
    (generate_msl_source, "constant int"),
], ids=["cuda", "hip", "opencl", "metal"])
def test_each_table_is_declared_once(generate, qualifier):
    tack.init(arch=tack.cpu)
    source = generate(_prepared(_faces, *_faces_args()))
    assert source.count(f"{qualifier} __tack_table_0__[24]") == 1
    assert source.count(f"{qualifier} __tack_table_1__[6]") == 1
    assert "__tack_table_2__" not in source
    assert source.count("__tack_table_0__[(") == 4


def test_opencl_tables_compile(tmp_path):
    clang = require_clang()
    tack.init(arch=tack.cpu)
    index = tack.field(tack.i32, shape=(4,))
    args = (index, tack.field(tack.i32, shape=(4,)), tack.field(tack.i64, shape=(4,)),
            tack.field(tack.u32, shape=(4,)), tack.field(tack.f32, shape=(4,)))
    for kernel, kernel_args in ((_faces, _faces_args()), (_lookups, args)):
        path = tmp_path / f"{kernel.name}.cl"
        path.write_text(generate_opencl_source(_prepared(kernel, *kernel_args)))
        result = subprocess.run(
            [clang, "-target", "x86_64-unknown-linux-gnu", "-x", "cl", "-cl-std=CL2.0",
             "-fsyntax-only", str(path)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
