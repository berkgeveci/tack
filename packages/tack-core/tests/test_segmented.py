"""Segmented operations (``tack.algorithms.segmented``), checked against NumPy.

Segments are CSR offsets over flat values. The shapes are the ones the layer is
for and the ones that break such layers: many short segments, empty ones (at
the start, between, at the end), one enormous segment over several levels of
chunks, heavily skewed lengths, and lengths at and around the 256-element chunk
and short-segment limit. Integers make every reduction exact, so each case is
held to NumPy exactly on every available backend.
"""

import itertools

import numpy as np
import pytest

import tack
from tack.algorithms import segment_broadcast, segment_ids, segment_reduce, segment_scan

pytestmark = pytest.mark.quick

_RNG = np.random.default_rng(7)
SHAPES = {
    "many short, some empty": _RNG.integers(0, 5, 3000),
    "one enormous": np.array([400_000]),                  # three levels of chunks
    "enormous between empties": np.array([0, 0, 70_000, 0, 3, 0, 200_000, 0]),
    "skewed": np.r_[_RNG.integers(0, 3, 500), [90_000], _RNG.integers(0, 700, 200), [0]],
    "around the chunk": _RNG.choice([0, 1, 255, 256, 257, 511, 512, 513], 300),
    "all empty": np.array([0, 0, 0]),
    "none": np.zeros(0, int),
}
_OPS = {"sum": (np.add, 0), "min": (np.minimum, np.iinfo(np.int32).max),
        "max": (np.maximum, np.iinfo(np.int32).min)}


def _case(lengths, seed=0):
    lengths = np.asarray(lengths, np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32)
    values = np.random.default_rng(seed).integers(-1000, 1000, int(offsets[-1])).astype(np.int32)
    return lengths, offsets, values


def _per_segment(offsets, f):
    return [f(a, b) for a, b in itertools.pairwise(offsets)]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("op", _OPS)
@pytest.mark.parametrize("ordered", [False, True])
def test_reductions_are_numpys(backend, shape, op, ordered):
    lengths, offsets, values = _case(SHAPES[shape])
    if ordered and backend != "cpu" and lengths.max(initial=0) > 100_000:
        pytest.skip("ordered gives a long segment one GPU thread: what the layer avoids")
    f, identity = _OPS[op]
    expected = np.array(_per_segment(offsets, lambda a, b: f.reduce(values[a:b])
                                     if b > a else identity), np.int32)
    got = segment_reduce(tack.array(values), tack.array(offsets), op=op, ordered=ordered)
    np.testing.assert_array_equal(got.numpy(), expected.reshape(-1))


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("op", _OPS)
def test_scans_restart_at_every_segment(backend, shape, op):
    lengths, offsets, values = _case(SHAPES[shape])
    f, _ = _OPS[op]
    parts = [f.accumulate(values[a:b]) for a, b in itertools.pairwise(offsets) if b > a]
    expected = np.concatenate(parts) if parts else np.zeros(0, np.int32)
    got = segment_scan(tack.array(values), tack.array(offsets), op=op)
    np.testing.assert_array_equal(got.numpy(), expected)
    if op == "sum":
        parts = [np.cumsum(values[a:b]) - values[a:b]
                 for a, b in itertools.pairwise(offsets) if b > a]
        exclusive = np.concatenate(parts) if parts else np.zeros(0, np.int32)
        np.testing.assert_array_equal(
            segment_scan(tack.array(values), tack.array(offsets), inclusive=False).numpy(),
            exclusive)
        in_place = tack.array(values)
        segment_scan(in_place, tack.array(offsets), out=in_place)
        np.testing.assert_array_equal(in_place.numpy(), expected)


@pytest.mark.parametrize("shape", SHAPES)
def test_broadcast_and_ids_spread_each_segment_over_its_elements(backend, shape):
    lengths, offsets, values = _case(SHAPES[shape])
    n = int(offsets[-1])
    np.testing.assert_array_equal(segment_ids(tack.array(offsets), n).numpy(),
                                  np.repeat(np.arange(len(lengths)), lengths))
    sums = segment_reduce(tack.array(values), tack.array(offsets))
    np.testing.assert_array_equal(segment_broadcast(sums, tack.array(offsets), n).numpy(),
                                  np.repeat(sums.numpy(), lengths))


def test_elements_outside_every_segment_are_left_alone(backend):
    """Offsets need not start at 0 nor end at the last value: ids are -1 there, and a
    scan or broadcast leaves the elements as they are."""
    offsets = tack.array(np.array([300, 310, 310, 1000], np.int32))
    values = np.arange(1200, dtype=np.int32)
    ids = segment_ids(offsets, 1200).numpy()
    assert (ids[:300] == -1).all() and (ids[1000:] == -1).all()
    np.testing.assert_array_equal(ids[300:1000], np.repeat([0, 2], [10, 690]))
    out = tack.array(np.full(1200, 7, np.int32))
    segment_scan(tack.array(values), offsets, out=out)
    assert (out.numpy()[:300] == 7).all() and (out.numpy()[1000:] == 7).all()
    np.testing.assert_array_equal(segment_reduce(tack.array(values), offsets).numpy(),
                                  [values[300:310].sum(), 0, values[310:1000].sum()])


def test_vectors_reduce_componentwise(backend):
    lengths, offsets, _ = _case(np.r_[_RNG.integers(0, 4, 200), [5000], [0]])
    values = np.random.default_rng(1).integers(-50, 50, (int(offsets[-1]), 3)).astype(np.float32)
    array = tack.empty((len(values),), tack.vec(3, tack.f32))
    tack.copyto(array, values)
    for op, (f, _) in _OPS.items():
        identity = {"sum": 0.0, "min": np.inf, "max": -np.inf}[op]
        expected = np.array([f.reduce(values[a:b], axis=0) if b > a else np.full(3, identity)
                             for a, b in itertools.pairwise(offsets)])
        np.testing.assert_array_equal(segment_reduce(array, tack.array(offsets), op=op).numpy(),
                                      expected)


def test_empty_segments_take_the_identity_or_what_is_given(backend):
    offsets = tack.array(np.array([0, 0, 2, 2], np.int32))
    values = tack.array(np.array([1.5, 2.5], np.float32))
    assert segment_reduce(values, offsets).numpy().tolist() == [0.0, 4.0, 0.0]
    assert segment_reduce(values, offsets, op="min").numpy().tolist() == [np.inf, 1.5, np.inf]
    assert segment_reduce(values, offsets, op="max", empty=-1.0).numpy().tolist() == [-1.0, 2.5,
                                                                                       -1.0]


def test_float_sums_are_the_same_bits_everywhere(backend):
    """Chunks group a long segment's float sum the same way on every backend; a
    short segment is summed in element order, as ``ordered=True`` sums."""
    values = np.random.default_rng(3).random(1_000_003).astype(np.float32)
    offsets = np.array([0, 17, 500_000, 1_000_003], np.int32)
    with tack.device("cpu"):
        reference = segment_reduce(tack.array(values), tack.array(offsets)).numpy()
    got = segment_reduce(tack.array(values), tack.array(offsets)).numpy()
    np.testing.assert_array_equal(got.view(np.uint32), reference.view(np.uint32))
    ordered = segment_reduce(tack.array(values), tack.array(offsets), ordered=True).numpy()
    assert got[0] == ordered[0]                                   # short: element order
    np.testing.assert_allclose(got, ordered, rtol=1e-4)


def test_the_arguments_are_checked():
    values = tack.array(np.arange(4, dtype=np.int32))
    with pytest.raises(ValueError, match="op is one of"):
        segment_reduce(values, tack.array(np.array([0, 4], np.int32)), op="median")
    with pytest.raises(ValueError, match="exclusive scan is a sum's"):
        segment_scan(values, tack.array(np.array([0, 4], np.int32)), op="min", inclusive=False)
    with pytest.raises(TypeError, match="i32 or i64"):
        segment_reduce(values, tack.array(np.array([0.0, 4.0], np.float32)))
    with pytest.raises(ValueError, match="at least one entry"):
        segment_reduce(values, tack.empty((0,), tack.i32))
    with pytest.raises(ValueError, match="out has"):
        segment_reduce(values, tack.array(np.array([0, 2, 4], np.int32)),
                       out=tack.empty((1,), tack.i32))


def test_debug_mode_checks_the_offsets():
    from tack.runtime import dispatch

    was = dispatch._debug
    tack.set_debug(True)
    try:
        with pytest.raises(ValueError, match="non-decreasing"):
            segment_reduce(tack.array(np.arange(4, dtype=np.int32)),
                           tack.array(np.array([0, 3, 2, 4], np.int32)))
    finally:
        tack.set_debug(was)


@pytest.mark.parametrize("shape", SHAPES)
def test_means_and_keeping_what_empty_segments_hold(backend, shape):
    """``op="mean"``: the sum over the count, a long segment's divided as it is placed;
    ``empty="keep"`` leaves an empty segment's ``out`` as it was (a point with no
    cells keeps its value), where by default a floating mean is NaN."""
    lengths, offsets, values = _case(SHAPES[shape])
    floats = values.astype(np.float32)                 # every backend has f32
    expected = np.array([floats[a:b].astype(np.float64).mean() if b > a else np.nan
                         for a, b in itertools.pairwise(offsets)])
    got = segment_reduce(tack.array(floats), tack.array(offsets), op="mean").numpy()
    np.testing.assert_allclose(got, expected.reshape(-1), rtol=1e-5, atol=1e-3)
    out = tack.array(np.full(len(lengths), -7.0, np.float32))
    segment_reduce(tack.array(floats), tack.array(offsets), op="mean", out=out, empty="keep")
    kept = np.where(lengths > 0, expected.reshape(-1), -7.0) if len(lengths) else expected
    np.testing.assert_allclose(out.numpy(), kept, rtol=1e-5, atol=1e-3)
