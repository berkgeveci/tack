"""Debug mode: every field and array index is checked, one dimension at a time.

With ``tack.init(debug=True)`` an out-of-bounds access is skipped (a load
gives 0) and the launch raises ``IndexError`` naming the first one: kernel
or device function, file and line, source text, index and extent. In-bounds
kernels give the results they give without debug mode, and debug mode off
generates the code it always did.
"""

import threading

import numpy as np
import pytest

import tack

pytestmark = pytest.mark.quick      # the inner-loop set; see CLAUDE.md, Tests

_FILE = __file__


@pytest.fixture
def debug(backend):
    tack.init(arch=backend, debug=True)
    yield backend
    tack.init(arch=backend)


def _line(text):
    """The line of this file that ends with the comment ``text``."""
    lines = [n for n, line in enumerate(open(_FILE), 1) if line.rstrip().endswith(text)]
    assert len(lines) == 1, text
    return lines[0]


def _field(values, dtype=tack.f32):
    values = np.asarray(values)
    f = tack.field(dtype, values.shape)
    f.from_numpy(values.astype(dtype.numpy_dtype))
    return f


@tack.kernel
def _shift(x, y, k):
    for i in range(y.shape[0]):
        y[i] = x[i + k]  # shift load


@tack.kernel
def _right(a, b):
    for i, j in tack.ndrange(a.shape[0], a.shape[1]):
        b[i, j] = a[i, j + 1]  # right load


@tack.kernel
def _down(a, b, k):
    for i, j in tack.ndrange(b.shape[0], b.shape[1]):
        b[i, j] = a[i + k, j]  # down load


@tack.kernel
def _scatter(x, y, k):
    for i in range(x.shape[0]):
        y[i + k] = x[i]  # scatter store


def test_a_flat_index_past_the_end_raises_and_reads_zero(debug):
    x, y = _field(np.arange(8)), _field(np.full(8, -1))
    _shift(x, y, 0)
    np.testing.assert_array_equal(y.to_numpy(), np.arange(8))
    with pytest.raises(IndexError) as error:
        _shift(x, y, 2)
    # Which violation is first depends on the order threads ran in.
    message = str(error.value)
    assert message.startswith(
        f"Kernel '_shift': x[i + k] at {_FILE}:{_line('# shift load')}, column 16: index ")
    assert message.removeprefix(message[:message.index("index ")]) in (
        f"index {n} is outside [0, 8). The access was skipped (a load gives 0). "
        "1 more out-of-bounds access in the launch was skipped too." for n in (8, 9))
    np.testing.assert_array_equal(y.to_numpy(), [2, 3, 4, 5, 6, 7, 0, 0])
    # The record is cleared: a launch in bounds raises nothing.
    _shift(x, y, 0)


def test_a_negative_index_raises(debug):
    x, y = _field(np.arange(8)), _field(np.zeros(8))
    with pytest.raises(IndexError, match=r"index -[123] is outside \[0, 8\)"):
        _shift(x, y, -3)
    np.testing.assert_array_equal(y.to_numpy(), [0, 0, 0, 0, 1, 2, 3, 4])


def test_each_dimension_is_checked_even_when_the_flat_index_is_inside(debug):
    """a[i, 4] of a 4x4 field is element (i + 1, 0): without debug mode it reads
    the next row silently. Debug mode names dimension 1."""
    a = _field(np.arange(16).reshape(4, 4))
    b = _field(np.full((4, 4), -1))
    with pytest.raises(IndexError) as error:
        _right(a, b)
    assert (f"Kernel '_right': a[i, j + 1] at {_FILE}:{_line('# right load')}, column 19: "
            "index 4 of dimension 1 is outside [0, 4)") in str(error.value)
    assert "3 more out-of-bounds accesses" in str(error.value)
    expected = np.arange(16).reshape(4, 4) + 1
    expected[:, 3] = 0
    np.testing.assert_array_equal(b.to_numpy(), expected)


def test_the_leading_dimension_is_checked_without_compiling_per_length(debug):
    from tack.runtime.dispatch import get_backend
    for rows in (3, 5, 9):
        a, b = _field(np.ones((rows, 2))), _field(np.zeros((rows, 2)))
        _down(a, b, 0)
        np.testing.assert_array_equal(b.to_numpy(), np.ones((rows, 2)))
        with pytest.raises(IndexError, match=rf"index {rows} of dimension 0 is outside "
                                             rf"\[0, {rows}\)"):
            _down(a, b, 1)
    assert len(get_backend()._cache.get(_down)) == 1


def test_an_out_of_bounds_store_is_skipped(debug):
    """The stores past the end would land in whatever follows the field."""
    x = _field(np.arange(4))
    y = _field(np.zeros(4))
    after = _field(np.full(4, 7))
    with pytest.raises(IndexError, match=rf"y\[i \+ k\] at {_FILE}:{_line('# scatter store')}"
                                         r", column 9: index [45] is outside \[0, 4\)"):
        _scatter(x, y, 2)
    np.testing.assert_array_equal(y.to_numpy(), [0, 0, 0, 1])
    np.testing.assert_array_equal(after.to_numpy(), np.full(4, 7))


@tack.kernel
def _count(keys, hist):
    for i in range(keys.shape[0]):
        tack.atomic_add(hist, keys[i], 1)


@tack.kernel
def _count_old(keys, hist, olds):
    for i in range(keys.shape[0]):
        olds[i] = tack.atomic_add(hist, keys[i], 1) + 100


def test_atomics_are_checked_as_statements_and_as_values(debug):
    keys = _field([0, 1, 4, 1], tack.i32)
    hist = _field(np.zeros(4), tack.i32)
    with pytest.raises(IndexError, match=r"tack\.atomic_add\(hist, keys\[i\], 1\) at .*: "
                                         r"index 4 is outside \[0, 4\)"):
        _count(keys, hist)
    np.testing.assert_array_equal(hist.to_numpy(), [1, 2, 0, 0])
    hist = _field(np.zeros(4), tack.i32)
    olds = _field(np.zeros(4), tack.i32)
    with pytest.raises(IndexError, match=r"index 4 is outside \[0, 4\)"):
        _count_old(keys, hist, olds)
    # A skipped atomic's value is 0.
    assert olds.to_numpy()[2] == 100
    assert sorted(olds.to_numpy()[[1, 3]]) == [100, 101]


@tack.kernel
def _bump(v, k):
    for i in range(v.shape[0]):
        v[i + k] += tack.Vector([1.0, 2.0, 3.0])


@tack.kernel
def _bump_component(v, flat):
    for i in range(1):
        tack.atomic_add(v, flat, 1.0)


def test_vector_fields_are_checked_by_element_and_by_component(debug):
    v = tack.Vector.field(3, tack.f32, (4,))
    v.from_numpy(np.zeros((4, 3), np.float32))
    with pytest.raises(IndexError, match=r"v\[i \+ k\] at .*: index 4 is outside \[0, 4\)"):
        _bump(v, 1)
    expected = np.zeros((4, 3))
    expected[1:] = [1, 2, 3]
    np.testing.assert_array_equal(v.to_numpy(vectors=True), expected)
    # A scalar atomic on a vector field takes a component's flat index.
    _bump_component(v, 11)
    assert v.to_numpy()[11] == 4
    with pytest.raises(IndexError, match=r"index 12 is outside \[0, 12\)"):
        _bump_component(v, 12)


@tack.kernel
def _local(out, k):
    for i in range(out.shape[0]):
        tmp = tack.local_array(tack.f32, 4)
        for j in range(4):
            tmp[j] = 1.0
        out[i] = tmp[i + k]  # local load


def test_local_arrays_are_checked_against_their_size(debug):
    out = _field(np.zeros(4))
    with pytest.raises(IndexError, match=rf"tmp\[i \+ k\] at {_FILE}:{_line('# local load')}, "
                                         r"column 18: index 4 is outside \[0, 4\)"):
        _local(out, 1)
    np.testing.assert_array_equal(out.to_numpy(), [1, 1, 1, 0])


@tack.func
def _tap(a, i, j):
    return a[i, j + 1]  # device tap


@tack.kernel
def _through_device(a, b):
    for i, j in tack.ndrange(a.shape[0], a.shape[1]):
        b[i, j] = _tap(a, i, j)


def test_an_access_in_a_device_function_names_it(debug):
    a, b = _field(np.ones((2, 2))), _field(np.zeros((2, 2)))
    with pytest.raises(IndexError) as error:
        _through_device(a, b)
    assert str(error.value).startswith(
        "Device function '_tap' (inlined into kernel '_through_device'): "
        f"a[i, j + 1] at {_FILE}:{_line('# device tap')}, column 12: "
        "index 2 of dimension 1 is outside [0, 2)")


@tack.kernel
def _guarded(x, out):
    for i in range(out.shape[0]):
        if i < x.shape[0] and x[i] > 0:
            out[i] = 1.0


def test_a_short_circuit_guard_keeps_the_access_from_being_checked(debug):
    x, out = _field(np.ones(4)), _field(np.zeros(8))
    _guarded(x, out)
    np.testing.assert_array_equal(out.to_numpy(), [1, 1, 1, 1, 0, 0, 0, 0])


@tack.device_class
class _Grid:
    def __init__(self, values):
        self.values = values

    @tack.func
    def at(self, i):
        return self.values[i]


@tack.kernel
def _sample(grid, out, k):
    for i in range(out.shape[0]):
        out[i] = grid.at(i + k)


def test_template_fields_are_checked(debug):
    grid, out = _Grid(_field(np.arange(4))), _field(np.zeros(4))
    with pytest.raises(IndexError, match=r"index 4 is outside \[0, 4\)"):
        _sample(grid, out, 1)
    np.testing.assert_array_equal(out.to_numpy(), [1, 2, 3, 0])


@tack.kernel
def _total(x) -> tack.f32:
    for i in range(x.shape[0]):
        tack.atomic_add(x, i + 1, 0.0)
    return x[0]


def test_a_kernel_returning_a_value_raises_for_its_loop(debug):
    with pytest.raises(IndexError, match=r"index 4 is outside \[0, 4\)"):
        _total(_field(np.ones(4)))


def test_without_debug_mode_nothing_is_checked(backend):
    tack.init(arch=backend, debug=False)
    a = _field(np.arange(16).reshape(4, 4))
    b = _field(np.zeros((4, 4)))
    _right(a, b)        # reads the next row's first element, silently
    assert b.to_numpy()[0, 3] == 4
    if backend == 'cpu':
        assert '__tack_in_bounds__' not in tack.inspect(_right, a, b, mode='source')
    tack.init(arch=backend)


def test_switching_modes_reuses_no_variant_of_the_other(backend):
    """A warm call reuses its variant by argument signature (the dispatch
    fast path); a call in the other mode with the same signature must not
    find it there. Debug variants take two hidden arguments, and the
    others check nothing."""
    x, y = _field(np.arange(8)), _field(np.zeros(8))
    try:
        tack.init(arch=backend, debug=False)
        for _ in range(2):
            _shift(x, y, 0)
        tack.init(arch=backend, debug=True)
        with pytest.raises(IndexError):
            _shift(x, y, 2)
        for _ in range(2):
            _shift(x, y, 0)
        np.testing.assert_array_equal(y.to_numpy(), np.arange(8))
        tack.init(arch=backend, debug=False)
        for _ in range(2):
            _shift(x, y, 0)
        np.testing.assert_array_equal(y.to_numpy(), np.arange(8))
    finally:
        tack.init(arch=backend)


def test_inspect_shows_the_checks(debug):
    a, b = _field(np.zeros((2, 2))), _field(np.zeros((2, 2)))
    assert '__tack_in_bounds__' in tack.inspect(_right, a, b, mode='source')
    # The IR is shown resolved: the leading extent is a load from the hidden
    # extents field, and dimension 1 is baked in.
    ir_text = tack.inspect(_right, a, b, mode='ir')
    assert 'a[Bounds[0](i < __tack_extents__[0], (j + 1) < 2; ((i * 2) + (j + 1)))]' in ir_text
    assert '<unknown' not in ir_text
    from tack.lang.ir import dump
    lowered = _right.get_ir(debug=True).functions[0]
    assert 'Bounds[0](i < Extent(a, leading), (j + 1) < DimSize(a, 1); ' in dump(lowered)


def test_debug_mode_from_the_environment(backend, monkeypatch):
    monkeypatch.setenv("TACK_DEBUG", "1")
    tack.init(arch=backend)
    try:
        with pytest.raises(IndexError):
            _shift(_field(np.arange(4)), _field(np.zeros(4)), 1)
    finally:
        monkeypatch.delenv("TACK_DEBUG")
        tack.init(arch=backend)
    _shift(_field(np.arange(4)), _field(np.zeros(4)), 1)


def test_a_wrong_argument_count_names_the_kernels_own_parameters(debug):
    with pytest.raises(TypeError, match="expects 3 arguments, got 2"):
        _shift(_field(np.zeros(4)), _field(np.zeros(4)))


def _bind_device_to_this_thread():
    """CUDA's context is current only on the thread that initialized Tack;
    a new thread must make it current before it allocates or launches."""
    from tack.runtime.dispatch import get_backend
    backend = get_backend()
    if backend.name == "cuda":
        from cuda.bindings import driver
        (err,) = driver.cuCtxSetCurrent(backend._context)
        assert err == driver.CUresult.CUDA_SUCCESS, err


def test_threads_report_their_own_launches(debug):
    errors = {}

    def run(k):
        _bind_device_to_this_thread()
        try:
            for _ in range(20):
                _shift(_field(np.arange(8)), _field(np.zeros(8)), k)
        except IndexError as e:
            errors[k] = str(e)

    threads = [threading.Thread(target=run, args=(k,)) for k in (0, 3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert set(errors) == {3}
    assert "is outside [0, 8)" in errors[3]


@tack.kernel
def _fill_second(a, b):
    for i in range(b.shape[0]):
        b[i] = a[0]


def test_each_parameter_has_its_own_extent_when_a_variant_is_reused(debug, monkeypatch):
    """A variant compiled for a call passing one field as both arguments checked
    the second against the first's extent ever after: the extents slot came
    from the field object, which only that call had in common. The CPU compiles
    such a call separately (``_SPECIALIZE_DISJOINT``), so it is turned off here,
    as GPU backends never had it; Metal's polyhedral contour found it."""
    import tack.runtime.cpu as cpu
    monkeypatch.setattr(cpu, "_SPECIALIZE_DISJOINT", False)
    shared = _field(np.full(2, 3.0))
    _fill_second(shared, shared)
    small, large = _field(np.full(2, 5.0)), _field(np.zeros(8))
    _fill_second(small, large)
    np.testing.assert_array_equal(large.to_numpy(), np.full(8, 5.0))
