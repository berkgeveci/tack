"""Structure-of-arrays storage: a vector or matrix array kept one array per component.

`tack.soa(x, y, z)` wraps arrays that exist already; `layout="soa"` on a
constructor makes new ones. Either way it is an `Array`: kernels index it
as they index an interleaved vector array, and the host API reads and
writes it the same way. Only the generated code differs, reading
component c of element i from the c-th component array.
"""

import asyncio

import numpy as np
import pytest

import tack
from tack import algorithms as alg
from tack import arrays
from tack.runtime.kernel_utils import drop_variants

pytestmark = pytest.mark.quick


def _xyz(n=4):
    return (tack.arange(n, dtype=tack.f32), tack.array(10 + np.arange(n, dtype=np.float32)),
            tack.array(20 + np.arange(n, dtype=np.float32)))


def _expected(n=4):
    return np.stack([np.arange(n), 10 + np.arange(n), 20 + np.arange(n)], axis=-1)


@tack.kernel
def _scale(a, out, s):
    for i in range(a.shape[0]):
        out[i] = a[i] * s


@tack.kernel
def _update(a):
    for i in range(len(a)):
        a[i] += tack.vec3(1.0, 2.0, 3.0)
        a[i][1] = a[i].x * 100.0
        a[i].z += 0.5


@tack.kernel
def _set_component(a, c, value):
    for i in range(len(a)):
        a[i][c] = value


@tack.kernel
def _broadcast(a, value):
    for i in range(len(a)):
        a[i] = value


# --- Construction ---

def test_soa_wraps_its_components(backend):
    x, y, z = _xyz()
    p = tack.soa(x, y, z)
    assert isinstance(p, tack.Array) and p.layout == "soa"
    assert p.dtype == tack.vec3() and p.shape == (4,) and p.size == 4
    assert p.components[0] is x and p.components[2] is z
    assert p.device == x.device
    np.testing.assert_allclose(p.numpy(), _expected())
    assert "layout='soa'" in repr(p)


def test_constructors_take_a_layout(backend):
    for make in (lambda: tack.empty((3,), tack.vec2(), layout="soa"),
                 lambda: tack.zeros((3,), tack.vec2(), layout="soa"),
                 lambda: tack.ones((3,), tack.vec2(), layout="soa"),
                 lambda: tack.full((3,), [1.0, 2.0], tack.vec2(), layout="soa"),
                 lambda: tack.array(np.ones((3, 2), np.float32), tack.vec2(), layout="soa")):
        a = make()
        assert a.layout == "soa" and a.dtype == tack.vec2() and len(a.components) == 2
        assert all(c.shape == (3,) and c.dtype == tack.f32 for c in a.components)
    np.testing.assert_allclose(tack.zeros((3,), tack.vec2(), layout="soa").numpy(), 0)
    np.testing.assert_allclose(tack.full((2,), [1.0, 2.0], tack.vec2(), layout="soa").numpy(),
                               [[1, 2], [1, 2]])
    p = tack.zeros((3,), tack.vec2(), layout="soa")
    assert tack.empty_like(p).layout == "soa" and tack.ones_like(p).layout == "soa"
    assert tack.zeros_like(p, layout="aos").layout == "aos"
    assert tack.full_like(p, 7.0, dtype=tack.f64).layout == "soa"
    assert tack.zeros((3,), tack.vec2()).layout == "aos" and tack.zeros((3,)).layout == "aos"


def test_converting_between_layouts(backend):
    aos = tack.array(np.arange(12, dtype=np.float32).reshape(4, 3), tack.vec3())
    soa = tack.array(aos, layout="soa")
    assert soa.layout == "soa"
    np.testing.assert_allclose(soa.components[1].numpy(), [1, 4, 7, 10])
    back = tack.asarray(soa, layout="aos")
    assert back.layout == "aos"
    np.testing.assert_allclose(back.numpy(), aos.numpy())
    assert tack.asarray(soa) is soa and tack.asarray(soa, layout="soa") is soa
    out = tack.zeros((4,), tack.vec3(), layout="soa")
    tack.copyto(out, aos)
    np.testing.assert_allclose(out.numpy(), aos.numpy())
    tack.copyto(aos, tack.full((4,), [1.0, 2.0, 3.0], tack.vec3(), layout="soa"))
    np.testing.assert_allclose(aos.numpy(), [[1, 2, 3]] * 4)


def test_soa_refuses_what_it_cannot_hold(backend):
    with pytest.raises(ValueError, match="share a dtype and a shape"):
        tack.soa(tack.zeros((3,)), tack.zeros((4,)))
    with pytest.raises(ValueError, match="share a dtype and a shape"):
        tack.soa(tack.zeros((3,)), tack.zeros((3,), tack.i32))
    with pytest.raises(TypeError, match="arrays of scalars"):
        tack.soa(tack.zeros((3,), tack.vec2()))
    with pytest.raises(TypeError, match="takes Tack arrays, not ndarray"):
        tack.soa(np.zeros(3))
    with pytest.raises(TypeError, match="dtype must be a vector or matrix of 2"):
        tack.soa(tack.zeros((3,)), tack.zeros((3,)), dtype=tack.mat2())
    with pytest.raises(TypeError, match="has no components"):
        tack.empty((3,), layout="soa")
    with pytest.raises(ValueError, match="layout is 'aos' or 'soa'"):
        tack.empty((3,), tack.vec2(), layout="planar")
    if backend != "cpu":
        with pytest.raises(ValueError):
            tack.soa(tack.zeros((3,)), tack.zeros((3,), device="cpu"))


def test_matrices(backend):
    m = tack.soa(*[tack.full((3,), float(k)) for k in range(4)], dtype=tack.mat2())
    assert m.dtype == tack.mat2() and m.numpy().shape == (3, 2, 2)
    np.testing.assert_allclose(m.numpy()[0], [[0, 1], [2, 3]])

    @tack.kernel
    def transpose(a, out):
        for i in range(a.shape[0]):
            out[i] = a[i].transpose()
            out[i][0, 1] = a[i][1, 0] + 100.0

    out = tack.zeros((3,), tack.mat2(), layout="soa")
    transpose(m, out)
    np.testing.assert_allclose(out.numpy()[1], [[0, 102], [1, 3]])


# --- The host API ---

def test_host_reads_writes_and_reductions(backend):
    p = tack.soa(*_xyz())
    np.testing.assert_allclose(p[2], [2, 12, 22])
    np.testing.assert_allclose(asyncio.run(p.numpy_async()), _expected())
    np.testing.assert_allclose(np.asarray(p), _expected())
    assert p.sum() == _expected().sum() and p.mean() == _expected().mean()
    assert p.min() == 0 and p.max() == 23
    p.fill([1.0, 2.0, 3.0])
    np.testing.assert_allclose(p.components[2].numpy(), 3)
    p.fill(5.0)
    np.testing.assert_allclose(p.numpy(), 5)
    tack.copyto(p, _expected())
    np.testing.assert_allclose(p.numpy(), _expected())
    tack.copyto(p, [7.0, 8.0, 9.0])                     # broadcast, as np.copyto
    np.testing.assert_allclose(p.components[1].numpy(), 8)
    with pytest.raises(ValueError, match="elements of this array have shape"):
        p.fill([1.0, 2.0])


def test_copies_reshapes_and_conversions(backend):
    p = tack.soa(*_xyz())
    q = p.copy()
    assert q.layout == "soa" and q.components[0] is not p.components[0]
    np.testing.assert_allclose(q.numpy(), _expected())
    r = p.reshape(2, 2)
    assert r.shape == (2, 2) and r.layout == "soa"
    np.testing.assert_allclose(r.numpy(), _expected().reshape(2, 2, 3))
    if backend in ("cpu",):
        assert p.astype(tack.f64).components[0].dtype == tack.f64
    assert p.astype(tack.i32).dtype == tack.vec3(tack.i32)
    joined = tack.concat([p, p])
    assert joined.layout == "aos"
    np.testing.assert_allclose(joined.numpy(), np.concatenate([_expected()] * 2))
    assert p.to_device(p.device) is p


def test_there_is_no_one_buffer_to_export(backend):
    p = tack.soa(*_xyz())
    with pytest.raises(BufferError, match="export its components"):
        p.__dlpack__()
    with pytest.raises(BufferError, match="export its components"):
        p.export_memory()


def test_components_from_another_library_share_its_memory():
    tack.set_default_device("cpu")
    hx, hy = np.arange(3, dtype=np.float32), np.ones(3, np.float32)
    w = tack.soa(tack.from_dlpack(hx), tack.from_dlpack(hy))

    @tack.kernel
    def double(a):
        for i in range(len(a)):
            a[i] *= 2.0

    double(w)
    np.testing.assert_allclose(hx, [0, 2, 4])
    np.testing.assert_allclose(hy, [2, 2, 2])


# --- Kernels ---

def test_kernels_read_and_write_either_layout(backend):
    for layout in ("aos", "soa", "aos", "soa"):        # each its own variant
        a = tack.array(_expected(), tack.vec3(), layout=layout)
        out = tack.zeros((4,), tack.vec3(), layout=layout)
        _scale(a, out, 2.0)
        np.testing.assert_allclose(out.numpy(), 2 * _expected())
    out = tack.zeros((4,), tack.vec3())
    _scale(tack.soa(*_xyz()), out, 1.0)                # mixed layouts in one call
    np.testing.assert_allclose(out.numpy(), _expected())


def test_the_generated_code_reads_each_component_array():
    tack.set_default_device("cpu")
    ir = tack.inspect(_scale, tack.zeros((4,), tack.vec3(), layout="soa"),
                      tack.zeros((4,), tack.vec3()), 2.0, mode="ir")
    assert "scale(__soa0__a, __soa1__a, __soa2__a, out, s)" in ir
    assert "__soa1__a[Cast(i, i64)]" in ir


def test_stores_updates_and_components(backend):
    x, y, z = _xyz()
    _update(tack.soa(x, y, z))
    np.testing.assert_allclose(x.numpy(), np.arange(4) + 1)
    np.testing.assert_allclose(y.numpy(), 100 * (np.arange(4) + 1))
    np.testing.assert_allclose(z.numpy(), 20 + np.arange(4) + 3.5)
    _set_component(tack.soa(x, y, z), 2, -1.0)           # a runtime component
    np.testing.assert_allclose(z.numpy(), -1)
    _broadcast(tack.soa(x, y, z), 4.0)                    # a scalar sets every component
    np.testing.assert_allclose([x.numpy(), y.numpy(), z.numpy()], 4)


def test_atomics(backend):
    p = tack.zeros((2,), tack.vec3(), layout="soa")
    old = tack.zeros((3,), tack.vec3())

    @tack.kernel
    def accumulate(a, ids, old):
        for i in range(ids.shape[0]):
            old[i] = tack.atomic_add(a, ids[i], tack.vec3(1.0, 2.0, 3.0))

    accumulate(p, tack.array([0, 0, 1]), old)
    np.testing.assert_allclose(p.numpy(), [[2, 4, 6], [1, 2, 3]])
    assert sorted(old.numpy()[:2, 0].tolist()) == [0, 1]

    @tack.kernel
    def flat(a):
        for i in range(3):
            tack.atomic_add(a, 2 * i, 1.0)

    with pytest.raises(tack.lang.source_validation.UnsupportedSyntaxError,
                       match="has no flat component index; give it a 3-vector"):
        flat(p)


def test_multidimensional_arrays(backend):
    g = tack.zeros((2, 3), tack.vec2(), layout="soa")

    @tack.kernel
    def fill(g):
        for i, j in tack.ndindex(g.shape[0], g.shape[1]):
            g[i, j] = tack.vec2(tack.f32(i), tack.f32(j))

    fill(g)
    assert g.components[0].shape == (2, 3)
    np.testing.assert_allclose(g.numpy()[..., 0], [[0, 0, 0], [1, 1, 1]])
    np.testing.assert_allclose(g.numpy()[..., 1], [[0, 1, 2]] * 2)


@tack.func
def _norm2(v, i):
    return v[i].dot(v[i])


@tack.func
def _put(v, i, w):
    v[i] = w


def test_device_functions_take_soa_arrays(backend):
    @tack.kernel
    def lengths(a, out):
        for i in range(len(a)):
            out[i] = _norm2(a, i)
            _put(a, i, a[i] * 2.0)

    p = tack.soa(tack.array([1.0, 0.0]), tack.array([2.0, 3.0]), tack.array([2.0, 4.0]))
    out = tack.zeros((2,))
    lengths(p, out)
    np.testing.assert_allclose(out.numpy(), [9, 25])
    np.testing.assert_allclose(p.numpy(), [[2, 4, 4], [0, 6, 8]])


@tack.device_class
class _Points:
    def __init__(self, xyz):
        self.xyz = xyz

    @tack.func
    def __getitem__(self, i):
        return self.xyz[i] + 1.0

    @tack.func
    def __setitem__(self, i, v):
        self.xyz[i] = v

    @tack.func
    def __len__(self):
        return self.xyz.shape[0]


@tack.kernel
def _copy(a, out):
    for i in range(len(a)):
        out[i] = a[i]


def test_device_classes_and_implicit_arrays_hold_soa_arrays(backend):
    p = tack.soa(*_xyz())
    out = tack.zeros((4,), tack.vec3())
    _copy(_Points(p), out)
    np.testing.assert_allclose(out.numpy(), _expected() + 1)
    _copy(_Points(tack.array(p, layout="aos")), out)      # the other layout, another variant
    np.testing.assert_allclose(out.numpy(), _expected() + 1)
    _copy(arrays.Permutation(tack.array([3, 0]), p), out)
    np.testing.assert_allclose(out.numpy()[:2], _expected()[[3, 0]])
    _broadcast(_Points(p), 9.0)
    np.testing.assert_allclose(p.numpy(), 9)


def test_one_variant_per_layout(backend):
    from tack.runtime.dispatch import get_backend
    drop_variants(_scale, lambda key: True)
    for n in (3, 5):
        for layout in ("aos", "soa"):
            _scale(tack.zeros((n,), tack.vec3(), layout=layout),
                   tack.zeros((n,), tack.vec3()), 1.0)
    assert len(get_backend()._cache[_scale]) == 2


def test_an_soa_array_is_not_a_value():
    tack.set_default_device("cpu")

    @tack.kernel
    def alias(a, out):
        for i in range(out.shape[0]):
            b = a
            out[i] = b[i][0]

    with pytest.raises(tack.lang.source_validation.UnsupportedSyntaxError,
                       match="'a' is an SOA array"):
        alias(tack.zeros((3,), tack.vec2(), layout="soa"), tack.zeros((3,)))


def test_a_read_only_component_is_named():
    tack.set_default_device("cpu")
    read_only = tack.array_from_ptr(np.zeros(3, np.float32), (3,), tack.f32)
    with pytest.raises(ValueError, match="component 1 of parameter 'a' may be written"):
        _broadcast(tack.soa(tack.zeros((3,)), read_only), 1.0)
    with pytest.raises(RuntimeError, match="read-only"):
        tack.soa(tack.zeros((3,)), read_only).fill(1.0)


def test_debug_mode_checks_soa_accesses(backend):
    @tack.kernel
    def past(a, n):
        for i in range(n):
            a[i] = tack.vec2(1.0, 2.0)

    p = tack.zeros((3,), tack.vec2(), layout="soa")
    tack.set_debug(True)
    try:
        with pytest.raises(IndexError, match="index 3 is outside"):
            past(p, 4)
        past(p, 3)
    finally:
        tack.set_debug(False)
    np.testing.assert_allclose(p.numpy(), [[1, 2]] * 3)


def test_statistics_pair_components_as_interleaved_arrays_do(backend):
    p = tack.soa(*_xyz())
    aos = tack.array(p, layout="aos")
    assert alg.dot(p, p) == alg.dot(aos, aos) == alg.dot(p, aos)
    assert alg.absmax(p) == 23
