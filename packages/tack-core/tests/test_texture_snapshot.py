"""A texture samples a snapshot of its field, refreshed by ``update()``.

The hardware paths used to copy the field into a texture image on first
dispatch and cache it by device address, forever: CUDA kept sampling the
first data after the field was refilled, while CPU, which interpolates in
software, read the field live. Every backend now copies at creation and
again on ``tex.update()``, and nothing in between reaches the texture.
"""

import gc
import weakref

import numpy as np
import pytest

import tack

W, H, D = 4, 4, 4
N = W * H * D


@tack.kernel
def _sample_center(out, tex, count):
    for i in range(count):
        out[i] = tex.sample(0.5, 0.5, 0.5)


@tack.kernel
def _sample_pair(out, a, b, count):
    for i in range(count):
        out[0] = a.sample(0.5, 0.5, 0.5)
        out[1] = b.sample(0.5, 0.5, 0.5)


@tack.kernel
def _fill(data, value, count):
    for i in range(count):
        data[i] = value


def _constant(value):
    data = tack.field(dtype=tack.f32, shape=(N,))
    data.from_numpy(np.full(N, value, dtype=np.float32))
    return data


def _sample(tex):
    out = tack.field(dtype=tack.f32, shape=(1,))
    _sample_center(out, tex, 1)
    return float(out.to_numpy()[0])


def test_writes_reach_texture_only_through_update(backend):
    data = _constant(1.0)
    tex = tack.texture3d(data, shape=(W, H, D))
    assert _sample(tex) == pytest.approx(1.0)

    data.from_numpy(np.full(N, 7.0, dtype=np.float32))
    assert _sample(tex) == pytest.approx(1.0)
    tex.update()
    assert _sample(tex) == pytest.approx(7.0)

    # A kernel store is a write like any other.
    _fill(data, 3.0, N)
    assert _sample(tex) == pytest.approx(7.0)
    tex.update()
    assert _sample(tex) == pytest.approx(3.0)


def test_textures_over_one_field_are_separate_snapshots(backend):
    """Two textures over the same storage were one cache entry on GPU."""
    data = _constant(1.0)
    first = tack.texture3d(data, shape=(W, H, D))
    data.from_numpy(np.full(N, 2.0, dtype=np.float32))
    second = tack.texture3d(data, shape=(W, H, D))

    out = tack.field(dtype=tack.f32, shape=(2,))
    _sample_pair(out, first, second, 1)
    np.testing.assert_allclose(out.to_numpy(), [1.0, 2.0])


def test_texture_is_not_served_for_reused_memory(backend):
    """A new field at a freed field's address gets its own texture.

    Allocators hand a just-freed block of the same size straight back, so
    the second field usually sits where the first one did. Keyed by
    address, the compiled kernel served the first texture's image again.
    """
    data = _constant(1.0)
    tex = tack.texture3d(data, shape=(W, H, D))
    assert _sample(tex) == pytest.approx(1.0)
    del tex, data
    gc.collect()

    data = _constant(5.0)
    tex = tack.texture3d(data, shape=(W, H, D))
    assert _sample(tex) == pytest.approx(5.0)


def test_texture_storage_is_released_with_the_texture(backend):
    """Dispatch keeps no reference to the snapshot it bound."""
    tex = tack.texture3d(_constant(1.0), shape=(W, H, D))
    assert _sample(tex) == pytest.approx(1.0)
    storage = weakref.ref(tex._storage)
    del tex
    gc.collect()
    assert storage() is None


def test_texture_passed_as_template_parameter(backend):
    """The volume renderer's form: a texture in a `tack.template()` slot."""
    @tack.kernel
    def sample_template(out, tex: tack.template(), count):
        for i in range(count):
            out[i] = tex.sample(0.5, 0.5, 0.5)

    data = _constant(2.0)
    tex = tack.texture3d(data, shape=(W, H, D))
    out = tack.field(dtype=tack.f32, shape=(1,))
    sample_template(out, tex, 1)
    assert out.to_numpy()[0] == pytest.approx(2.0)
    data.from_numpy(np.full(N, 4.0, dtype=np.float32))
    tex.update()
    sample_template(out, tex, 1)
    assert out.to_numpy()[0] == pytest.approx(4.0)


def test_f64_field_is_rejected():
    tack.init(arch=tack.cpu)
    data = tack.field(dtype=tack.f64, shape=(N,))
    with pytest.raises(ValueError, match='f32'):
        tack.texture3d(data, shape=(W, H, D))


@pytest.mark.parametrize('interp', ['nearest', 'cubic'])
def test_only_linear_interpolation_is_accepted(interp):
    tack.init(arch=tack.cpu)
    with pytest.raises(ValueError, match="'linear'"):
        tack.texture3d(_constant(1.0), shape=(W, H, D), interp=interp)


def test_shape_must_cover_the_field():
    tack.init(arch=tack.cpu)
    with pytest.raises(ValueError, match='64 elements'):
        tack.texture3d(_constant(1.0), shape=(W, H, D + 1))
