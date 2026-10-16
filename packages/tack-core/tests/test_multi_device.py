"""Two devices of one type in one process (CUDA, HIP, Level Zero).

Each test runs on every type with at least two devices here, and skips
elsewhere. Run on a multi-GPU machine with

    uv run pytest packages/tack-core/tests/test_multi_device.py -v -rs

The CUDA backend switches contexts and the HIP backend switches devices
around every operation once a second device is in use, so these interleave
allocations, copies, launches, reductions and host reads between devices,
from one thread and from several. `test_multi_device_switching.py` checks
the same switching against fake runtimes, on any machine.
"""

import threading

import numpy as np
import pytest

import tack
from tack.algorithms import argsort, exclusive_scan

_TYPES = [pytest.param(kind, marks=pytest.mark.skipif(
    tack.device_count(kind) < 2, reason=f"needs two {kind} devices"))
    for kind in ("cuda", "hip", "level_zero")]


@pytest.fixture(params=_TYPES)
def pair(request):
    kind = request.param
    tack.set_default_device("cpu")
    yield tack.device(f"{kind}:0"), tack.device(f"{kind}:1")
    tack.set_default_device("cpu")


@tack.kernel
def _affine(x, out, a, b):
    for i in range(x.shape[0]):
        out[i] = x[i] * a + b


@tack.kernel
def _add_into(x, total):
    for i in range(x.shape[0]):
        tack.atomic_add(total, 0, x[i])


def _on(dev, values, dtype=np.float32):
    return tack.array(np.asarray(values, dtype), device=dev)


def test_each_device_is_its_own_value(pair):
    first, second = pair
    assert first != second and (first.index, second.index) == (0, 1)
    assert str(second) == f"{second.type}:1"
    assert first._backend.memory_domain != second._backend.memory_domain
    assert tack.device(f"{first.type}:1") is second
    with pytest.raises(ValueError, match="has"):
        tack.device(f"{first.type}:{tack.device_count(first.type)}")


def test_kernels_run_on_each_device(pair):
    for dev, (a, b) in zip(pair, ((2.0, 1.0), (3.0, -1.0))):
        x = _on(dev, np.arange(1000))
        out = tack.zeros((1000,), device=dev)
        _affine(x, out, a, b)
        np.testing.assert_allclose(out.numpy(), np.arange(1000) * a + b)
        assert out.device == dev
    assert all(dev._backend._cache.get(_affine) for dev in pair)


def test_interleaved_work_stays_on_its_device(pair):
    """Allocate, write, launch and read on alternating devices, many times:
    every operation must find its own context or device."""
    xs = [_on(dev, np.full(256, k + 1)) for k, dev in enumerate(pair)]
    outs = [tack.zeros((256,), device=dev) for dev in pair]
    for step in range(50):
        for k in (0, 1):
            _affine(xs[k], outs[k], 1.0, float(step))
            tack.copyto(xs[k], outs[k].numpy())
    for k in (0, 1):
        np.testing.assert_allclose(xs[k].numpy(), k + 1 + sum(range(50)))
        assert xs[k][3] == k + 1 + sum(range(50))


def test_arrays_on_both_devices_are_refused(pair):
    first, second = pair
    with pytest.raises(ValueError, match="arrays on two devices"):
        _affine(_on(first, [1.0]), tack.zeros((1,), device=second), 1.0, 0.0)


def test_the_second_device_works_while_the_first_is_the_default(pair):
    first, second = pair
    tack.set_default_device(first)
    x = _on(second, np.arange(64))
    total = tack.zeros((1,), device=second)
    _add_into(x, total)
    assert total.numpy()[0] == sum(range(64))
    assert x.sum() == sum(range(64)) and x.max() == 63
    x.fill(2.0)
    assert x.numpy().tolist() == [2.0] * 64
    with second:
        assert tack.zeros((4,)).device == second
    second.synchronize()
    tack.sync()


def test_library_code_follows_its_inputs(pair):
    for dev in pair:
        keys = _on(dev, [5, 1, 4, 1, 3], np.int32)
        order = argsort(keys)
        assert order.device == dev
        np.testing.assert_array_equal(order.numpy(), [1, 3, 4, 2, 0])
        sums = tack.zeros((5,), tack.i32, device=dev)
        exclusive_scan(keys, sums, 5)
        np.testing.assert_array_equal(sums.numpy(), [0, 5, 6, 10, 11])


def test_moving_between_devices(pair):
    first, second = pair
    src = _on(first, np.linspace(0, 1, 100))
    dst = src.to_device(second)
    assert dst.device == second
    np.testing.assert_array_equal(dst.numpy(), src.numpy())
    assert src.copy().device == first and dst.astype(tack.i32).device == second


@tack.kernel
def _sample_center(tex, out):
    for i in range(out.shape[0]):
        out[i] = tex.sample(0.5, 0.5, 0.5)


def test_textures_on_each_device(pair):
    for k, dev in enumerate(pair):
        volume = _on(dev, np.full(8, k + 1.0))
        out = tack.zeros((4,), device=dev)
        _sample_center(tack.texture3d(volume, (2, 2, 2)), out)
        np.testing.assert_allclose(out.numpy(), k + 1.0, rtol=1e-6)


def test_a_thread_per_device(pair):
    errors, results = [], {}

    def work(k, dev):
        try:
            x = _on(dev, np.full(4096, float(k)))
            out = tack.zeros((4096,), device=dev)
            for _ in range(100):
                _affine(x, out, 1.0, 1.0)
                tack.copyto(x, out.numpy())
            results[k] = x.numpy()
        except Exception as e:      # reported below, with the thread's device
            errors.append((k, repr(e)))

    threads = [threading.Thread(target=work, args=(k, dev)) for k, dev in enumerate(pair)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    for k in (0, 1):
        np.testing.assert_allclose(results[k], k + 100.0)


def test_wrapping_a_second_device_pointer(pair):
    first, second = pair
    owner = _on(second, np.arange(16))
    assert tack.memory_space(owner._buffer.address, device=second) == second.type
    view = tack.array_from_ptr(owner._buffer.address, (16,), tack.f32, device=second)
    np.testing.assert_array_equal(view.numpy(), np.arange(16))
    assert view.device == second


def test_dlpack_carries_the_device_number(pair):
    first, second = pair
    owner = _on(second, np.arange(16))
    assert owner.__dlpack_device__()[1] == 1
    tack.set_default_device(first)
    imported = tack.from_dlpack(owner)
    assert imported.device == second
    np.testing.assert_array_equal(imported.numpy(), np.arange(16))


def test_every_device_runs_a_kernel(pair):
    """Every device of the type, not just the first two (Perlmutter has four)."""
    kind = pair[0].type
    for k in range(tack.device_count(kind)):
        dev = tack.device(f"{kind}:{k}")
        x = _on(dev, np.arange(512))
        out = tack.zeros((512,), device=dev)
        _affine(x, out, float(k + 1), 0.5)
        np.testing.assert_allclose(out.numpy(), np.arange(512) * (k + 1) + 0.5)
        assert out.device.index == k


# Another library makes device 1 the thread's before Tack first touches the
# runtime: an embedding framework's CUDA context there, or hipSetDevice(1).
# A bare spec then means device 1, and work on device 0 leaves the thread's
# device as it was. In a fresh process, since a device lives as long as the
# process and this one has made its devices already.
_BARE_SPEC_SCRIPTS = {
    "cuda": """
from cuda.bindings import driver
assert driver.cuInit(0)[0] == driver.CUresult.CUDA_SUCCESS
err, dev = driver.cuDeviceGet(1)
assert err == driver.CUresult.CUDA_SUCCESS, err
err, foreign = driver.cuCtxCreate(None, 0, dev)
assert err == driver.CUresult.CUDA_SUCCESS, err

def threads_own():
    err, ctx = driver.cuCtxGetCurrent()
    return int(ctx) == int(foreign)
""",
    "hip": """
from hip import hip

def ok(result):
    err = result[0] if isinstance(result, tuple) else result
    return err == hip.hipError_t.hipSuccess

assert ok(hip.hipSetDevice(1))

def threads_own():
    err, device = hip.hipGetDevice()
    return ok(err) and int(device) == 1
""",
}

_BARE_SPEC_CHECK = """
import numpy as np
import tack

@tack.kernel
def affine(x, out, a, b):
    for i in range(x.shape[0]):
        out[i] = x[i] * a + b

bare = tack.device("{kind}")
assert bare.index == 1, f"'{kind}' made {{bare}}, not the thread's device 1"
{adopted}
first = tack.device("{kind}:0")
for dev, a in ((bare, 2.0), (first, 3.0), (bare, 4.0)):
    x = tack.array(np.arange(100, dtype=np.float32), device=dev)
    out = tack.zeros((100,), device=dev)
    affine(x, out, a, 1.0)
    assert np.allclose(out.numpy(), np.arange(100) * a + 1.0), (dev, out.numpy()[:4])
    assert threads_own(), f"work on {{dev}} changed the thread's device"
print("ok")
"""

_ADOPTED = {
    "cuda": 'assert int(bare._backend._context) == int(foreign), "the thread\'s context was not adopted"',
    "hip": "",
}


@pytest.mark.parametrize("kind", [p for p in _TYPES if p.values[0] != "level_zero"])
def test_a_bare_spec_takes_the_threads_device(kind, tmp_path):
    import subprocess
    import sys

    script = tmp_path / "bare_spec.py"
    script.write_text(_BARE_SPEC_SCRIPTS[kind]
                      + _BARE_SPEC_CHECK.format(kind=kind, adopted=_ADOPTED[kind]))
    proc = subprocess.run([sys.executable, "-u", str(script)],
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0 and proc.stdout.strip() == "ok", proc.stdout + proc.stderr


@pytest.mark.skipif(not (tack.device_count("cuda") and tack.device_count("hip")),
                    reason="needs a CUDA and a HIP device")
def test_cuda_and_hip_in_one_process():
    """Both runtimes in one process (illyad: an H100 beside the MI210s),
    work interleaved between them."""
    tack.set_default_device("cpu")
    devices = tack.device("cuda:0"), tack.device("hip:0")
    xs = [_on(dev, np.full(256, k + 1)) for k, dev in enumerate(devices)]
    outs = [tack.zeros((256,), device=dev) for dev in devices]
    for step in range(20):
        for k in (0, 1):
            _affine(xs[k], outs[k], 1.0, float(step))
            tack.copyto(xs[k], outs[k].numpy())
    for k in (0, 1):
        np.testing.assert_allclose(xs[k].numpy(), k + 1 + sum(range(20)))
    with pytest.raises(ValueError, match="arrays on two devices"):
        _affine(xs[0], outs[1], 1.0, 0.0)
    moved = xs[0].to_device(devices[1])
    assert moved.device == devices[1]
    np.testing.assert_array_equal(moved.numpy(), xs[0].numpy())
    tack.sync()
