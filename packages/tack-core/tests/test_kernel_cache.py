"""Compiled-kernel cache correctness.

Regression tests for the cache key.  The cache was once keyed on
``id(kernel)``; because ``id()`` is a memory address and the cache held no
reference to the kernel, a garbage-collected kernel freed its address for
reuse and a later kernel with the same name and type signature silently
executed the previous kernel's compiled code.
"""

import gc
import importlib.util
import sys
import textwrap

import numpy as np
import pytest

import tack

backends = []
for _arch in ["cpu", "metal", "cuda", "hip", "level_zero"]:
    try:
        tack.init(arch=getattr(tack, _arch))
        backends.append(_arch)
    except (ImportError, RuntimeError, OSError):
        pass


@pytest.fixture(params=backends)
def backend(request):
    tack.init(arch=getattr(tack, request.param))
    return request.param


def _build_kernel(tmp_path, factor, index):
    """Compile a fresh Kernel object that scales by ``factor``.

    Each kernel has the same name and the same argument types, so the only
    thing separating them in the cache is kernel identity.
    """
    path = tmp_path / f"gen_{index}.py"
    path.write_text(textwrap.dedent(f"""
        import tack

        @tack.kernel
        def scale(x, out):
            for i in range(x.shape[0]):
                out[i] = x[i] * {factor}.0
    """))
    name = f"_tack_gen_{index}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    kern = mod.scale
    sys.modules.pop(name, None)
    return kern


def test_recycled_kernel_id_does_not_reuse_compiled_code(backend, tmp_path):
    """Distinct kernels must never share a cache entry.

    Kernels are built and dropped one at a time so CPython reuses freed
    addresses -- on the old id()-keyed cache this produced a silent wrong
    answer within ~30 iterations.
    """
    n = 4
    x = tack.field(dtype=tack.f32, shape=(n,))
    x.from_numpy(np.ones(n, dtype=np.float32))
    out = tack.field(dtype=tack.f32, shape=(n,))

    for i, factor in enumerate(range(2, 62)):
        kern = _build_kernel(tmp_path, factor, i)
        kern(x, out)
        got = out.to_numpy()
        np.testing.assert_allclose(
            got, np.full(n, float(factor), dtype=np.float32), rtol=1e-6,
            err_msg=(f"kernel scaling by {factor} returned {got[0]} -- "
                     f"stale compiled code from a recycled kernel id"),
        )
        del kern
        gc.collect()


def test_same_kernel_reuses_one_cache_entry(backend, tmp_path):
    """The fix must not defeat caching: repeat calls stay on one entry."""
    from tack.runtime.dispatch import get_backend

    be = get_backend()
    n = 8
    x = tack.field(dtype=tack.f32, shape=(n,))
    x.from_numpy(np.ones(n, dtype=np.float32))
    out = tack.field(dtype=tack.f32, shape=(n,))

    kern = _build_kernel(tmp_path, 3, 900)
    for _ in range(10):
        kern(x, out)
    np.testing.assert_allclose(out.to_numpy(), 3.0, rtol=1e-6)

    assert len(be._cache[kern]) == 1, "one kernel + one signature = one entry"


def test_compiled_code_is_released_with_the_kernel(backend, tmp_path):
    """Cache entries are weakly held, so dead kernels do not accumulate."""
    from tack.runtime.dispatch import get_backend

    be = get_backend()
    n = 4
    x = tack.field(dtype=tack.f32, shape=(n,))
    x.from_numpy(np.ones(n, dtype=np.float32))
    out = tack.field(dtype=tack.f32, shape=(n,))

    before = len(be._cache)
    kern = _build_kernel(tmp_path, 7, 901)
    kern(x, out)
    assert len(be._cache) == before + 1

    del kern
    gc.collect()
    assert len(be._cache) == before, "dropped kernel should leave no entry"
