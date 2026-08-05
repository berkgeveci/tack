"""Shared fixtures for tack-vis tests.

Discovers the backends available on this machine once, at import, and
exposes them as fixtures so individual test modules don't each re-roll
the try-init-append loop.

Fixtures
--------
backend
    Parametrized over every available backend.
f64_backend
    Parametrized over backends that support f64 (Metal does not).
"""

import pytest
import tack

_ALL_ARCHES = ["cpu", "metal", "cuda", "hip", "level_zero"]

available_backends = []
f64_backends = []

for _arch in _ALL_ARCHES:
    try:
        tack.init(arch=getattr(tack, _arch))
    except (ImportError, RuntimeError, OSError, AttributeError):
        continue
    available_backends.append(_arch)

    from tack.runtime.dispatch import get_backend as _get_backend
    _be = _get_backend()
    # Metal has no f64; it does not declare supports_f64, so name it here
    # until the capability attribute lands on every backend.
    if _arch != "metal" and getattr(_be, "supports_f64", True):
        f64_backends.append(_arch)


@pytest.fixture(params=available_backends)
def backend(request):
    """Run a test once per available backend."""
    tack.init(arch=getattr(tack, request.param))
    return request.param


@pytest.fixture(params=f64_backends)
def f64_backend(request):
    """Run a test once per backend that supports f64."""
    tack.init(arch=getattr(tack, request.param))
    return request.param
