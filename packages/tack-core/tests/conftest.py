"""Shared fixtures for tack-core tests.

Discovers the backends available on this machine once, at import, and
exposes them as fixtures so individual test modules don't each re-roll the
try-init-append loop.

Capability-based skips read the backend's declared attributes. They used
to be written as `if _arch == "metal": continue  # Metal lacks f64`,
because `supports_f64` existed on Level Zero alone and `getattr(backend,
'supports_f64', True)` answered True for everyone else. It is derived
from `supported_dtypes` now, so asking the backend works.
"""

import pytest
import tack

_ALL_ARCHES = ["cpu", "metal", "cuda", "hip", "level_zero"]

available_backends = []
f64_backends = []
reduction_backends = []

for _arch in _ALL_ARCHES:
    try:
        tack.init(arch=getattr(tack, _arch))
    except (ImportError, RuntimeError, OSError, AttributeError):
        continue
    available_backends.append(_arch)

    from tack.runtime.dispatch import get_backend as _get_backend
    _be = _get_backend()
    if _be.supports_f64:
        f64_backends.append(_arch)
    if _be.supports_device_reductions:
        reduction_backends.append(_arch)


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


@pytest.fixture(params=reduction_backends)
def reduction_backend(request):
    """Run a test once per backend that reduces on the device."""
    tack.init(arch=getattr(tack, request.param))
    return request.param
