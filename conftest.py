"""Backend fixtures, shared by every package's tests.

Discovers what this machine can run once, at import, and hands it out as
fixtures. It lives at the repository root because all three packages need
the same thing and pytest's rootdir is here: a conftest per package meant
three copies of the loop below, which is the same duplication the fixtures
exist to remove, one level up.

Fixtures
--------
backend
    Every backend available here.
f64_backend
    Those declaring f64 support — Metal has none, and asking for one is
    better than skipping inside the test.
reduction_backend
    Those that reduce on the device rather than through numpy.

A test that names one of these runs once per matching backend, so which
backends exist is a property of the machine and never of the test file.
Files used to pin their own list, and eleven of them had settled on
["cpu", "metal"] — not a decision about what they needed, just what the
machine they were written on could run. On a CUDA box those tests said
nothing at all.

Capability skips read declared attributes rather than comparing arch names.
`if arch == "metal": continue  # no f64` was the old spelling, and it was
wrong twice over: it named one backend for a property several share, and it
went stale the moment another arrived.
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
        # Not built, no device, or no driver. All three mean the same here.
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
