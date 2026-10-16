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
workgroup_backend
    Those with native workgroup execution for cooperative kernels.
require_idle
    A function: ``require_idle(threads=None)`` skips the test, with the
    numbers, when the load average is above half of ``threads`` (the CPUs
    this process may use, by default). For a test that asserts what the
    scheduler or the OS did with the machine.

Options
-------
--ids=i64
    Every ``tack.dataset`` topology takes 64-bit ids, however small, so the
    dataset tests run each path in i64 as well as (by default) i32.

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

import os

import pytest

# Tests compile from scratch: entries a previous run stored on disk would
# hide a test that patches the compiler. test_disk_cache.py turns the cache
# on for itself; `TACK_CACHE=write pytest` runs the suite through it.
os.environ.setdefault("TACK_CACHE", "off")

import tack

_ALL_ARCHES = ["cpu", "metal", "cuda", "hip", "level_zero"]

available_backends = []
f64_backends = []
reduction_backends = []
workgroup_backends = []

for _arch in _ALL_ARCHES:
    try:
        tack.set_default_device(_arch)
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
    if _be.supports_workgroups:
        workgroup_backends.append(_arch)

# Probing left the last backend found as the default; arrays a test module
# makes at import (parametrize lists, module globals) go to the CPU instead,
# whatever this machine has -- if it has the CPU backend: an install for a
# GPU alone has no llvmlite.
if "cpu" in available_backends:
    tack.set_default_device("cpu")


def pytest_addoption(parser):
    parser.addoption("--ids", choices=("i32", "i64"), default="i32",
                     help="the narrowest id type tack.dataset topologies take")


def pytest_configure(config):
    if config.getoption("--ids") == "i64":
        try:
            from tack.dataset import ids
        except ImportError:                    # tack-vis is not installed
            return
        ids._minimum = tack.i64


@pytest.fixture(params=available_backends)
def backend(request):
    """Run a test once per available backend."""
    tack.set_default_device(request.param)
    return request.param


@pytest.fixture
def no_disk_cache():
    """For a test that counts compile work in this process, which a disk-cache
    hit would skip when the suite runs with the cache on."""
    from tack import cache
    cache.configure(mode="off")
    yield
    cache.configure()


@pytest.fixture(params=f64_backends)
def f64_backend(request):
    """Run a test once per backend that supports f64."""
    tack.set_default_device(request.param)
    return request.param


@pytest.fixture(params=reduction_backends)
def reduction_backend(request):
    """Run a test once per backend that reduces on the device."""
    tack.set_default_device(request.param)
    return request.param


@pytest.fixture(params=workgroup_backends)
def workgroup_backend(request):
    """Run a test once per available backend with workgroup execution."""
    tack.set_default_device(request.param)
    return request.param


@pytest.fixture
def require_idle():
    """For a test whose assertion is about what the scheduler or the OS did
    (timings, wake-ups, which node a page landed on): skip it, with the
    numbers, on a busy host rather than fail. A tolerance that a loaded
    host passes is better where one exists; this is for where none does."""
    def require(threads=None):
        try:
            load = os.getloadavg()[0]
        except (AttributeError, OSError):
            return
        if threads is None:
            threads = (len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity")
                       else os.cpu_count() or 1)
        if load > threads / 2:
            pytest.skip(f"needs an idle host: load average {load:.1f} "
                        f"against {threads} threads")
    return require
