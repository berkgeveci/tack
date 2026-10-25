"""A pytest plugin that runs the suite on the Warp backend (a prototype).

    PYTHONPATH=packages/tack-core/tests uv run pytest -p _warp_default packages/tack-vis/tests -m quick

Every test starts with ``"warp"`` as the default device, and the root
conftest's `backend` fixture, which sets the CPU after autouse fixtures
run, is redirected to Warp too. Arrays a module makes at import, and tests
that name the CPU, stay on the CPU, so a test asserting that data is on
the CPU fails under it.
"""

import pytest


@pytest.fixture(autouse=True)
def _warp_default(monkeypatch):
    import tack
    real = tack.set_default_device
    monkeypatch.setattr(tack, "set_default_device",
                        lambda spec, *a, **k: real("warp" if spec == "cpu" else spec, *a, **k))
    real("warp")
    yield


def pytest_sessionfinish(session):
    from tack.runtime import warp_backend
    print(f"\nWarp variants compiled: {next(warp_backend._modules)}")
