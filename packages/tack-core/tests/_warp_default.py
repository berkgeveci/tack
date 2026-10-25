"""A pytest plugin that runs the suite on the Warp backend (a prototype).

    PYTHONPATH=packages/tack-core/tests uv run pytest -p _warp_default packages/tack-vis/tests -m quick

Every test starts with ``"warp"`` as the default device, and the root
conftest's `backend` fixture, which sets the CPU after autouse fixtures
run, is redirected to Warp too. Arrays a module makes at import, and tests
that name the CPU, stay on the CPU, so a test asserting that data is on
the CPU fails under it. A test that reaches `tack.inspect` on Warp is
skipped: the Warp backend does not offer it.
"""

import pytest


@pytest.fixture(autouse=True)
def _warp_default(monkeypatch):
    import tack
    # The Warp backend does not offer tack.inspect (`supports_inspect`): a test
    # that reaches it on Warp is skipped, not failed.
    inspect = tack.inspect

    def inspect_or_skip(kernel, *args, **kwargs):
        try:
            return inspect(kernel, *args, **kwargs)
        except NotImplementedError as error:
            if "tack.inspect does not show kernels" in str(error):
                pytest.skip(str(error))
            raise
    monkeypatch.setattr(tack, "inspect", inspect_or_skip)
    real = tack.set_default_device
    monkeypatch.setattr(tack, "set_default_device",
                        lambda spec, *a, **k: real("warp" if spec == "cpu" else spec, *a, **k))
    real("warp")
    yield


def pytest_sessionfinish(session):
    from tack.runtime import warp_backend
    print(f"\nWarp variants compiled: {next(warp_backend._modules)}")
