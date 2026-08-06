"""Errors have to say what to do about them.

Two habits made Tack's failures harder to act on than they needed to be.

`Kernel.__call__` wrapped every error and re-raised it `from None`, which
discards the chain. Naming the kernel is worth doing, but throwing away
where the failure came from meant debugging a codegen bug started with
editing the library to see the real traceback.

And a backend that will not start used to say "Requires AMD GPU with ROCm
and hip-python", which sends you to `pip install hip-python` — a command
that fails, because hip-python is on Test PyPI. The message now carries
the command that works.
"""

import pytest

import tack


@pytest.fixture(autouse=True)
def cpu():
    tack.init(arch=tack.cpu)


# ── The chain survives ───────────────────────────────────────────────

def test_dispatch_errors_keep_their_cause():
    """__cause__ is what `raise ... from e` sets and `from None` clears."""

    @tack.kernel
    def dynamic_dim(x, out, d):
        for i in range(x.shape[d]):
            out[i] = x[i]

    x = tack.field(dtype=tack.f32, shape=(4,))
    out = tack.field(dtype=tack.f32, shape=(4,))

    with pytest.raises(RuntimeError) as excinfo:
        dynamic_dim(x, out, 0)

    assert excinfo.value.__cause__ is not None, \
        "the original error was discarded — debugging starts from nothing"


def test_type_errors_keep_their_cause():
    @tack.kernel
    def double_it(x, out, n):
        for i in range(n):
            out[i] = x[i] * 2.0

    x = tack.field(dtype=tack.f64, shape=(4,))
    out = tack.field(dtype=tack.f64, shape=(4,))

    from tack.runtime.dispatch import get_backend
    if get_backend().supports_f64:
        pytest.skip("needs a backend that rejects f64")

    with pytest.raises(TypeError) as excinfo:
        double_it(x, out, 4)
    assert excinfo.value.__cause__ is not None


def test_the_readable_summary_is_still_the_last_line():
    """Keeping the chain must not bury the message underneath it.

    Python prints the cause first and the raised exception last, so the
    clean, kernel-named summary is what a reader sees at the bottom.
    """

    @tack.kernel
    def dynamic_dim(x, out, d):
        for i in range(x.shape[d]):
            out[i] = x[i]

    x = tack.field(dtype=tack.f32, shape=(4,))
    out = tack.field(dtype=tack.f32, shape=(4,))

    with pytest.raises(RuntimeError) as excinfo:
        dynamic_dim(x, out, 0)
    assert str(excinfo.value).startswith("Kernel 'dynamic_dim'")


# ── Unavailable backends explain themselves ──────────────────────────

def test_hip_message_gives_the_command_that_works():
    """`pip install hip-python` fails; the message must not imply it works."""
    from tack.runtime.dispatch import _BACKEND_HELP
    help_text = _BACKEND_HELP["hip"]
    assert "test.pypi.org" in help_text, \
        "hip-python is not on PyPI; the message has to say where it is"
    assert "pip install" in help_text


def test_level_zero_message_says_the_extra_installs_nothing():
    """The [level_zero] extra is empty because the deps are system libraries."""
    from tack.runtime.dispatch import _BACKEND_HELP
    help_text = _BACKEND_HELP["level_zero"]
    assert "libze_loader" in help_text
    assert "system librar" in help_text.lower()


@pytest.mark.parametrize("arch", ["cpu", "metal", "cuda", "hip", "level_zero"])
def test_every_backend_has_install_guidance(arch):
    from tack.runtime.dispatch import _BACKEND_HELP
    assert arch in _BACKEND_HELP
    assert len(_BACKEND_HELP[arch]) > 20


def test_unavailable_backend_raises_with_the_guidance_attached():
    """The help text has to reach the user, not just live in a dict."""
    unavailable = None
    for arch in ("hip", "level_zero", "cuda"):
        try:
            tack.init(arch=getattr(tack, arch))
        except (RuntimeError, ImportError) as e:
            unavailable = (arch, str(e))
            break
        finally:
            tack.init(arch=tack.cpu)

    if unavailable is None:
        pytest.skip("every backend is available on this machine")

    arch, message = unavailable
    from tack.runtime.dispatch import _BACKEND_HELP
    first_line = _BACKEND_HELP[arch].splitlines()[0]
    assert first_line in message, \
        f"init() error for '{arch}' did not carry its install guidance"


def test_unknown_arch_lists_the_real_ones():
    with pytest.raises(ValueError) as excinfo:
        tack.init(arch="quantum")
    message = str(excinfo.value)
    for arch in ("cpu", "metal", "cuda", "hip", "level_zero"):
        assert arch in message
