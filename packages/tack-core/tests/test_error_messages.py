"""Errors have to say what to do about them.

Two habits made Tack's failures harder to act on than they needed to be.

`Kernel.__call__` wrapped every error and re-raised it `from None`, which
discards the chain. Naming the kernel is worth doing, but throwing away
where the failure came from meant debugging a codegen bug started with
editing the library to see the real traceback.

And a backend that will not start used to say "Requires AMD GPU with ROCm
and hip-python", which sends you to `pip install hip-python` — a command
that failed at the time, because hip-python was only on Test PyPI. The
message now carries the command that works.

Which turned out to be the harder half. The first fix wrote the Test-PyPI
incantation into the message, and the test below asserted it was there —
so when hip-python moved to PyPI proper and the [hip] extra started
declaring it for real, the message kept pointing at the wrong index and
the test kept that in place. A test that pins a fact about the world
outside the repo goes stale with it, silently, because it still passes.
What these pin now is that the message and pyproject.toml agree with each
other, which is a fact about this repo and cannot drift on its own.
"""

import pathlib
import tomllib

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


@pytest.mark.parametrize("decorator,kind", [
    ("kernel", "kernel"), ("func", "device function")])
def test_unreadable_source_is_explained_the_same_way(decorator, kind):
    """@tack.func raised the raw OSError where @tack.kernel explained it."""
    namespace = {}
    exec(compile("def generated(x):\n    return x\n", "<generated>", "exec"),
         namespace)
    with pytest.raises(RuntimeError, match=f"source of {kind} 'generated'") \
            as excinfo:
        getattr(tack, decorator)(namespace["generated"])
    assert isinstance(excinfo.value.__cause__, OSError)


# ── Unavailable backends explain themselves ──────────────────────────

def _extra_requirements(package: str, extra: str) -> list[str]:
    """What `pyproject.toml` actually declares for one extra."""
    pyproject = (pathlib.Path(__file__).resolve().parents[3]
                 / "packages" / package / "pyproject.toml")
    with pyproject.open("rb") as f:
        data = tomllib.load(f)
    return data["project"]["optional-dependencies"][extra]


def test_hip_message_matches_what_the_extra_installs():
    """The message and the packaging must not contradict each other.

    They did, for a while: the extra declared hip-python and the message
    said it could not. Asking pyproject.toml rather than asserting a
    remembered fact means the next move — a new index, a dropped
    dependency — fails here instead of misleading somebody.
    """
    from tack.runtime.dispatch import _BACKEND_HELP
    help_text = _BACKEND_HELP["hip"]
    requirements = _extra_requirements("tack-core", "hip")

    if any("hip-python" in req for req in requirements):
        assert "tack-core[hip]" in help_text, \
            "the extra installs hip-python; the message has to say so"
        assert "test.pypi.org" not in help_text, \
            "hip-python is on PyPI proper — Test PyPI is the stale answer"
    else:
        assert "hip-python" in help_text, \
            "the extra installs nothing, so the message must name the dep"

    assert "pip install" in help_text


def test_hip_message_does_not_pin_against_the_extra():
    """The extra sets a lower bound because the pin belongs to the machine.

    hip-python's version tracks the ROCm it binds to, so the right one is
    whichever matches the system. A message that hands over a specific
    version as *the* command contradicts that; naming one as the override
    for a mismatched ROCm does not.
    """
    from tack.runtime.dispatch import _BACKEND_HELP
    requirements = _extra_requirements("tack-core", "hip")
    hip_req = next((r for r in requirements if "hip-python" in r), None)
    if hip_req is None:
        pytest.skip("the [hip] extra no longer declares hip-python")

    # Split the environment marker off first — it carries `==` of its own
    # (`sys_platform == 'linux'`), which is not a version pin.
    specifier = hip_req.split(";")[0]
    assert "==" not in specifier and "~=" not in specifier, \
        "pyproject pinned hip-python; this test's premise is gone"
    assert "tack-core[hip]" in _BACKEND_HELP["hip"].splitlines()[1], \
        "the first command offered should be the unpinned extra"


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
