"""Every numbered example must still run.

The examples are the code new users copy, and they exercise combinations
the unit tests do not, such as templates feeding a solver or a dataset
filter feeding VTK. Before this sweep, nothing noticed when one broke.

Marked `slow` and deselected from the default run, because the full sweep
takes about a minute against an 18-second suite. CI runs it as its own
job; locally, `pytest -m slow`.

Examples that need an optional third-party package, or that do not offer
the selected backend (CPU unless `TACK_EXAMPLES_ARCH` names another), skip
themselves with the reason. That keeps the suite honest
about what it actually checked, and means adding an example with a new
optional dependency will not turn CI red. A missing *tack* module is not
treated as optional — that is a real failure.

This lives under tack-core/tests because it spans both packages and
`testpaths` only covers the package test directories.
"""

import os
import pathlib
import re
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[3]

EXAMPLES = sorted(REPO.glob("packages/*/examples/[0-9]*.py"))

# CI has no GPU, so the sweep runs on CPU unless told otherwise. On a machine
# with a device, `TACK_EXAMPLES_ARCH=metal pytest -m slow` (or cuda, hip,
# level_zero) runs every example there instead.
ARCH = os.environ.get("TACK_EXAMPLES_ARCH", "cpu")

_MISSING_MODULE = re.compile(r"ModuleNotFoundError: No module named '([\w.]+)'")
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _example_id(path: pathlib.Path) -> str:
    return f"{path.parent.parent.name}/{path.name}"


def test_examples_were_found():
    """A glob that silently matches nothing would make every test below vacuous."""
    assert len(EXAMPLES) >= 30, f"only found {len(EXAMPLES)} examples"


@pytest.mark.slow
@pytest.mark.parametrize("path", EXAMPLES, ids=_example_id)
def test_example_runs(path):
    # Several examples save output here; it is gitignored.
    (REPO / "results").mkdir(exist_ok=True)

    # Python 3.13 colours tracebacks, and FORCE_COLOR -- which several
    # terminals and CI runners export -- makes it do so even into a pipe.
    # The escape codes land between "ModuleNotFoundError" and its message,
    # so every skip below stops matching and an absent optional dependency
    # reads as a broken example.  Ask for plain text, and strip anything
    # that arrives coloured anyway.
    # Plotting examples must render without waiting for a GUI window to close.
    env = {**os.environ, "PYTHON_COLORS": "0", "NO_COLOR": "1",
           "MPLBACKEND": "Agg"}
    env.pop("FORCE_COLOR", None)

    proc = subprocess.run(
        [sys.executable, str(path), "--arch", ARCH],
        capture_output=True, text=True, cwd=REPO, timeout=300, env=env,
    )
    if proc.returncode == 0:
        return

    combined = _ANSI.sub("", proc.stdout + proc.stderr)

    missing = _MISSING_MODULE.search(combined)
    if missing and not missing.group(1).startswith("tack"):
        pytest.skip(f"needs optional dependency '{missing.group(1)}'")

    if f"invalid choice: '{ARCH}'" in combined:
        pytest.skip(f"example does not offer the {ARCH} backend")

    pytest.fail(f"{_example_id(path)} exited {proc.returncode}\n"
                f"{combined[-3000:]}")
