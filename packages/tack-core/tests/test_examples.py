"""Every numbered example must still run.

49 examples ship with Tack and none of them were executed by the test
suite. They are the code new users copy, they exercise combinations the
unit tests do not — templates feeding a path tracer, flying edges into a
renderer — and nothing noticed when one broke.

Marked `slow` and deselected from the default run, because the full sweep
takes about a minute against an 18-second suite. CI runs it as its own
job; locally, `pytest -m slow`.

Examples that need an optional third-party package, or that do not offer a
CPU backend, skip themselves with the reason. That keeps the suite honest
about what it actually checked, and means adding an example with a new
optional dependency will not turn CI red. A missing *tack* module is not
treated as optional — that is a real failure.

This lives under tack-core/tests because it spans all three packages and
`testpaths` only covers the package test directories.
"""

import pathlib
import re
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[3]

EXAMPLES = (sorted(REPO.glob("packages/*/examples/[0-9]*.py"))
            + sorted(REPO.glob("examples/[0-9]*.py")))

_MISSING_MODULE = re.compile(r"ModuleNotFoundError: No module named '([\w.]+)'")


def _example_id(path: pathlib.Path) -> str:
    package = path.parent.parent.name
    return f"{package}/{path.name}" if package != "tack" else path.name


def test_examples_were_found():
    """A glob that silently matches nothing would make every test below vacuous."""
    assert len(EXAMPLES) >= 45, f"only found {len(EXAMPLES)} examples"


@pytest.mark.slow
@pytest.mark.parametrize("path", EXAMPLES, ids=_example_id)
def test_example_runs(path):
    # Several examples save output here; it is gitignored.
    (REPO / "results").mkdir(exist_ok=True)

    proc = subprocess.run(
        [sys.executable, str(path), "--arch", "cpu"],
        capture_output=True, text=True, cwd=REPO, timeout=300,
    )
    if proc.returncode == 0:
        return

    combined = proc.stdout + proc.stderr

    missing = _MISSING_MODULE.search(combined)
    if missing and not missing.group(1).startswith("tack"):
        pytest.skip(f"needs optional dependency '{missing.group(1)}'")

    if "invalid choice: 'cpu'" in combined:
        pytest.skip("example does not offer a CPU backend")

    pytest.fail(f"{_example_id(path)} exited {proc.returncode}\n"
                f"{combined[-3000:]}")
