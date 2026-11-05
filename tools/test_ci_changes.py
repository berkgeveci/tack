# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""``tools/ci_changes.py``: a pull request's paths choose CI's jobs."""

import subprocess
import sys

import ci_changes
import pytest


@pytest.mark.parametrize("path, groups", [
    ("packages/tack-core/src/tack/runtime/cuda_backend.py", {"code"}),
    ("packages/tack-vis/tests/test_streamlines.py", {"code"}),
    ("packages/tack-vis/tests/data/vtk/README.md", {"code"}),
    ("packages/tack-core/README.md", {"code"}),       # the package's description
    ("packages/tack-vis/tools/make_tables.py", {"code"}),
    ("pyproject.toml", {"code"}),
    ("uv.lock", {"code"}),
    ("conftest.py", {"code"}),
    ("test-durations.json", {"code"}),
    (".github/workflows/ci.yml", {"code"}),
    ("benchmarks/threading_decisions.py", {"code"}),  # test_threading_benchmark runs it
    ("something/new.txt", {"code"}),
    ("packages/tack-core/examples/01_hello_tack.py", {"examples"}),
    ("packages/tack-vis/examples/data/README.md", {"examples"}),
    ("docs/vis-guide/reference/algorithms.md", {"docs"}),
    ("docs/examples/validate.py", {"docs"}),
    ("docs/_ext/tack_namespace.py", {"docs"}),
    ("mkdocs.yml", {"docs"}),
    ("overrides/main.html", {"docs"}),
    ("README.md", {"docs"}),
    ("CONTRIBUTING.md", {"docs"}),
    ("CLAUDE.md", {"docs"}),
    ("packages/tack-vis/CLAUDE.md", {"docs"}),
    ("changelog.d/ci-path-filter.md", set()),
    ("tools/changelog.py", set()),
    ("archive/tack-rendering/NOTES.md", set()),
    ("LICENSE", set()),
])
def test_each_path_has_its_groups(path, groups):
    assert ci_changes.groups_of(path) == groups


def test_a_change_needs_the_union_of_its_paths():
    assert ci_changes.classify(["docs/install.md", "packages/tack-core/examples/02_saxpy.py"]) == {
        "code": False, "docs": True, "examples": True}
    assert ci_changes.classify(["changelog.d/x.md", "tools/spdx_headers.py"]) == {
        "code": False, "docs": False, "examples": False}
    assert ci_changes.classify(["docs/index.md", "uv.lock"])["code"]


def test_prints_one_output_a_group():
    out = subprocess.run([sys.executable, "ci_changes.py"], input="docs/index.md\n\n",
                         capture_output=True, text=True, check=True).stdout
    assert out.splitlines() == ["code=false", "docs=true", "examples=false"]
