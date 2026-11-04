# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""``tools/spdx_headers.py``: every source file opens with an SPDX header."""

import subprocess

import pytest
import spdx_headers

KITWARE = ("# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.\n"
           "# SPDX-License-Identifier: BSD-3-Clause\n")


@pytest.mark.parametrize("text", [
    KITWARE,
    KITWARE + '\n"""A module."""\n',
    "#!/usr/bin/env python3\n" + KITWARE + "\nimport sys\n",
    ("# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.\n"
     "# SPDX-FileCopyrightText: Copyright (c) The Taichi Authors\n"
     "# SPDX-License-Identifier: Apache-2.0\n"),
])
def test_headers_with_any_holders_and_license_are_accepted(text):
    assert spdx_headers.problem(text, ".py") is None


@pytest.mark.parametrize("text, message", [
    ("", "no SPDX header"),
    ('"""A module."""\n' + KITWARE, "no SPDX header"),
    ("# Copyright (c) 2026 Kitware, Inc.\n# SPDX-License-Identifier: BSD-3-Clause\n",
     "no SPDX header"),
    ("# SPDX-License-Identifier: BSD-3-Clause\n", "no copyright above it"),
    ("# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.\nimport sys\n", "no SPDX-License"),
    ("# SPDX-FileCopyrightText: Copyright (c) \n# SPDX-License-Identifier: MIT\n",
     "no SPDX header"),
])
def test_what_is_not_a_header_is_refused_with_why(text, message):
    assert message in spdx_headers.problem(text, ".py")


@pytest.mark.parametrize("suffix", sorted(spdx_headers.COMMENTS))
def test_the_header_written_is_the_header_checked(suffix):
    for text in ("", "body\n", "\n\nbody\n"):
        written = spdx_headers.with_header(text, suffix)
        assert spdx_headers.problem(written, suffix) is None
        assert written.endswith(text)
    assert spdx_headers.with_header("", suffix).count("\n") == 2


def test_the_header_follows_a_shebang():
    written = spdx_headers.with_header("#!/usr/bin/env python3\nimport sys\n", ".py")
    assert written == "#!/usr/bin/env python3\n" + KITWARE + "\nimport sys\n"


def test_add_writes_kitwares_header_and_check_then_passes(tmp_path):
    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    (tmp_path / "archive").mkdir()
    files = {"a.py": "x = 1\n", "b.js": "f();\n", "notes.md": "# Notes\n",
             "archive/old.py": "y = 2\n", "c.py": KITWARE + "\nz = 3\n"}
    for name, text in files.items():
        (tmp_path / name).write_text(text)
    git("add", ".")
    assert spdx_headers.main(["check"], root=tmp_path) == 1
    assert spdx_headers.main(["add"], root=tmp_path) == 0
    assert (tmp_path / "a.py").read_text() == KITWARE + "\nx = 1\n"
    assert (tmp_path / "b.js").read_text().startswith("// SPDX-FileCopyrightText")
    for untouched in ("notes.md", "archive/old.py", "c.py"):
        assert (tmp_path / untouched).read_text() == files[untouched]
    assert spdx_headers.main(["check"], root=tmp_path) == 0


def test_a_malformed_header_is_reported_not_rewritten(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    text = "# SPDX-License-Identifier: BSD-3-Clause\nx = 1\n"
    (tmp_path / "a.py").write_text(text)
    subprocess.run(["git", "add", "a.py"], cwd=tmp_path, check=True)
    assert spdx_headers.main(["add"], root=tmp_path) == 1
    assert (tmp_path / "a.py").read_text() == text


def test_every_source_file_in_the_repository_has_a_header():
    assert spdx_headers.main(["check"]) == 0
