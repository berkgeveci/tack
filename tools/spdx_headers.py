# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Every source file opens with an SPDX header, as ParaView's and VTK's do::

    # SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
    # SPDX-License-Identifier: BSD-3-Clause

A file with code from elsewhere names each holder on a line of its own and
the license it is under, in the same form::

    # SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
    # SPDX-FileCopyrightText: Copyright (c) The Taichi Authors
    # SPDX-License-Identifier: Apache-2.0

The files are the record: this tool writes Kitware's header where a file has
none and checks that every one has a header, whatever it names. Sources are
the tracked ``.py``, ``.js``, ``.css`` and ``.html`` files outside
``archive/``; data, generated files and prose (Markdown) carry none, as in
ParaView. The header comes first, after a ``#!`` line if there is one.

Commands (from the repository root)::

    python tools/spdx_headers.py check     # every source file has a header
    python tools/spdx_headers.py add       # write Kitware's where one is missing
"""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HOLDER = "Kitware Inc."
LICENSE = "BSD-3-Clause"
#: Comment syntax per suffix: (opening, closing).
COMMENTS = {
    ".py": ("# ", ""),
    ".js": ("// ", ""),
    ".css": ("/* ", " */"),
    ".html": ("{# ", " #}"),
}
EXCLUDED = ("archive/",)
_COPYRIGHT = "SPDX-FileCopyrightText: Copyright (c) "
_LICENSE = "SPDX-License-Identifier: "


def header(suffix, holders=(HOLDER,), license=LICENSE):
    """The header's lines for a file of ``suffix``."""
    opening, closing = COMMENTS[suffix]
    lines = [f"{opening}{_COPYRIGHT}{holder}{closing}" for holder in holders]
    return [*lines, f"{opening}{_LICENSE}{license}{closing}"]


def problem(text, suffix):
    """Why ``text`` has no well-formed header, or None if it has one."""
    opening, closing = COMMENTS[suffix]
    lines = text.splitlines()
    if lines and lines[0].startswith("#!"):
        lines = lines[1:]
    wrapped = re.compile(re.escape(opening) + "(.*)" + re.escape(closing) + "$")
    holders = 0
    for line in lines:
        match = wrapped.match(line)
        body = match.group(1) if match else None
        if body is not None and body.startswith(_COPYRIGHT) and body[len(_COPYRIGHT):].strip():
            holders += 1
            continue
        if body is not None and body.startswith(_LICENSE) and body[len(_LICENSE):].strip():
            return None if holders else "an SPDX-License-Identifier with no copyright above it"
        break
    if holders:
        return "no SPDX-License-Identifier after the copyright lines"
    return "no SPDX header at the top"


def with_header(text, suffix):
    """``text`` with Kitware's header first (after any ``#!`` line)."""
    lines = text.splitlines(keepends=True)
    first = lines[:1] if lines and lines[0].startswith("#!") else []
    rest = "".join(lines[len(first):])
    block = "".join(line + "\n" for line in header(suffix))
    return "".join(first) + block + ("\n" + rest if rest.strip() else "")


def sources(root=ROOT):
    """The tracked source files that carry a header, relative to ``root``."""
    listed = subprocess.run(["git", "ls-files", "-z"], cwd=root, check=True,
                            capture_output=True, text=True).stdout.split("\0")
    return sorted(Path(name) for name in listed
                  if name and Path(name).suffix in COMMENTS
                  and not name.startswith(EXCLUDED) and (root / name).is_file())


def main(argv, root=ROOT):
    if argv not in (["check"], ["add"]):
        print(__doc__.split("Commands")[1].strip(), file=sys.stderr)
        return 2
    missing = []
    for path in sources(root):
        text = (root / path).read_text(encoding="utf-8")
        why = problem(text, path.suffix)
        if why is None:
            continue
        if argv[0] == "add" and why == "no SPDX header at the top":
            (root / path).write_text(with_header(text, path.suffix), encoding="utf-8")
            print(f"{path}: added Kitware's header")
        else:
            missing.append(f"{path}: {why}")
    for line in missing:
        print(line, file=sys.stderr)
    if missing:
        print(f"{len(missing)} files without a well-formed SPDX header; "
              f"`python tools/spdx_headers.py add` writes Kitware's ({LICENSE}). "
              "Code from elsewhere names its holders and license instead.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
