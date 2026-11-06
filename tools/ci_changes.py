# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Which of CI's jobs a pull request's changed files need.

Reads the changed paths (one a line, ``git diff --name-only``) on standard
input and prints ``group=true`` or ``group=false`` for each group, which
ci.yml's ``changes`` job appends to ``$GITHUB_OUTPUT``:

- ``code``: the packages, their tests, the build and the workflows. Every
  test job, the packaging jobs and Metal.
- ``docs``: the site's pages and the prose around the code. The docs job.
- ``examples``: the shipped examples. The examples job.

Lint runs on every change and is no group. A path no rule names is code, so
a new kind of file runs everything until a rule says otherwise.

    python3 tools/ci_changes.py < changed.txt
"""

import sys

GROUPS = ("code", "docs", "examples")

# (test, groups), first match wins. A test is a prefix ("docs/"), a whole path
# ("mkdocs.yml") or, for the packages' examples, a function.


def _example(path):
    parts = path.split("/")
    return len(parts) >= 4 and parts[0] == "packages" and parts[2] == "examples"


def _prose(path):
    """Markdown outside the packages, which the site or a reader reads, nothing runs."""
    return (path.endswith(".md") and not path.startswith("packages/")) or path.endswith("/CLAUDE.md")


RULES = (
    # Outside the workspace, ruff and the SPDX check; nothing runs it.
    ("archive/", ()),
    # Tested by lint alone (tools/ tests itself there).
    ("tools/", ()),
    (_example, ("examples",)),
    ("docs/", ("docs",)),
    ("overrides/", ("docs",)),
    ("mkdocs.yml", ("docs",)),
    ("changelog.d/", ()),           # lint checks the entries
    ("LICENSE", ()),
    (_prose, ("docs",)),
)


def _matches(test, path):
    if callable(test):
        return test(path)
    return path.startswith(test) if test.endswith("/") else path == test


def groups_of(path):
    """The groups one changed path needs."""
    for test, groups in RULES:
        if _matches(test, path):
            return set(groups)
    return {"code"}


def classify(paths):
    """``{group: needed}`` for a change's paths."""
    needed = set()
    for path in paths:
        needed |= groups_of(path)
    return {group: group in needed for group in GROUPS}


def main():
    paths = [line.strip() for line in sys.stdin if line.strip()]
    for path in paths:
        print(f"{path}: {', '.join(sorted(groups_of(path))) or 'lint'}", file=sys.stderr)
    for group, needed in classify(paths).items():
        print(f"{group}={'true' if needed else 'false'}")


if __name__ == "__main__":
    main()
