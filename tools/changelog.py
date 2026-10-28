"""The CHANGELOG's entries, one file per change, gathered at release.

A pull request does not edit ``CHANGELOG.md``: it adds a file to
``changelog.d/`` (any name ending ``.md`` but ``README.md``) holding its
entries under the CHANGELOG's own headings::

    ### Fixed

    - `contour` interpolates each crossing from ...

A file may hold several sections (``### Added``, ``### Changed``,
``### Fixed``), each a list of bullets. Since every change adds its own
file, no two pull requests touch one file, and none conflicts with
another however they are merged.

Commands (from the repository root)::

    python tools/changelog.py check              # every file parses
    python tools/changelog.py preview            # the next release's sections
    python tools/changelog.py release 0.7.0 2026-11-30

``release`` writes ``## 0.7.0 — 2026-11-30`` above the last release, the
entries grouped by section and in the order they reached ``main``, Changed's
**Breaking** entries first, and deletes the files; the summary paragraph
under the heading is written by hand, as before.
"""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIRECTORY = ROOT / "changelog.d"
CHANGELOG = ROOT / "CHANGELOG.md"
SECTIONS = ("Added", "Changed", "Fixed")
_HEADING = re.compile(r"^### (\S.*?)\s*$")


class EntryError(ValueError):
    """A file in ``changelog.d/`` that does not parse, saying where."""


def parse(text, name="<entry>"):
    """``{section: [entry, ...]}`` of one file, each entry a bullet with its
    continuation lines, as written."""
    sections = {}
    current = None
    entry = None
    for number, line in enumerate(text.splitlines(), 1):
        where = f"{name}:{number}"
        heading = _HEADING.match(line)
        if heading:
            current = heading.group(1)
            if current not in SECTIONS:
                raise EntryError(f"{where}: section {current!r} is not one of "
                                 f"{', '.join(SECTIONS)}")
            if current in sections:
                raise EntryError(f"{where}: a second {current} section")
            sections[current] = []
            entry = None
        elif line.startswith("#"):
            raise EntryError(f"{where}: only '### Section' headings belong here, "
                             f"not {line.split()[0]!r}")
        elif line.startswith("- "):
            if current is None:
                raise EntryError(f"{where}: an entry before any '### Section' heading")
            entry = [line]
            sections[current].append(entry)
        elif not line.strip():
            if entry is not None:
                entry.append(line)
        elif line.startswith("  ") and entry is not None:
            entry.append(line)
        else:
            raise EntryError(f"{where}: text outside an entry (entries are '- ' "
                             "bullets, continued by indented lines)")
    if not sections:
        raise EntryError(f"{name}: no '### Section' heading")
    for section, entries in sections.items():
        if not entries:
            raise EntryError(f"{name}: {section} has no entries")
    return {section: ["\n".join(lines).rstrip() for lines in entries]
            for section, entries in sections.items()}


def files(directory=DIRECTORY):
    """The entry files, in the order they reached the current branch (the
    commit that added each, first parent first), then any not committed, by
    name."""
    found = sorted(p for p in directory.glob("*.md") if p.name != "README.md")

    def added(path):
        try:
            out = subprocess.run(
                ["git", "log", "--first-parent", "-m", "--diff-filter=A",
                 "--format=%ct", "--", str(path)],
                cwd=directory, capture_output=True, text=True, check=True).stdout.split()
        except (OSError, subprocess.CalledProcessError):
            out = []
        return (0, int(out[-1]), path.name) if out else (1, 0, path.name)

    return sorted(found, key=added)


def gather(paths):
    """Every file's entries, by section, in the files' order; Changed's
    **Breaking** entries first."""
    gathered = {section: [] for section in SECTIONS}
    for path in paths:
        for section, entries in parse(path.read_text(encoding="utf-8"), path.name).items():
            gathered[section].extend(entries)
    gathered["Changed"].sort(key=lambda entry: not entry.startswith("- **Breaking"))
    return gathered


def render(gathered):
    """The sections as the CHANGELOG writes them; empty ones left out."""
    parts = []
    for section in SECTIONS:
        if gathered[section]:
            parts.append(f"### {section}\n\n" + "\n".join(gathered[section]))
    return "\n\n".join(parts)


def insert(changelog, version, date, body):
    """``changelog`` with ``## version — date`` and ``body`` above its first
    release."""
    heading = f"## {version} — {date}"
    if re.search(rf"^## {re.escape(version)} ", changelog, re.MULTILINE):
        raise EntryError(f"CHANGELOG.md already has {version}")
    first = re.search(r"^## ", changelog, re.MULTILINE)
    at = first.start() if first else len(changelog)
    return changelog[:at] + f"{heading}\n\n{body}\n\n" + changelog[at:]


def main(argv):
    if not argv or argv[0] not in ("check", "preview", "release"):
        print(__doc__)
        return 2
    try:
        paths = files()
        gathered = gather(paths)
        if argv[0] == "check":
            print(f"changelog.d: {len(paths)} files, "
                  + ", ".join(f"{len(gathered[s])} {s}" for s in SECTIONS))
        elif argv[0] == "preview":
            print(render(gathered) or "(no entries)")
        else:
            if len(argv) != 3 or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", argv[2]):
                print("usage: tools/changelog.py release VERSION YYYY-MM-DD")
                return 2
            if not paths:
                raise EntryError("changelog.d has no entries to release")
            text = CHANGELOG.read_text(encoding="utf-8")
            CHANGELOG.write_text(insert(text, argv[1], argv[2], render(gathered)),
                                 encoding="utf-8")
            for path in paths:
                path.unlink()
            print(f"CHANGELOG.md: {argv[1]} from {len(paths)} files, which are deleted; "
                  "write the summary under its heading")
    except EntryError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
