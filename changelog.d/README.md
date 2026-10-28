# Changelog entries

Each pull request adds its CHANGELOG entries here, in a file of its own,
rather than editing `CHANGELOG.md`, so pull requests never conflict over it.
Name the file after the change (`contour-crossing-order.md`); write its
entries as the CHANGELOG writes them, under `### Added`, `### Changed` or
`### Fixed` (one file may have several):

```markdown
### Fixed

- `contour`, `slice` and `clip` interpolate each crossing from its edge's
  end first by position, so pieces compute a shared crossing to the same
  bits.
```

A breaking change is a `### Changed` entry starting `- **Breaking:**`.

`python tools/changelog.py check` checks every file (CI's lint job runs it),
`preview` prints the next release's sections, and at a release
`release VERSION YYYY-MM-DD` writes them into `CHANGELOG.md` and deletes the
files.
