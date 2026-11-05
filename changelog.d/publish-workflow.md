### Added

- A publish workflow for tack-core (`.github/workflows/publish.yml`). On a
  release tag it checks the tag is the version, builds the wheel and the sdist,
  `twine check`s them, installs tack-core alone from each, and uploads to PyPI
  with trusted publishing after the `pypi` environment's approval. Run by hand,
  it uploads to TestPyPI, the dry run. `RELEASING.md` has the one-time setup and
  a release's steps. tack-vis is not published; it is installed from the
  repository at the release tag.

### Changed

- CI's wheel job also builds tack-core's sdist, checks both with `twine check`,
  and installs tack-core alone, from the wheel and from the sdist
  (`tools/check_core_install.py`). Asking for `tack.dataset` without tack-vis
  names the command that installs it from the repository.
