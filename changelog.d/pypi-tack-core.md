### Added

- `tack-core` is published to PyPI: `pip install "tack-core[cpu]"` (and
  `[cuda]`, `[metal]`, `[hip]`). `tack-vis` is installed from the repository at
  the release tag (`docs/install.md`). The publish workflow
  (`.github/workflows/publish.yml`) builds and checks the wheel and the sdist on
  a release tag and uploads them with trusted publishing; run by hand, it
  uploads to TestPyPI. `RELEASING.md` has the steps.

### Changed

- The install docs, the READMEs and the messages for a missing backend give
  `pip install "tack-core[...]"` instead of an install from a clone (a clone's
  `uv sync` stays, for working on Tack). Asking for `tack.dataset` without
  tack-vis names the command that installs it from the repository.
- CI's wheel job also builds tack-core's sdist, checks both with
  `twine check`, and installs tack-core alone, from the wheel and from the
  sdist (`tools/check_core_install.py`).
