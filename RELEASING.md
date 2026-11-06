# Releasing Tack

`tack-core` is published to PyPI by `.github/workflows/publish.yml`. `tack-vis`
is not published: users install it from this repository at the release tag.

## Once: PyPI, TestPyPI and GitHub

1. **PyPI and TestPyPI.** On each, add a trusted publisher for the project
   `tack-core` (a "pending publisher" before the first upload):
   - owner `Kitware`, repository `tack`;
   - workflow `publish.yml`;
   - environment `pypi` on PyPI, `testpypi` on TestPyPI.
2. **GitHub.** In the repository's settings, under Environments, add `pypi`
   and `testpypi`. Give `pypi` required reviewers, so a tag cannot publish
   without an approval.

No API token is stored anywhere: the upload authenticates with GitHub's OIDC
token.

## Each release

1. **Versions.**
   - Set `version` in `packages/tack-core/pyproject.toml` and
     `packages/tack-vis/pyproject.toml` (say `0.7.0`).
   - Raise tack-vis's `tack-core>=` requirement to the new version, so a
     tagged tack-vis never pairs with an older tack-core from PyPI.
2. **The install commands.** They name the release tag (`@v0.7.0`): in
   `docs/install.md`, `README.md` and `packages/tack-core/README.md`.
3. **The CHANGELOG.** Run `python tools/changelog.py release 0.7.0 YYYY-MM-DD`,
   and write the release's summary in `CHANGELOG.md`.
4. **Merge** those changes to `main`, through a pull request as usual.
5. **The dry run.** Run Actions > Publish > Run workflow on `main`. It builds
   and checks tack-core, then uploads it to TestPyPI. Then install it in a
   clean environment:

   ```bash
   python -m venv /tmp/t && /tmp/t/bin/pip install \
     --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ \
     'tack-core[cpu]==0.7.0'
   cd /tmp && /tmp/t/bin/python /path/to/tack/tools/check_core_install.py
   ```

   TestPyPI keeps a version once uploaded. To run a second dry run of one
   version, upload a dev version (`0.7.0.dev1`) from a branch.
6. **Tag and push.** Push `v0.7.0` on the merge commit. The workflow checks
   that the tag is the package's version, builds and checks again, and waits
   for the `pypi` environment's approval. Approve, and it publishes.
7. **The GitHub Release.** Publish a release from the tag, titled
   `Tack 0.7.0`, its notes the version's section of `CHANGELOG.md` (the
   summary and the entries, without the heading):

   ```bash
   python3 - > /tmp/notes.md <<'EOF'
   s = open("CHANGELOG.md").read()
   start = s.index("\n", s.index("## 0.7.0 ")) + 1
   print(s[start:s.index("\n## ", start)].strip())
   EOF
   gh release create v0.7.0 --verify-tag --title "Tack 0.7.0" --notes-file /tmp/notes.md
   ```

   Publishing it runs `.github/workflows/docs-versions.yml`, which builds the
   tag's documentation as `0.7` on GitHub Pages and makes it `latest`, the
   site's default; the tag alone builds no documentation. GitHub limits the
   notes to 125,000 characters.
8. **Check PyPI.**
   - Run `pip install 'tack-core[cpu]'` in a clean environment, then the
     tack-vis command from `docs/install.md`.
   - The GPU extras are checked on the GPU machines (`[cuda]` on jura, `[hip]`
     on illyad, `[level_zero]` on headroom).
