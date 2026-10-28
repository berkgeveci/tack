"""``tools/changelog.py``: entry files parse, gather and land in the CHANGELOG."""

import subprocess

import changelog
import pytest


def test_entries_keep_their_text_and_sections():
    text = ("### Added\n\n- One,\n  continued.\n\n- Two.\n\n"
            "### Fixed\n\n- Three\n\n  with a second paragraph.\n")
    assert changelog.parse(text) == {
        "Added": ["- One,\n  continued.", "- Two."],
        "Fixed": ["- Three\n\n  with a second paragraph."],
    }


@pytest.mark.parametrize("text, message", [
    ("- An entry.\n", "before any"),
    ("### Removed\n\n- Gone.\n", "not one of"),
    ("### Added\n\n", "has no entries"),
    ("### Added\n\n- A.\n\n### Added\n\n- B.\n", "a second Added"),
    ("## 0.7.0\n\n### Added\n\n- A.\n", "only '### Section'"),
    ("### Added\n\nSome prose.\n", "text outside an entry"),
    ("", "no '### Section'"),
])
def test_what_does_not_parse_is_refused_with_where(text, message):
    with pytest.raises(changelog.EntryError, match=message):
        changelog.parse(text, "x.md")


def test_a_release_goes_above_the_last_one_breaking_changes_first(tmp_path):
    (tmp_path / "a.md").write_text("### Changed\n\n- Plain.\n")
    (tmp_path / "b.md").write_text("### Fixed\n\n- Fix.\n\n### Changed\n\n"
                                   "- **Breaking:** renamed.\n")
    (tmp_path / "README.md").write_text("not an entry\n")
    body = changelog.render(changelog.gather(changelog.files(tmp_path)))
    assert body == ("### Changed\n\n- **Breaking:** renamed.\n- Plain.\n\n"
                    "### Fixed\n\n- Fix.")
    old = "# Changelog\n\nPreamble.\n\n## 0.6.0 — 2026-10-28\n\nOld.\n"
    new = changelog.insert(old, "0.7.0", "2026-11-30", body)
    assert new.startswith("# Changelog\n\nPreamble.\n\n## 0.7.0 — 2026-11-30\n\n"
                          "### Changed\n")
    assert new.endswith("## 0.6.0 — 2026-10-28\n\nOld.\n")
    with pytest.raises(changelog.EntryError, match="already has"):
        changelog.insert(new, "0.7.0", "2026-11-30", body)


def test_files_come_in_the_order_they_were_committed(tmp_path):
    def git(*args, when=None):
        env = {"GIT_AUTHOR_DATE": when, "GIT_COMMITTER_DATE": when} if when else {}
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True,
                       env={**env, "PATH": "/usr/bin:/bin:/usr/local/bin",
                            "HOME": str(tmp_path)})

    git("init", "-q")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    for name, when in (("zeta.md", "2026-01-01T00:00:00"), ("alpha.md", "2026-01-02T00:00:00")):
        (tmp_path / name).write_text("### Added\n\n- x\n")
        git("add", name)
        git("commit", "-q", "-m", name, when=when)
    (tmp_path / "middle.md").write_text("### Added\n\n- x\n")   # not committed: last
    assert [p.name for p in changelog.files(tmp_path)] == ["zeta.md", "alpha.md", "middle.md"]


def test_the_repositorys_entries_parse():
    changelog.gather(changelog.files())
