# Governance

Tack is a young project, started at Kitware. This document says who makes
decisions, how they are made, and how people come to share in them. It is
deliberately small and will grow with the project (see the last section).

## Roles

**Contributors** are everyone who opens an issue, submits a pull request or
reviews one. [CONTRIBUTING.md](CONTRIBUTING.md) says how, and the
[Code of Conduct](CODE_OF_CONDUCT.md) applies to everyone.

**Maintainers** review and merge pull requests, triage issues, look after
CI, and plan releases with the lead. They act for the project, not for
their employers.

**The project lead** is a maintainer who sets the project's direction,
decides when maintainers cannot agree, and is responsible for releases.

| Role         | Name            | Organization | GitHub                                              |
|--------------|-----------------|--------------|-----------------------------------------------------|
| Project lead | Berk Geveci     | Kitware      | [berkgeveci](https://github.com/berkgeveci)         |
| Maintainer   | Jeff Lee        | Kitware      | [jlee-kitware](https://github.com/jlee-kitware)     |
| Maintainer   | Utkarsh Ayachit | NVIDIA       | [utkarshayachit](https://github.com/utkarshayachit) |

## How decisions are made

Most decisions are made in pull requests. A pull request is merged when CI
passes, a maintainer other than its author has approved it, and no
maintainer's objection is left unanswered. A change to documentation or CI
configuration alone may be merged by its author once CI passes. While the
lead is the only maintainer taking part, the lead merges their own pull
requests once CI passes.

Larger changes start with an issue, before the code: a new backend, a
change to the public API that breaks existing callers, a new required
dependency, or a change to the license or to this document. That lets the
other maintainers weigh in while changing course is still cheap. Longer
design notes go in `docs/design/`.

Maintainers try to reach agreement in the issue or pull request. When they
cannot, the lead decides and says why in the same thread.

## Roadmap and releases

The roadmap is the issue tracker. Each planned release is a milestone, and
the work planned for it is the issues and pull requests assigned to that
milestone. Anyone may propose work by opening an issue; maintainers decide
what goes into a milestone, and the lead decides when a release is made.

The lead, or a maintainer the lead asks, makes a release: gathers the
CHANGELOG with `python tools/changelog.py release VERSION DATE`, tags the
commit, and publishes a GitHub Release, which publishes that version's
documentation.

## Becoming a maintainer

Anyone with a record of good contributions (code, reviews, documentation,
answers to other people's questions) can become a maintainer. A maintainer
nominates them in an issue; they are added if no maintainer objects within
a week, and the lead decides if one does.

Maintainership belongs to the person, not their employer: it does not pass
to a colleague, and it stays with someone who changes jobs. Maintainers who
can no longer take part step down by a pull request to this file, with the
project's thanks.

## Copyright

Contributors, or their employers, keep the copyright to their work, and
license it under the files' license, as [CONTRIBUTING.md](CONTRIBUTING.md)
describes. A file's header names the holders of its copyright: Kitware's
line by default, and an author from another organization adds their own
line to the files they write or substantially change.

## Changing this document

By a pull request, approved by the lead. This model suits a handful of
maintainers. It should be revisited when maintainers come from a third
organization, or before Tack 1.0, whichever comes first; a more formal
model, such as a steering group or a foundation, may suit the project then.
