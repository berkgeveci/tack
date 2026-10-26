"""The visualization guide's reference pages name everything ``tack.dataset`` exports.

``docs/vis-guide/reference/algorithms.md`` renders each algorithm's docstring
(``::: tack.dataset.algorithms.<module>.<name>``) and ``data-model.md`` each
class and constructor (``::: tack.dataset.<name>``), but only the ones they
list: an algorithm added without its entry would build cleanly and be missing
from the site. These tests make adding the entry part of adding the name.
"""

import pathlib
import re

import pytest

import tack.dataset as td
import tack.dataset.algorithms as algorithms

pytestmark = pytest.mark.quick      # the inner-loop set; see CLAUDE.md, Tests

REFERENCE = pathlib.Path(__file__).resolve().parents[3] / "docs" / "vis-guide" / "reference"


def _entries(page):
    """The objects a page renders, by their ``:::`` paths."""
    return re.findall(r"^::: (\S+)$", (REFERENCE / page).read_text(), re.MULTILINE)


def test_every_algorithm_has_an_entry_on_the_algorithms_page():
    entries = _entries("algorithms.md")
    paths = {f"{getattr(algorithms, name).__module__}.{name}" for name in algorithms.__all__}
    assert sorted(paths - set(entries)) == []
    assert sorted(set(entries) - paths) == []
    assert len(entries) == len(set(entries))


def test_every_algorithm_has_a_row_in_its_groups_table():
    text = (REFERENCE / "algorithms.md").read_text()
    rows = set(re.findall(r"^\| \[`(\w+)`\]\(#", text, re.MULTILINE))
    functions = {name for name in algorithms.__all__ if not isinstance(getattr(algorithms, name), type)}
    assert sorted(functions - rows) == []


def test_everything_else_tack_dataset_exports_has_an_entry_on_the_data_model_page():
    entries = set(_entries("data-model.md"))
    rest = set(td.__all__) - set(algorithms.__all__) - {"algorithms"}
    missing = [name for name in sorted(rest) if f"tack.dataset.{name}" not in entries]
    assert missing == []
