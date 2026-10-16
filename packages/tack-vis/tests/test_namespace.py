"""The ``tack`` namespace as tack-vis extends it.

tack-core and tack-vis share the ``tack`` namespace through ``extend_path``,
which merges the directories' *contents* but runs only one ``__init__.py``,
so only one package may ship a given ``tack/<name>/``. ``tack.algorithms`` is
tack-core's primitives alone; tack-vis's algorithms are ``tack.dataset``.
These tests pin that, and that retired packages stay gone.
"""

import pathlib

import pytest

pytestmark = pytest.mark.quick      # the inner-loop set; see CLAUDE.md, Tests


def test_tack_algorithms_is_core_alone():
    import tack.algorithms

    assert "flying_edges" not in tack.algorithms.__all__
    assert callable(tack.algorithms.exclusive_scan)


def test_tack_vis_ships_no_algorithms_directory():
    """A second ``tack/algorithms/__init__.py`` would be dead code: only the
    first one found on the path runs. A directory left holding only
    ``__pycache__`` (an old checkout) ships nothing."""
    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "tack"
    assert not list((src / "algorithms").glob("**/*.py"))


def test_tack_vis_ships_no_tack_init_and_core_initializes_first_or_second():
    """tack-vis's ``tack/`` is a namespace portion, with no ``__init__.py``:
    one there (holding only ``extend_path``) was the package Python ran when
    tack-vis came first on the path, so tack-core's never ran and ``tack``
    had no ``kernel``. Without it, tack-core's runs in either order and
    ``extend_path`` adds tack-vis's portion."""
    import os
    import subprocess
    import sys

    packages = pathlib.Path(__file__).resolve().parents[2]
    vis, core = packages / "tack-vis" / "src", packages / "tack-core" / "src"
    assert not (vis / "tack" / "__init__.py").exists()
    check = ("import tack, tack.dataset, tack.algorithms; "
             "print(callable(tack.kernel), tack.dataset.__name__)")
    for order in ((vis, core), (core, vis)):
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(str(p) for p in order))
        run = subprocess.run([sys.executable, "-c", check], env=env,
                             capture_output=True, text=True, timeout=120)
        assert run.stdout.split() == ["True", "tack.dataset"], run.stdout + run.stderr


def test_dataset_package_exports_the_canonical_model():
    import importlib

    from tack import dataset
    from tack.dataset.dataset import DataSet, Field

    assert dataset.DataSet is DataSet
    assert dataset.Field is Field
    assert DataSet.__module__ == "tack.dataset.dataset"
    assert Field.__module__ == "tack.dataset.dataset"
    assert dataset.contour is importlib.import_module(
        "tack.dataset.algorithms.contour").contour
    for name in dataset.__all__:
        assert hasattr(dataset, name), name


@pytest.mark.parametrize("name", ["data", "deprecated", "fe"])
def test_retired_packages_are_not_shipped(name):
    import importlib.util

    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "tack"
    assert not list((src / name).glob("**/*.py"))
    assert importlib.util.find_spec(f"tack.{name}") is None
