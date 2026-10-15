"""The ``tack.algorithms`` and ``tack.deprecated`` namespaces.

tack-core and tack-vis both used to ship a ``tack/algorithms/`` directory.
``extend_path`` merges the directories' *contents*, but only one
``__init__.py`` runs -- tack-core's -- so tack-vis's worklets reached
``tack.algorithms`` only through a guarded re-export there. The worklets now
live in ``tack.deprecated.algorithms`` (superseded by ``tack.dataset``), tack-vis
ships no ``tack/algorithms/`` at all, and ``tack.algorithms`` is tack-core's
primitives alone. These tests pin both halves.
"""

import pathlib
import types

import pytest

pytestmark = pytest.mark.quick      # the inner-loop set; see CLAUDE.md, Tests


def test_the_worklets_import_from_tack_deprecated():
    from tack.deprecated.algorithms import (
        cell_to_point,
        compute_normals,
        flying_edges,
        flying_edges_multiblock,
    )
    for f in (cell_to_point, compute_normals, flying_edges, flying_edges_multiblock):
        assert callable(f) and not isinstance(f, types.ModuleType)
    from tack.deprecated.fe import HexBasis, QuadBasis  # noqa: F401


def test_tack_algorithms_is_core_alone():
    import tack.algorithms

    assert "flying_edges" not in tack.algorithms.__all__
    assert callable(tack.algorithms.exclusive_scan)


def test_tack_vis_ships_no_algorithms_directory():
    """A second ``tack/algorithms/__init__.py`` would be dead code: only the
    first one found on the path runs. A directory left holding only
    ``__pycache__`` (a checkout from before the worklets moved) ships nothing."""
    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "tack"
    assert not list((src / "algorithms").glob("**/*.py"))


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


def test_retired_data_package_is_not_shipped():
    import importlib.util

    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "tack"
    assert not list((src / "data").glob("**/*.py"))
    assert importlib.util.find_spec("tack.data") is None
