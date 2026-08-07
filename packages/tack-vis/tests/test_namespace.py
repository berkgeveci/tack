"""The merged `tack.algorithms` namespace.

tack-core and tack-vis both shipped a `tack/algorithms/__init__.py`.
`extend_path` merges the two directories' *contents*, but only one
`__init__.py` ever runs — the first found on the path, which is
tack-core's. So tack-vis's re-exports were dead code, and the import the
docs promise raised ImportError in a fresh process:

    >>> from tack.algorithms import flying_edges
    ImportError: cannot import name 'flying_edges'

Confusingly it appeared to work once anything had imported the submodule,
because that binds the *module* under the name rather than the function —
so `flying_edges(...)` then failed with "module is not callable" instead.

The worklets are re-exported from tack-core's `__init__` now, guarded so
tack-core alone still works. These tests pin both halves.
"""

import pytest


def test_worklets_import_from_the_documented_path():
    """The import the README, CLAUDE.md and docs all promise."""
    from tack.algorithms import (
        cell_to_point,
        compute_normals,
        flying_edges,
        flying_edges_multiblock,
    )
    assert callable(flying_edges)
    assert callable(flying_edges_multiblock)
    assert callable(compute_normals)
    assert callable(cell_to_point)


def test_the_names_are_functions_not_modules():
    """The failure mode that made this look like it worked.

    Importing `tack.algorithms.flying_edges` binds the module under that
    attribute. If the re-export is missing, the name resolves to a module
    that is not callable — which reads as a very strange error.
    """
    import types

    import tack.algorithms.flying_edges  # noqa: F401  (bind the module)
    from tack.algorithms import flying_edges
    assert not isinstance(flying_edges, types.ModuleType)


def test_core_primitives_are_still_there():
    """Merging the namespaces must not displace what was already in it."""
    from tack.algorithms import (
        exclusive_scan,
        histogram,
    )
    assert callable(exclusive_scan)
    assert callable(histogram)


def test_all_covers_both_packages():
    import tack.algorithms as algorithms
    for name in ("exclusive_scan", "histogram",
                 "flying_edges", "compute_normals", "cell_to_point"):
        assert name in algorithms.__all__, name


@pytest.mark.parametrize("name", [
    "exclusive_scan", "inclusive_scan", "copy", "fill_value",
    "var", "std", "norm", "absmax", "count_nonzero", "dot", "histogram",
    "flying_edges", "flying_edges_multiblock", "compute_normals",
    "cell_to_point", "UniformGrid", "MCTables",
])
def test_every_advertised_name_resolves(name):
    """__all__ must not promise anything it cannot deliver."""
    import tack.algorithms as algorithms
    assert hasattr(algorithms, name), f"__all__ lists {name} but it is missing"


def test_only_one_algorithms_init_exists():
    """A second __init__.py would be dead code again, silently.

    Whichever one is not first on the path never executes, so having two
    is the bug this test exists to prevent coming back.
    """
    import pathlib
    repo = pathlib.Path(__file__).resolve().parents[3]
    inits = list(repo.glob("packages/*/src/tack/algorithms/__init__.py"))
    assert len(inits) == 1, (
        f"{len(inits)} algorithms/__init__.py files exist; only the first on "
        f"the path runs, so the others are dead code: {inits}")
