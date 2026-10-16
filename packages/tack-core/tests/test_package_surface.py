"""What ``import tack`` offers: ``__all__``, submodules on first use, typing.

``tack`` is marked typed (``py.typed``), and a typed package's ``__init__``
re-exports only what ``__all__`` names, as far as type checkers go: a name
missing from it reads to pyright as private.
"""

import pathlib
import subprocess
import sys

import numpy as np
import pytest

import tack

pytestmark = pytest.mark.quick      # the inner-loop set; see CLAUDE.md, Tests


def test_all_names_exist_and_cover_the_vocabulary():
    assert not [n for n in tack.__all__ if not hasattr(tack, n)]
    for name in ("kernel", "func", "device_class", "Array", "empty", "ScalarType", "f32",
                 "vec3", "set_default_device", "sync"):
        assert name in tack.__all__


def test_public_submodules_load_on_first_use():
    """``import tack`` alone reaches them; it used to need ``import tack.arrays``."""
    check = ("import tack; print(tack.arrays.CountingArray(3).size, "
             "tack.algorithms.exclusive_scan.__name__, tack.math.__name__, "
             "tack.random.__name__, tack.cache.__name__)")
    run = subprocess.run([sys.executable, "-c", check], capture_output=True, text=True,
                         timeout=120)
    assert run.stdout.split() == ["3", "exclusive_scan", "tack.math", "tack.random",
                                  "tack.cache"], run.stderr
    with pytest.raises(AttributeError, match="no attribute 'nope'"):
        _ = tack.nope


def test_a_cast_on_the_host_says_what_to_use():
    with pytest.raises(RuntimeError, match=r"np\.float32\(x\) or array\.astype\(tack\.f32\)"):
        tack.f32(1.0)


@tack.kernel
def _truncate(x, out):
    for i in range(x.shape[0]):
        out[i] = tack.f32(tack.i32(x[i]))


def test_casts_inside_kernels_are_unchanged(backend):
    x = tack.array(np.array([1.7, -2.5, 3.2], np.float32))
    out = tack.zeros((3,))
    _truncate(x, out)
    np.testing.assert_array_equal(out.numpy(), [1, -2, 3])


def test_both_packages_are_marked_typed():
    packages = pathlib.Path(__file__).resolve().parents[2]
    assert (packages / "tack-core" / "src" / "tack" / "py.typed").exists()
    for sub in ("dataset", "interop"):
        assert (packages / "tack-vis" / "src" / "tack" / sub / "py.typed").exists()
