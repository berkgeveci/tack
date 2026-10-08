"""``for c in obj``: a data-oriented object supplies the loop's iteration space.

The class declares ``__tack_iterate__``, one to three of its scalar
attributes, fastest first. One makes ``c`` an integer index over
``range``; two or three make ``c`` the vector of indices over an
``ndrange``, which at the top of a kernel launches in its own shape. The
object's methods take ``c`` as it comes, so one kernel source runs over
objects that iterate differently -- which is what lets a cell set iterate
structured cells as (i, j, k) without dividing.
"""

import numpy as np
import pytest

import tack


@tack.data_oriented
class Line:
    __tack_iterate__ = "n"

    def __init__(self, n):
        self.n = n

    @tack.func
    def flat(self, c):
        return c


@tack.data_oriented
class Grid:
    __tack_iterate__ = ("nx", "ny")

    def __init__(self, nx, ny):
        self.nx = nx
        self.ny = ny

    @tack.func
    def flat(self, c):
        return c[0] + self.nx * c[1]


@tack.data_oriented
class Volume:
    __tack_iterate__ = ("nx", "ny", "nz")

    def __init__(self, nx, ny, nz):
        self.nx = nx
        self.ny = ny
        self.nz = nz

    @tack.func
    def flat(self, c):
        return c[0] + self.nx * (c[1] + self.ny * c[2])


@tack.data_oriented
class FixedGrid:
    """Class constants: the extents are compiled in."""
    __tack_iterate__ = ("NX", "NY")
    NX = 6
    NY = 4

    @tack.func
    def flat(self, c):
        return c[0] + self.NX * c[1]


@tack.kernel
def _label(cells, out):
    for c in cells:
        out[cells.flat(c)] = cells.flat(c) + 1


@pytest.mark.parametrize("obj, n", [
    (Line(9), 9), (Grid(7, 5), 35), (Grid(300, 2), 600), (Grid(1, 40), 40),
    (Volume(4, 5, 6), 120), (Volume(65, 3, 2), 390), (FixedGrid(), 24),
], ids=lambda v: f"{type(v).__name__}" if not isinstance(v, int) else str(v))
def test_one_kernel_over_every_iteration_space(backend, obj, n):
    out = tack.field(tack.i32, shape=(n,))
    out.fill(-1)
    _label(obj, out)
    np.testing.assert_array_equal(out.to_numpy(), np.arange(n) + 1)


def test_multi_dimensional_spaces_launch_in_their_shape():
    tack.init(arch=tack.cpu)
    for obj, n in ((Grid(3, 2), 2), (Volume(2, 2, 2), 3)):
        ir_func = tack.inspect(_label, obj, tack.field(tack.i32, shape=(8,)), mode="ir")
        assert ir_func.startswith("Function") and "ParallelFor (" in ir_func
        assert ir_func.count(" < ") >= n
    assert "ParallelFor (" not in tack.inspect(_label, Line(4), tack.field(tack.i32, shape=(4,)),
                                               mode="ir")


def test_instance_extents_do_not_recompile(backend):
    from tack.runtime.dispatch import get_backend
    _label(Grid(3, 5), tack.field(tack.i32, shape=(15,)))
    before = len(get_backend()._cache[_label])
    for nx, ny in ((8, 8), (100, 1), (11, 13)):
        out = tack.field(tack.i32, shape=(nx * ny,))
        _label(Grid(nx, ny), out)
        np.testing.assert_array_equal(out.to_numpy(), np.arange(nx * ny) + 1)
    assert len(get_backend()._cache[_label]) == before


@tack.kernel
def _rows(cells, out, m):
    for r in range(m):
        for c in cells:                      # a sequential loop over the object
            out[r * cells.nx * cells.ny + cells.flat(c)] = r


def test_iterating_inside_the_parallel_loop(backend):
    out = tack.field(tack.i32, shape=(3 * 12,))
    _rows(Grid(4, 3), out, 3)
    np.testing.assert_array_equal(out.to_numpy(), np.repeat(np.arange(3), 12))


@tack.data_oriented
class Particles:
    __tack_iterate__ = "count"

    def __init__(self, count):
        self.count = count
        self.x = tack.field(tack.f32, shape=(count,))

    @tack.kernel
    def advance(self, dt: tack.f32):
        for p in self:
            self.x[p] += dt


def test_a_kernel_method_iterates_its_own_object(backend):
    particles = Particles(10)
    particles.advance(0.5)
    particles.advance(0.25)
    np.testing.assert_allclose(particles.x.to_numpy(), 0.75)


# ── Errors ──────────────────────────────────────────────────────────

@tack.data_oriented
class Undeclared:
    def __init__(self):
        self.n = 3


@tack.data_oriented
class TooMany:
    __tack_iterate__ = ("a", "b", "c", "d")
    a = b = c = d = 2


@tack.data_oriented
class NotAScalar:
    __tack_iterate__ = "values"

    def __init__(self):
        self.values = tack.field(tack.f32, shape=(4,))


@tack.kernel
def _pairs(cells, out):
    for c, d in cells:
        out[c] = d


@pytest.mark.parametrize("obj, message", [
    (Undeclared(), "declares no __tack_iterate__"),
    (TooMany(), "must name one to three attributes"),
    (NotAScalar(), "'values', which is not a scalar attribute"),
], ids=["undeclared", "too-many", "not-a-scalar"])
def test_what_cannot_be_iterated(obj, message):
    tack.init(arch=tack.cpu)
    with pytest.raises(TypeError, match=message):
        _label(obj, tack.field(tack.i32, shape=(16,)))


def test_the_loop_binds_one_name():
    tack.init(arch=tack.cpu)
    with pytest.raises(TypeError, match="binds one name, the index"):
        _pairs(Grid(2, 2), tack.field(tack.i32, shape=(4,)))


def test_the_declaration_is_read_from_the_class():
    from tack.lang.template_rewrite import iteration_space
    assert iteration_space(Line) == ("n",)
    assert iteration_space(Volume) == ("nx", "ny", "nz")
