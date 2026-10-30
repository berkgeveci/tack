"""``L2``'s layout (the vault's coefficient-layout proposal): ``"blocks"`` keeps each
cell's values together, ``"cells"`` stores them cell-fastest, ``"auto"`` chooses by
device. Every operation must give the same answer either way; only where the
values sit differs, which ``space.blocks`` and ``space.indices`` undo."""

import numpy as np
import pytest

import tack
import tack.dataset as td
from tack.dataset import spaces

pytestmark = pytest.mark.quick


@tack.func
def _cubic(x):
    return x[0] * x[0] * x[0] - 2.0 * x[1] * x[2] + 0.5 * x[2] * x[2] * x[0] + 1.0


def _cubic_at(p):
    return p[:, 0] ** 3 - 2 * p[:, 1] * p[:, 2] + 0.5 * p[:, 2] ** 2 * p[:, 0] + 1


def _hexes():
    grid = td.uniform_grid((4, 3, 3), spacing=(0.5, 0.5, 0.5), dtype=tack.f64)
    return td.extract_cells(grid, np.arange(grid.num_cells))


def _both(data, order=3, basis="gauss_legendre"):
    out = {}
    for layout in ("blocks", "cells"):
        space = td.L2(data, order=order, basis=basis, layout=layout)
        out[layout] = td.interpolate(data, _cubic, space)
    return out


def _points(data, n=200):
    x = data.positions()
    return np.random.default_rng(0).uniform(x.min(0), x.max(0), (n, 3))


def test_the_parameter(f64_backend, monkeypatch):
    monkeypatch.delenv("TACK_L2_LAYOUT", raising=False)             # "auto" by device
    data = _hexes()
    gpu = data.device.type != "cpu"
    assert td.L2(data, order=3).layout == ("cells" if gpu else "blocks")    # auto
    assert td.L2(data, order=2).layout == "blocks"                 # auto, everywhere
    assert td.L2(data, order=3, layout="cells").layout == "cells"
    assert td.L2(data, order=3, layout="auto") is td.L2(data, order=3)
    assert repr(td.L2(data, order=3, layout="cells")) == \
        "L2(order=3, basis='gauss_lobatto', layout='cells')"
    assert td.L2(data).layout == "blocks"                          # corner values
    with pytest.raises(ValueError, match="for tensor bases"):
        td.L2(data, order=1, layout="cells")
    with pytest.raises(ValueError, match="no common stride"):
        td.L2(data, order=np.ones(data.num_cells, int), layout="cells")
    with pytest.raises(ValueError, match="layout is one of"):
        td.L2(data, order=3, layout="rows")


def test_the_same_values_where_each_layout_keeps_them(f64_backend):
    data = _hexes()
    fields = _both(data)
    blocks, cells = (fields[k] for k in ("blocks", "cells"))
    np.testing.assert_array_equal(blocks.space.blocks(blocks.values.numpy()),
                                  cells.space.blocks(cells.values.numpy()))
    ids = cells.space.indices()
    np.testing.assert_array_equal(cells.values.numpy()[ids], blocks.space.blocks(
        blocks.values.numpy()))


def test_every_operation_gives_the_same_answer(f64_backend):
    data = _hexes()
    points = _points(data)
    results = {}
    for layout, field in _both(data).items():
        data.fields["u"] = field
        target = td.L2(data, order=4, basis="gauss_lobatto", layout=layout)
        kept = td.extract_cells(data, np.array([1, 4, 7]))
        results[layout] = (
            td.probe(data, points).fields["u"].values.numpy(),
            td.gradients(data, "u", at="cells").values.numpy(),
            target.blocks(td.project(data, field, target).values.numpy()),
            kept.fields["u"].space.blocks(kept.fields["u"].values.numpy()),
            np.sort(td.contour(data, "u", 1.2).positions(), 0),
            td.subdivide(data, 2, fields=["u"]).fields["u"].values.numpy(),
        )
        assert kept.fields["u"].space.basis == "gauss_legendre"         # the basis kept
        assert kept.fields["u"].space.layout == layout
    for a, b in zip(results["blocks"], results["cells"]):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(results["cells"][0], _cubic_at(points), atol=1e-12)


def test_a_filter_keeps_the_basis(f64_backend):
    """A filter carrying an ``L2`` field kept its order but not its nodes: a
    Gauss-Legendre field came out labelled Gauss-Lobatto, another basis."""
    data = _hexes()
    data.fields["u"] = td.interpolate(data, _cubic, td.L2(data, order=3, basis="gauss_legendre"))
    kept = td.extract_cells(data, np.array([0, 5]))
    assert kept.fields["u"].space.basis == "gauss_legendre"
    points = _points(data, 400)
    out = td.probe(kept, points)
    held = out.fields["valid"].values.numpy() == 1
    assert held.any()
    np.testing.assert_allclose(out.fields["u"].values.numpy()[held], _cubic_at(points[held]),
                               atol=1e-12)


def test_owned_values_drop_ghost_cells_values(f64_backend):
    data = _hexes()
    marks = np.zeros(data.num_cells, np.uint8)
    marks[[2, 3]] = td.Ghosts.DUPLICATECELL
    data.ghosts = td.Ghosts(cells=tack.array(marks), points=tack.zeros((data.num_points,),
                                                                        tack.u8))
    owned = {}
    for layout, field in _both(data).items():
        data.fields["u"] = field
        owned[layout] = np.sort(td.owned_values(data, "u").numpy())
    np.testing.assert_array_equal(owned["blocks"], owned["cells"])
    assert len(owned["cells"]) == (data.num_cells - 2) * 64


def test_moving_to_a_gpu_lays_out_again(f64_backend, monkeypatch):
    """``"auto"`` resolves on the destination: a field moved where it chooses
    cell-fastest is transposed, and back again (a second CPU stands in for a GPU)."""
    from tack.runtime import dispatch
    from tack.runtime.cpu import CPUBackend

    class _Second(CPUBackend):
        @property
        def memory_domain(self):
            return ("cpu", 9)

    second = dispatch.as_device(_Second())
    monkeypatch.delenv("TACK_L2_LAYOUT", raising=False)             # "auto" by device
    monkeypatch.setattr(spaces, "_on_gpu", lambda topology: topology.device == second)
    data = _hexes()
    data.fields["u"] = td.interpolate(data, _cubic, td.L2(data, order=3))
    assert data.fields["u"].space.layout == "blocks"
    there = data.to_device(second)
    assert there.fields["u"].space.layout == "cells"
    back = there.to_device(tack.device("cpu"))
    np.testing.assert_array_equal(back.fields["u"].values.numpy(),
                                  data.fields["u"].values.numpy())
    np.testing.assert_array_equal(
        there.fields["u"].space.blocks(there.fields["u"].values.numpy()),
        data.fields["u"].space.blocks(data.fields["u"].values.numpy()))


def test_the_environment_overrides_auto(f64_backend, monkeypatch):
    data = _hexes()
    monkeypatch.setenv("TACK_L2_LAYOUT", "cells")
    assert td.L2(data, order=2).layout == "cells"
    assert td.L2(data, order=2, layout="blocks").layout == "blocks"    # explicit stays
    assert td.L2(data).layout == "blocks"                             # corner values
    monkeypatch.setenv("TACK_L2_LAYOUT", "diagonal")
    with pytest.raises(ValueError, match="TACK_L2_LAYOUT"):
        _ = td.L2(data, order=3, basis="equispaced").layout


def test_mfem_reads_and_writes_either_layout(f64_backend, monkeypatch):
    mfem = pytest.importorskip("mfem.ser")
    from tack.interop.mfem import mfem_grid_function, mfem_to_dataset

    mesh = mfem.Mesh.MakeCartesian3D(2, 2, 2, mfem.Element.HEXAHEDRON, 1.0, 1.0, 1.0)
    fec = mfem.L2_FECollection(3, 3)
    fes = mfem.FiniteElementSpace(mesh, fec)
    gf = mfem.GridFunction(fes)

    class F(mfem.PyCoefficient):
        def EvalValue(self, x):
            return x[0] ** 3 - 2 * x[1] * x[2] + 0.5 * x[2] ** 2 * x[0] + 1

    gf.ProjectCoefficient(F())
    for layout in ("blocks", "cells"):
        monkeypatch.setenv("TACK_L2_LAYOUT", layout)
        data = mfem_to_dataset(mesh, {"u": gf}, dtype=tack.f64)
        assert data.fields["u"].space.layout == layout
        points = np.random.default_rng(1).uniform(0.05, 0.95, (50, 3))
        np.testing.assert_allclose(td.probe(data, points).fields["u"].values.numpy(),
                                   _cubic_at(points), atol=1e-10)
        back = mfem_grid_function(data, data.fields["u"], mesh)
        np.testing.assert_array_equal(back.GetDataArray(), gf.GetDataArray())
