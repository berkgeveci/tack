"""The coefficient layout of an ``L2`` field (question 1 of the vault's Dataset API todo):
each cell's values together (Tack's layout, MFEM's and VTK cell grids': cell ``c``'s
value ``n`` at ``c * N + n``) against cell-fastest (value ``n`` of every cell together,
``n * C + c``), timed through Tack's own kernels on one device.

Cell-fastest is prototyped here as a subclass of ``L2`` whose field view reads
``get(n * num_cells + cell)``; nothing else changes, so every operation runs the
code it runs on Tack's layout. Per order, on ``cells^3`` hexahedra of an
``L2`` field on Gauss-Legendre nodes, warm medians (ms) of:

- ``value``: each cell evaluated at a point of its own (a probe's evaluation);
- ``gradient``: its parametric gradient there;
- ``bound``: its Bernstein coefficients' range (subdivision's culling);
- ``write``: every value written by the cell that owns it (a projection's store);
- ``probe``: 100k random points located and evaluated;
- ``subdivide``: the field at the lattice of 2 steps a cell.

Both layouts' results are checked equal before their times count.

    python benchmarks/coefficient_layout.py --arch metal --cells 48 --orders 2,3,4,6
"""

import argparse
import json
import time

import numpy as np

import tack
import tack.dataset as td
from tack.dataset import views


class _H1InteriorCellsField(views._H1TensorField):
    """``H1`` of a tensor basis with its cell interiors cell-fastest (the proposal's D3,
    prototyped): interior value ``a`` of cell ``c`` at ``interior_base + a *
    interior_cells + c``; points, edges and faces as ``H1`` keeps them."""

    @tack.func
    def dof_index(self, c, n):
        kind = self.node_kind[n]
        e = self.node_entity[n]
        out = self.interior_base + self.node_a[n] * self.interior_cells + self.entity_id(c)
        if kind == 0:
            out = self.point_id(c, e)
        if kind == 1:
            out = self.tensor_edge_dof(c, e, self.node_a[n])
        if kind == 2:
            out = self.tensor_face_dof(c, e, self.node_a[n], self.node_b[n])
        return out


class H1Cells(td.H1):
    """The prototype space: ``H1`` of a tensor basis, its cell interiors cell-fastest
    (every cell has an interior here: one shape)."""

    @property
    def mixin(self):
        return views.of_order(_H1InteriorCellsField, self.order)

    def mixin_for(self, key, shape=None):
        return self.mixin

    def attributes(self, group=None, dtype=None):
        out = super().attributes(group, dtype)
        out["interior_base"] = int(self._layout()[1].numpy()[0])
        out["interior_cells"] = self.topology.num_cells
        return out


def _spaces(data, family, order):
    """The two layouts' spaces and a map from the blocks field's values to the other's."""
    n = data.num_cells
    if family == "l2":
        blocks = td.L2(data, order=order, nodes="gauss_legendre", layout="blocks")
        cells = td.L2(data, order=order, nodes="gauss_legendre", layout="cells")
        per_cell = (order + 1) ** 3
        where = np.arange(per_cell)[None, :] * n + np.arange(n)[:, None]   # [c, k]
        return blocks, cells, lambda v: _scatter(v, where.reshape(-1), per_cell * n)
    blocks = td.H1(data, order=order)
    cells = H1Cells(data, order=order)
    base = int(blocks._layout()[1].numpy()[0])
    m = (order - 1) ** 3
    new = np.arange(blocks.size)
    new[base:] = base + (np.arange(m)[None, :] * n + np.arange(n)[:, None]).reshape(-1)
    return blocks, cells, lambda v: _scatter(v, new, blocks.size)


def _scatter(values, where, size):
    out = np.empty(size, values.dtype)
    out[where] = values
    return out


@tack.kernel
def _value(cells, u, pcs, out):
    for c in cells:
        e = cells.entity_id(c)
        out[e] = u.value(c, pcs[e])


@tack.kernel
def _gradient(cells, u, pcs, out):
    for c in cells:
        e = cells.entity_id(c)
        out[e] = u.parametric_gradient(c, pcs[e])


@tack.kernel
def _bound(cells, u, lo, hi):
    for c in cells:
        e = cells.entity_id(c)
        a, b = u.value_range(c)
        lo[e] = a
        hi[e] = b


@tack.kernel
def _write_blocks(cells, per_cell, out):
    # Each cell writes its own values: per cell, c * N + n.
    for c in cells:
        e = cells.entity_id(c)
        for n in range(per_cell):
            out[e * per_cell + n] = tack.f32(e + n)


@tack.kernel
def _write_cells(cells, per_cell, ncells, out):
    # Cell-fastest: n * C + c.
    for c in cells:
        e = cells.entity_id(c)
        for n in range(per_cell):
            out[n * ncells + e] = tack.f32(e + n)


@tack.func
def _smooth(x):
    return x.x * x.x * x.x - x.y * x.z + 0.5 * x.y * x.y + 0.25 * x.z * x.z * x.x


def _timed(f, runs):
    f()
    tack.sync()
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        r = f()
        tack.sync()
        times.append(1e3 * (time.perf_counter() - t))
    return r, float(np.median(times))


@tack.kernel
def _write_interiors(cells, base, count, per_cell, cell_fastest, out):
    # Each cell writes its own values (an L2 field's, or an H1 field's interior):
    # blocks from base + c * per_cell, or cell-fastest at base + n * cells + c.
    for c in cells:
        e = cells.entity_id(c)
        for n in range(per_cell):
            at = base + n * count + e if cell_fastest == 1 else base + e * per_cell + n
            out[at] = tack.f32(e + n)


def run(arch, cells, orders, runs, family="l2"):
    tack.set_default_device(arch)
    grid = td.uniform_grid((cells + 1,) * 3, spacing=(1.0 / cells,) * 3)
    data = td.extract_cells(grid, np.arange(grid.num_cells))
    n = data.num_cells
    rng = np.random.default_rng(0)
    host_pcs = rng.uniform(0.05, 0.95, (n, 3)).astype(np.float32)
    pcs = tack.empty((n,), tack.vec3())
    tack.copyto(pcs, host_pcs)
    points = rng.uniform(0.0, 1.0, (100_000, 3)).astype(np.float32)
    rows = []
    for order in orders:
        blocks_space, cells_space, relaid = _spaces(data, family, order)
        blocks = td.interpolate(data, _smooth, blocks_space)
        moved = tack.empty((blocks_space.size,), tack.f32)
        tack.copyto(moved, relaid(blocks.values.numpy()))
        fields = {"blocks": blocks, "cells": td.Field(cells_space, moved)}
        if family == "l2":
            base, per_cell = 0, (order + 1) ** 3
        else:
            base, per_cell = int(blocks_space._layout()[1].numpy()[0]), (order - 1) ** 3
        results = {}
        for layout, field in fields.items():
            def each(kernel, *outputs):
                for group in data.launch_groups("cells", [field]):
                    if group.count:
                        kernel(data.domain_view("cells", group), field.view(group), *outputs)

            out = tack.empty((n,), tack.f32)
            grad = tack.empty((n,), tack.vec3())
            lo, hi = tack.empty((n,), tack.f32), tack.empty((n,), tack.f32)
            written = tack.empty((blocks_space.size,), tack.f32)
            timings = {}
            _, timings["value"] = _timed(lambda: each(_value, pcs, out), runs)
            _, timings["gradient"] = _timed(lambda: each(_gradient, pcs, grad), runs)
            _, timings["bound"] = _timed(lambda: each(_bound, lo, hi), runs)

            def write():
                for group in data.launch_groups("cells", []):
                    if group.count:
                        _write_interiors(data.domain_view("cells", group), base, n, per_cell,
                                         int(layout == "cells"), written)

            _, timings["write"] = _timed(write, runs)
            data.fields["u"] = field
            probed, timings["probe"] = _timed(lambda: td.probe(data, points), runs)
            refined, timings["subdivide"] = _timed(
                lambda: td.subdivide(data, 2, fields=["u"]), runs)
            del data.fields["u"]
            results[layout] = (out.numpy(), grad.numpy(), lo.numpy(), hi.numpy(),
                               probed.fields["u"].values.numpy(),
                               refined.fields["u"].values.numpy())
            for what, t in timings.items():
                rows.append({"arch": arch, "family": family, "order": order,
                             "layout": layout, "what": what, "ms": t})
        same = all(np.array_equal(a, b) for a, b in zip(results["blocks"], results["cells"]))
        if not same:
            raise AssertionError(f"{family} order {order}: the two layouts' results differ")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arch", default="cpu")
    parser.add_argument("--cells", type=int, default=48, help="cells a side")
    parser.add_argument("--orders", default="2,3,4,6")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--family", default="l2", choices=["l2", "h1"],
                        help="l2: L2's values; h1: H1's cell interiors (orders 3 and up)")
    parser.add_argument("--json", help="write the rows here")
    args = parser.parse_args()
    orders = [int(o) for o in args.orders.split(",")]
    if args.family == "h1" and min(orders) < 3:
        parser.error("--family h1 measures cell interiors, which start at order 3")
    rows = run(args.arch, args.cells, orders, args.runs, args.family)
    print(f"{args.arch}, {args.family}, {args.cells}^3 hexahedra; ms, warm medians: per cell, "
          "cell-fastest, and their ratio")
    whats = list(dict.fromkeys(r["what"] for r in rows))
    print(f"{'order':>5} " + " ".join(f"{w:>22}" for w in whats))
    for order in dict.fromkeys(r["order"] for r in rows):
        cells_ = []
        for what in whats:
            b = next(r["ms"] for r in rows if r["order"] == order
                     and r["layout"] == "blocks" and r["what"] == what)
            c = next(r["ms"] for r in rows if r["order"] == order and r["layout"] == "cells"
                     and r["what"] == what)
            cells_.append(f"{b:8.2f} {c:8.2f} {c / b:4.2f}")
        print(f"{order:>5} " + " ".join(f"{x:>22}" for x in cells_))
    if args.json:
        json.dump(rows, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
