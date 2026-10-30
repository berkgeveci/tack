# Copyright (c) 2026 Kitware, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""MFEM's examples through Tack: each problem solved with PyMFEM (``_mfem``), its
mesh and solution imported with ``tack.interop.mfem``, and what Tack's filters
make of them drawn through VTK: contours and slices cut on sub-cells of curved
cells, curved and discontinuous meshes subdivided, a beam warped by its solution.

From the checkout, with the ``interop`` extra (writes
docs/assets/gallery/mfem-<scene>.jpg):
uv run python docs/gallery/mfem_examples.py [escher fichera beam star sphere advection dg-beam]
"""

import _mfem
import _scene
import numpy as np

import tack
import tack.dataset as td
from tack.dataset import implicit
from tack.interop.mfem import mfem_to_dataset

SCENES = ("escher", "fichera", "beam", "star", "sphere", "advection", "dg-beam")


def _range(values):
    v = np.asarray(values, np.float64)
    return float(v.min()), float(v.max())


def escher(out):
    # --8<-- [start:escher]
    # ex1 on the Escher knot: curved tetrahedra of order 3 (MFEM's "Cubic"
    # nodes), the solution in H1 of order 3. Each level is cut on sub-cells, 4
    # a side, its crossings placed on the field itself.
    mesh = _mfem.mesh("escher-p3.mesh", 1500)
    x = _mfem.poisson(mesh, 3)
    data = mfem_to_dataset(mesh, {"u": x})
    top = float(np.abs(x.GetDataArray()).max())
    levels = [0.15 * top, 0.45 * top, 0.75 * top]
    surfaces = [_scene.normals(td.contour(data, "u", v, subdivision=4)) for v in levels]
    # --8<-- [end:escher]
    parts = []
    for v, surface, opacity in zip(levels, surfaces, (0.25, 0.45, 1.0), strict=True):
        surface.fields["level"] = td.Field(td.Constant(surface), tack.array(
            np.full(surface.num_cells, v, np.float32)))
        parts.append((surface, "level", (levels[0], levels[-1]), {"opacity": opacity}))
    _scene.draw(out / "mfem-escher.jpg", parts, outline=_scene.outline(data),
                view=(20.0, 25.0))


def fichera(out):
    # --8<-- [start:fichera]
    # ex1 on the Fichera corner: curved hexahedra of order 3, the solution of
    # order 3, sliced through the re-entrant corner.
    mesh = _mfem.mesh("fichera-q3.mesh", 600)
    x = _mfem.poisson(mesh, 3)
    data = mfem_to_dataset(mesh, {"u": x})
    cut = td.slice(data, implicit.Plane((0.0, 0.0, 0.0), (1.0, -1.0, 1.0)), fields=["u"],
                   subdivision=6)
    # --8<-- [end:fichera]
    _scene.draw(out / "mfem-fichera.jpg", [(cut, "u", _range(x.GetDataArray()))],
                outline=_scene.outline(data), view=(35.0, 25.0), lit=False)


def beam(out):
    # --8<-- [start:beam]
    # ex2: the two-material beam of hexahedra, its displacement in H1 of order 2.
    # warp lifts the straight geometry into the displacement's space, so the
    # displaced beam is curved, MFEM's displaced mesh node for node.
    mesh = _mfem.mesh("beam-hex.mesh", 600)
    x = _mfem.elasticity(mesh, 2)
    data = mfem_to_dataset(mesh, {"u": x})
    data.fields["|u|"] = td.magnitude(data, "u")
    moved = td.warp(data, "u", scale=1.0)
    # --8<-- [end:beam]
    _scene.draw(out / "mfem-beam.jpg",
                [(_scene.boundary(moved), "|u|", _range(data.fields["|u|"].values.numpy()))],
                outline=_scene.outline(data), view=(-30.0, 20.0))


def star(out):
    # --8<-- [start:star]
    # ex6: the star refined where Zienkiewicz-Zhu's estimate is largest, to 3000
    # unknowns: a nonconforming mesh, its hanging nodes in the cells' edges.
    mesh = _mfem.mesh("star.mesh")
    x = _mfem.adaptive_poisson(mesh, 2, 3000)
    data = mfem_to_dataset(mesh, {"u": x})
    surface = td.subdivide(data, 4)
    # --8<-- [end:star]
    _scene.draw(out / "mfem-star.jpg", [(surface, "u", _range(x.GetDataArray()))],
                outline=_scene.outline(data), lit=False, flat=True)


def sphere(out):
    # --8<-- [start:sphere]
    # ex7: the cube's quads snapped onto the sphere, curved cells of order 3, and
    # the Laplace-Beltrami solution on them.
    mesh = _mfem.sphere(3)
    x = _mfem.laplace_beltrami(mesh, 3)
    data = mfem_to_dataset(mesh, {"u": x})
    surface = _scene.normals(td.subdivide(data, 6))
    # --8<-- [end:sphere]
    _scene.draw(out / "mfem-sphere.jpg", [(surface, "u", _range(x.GetDataArray()))],
                outline=_scene.outline(data), view=(30.0, 25.0))


def advection(out):
    # --8<-- [start:advection]
    # ex9: a DG bump of order 3 carried across the periodic hexagon. A periodic
    # mesh keeps its nodes per element, an L2 geometry: cells on two sides of the
    # seam share points in the topology and sit apart in space, so each cell is
    # subdivided on its own lattice.
    mesh = _mfem.mesh("periodic-hexagon.mesh")
    for _ in range(2):
        mesh.UniformRefinement()
    u = _mfem.advection(mesh, 0.5)
    data = mfem_to_dataset(mesh, {"u": u})
    surface = td.subdivide(data, 6)
    edges = td.extract_edges(td.subdivide(data, 1, fields=[]))  # each cell's own corners
    # --8<-- [end:advection]
    _scene.draw(out / "mfem-advection.jpg", [(surface, "u", _range(u.GetDataArray()))],
                outline=edges, lit=False, flat=True)


def dg_beam(out):
    # --8<-- [start:dg-beam]
    # ex17: the beam of quads displaced by its DG solution of order 2 (MFEM moves
    # the mesh's nodes into the DG space): a curved, discontinuous L2 geometry,
    # each cell where MFEM puts it.
    mesh = _mfem.mesh("beam-quad.mesh", 400)
    x = _mfem.dg_elasticity(mesh, 2)
    mesh.SetNodalFESpace(x.FESpace())
    nodes = mesh.GetNodes()
    nodes += x
    data = mfem_to_dataset(mesh, {"u": x})
    data.fields["|u|"] = td.magnitude(data, "u")
    surface = td.subdivide(data, 4)
    # --8<-- [end:dg-beam]
    _scene.draw(out / "mfem-dg-beam.jpg",
                [(surface, "|u|", _range(data.fields["|u|"].values.numpy()))],
                outline=td.extract_edges(td.subdivide(data, 1, fields=[])), lit=False,
                flat=True)


DRAW = {"escher": escher, "fichera": fichera, "beam": beam, "star": star, "sphere": sphere,
        "advection": advection, "dg-beam": dg_beam}


def main():
    args = _scene.arguments(__doc__, SCENES)
    for name in args.kinds:
        DRAW[name](args.output)


if __name__ == "__main__":
    main()
