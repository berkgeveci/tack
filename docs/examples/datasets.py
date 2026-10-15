# Copyright (c) 2026 Kitware, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Datasets and their algorithms (``tack.dataset``): the code of the User's Guide
chapter, run end to end."""

import argparse

import numpy as np

import tack


def run(arch="cpu", check=False):
    tack.init(arch=getattr(tack, arch))
    import tack.dataset as td

    # --8<-- [start:build]
    # --8<-- [start:unstructured]
    # A hexahedron with a pyramid on top: VTK cell types, offsets, connectivity.
    positions = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                          [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
                          [0.5, 0.5, 1.6]], np.float32)
    topology = td.UnstructuredTopology(types=[12, 14], offsets=[0, 8, 13],
                                       connectivity=[0, 1, 2, 3, 4, 5, 6, 7, 4, 5, 6, 7, 8])
    coordinate_values = tack.array(positions, dtype=tack.vec3())
    geometry = td.Field(td.H1(topology), coordinate_values)
    mesh = td.DataSet(topology, geometry)
    # --8<-- [end:unstructured]

    # --8<-- [start:fields]
    height = tack.empty(tack.f32, shape=(mesh.num_points,))
    height.from_numpy(positions[:, 2].copy())
    mesh.fields["height"] = td.Field(td.H1(mesh), height)            # point data
    material = tack.empty(tack.f32, shape=(mesh.num_cells,))
    material.from_numpy(np.array([1.0, 2.0], np.float32))
    mesh.fields["material"] = td.Field(td.Constant(mesh), material)  # cell data
    # --8<-- [end:fields]
    # --8<-- [end:build]

    # --8<-- [start:pipeline]
    data = td.wavelet(extent=10)                     # 21^3 points, field "RTData"
    surface = td.contour(data, "RTData", 157.0)      # triangles, RTData carried
    inside = td.threshold(data, "RTData", 200.0, 300.0)
    skin = td.external_faces(inside)                 # the kept region's boundary
    cut = td.slice_plane(data, origin=(0, 0, 0), normal=(0, 0, 1))
    stats = td.statistics(data, "RTData")            # a dict of Python values
    print(f"contour: {surface.num_cells} triangles; threshold: {inside.num_cells} "
          f"cells, {skin.num_cells} boundary faces; mean RTData {stats['mean']:.1f}")
    # --8<-- [end:pipeline]

    # --8<-- [start:spaces]
    on_points = td.to_points(mesh, "material")       # Constant -> H1: averaged
    on_cells = td.to_cells(mesh, "height")           # H1 -> Constant: corner means
    dg = td.project(mesh, "height", td.L2(mesh))     # each cell's own corner values
    slope = td.gradients(mesh, "height")             # a 3-vector per cell
    # --8<-- [end:spaces]

    # --8<-- [start:sets]
    mesh.sets["boundary"] = td.boundary_faces(mesh)  # a face Set
    mesh.sets["roof"] = td.Set(mesh, "cells", [1])   # a cell Set
    kept = td.threshold(mesh, "material", 1.5, 3.0)  # sets come along
    print(f"boundary: {len(mesh.sets['boundary'])} faces; the roof is cell "
          f"{kept.sets['roof'].to_numpy()} of the threshold")
    # --8<-- [end:sets]

    # --8<-- [start:kernel]
    @tack.kernel
    def centers(cells, u, out, where):
        # Once per shape: `cells` is that shape's view, `u` the field's.
        for c in cells:
            pc = cells.parametric_center()
            out[cells.entity_id(c)] = u.value(c, pc)
            where[cells.entity_id(c)] = cells.position(c, pc)

    values = tack.empty(tack.f32, shape=(mesh.num_cells,))
    where = tack.empty(dtype=tack.vec3(tack.f32), shape=(mesh.num_cells,))
    td.for_each(centers, mesh, "cells", mesh.fields["height"], values, where)
    # --8<-- [end:kernel]

    # --8<-- [start:faces]
    @tack.kernel
    def surface_areas(cells, out):
        # Every face of every cell, walked outward from the cell: the same code
        # for a hexahedron, a pyramid or a polyhedron.
        for c in cells:
            total = 0.0
            for k in range(cells.num_faces(c)):
                a = cells.side_position(c, k, 0)
                n = a - a
                for j in range(1, cells.side_num_points(c, k) - 1):
                    n += (cells.side_position(c, k, j) - a).cross(
                        cells.side_position(c, k, j + 1) - a)
                total += 0.5 * n.norm()
            out[cells.entity_id(c)] = total

    mesh.topology.faces()                            # the faces kernels walk
    areas = tack.empty(tack.f32, shape=(mesh.num_cells,))
    td.for_each(surface_areas, mesh, "cells", areas)

    polyhedra = td.as_polyhedra(mesh)                # every cell a polyhedron
    same = tack.empty(tack.f32, shape=(polyhedra.num_cells,))
    td.for_each(surface_areas, polyhedra, "cells", same)
    # --8<-- [end:faces]

    # --8<-- [start:lines]
    flow = td.uniform_grid((9, 9, 3), origin=(-1, -1, 0), spacing=(0.25, 0.25, 0.5))
    x = flow.positions()
    swirl = tack.empty(dtype=tack.vec3(tack.f32), shape=(flow.num_points,))
    swirl.from_numpy(np.c_[-x[:, 1], x[:, 0], np.zeros(len(x))].astype(np.float32).ravel())
    flow.fields["velocity"] = td.Field(td.H1(flow), swirl)
    paths = td.streamlines(flow, "velocity", seeds=[[0.5, 0, 0.5], [0, 0.75, 0.5]],
                           step_size=0.05, steps=40)
    lengths = td.cell_geometry(paths).measure        # a polyline's length
    # --8<-- [end:lines]

    # --8<-- [start:render]
    from tack.rendering import (
        Actor,
        Canvas,
        ColorTable,
        PerspectiveCamera,
        PointLight,
        Scene,
        render,
    )

    positions = surface.geometry.values              # 3-vectors, stored interleaved
    points = positions.reshape(positions.size)       # a flat view: nothing copied
    heights = tack.empty(tack.f32, shape=(surface.num_points,))
    heights.from_numpy(surface.positions()[:, 2].copy())
    scene = Scene()
    scene.add(Actor(points, surface.topology.connectivity, scalars=heights,
                    color_table=ColorTable("viridis"), smooth=True))
    scene.add(PointLight(position=(30, 40, 50), intensity=6000.0))
    size = 64 if check else 400
    camera = PerspectiveCamera((25, 20, 30), (0, 0, 0), width=size, height=size)
    canvas = Canvas(size, size)
    render(canvas, scene, camera, samples=1 if check else 8, max_bounces=1)
    image = canvas.to_numpy()                        # (size, size, 4) uint8
    # --8<-- [end:render]

    if check:
        assert surface.num_cells > 0 and inside.num_cells > 0 and skin.num_cells > 0
        assert set(skin.topology.types.to_numpy()) <= {9}               # quads
        assert cut.num_cells > 0
        np.testing.assert_allclose(surface.fields["RTData"].values.to_numpy(), 157.0,
                                   atol=1e-3)
        np.testing.assert_allclose(on_points.values.to_numpy()[[0, 4, 8]], [1.0, 1.5, 2.0])
        np.testing.assert_allclose(on_cells.values.to_numpy(), [0.5, (4 + 1.6) / 5],
                                   rtol=1e-6)
        assert dg.values.shape[0] == 8 + 5
        np.testing.assert_allclose(td.arrays.to_host(slope.values)[0], [0, 0, 1],
                                   atol=1e-6)
        assert len(mesh.sets["boundary"]) == 9                          # 5 + 4 faces
        np.testing.assert_array_equal(kept.sets["roof"].to_numpy(), [0])
        np.testing.assert_allclose(values.to_numpy()[0], 0.5)
        np.testing.assert_allclose(td.arrays.to_host(where)[0], [0.5, 0.5, 0.5])
        np.testing.assert_allclose(areas.to_numpy(), same.to_numpy(), rtol=1e-6)
        np.testing.assert_allclose(areas.to_numpy()[0], 6.0, rtol=1e-6)
        assert isinstance(paths.topology, td.PolylineTopology) and paths.num_cells == 2
        # Circles at unit angular speed for 40 steps of 0.05: 2 r long.
        np.testing.assert_allclose(lengths.values.to_numpy(), [2 * 0.5, 2 * 0.75], rtol=2e-3)
        assert image.shape == (size, size, 4)
        assert np.count_nonzero(canvas.depth_to_numpy() >= 0) > 100     # the surface is in view
    return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default="cpu", choices=["cpu", "metal", "cuda", "hip", "level_zero"])
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", help="Save the rendered image (requires Pillow)")
    args = parser.parse_args()
    image = run(args.arch, args.check)
    if args.output:
        from PIL import Image

        Image.fromarray(image).save(args.output)
    print("Datasets: check passed" if args.check else "Datasets: done")
