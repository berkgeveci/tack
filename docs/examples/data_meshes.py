# Copyright (c) 2026 Kitware, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Assemble datasets from topology and geometry, then check factory shortcuts."""

import argparse

import numpy as np

import tack
import tack.dataset as td


def run(arch="cpu", check=False):
    tack.set_default_device(arch)
    # --8<-- [start:uniform]
    point_dims = (3, 2, 2)
    uniform_topology = td.StructuredTopology(point_dims)
    uniform_coordinates = tack.arrays.UniformCoordinates(point_dims, spacing=(0.5, 1.0, 1.0))
    uniform_geometry = td.Field(td.H1(uniform_topology), uniform_coordinates)
    uniform = td.DataSet(uniform_topology, uniform_geometry)
    # --8<-- [end:uniform]
    # --8<-- [start:rectilinear]
    x_axis, y_axis, z_axis = [0.0, 0.25, 1.0], [0.0, 1.0], [0.0, 1.0]
    rectilinear_topology = td.StructuredTopology((len(x_axis), len(y_axis), len(z_axis)))
    rectilinear_coordinates = tack.arrays.CartesianProduct(x_axis, y_axis, z_axis)
    rectilinear_geometry = td.Field(td.H1(rectilinear_topology), rectilinear_coordinates)
    rectilinear = td.DataSet(rectilinear_topology, rectilinear_geometry)
    # --8<-- [end:rectilinear]
    # --8<-- [start:curvilinear]
    curvilinear_topology = td.StructuredTopology((3, 2, 2))
    # Explicit corner positions in x-fast order, with a bend in z.
    positions = np.array([[0.5 * i, j, k + 0.1 * i * j]
                          for k in range(2) for j in range(2) for i in range(3)], np.float32)
    curvilinear_coordinates = tack.array(positions, dtype=tack.vec3())
    curvilinear_geometry = td.Field(td.H1(curvilinear_topology), curvilinear_coordinates)
    curvilinear = td.DataSet(curvilinear_topology, curvilinear_geometry)
    # --8<-- [end:curvilinear]
    # --8<-- [start:polyhedron]
    xyz = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], np.float32)
    faces = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4],
             [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
    polyhedral_topology = td.PolyhedralTopology.from_cell_faces(
        [faces], num_points=8, positions=xyz, orient=True)
    polyhedral_coordinates = tack.array(xyz, dtype=tack.vec3())
    polyhedral_geometry = td.Field(td.H1(polyhedral_topology), polyhedral_coordinates)
    cube = td.DataSet(polyhedral_topology, polyhedral_geometry)
    # --8<-- [end:polyhedron]
    # --8<-- [start:polygon]
    ring_points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], np.float32)
    polygon_topology = td.PolygonalTopology([0, 4], [0, 1, 2, 3], num_points=4)
    polygon_geometry = td.Field(td.H1(polygon_topology), tack.array(ring_points, dtype=tack.vec3()))
    square = td.DataSet(polygon_topology, polygon_geometry)
    # --8<-- [end:polygon]
    # --8<-- [start:polyline]
    path_points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], np.float32)
    polyline_topology = td.PolylineTopology([0, 3], [0, 1, 2], num_points=3)
    polyline_geometry = td.Field(td.H1(polyline_topology), tack.array(path_points, dtype=tack.vec3()))
    path = td.DataSet(polyline_topology, polyline_geometry)
    # --8<-- [end:polyline]
    # --8<-- [start:grids]
    # Shortcuts for the same topology-and-geometry combinations built above.
    uniform_shortcut = td.uniform_grid((3, 2, 2), spacing=(0.5, 1.0, 1.0))
    rectilinear_shortcut = td.rectilinear_grid([0.0, 0.25, 1.0], [0.0, 1.0], [0.0, 1.0])
    # --8<-- [end:grids]
    if check:
        ijk = np.array([[i, j, k] for k in range(2) for j in range(2) for i in range(3)])
        expected_uniform = ijk * [0.5, 1, 1]
        np.testing.assert_allclose(uniform.positions(), expected_uniform)
        expected_rectilinear = np.c_[np.array(x_axis)[ijk[:, 0]], ijk[:, 1:]]
        np.testing.assert_allclose(rectilinear.positions(), expected_rectilinear)
        expected_curvilinear = expected_uniform.copy()
        expected_curvilinear[:, 2] += 0.1 * ijk[:, 0] * ijk[:, 1]
        np.testing.assert_allclose(curvilinear.positions(), expected_curvilinear)
        for data in (uniform, rectilinear, curvilinear):
            assert isinstance(data.topology, td.StructuredTopology)
            assert data.geometry.space is td.H1(data.topology)
            assert (data.num_points, data.num_cells) == (12, 2)
        assert isinstance(uniform.geometry.values, tack.arrays.UniformCoordinates)
        assert isinstance(rectilinear.geometry.values, tack.arrays.CartesianProduct)
        assert curvilinear.geometry.values.dtype is tack.vec3()
        for explicit, shortcut in ((uniform, uniform_shortcut), (rectilinear, rectilinear_shortcut)):
            assert type(explicit.topology) is type(shortcut.topology)
            assert type(explicit.geometry.values) is type(shortcut.geometry.values)
            assert explicit.topology.point_dims == shortcut.topology.point_dims
            assert explicit.geometry.space.order == shortcut.geometry.space.order == 1
            np.testing.assert_array_equal(explicit.positions(), shortcut.positions())
            np.testing.assert_allclose(td.cell_geometry(explicit).measure.values.numpy(),
                                       td.cell_geometry(shortcut).measure.values.numpy())
        for data, expected in ((cube, 1.0), (square, 1.0), (path, 2.0)):
            np.testing.assert_allclose(td.cell_geometry(data).measure.values.numpy(), [expected])
        assert len(td.check_winding(cube)) == 0
        assert len(td.boundary_faces(cube)) == 6
    print("Dataset construction: explicit topology and geometry; factory equivalence checked")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default="cpu", choices=["cpu", "metal", "cuda", "hip", "level_zero"])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    run(args.arch, args.check)
