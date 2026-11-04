# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Build datasets from your own numbers: point clouds, sources, fields from
arrays and functions, and named sets (the Data layer's "Build meshes and
fields" page)."""

import argparse

import numpy as np

import tack
import tack.dataset as td


# --8<-- [start:function]
@tack.func
def swirl(p):
    # A world position in, a 3-vector out.
    return tack.vec3(-p.y, p.x, 0.0 * p.z)
# --8<-- [end:function]


# --8<-- [start:kernel]
@tack.kernel
def distance_from_center(xyz, out):
    # xyz is the geometry's values: here an image's positions, computed as read.
    for p in range(out.shape[0]):
        x = xyz[p]
        out[p] = (x - tack.vec3(0.5)).norm()
# --8<-- [end:kernel]


def run(arch="cpu", check=False):
    tack.set_default_device(arch)

    # --8<-- [start:point-cloud]
    rng = np.random.default_rng(1)
    xyz = rng.random((1000, 3))                               # world positions, float64
    cloud = td.point_cloud(xyz, fields={"mass": rng.random(1000),
                                        "velocity": rng.random((1000, 3))},
                           units={"mass": "kg", "velocity": "m/s"}, unit="m")
    print(cloud.num_points, cloud.num_cells, cloud.fields["velocity"].values.dtype)
    # 1000 1000 tack.vec(3, tack.f32)
    # --8<-- [end:point-cloud]

    # --8<-- [start:sources]
    wave = td.wavelet(extent=10)               # 21^3 points, point field "RTData"
    knot = td.tangle(dims=(32, 32, 32))        # point field "tangle"
    pulse = td.amr_gaussian_pulse()            # an AMRDataSet of two levels
    print(wave.num_points, sorted(wave.fields), sorted(knot.fields))
    # 9261 ['RTData'] ['tangle']
    # --8<-- [end:sources]

    # --8<-- [start:from-arrays]
    data = td.uniform_grid((5, 4, 3), spacing=(0.25, 1 / 3, 0.5))
    x = data.positions()                                     # a host copy, (60, 3)

    # Point data: one value per point.
    height = tack.array(x[:, 2].astype(np.float32))
    data.fields["height"] = td.Field(td.H1(data), height, unit="m")

    # Cell data: one value per cell.
    material = tack.array(np.arange(data.num_cells) % 2, dtype=tack.i32)
    data.fields["material"] = td.Field(td.Constant(data), material)

    # A vector at each point: an array of 3-vectors.
    wind = tack.array(np.tile([1.0, 0.0, 0.0], (data.num_points, 1)).astype(np.float32),
                      dtype=tack.vec3())
    data.fields["wind"] = td.Field(td.H1(data), wind, unit="m/s")

    # One value per face, with no basis: an area, a flux, a label.
    faces = td.Values(data, "faces")
    data.fields["face_label"] = td.Field(faces, tack.zeros((faces.size,), tack.i32))
    print(data.num_points, data.num_cells, faces.size)
    # 60 24 98
    # --8<-- [end:from-arrays]

    # --8<-- [start:from-kernel]
    radius = tack.empty((data.num_points,))
    distance_from_center(data.geometry.values, radius)
    data.fields["radius"] = td.Field(td.H1(data), radius, unit="m")
    # --8<-- [end:from-kernel]

    # --8<-- [start:interpolate]
    flow = td.uniform_grid((9, 9, 3), origin=(-1, -1, 0), spacing=(0.25, 0.25, 0.5))
    # Point data: the function at each point.
    flow.fields["velocity"] = td.interpolate(flow, swirl, td.H1(flow), dtype=tack.vec3())
    # Cell data: the function at each cell's center.
    flow.fields["velocity", td.Constant] = td.interpolate(flow, swirl, td.Constant(flow),
                                                          dtype=tack.vec3())
    # A quadratic field: the function at every node of order 2.
    quadratic = td.H1(flow, order=2)
    flow.fields["velocity2"] = td.interpolate(flow, swirl, quadratic, dtype=tack.vec3())
    print(flow.num_points, quadratic.size)
    # 243 1445
    # --8<-- [end:interpolate]

    # --8<-- [start:sets]
    # Cells by id; duplicates collapse and the ids are kept sorted.
    data.sets["corner"] = td.Set(data, "cells", [5, 0, 0])
    # Cells by a flag per cell, as a kernel or NumPy computes it.
    data.sets["odd"] = td.Set.from_flags(data, "cells", material)
    # Faces by (cell, local face) pairs, as an Exodus side set gives them.
    data.sets["inflow"] = td.Set.from_sides(data, cells=[0, 4, 8], local_faces=[0, 0, 0])
    # The faces with one cell: the boundary.
    data.sets["boundary"] = td.boundary_faces(data)
    print({name: len(s) for name, s in data.sets.items()})
    # {'corner': 2, 'odd': 12, 'inflow': 3, 'boundary': 52}
    # --8<-- [end:sets]

    if check:
        assert cloud.num_points == cloud.num_cells == 1000
        assert cloud.fields["mass"].unit == "kg"
        assert cloud.fields["velocity"].values.dtype is tack.vec3()
        np.testing.assert_allclose(cloud.world_positions(), xyz, atol=1e-6)
        assert wave.num_points == 21 ** 3 and "RTData" in wave.fields
        assert knot.num_points == 32 ** 3 and "tangle" in knot.fields
        assert isinstance(pulse, td.AMRDataSet)
        assert (data.num_points, data.num_cells, faces.size) == (60, 24, 98)
        np.testing.assert_allclose(data.fields["radius"].values.numpy(),
                                   np.linalg.norm(x - 0.5, axis=1), rtol=1e-6)
        v = flow.fields["velocity", td.H1].values.numpy()
        p = flow.positions()
        np.testing.assert_allclose(v, np.c_[-p[:, 1], p[:, 0], np.zeros(len(p))], atol=1e-6)
        assert flow.fields["velocity", td.Constant].space.size == flow.num_cells
        assert quadratic.size == 17 * 17 * 5
        np.testing.assert_array_equal(data.sets["corner"].numpy(), [0, 5])
        assert len(data.sets["odd"]) == 12
        assert len(data.sets["inflow"]) == 3
        # An image of 4 x 3 x 2 cells: 2 * (12 + 8 + 6) boundary faces.
        assert len(data.sets["boundary"]) == 52
    print("Build meshes and fields: point clouds, sources, fields and sets checked")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default="cpu", choices=["cpu", "metal", "cuda", "hip", "level_zero"])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    run(args.arch, args.check)
