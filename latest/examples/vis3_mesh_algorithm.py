# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Mesh algorithms written with views: cell, face and node kernels on a mixed
mesh, each checked against an analytic answer or a built-in algorithm."""

import argparse

import numpy as np

import tack
import tack.dataset as td


# --8<-- [start:corner-mean]
@tack.kernel
def corner_mean(cells, u, out):
    for c in cells:
        total = 0.0
        for j in range(cells.NUM_POINTS):
            total += u.corner_value(c, j)
        out[cells.entity_id(c)] = total / cells.NUM_POINTS
# --8<-- [end:corner-mean]


# --8<-- [start:center]
@tack.kernel
def at_centers(cells, u, values, positions, gradients):
    for c in cells:
        e = cells.entity_id(c)
        pc = cells.parametric_center()          # in the cell's reference element
        values[e] = u.value(c, pc)
        positions[e] = cells.position(c, pc)    # mapped into the world
        gradients[e] = td.world_gradient(cells, u, c, pc)
# --8<-- [end:center]


# --8<-- [start:volumes]
@tack.kernel
def volumes(cells, out):
    # The divergence theorem for v = x, whose divergence is 3: the outward flux
    # through the cell's faces is three times its volume. Each face is fanned
    # into triangles from its first point.
    for c in cells:
        flux = 0.0
        for k in range(cells.num_faces(c)):
            a = cells.side_position(c, k, 0)
            for j in range(1, cells.side_num_points(c, k) - 1):
                b = cells.side_position(c, k, j)
                d = cells.side_position(c, k, j + 1)
                area = 0.5 * (b - a).cross(d - a)      # outward, times the area
                flux += area.dot(a + b + d) / 3.0      # v at the triangle's centroid
        out[cells.entity_id(c)] = flux / 3.0
# --8<-- [end:volumes]


# --8<-- [start:faces]
@tack.kernel
def differences(faces, cell_values, out):
    for f in faces:
        e = faces.entity_id(f)
        out[e] = 0.0                                   # a boundary face has one side
        if faces.num_sides(f) == 2:
            out[e] = cell_values[faces.side_cell(f, 1)] - cell_values[faces.side_cell(f, 0)]
# --8<-- [end:faces]


# --8<-- [start:nodes]
@tack.kernel
def sample(nodes, values):
    for i in nodes:
        for j in range(nodes.node_count(i)):
            x = nodes.node_position(i, j)
            values[nodes.node_dof(i, j)] = x[0] * x[1]
# --8<-- [end:nodes]


# --8<-- [start:field-nodes]
@tack.kernel
def add_at_nodes(cells, p, q, out):
    for c in cells:
        for k in range(q.NODES):
            out[q.dof_index(c, k)] = p.value(c, q.node_parametric(k)) + q.dof(c, k)
# --8<-- [end:field-nodes]


def run(arch="cpu", check=False):
    tack.set_default_device(arch)
    # --8<-- [start:data]
    # A unit cube (a hexahedron) and, beside it, a wedge sharing its x = 1 face.
    points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                       [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
                       [2, 0, 0], [2, 0, 1]], np.float32)
    topology = td.UnstructuredTopology(
        types=np.array([td.shapes.HEXAHEDRON, td.shapes.WEDGE], np.uint8),
        offsets=np.array([0, 8, 14]),
        connectivity=np.array([0, 1, 2, 3, 4, 5, 6, 7, 1, 8, 2, 5, 9, 6]),
        num_points=10)
    data = td.DataSet(topology, points)
    T = points[:, 0] + 2 * points[:, 1] + 3 * points[:, 2]      # linear: T = x + 2y + 3z
    data.fields["T"] = td.Field(td.H1(data), tack.array(T))
    # --8<-- [end:data]

    # --8<-- [start:run-corner-mean]
    means = tack.zeros((data.num_cells,), tack.f32)
    td.for_each(corner_mean, data, "cells", data.fields["T"], means)
    print(means.numpy())                    # [3.  3.5]: 8 corners, then 6
    # --8<-- [end:run-corner-mean]

    # --8<-- [start:groups]
    T_field = data.fields["T"]
    by_hand = tack.zeros((data.num_cells,), tack.f32)
    for group in data.launch_groups("cells", [T_field]):
        if group.count:
            print(group.shape.__name__, group.count)          # Hexahedron 1, then Wedge 1
            cells = data.domain_view("cells", group)          # this shape's cells
            corner_mean(cells, T_field.view(group), by_hand)  # T, for the same cells
    # --8<-- [end:groups]

    # --8<-- [start:subset]
    data.sets["wedge"] = td.Set(data, "cells", [1])
    wedge_mean = tack.zeros((data.num_cells,), tack.f32)       # room for every cell
    td.for_each(corner_mean, data, "wedge", data.fields["T"], wedge_mean)
    print(wedge_mean.numpy())               # [0.  3.5]: cell 0 keeps its zero
    # --8<-- [end:subset]

    # --8<-- [start:run-center]
    n = data.num_cells
    values = tack.zeros((n,), tack.f32)
    positions = tack.zeros((n,), tack.vec3())
    gradients = tack.zeros((n,), tack.vec3())
    td.for_each(at_centers, data, "cells", data.fields["T"], values, positions, gradients)
    print(gradients.numpy())                # (1, 2, 3) in both cells
    # --8<-- [end:run-center]

    # --8<-- [start:run-volumes]
    data.topology.faces()                   # derive the faces once, before the kernel
    volume = tack.zeros((data.num_cells,), tack.f32)
    td.for_each(volumes, data, "cells", volume)
    print(volume.numpy())                   # [1.  0.5]

    polyhedra = td.as_polyhedra(data)       # the same cells, as polyhedra
    same = tack.zeros((polyhedra.num_cells,), tack.f32)
    td.for_each(volumes, polyhedra, "cells", same)
    # --8<-- [end:run-volumes]

    # --8<-- [start:run-faces]
    faces = data.topology.faces()
    jumps = tack.zeros((faces.num_faces,), tack.f32)
    td.for_each(differences, data, "faces", means, jumps)
    print(faces.num_faces, jumps.numpy().nonzero()[0])   # 10 faces, one of them inside
    # --8<-- [end:run-faces]

    # --8<-- [start:run-nodes]
    space = td.H1(data, order=2)
    xy = tack.zeros((space.size,), tack.f32)
    td.for_each(sample, data, space, xy)
    q = td.Field(space, xy)                 # x * y, exact in an order-2 space
    # --8<-- [end:run-nodes]

    # --8<-- [start:run-field-nodes]
    p = data.fields["T"]                    # H1 of order 1
    total = tack.empty((space.size,), tack.f32)
    td.for_each(add_at_nodes, data, "cells", p, q, total)
    s = td.Field(space, total)              # T + x * y, of order 2
    # --8<-- [end:run-field-nodes]

    if check:
        np.testing.assert_allclose(means.numpy(), [3.0, 3.5])
        np.testing.assert_array_equal(by_hand.numpy(), means.numpy())
        np.testing.assert_allclose(wedge_mean.numpy(), [0.0, 3.5])
        centers = td.cell_centers(data).values.numpy()
        np.testing.assert_allclose(positions.numpy(), centers, atol=1e-6)
        np.testing.assert_allclose(values.numpy(), centers @ [1, 2, 3], atol=1e-5)
        np.testing.assert_allclose(gradients.numpy(), [[1, 2, 3], [1, 2, 3]], atol=1e-5)
        np.testing.assert_allclose(volume.numpy(), td.cell_geometry(data).measure.values.numpy(),
                                   atol=1e-6)
        np.testing.assert_allclose(same.numpy(), [1.0, 0.5], atol=1e-6)
        assert faces.num_faces == 10
        inside = jumps.numpy()[jumps.numpy() != 0]
        np.testing.assert_allclose(np.abs(inside), [0.5])
        @tack.func
        def x_times_y(x):
            return x[0] * x[1]
        np.testing.assert_array_equal(xy.numpy(), td.interpolate(data, x_times_y, space).values.numpy())
        want = td.project(data, p, space).values.numpy() + xy.numpy()
        np.testing.assert_allclose(s.values.numpy(), want, atol=1e-5)
    print(f"Mesh algorithm: corner means {means.numpy()}, volumes {volume.numpy()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default="cpu", choices=["cpu", "metal", "cuda", "hip", "level_zero"])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    run(args.arch, args.check)
