# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Cell averages, face values, traces, jumps, fluxes and divergence, on two mesh forms."""

import argparse

import numpy as np

import tack
import tack.dataset as td


# --8<-- [start:face-kernel]
@tack.kernel
def normal_components(faces, normals, areas, centers, normal_velocity, volume_flux,
                      radial_flux, vx, vy, vz):
    for f in faces:
        e = faces.entity_id(f)                      # the face's id among all faces
        un = normals[e].dot(tack.vec3(vx, vy, vz))  # a uniform velocity's normal component
        normal_velocity[e] = un
        volume_flux[e] = un * areas[e]
        # A second flow, u(x, y, z) = (x, y, z), whose divergence is 3.
        radial_flux[e] = centers[e].dot(normals[e]) * areas[e]
# --8<-- [end:face-kernel]


def analyze(data, label):
    # --8<-- [start:geometry]
    # One density average per cell, kept as a constant in that cell.
    data.fields["density"] = td.Field(td.Constant(data), tack.array([1.0, 2.0], tack.f32),
                                      unit="kg/m**3")
    cells = td.cell_geometry(data)          # measure (volume) and centroid of each cell
    faces = td.face_geometry(data)          # normal (out of side 0) and area of each face
    centers = td.face_centers(data)         # each face's area centroid
    boundary = td.boundary_faces(data)      # the faces with one side, as a face Set
    # --8<-- [end:geometry]

    # --8<-- [start:faces]
    num_faces = data.topology.faces().num_faces
    normal_velocity = tack.empty((num_faces,), data.dtype)
    volume_flux = tack.empty_like(normal_velocity)
    radial_flux = tack.empty_like(normal_velocity)
    velocity = (0.3, -1.2, 0.7)
    td.for_each(normal_components, data, "faces", faces.normal.values, faces.area.values,
                centers.values, normal_velocity, volume_flux, radial_flux, *velocity)

    oriented = td.Values(data, "faces", oriented=True)
    data.fields["normal_velocity"] = td.Field(oriented, normal_velocity, unit="m/s")
    data.fields["volume_flux"] = td.Field(oriented, volume_flux, unit="m**3/s")
    data.fields["radial_flux"] = td.Field(oriented, radial_flux, unit="m**3/s")
    # --8<-- [end:faces]

    # --8<-- [start:balance]
    net = td.net_flux(data, "radial_flux")             # each cell's outflow, m**3/s
    divergence = td.divergence(data, "radial_flux")    # the outflow over the volume, 1/s
    # --8<-- [end:balance]

    # --8<-- [start:perot]
    velocity_at_cells = td.perot(data, "normal_velocity")   # normal components, not fluxes
    # --8<-- [end:perot]

    # --8<-- [start:upwind]
    mass_flux = td.upwind_flux(data, "density", (velocity, "m/s"))   # kg/s through each face
    mass_outflow = td.net_flux(data, mass_flux)                       # kg/s out of each cell
    # --8<-- [end:upwind]

    volumes = cells.measure.values.numpy()
    boundary_outflow = radial_flux.numpy()[boundary.ids.numpy()].sum()
    np.testing.assert_allclose(volumes, [0.5, 0.5], atol=1e-6)
    np.testing.assert_allclose(cells.centroid.values.numpy(),
                               [[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]], atol=1e-6)
    np.testing.assert_allclose(net.values.numpy(), [1.5, 1.5], atol=1e-6)
    np.testing.assert_allclose(divergence.values.numpy(), [3.0, 3.0], atol=1e-5)
    np.testing.assert_allclose(net.values.numpy().sum(), boundary_outflow, atol=1e-6)
    np.testing.assert_allclose(velocity_at_cells.values.numpy(),
                               np.tile(velocity, (data.num_cells, 1)), atol=1e-5)
    np.testing.assert_allclose(td.divergence(data, "volume_flux").values.numpy(), 0.0,
                               atol=1e-5)
    np.testing.assert_allclose(mass_outflow.values.numpy(), [0.0, 0.3], atol=1e-5)
    assert velocity_at_cells.unit == "m/s" and mass_flux.unit == "kg/s"
    print(f"{label}: divergence {divergence.values.numpy()}, "
          f"boundary outflow {boundary_outflow:.3f}, "
          f"velocity at cells {velocity_at_cells.values.numpy()[0]}")


def traces_and_jumps():
    # --8<-- [start:traces]
    data = td.uniform_grid((3, 2, 2), spacing=(0.5, 1.0, 1.0), unit="m")
    x = data.positions()
    data.fields["density"] = td.Field(td.Constant(data), tack.array([1.0, 3.0], tack.f32))
    data.fields["T"] = td.Field(td.H1(data), tack.array((x[:, 0] + 2 * x[:, 1]).astype(np.float32)))

    sides = td.project(data, "density", td.SideTraces(data))   # each side's values on each face
    density_jump = td.jump(data, "density")                     # side 1 minus side 0
    T_jump = td.jump(data, "T")                                 # zero: T is continuous
    # --8<-- [end:traces]
    jumps = density_jump.values.numpy()
    assert np.count_nonzero(jumps) == 1 and np.isclose(np.abs(jumps).max(), 2.0)
    np.testing.assert_allclose(T_jump.values.numpy(), 0.0, atol=1e-6)
    per_face = sides.values.numpy().reshape(-1, 2, 4)
    interior = np.flatnonzero(jumps)[0]
    assert sorted(per_face[interior, :, 0].tolist()) == [1.0, 3.0]
    print("traces and jumps: one interior face, jump", jumps[interior])


def run(arch="cpu"):
    tack.set_default_device(arch)
    # --8<-- [start:mesh]
    # Two hexahedra in a unit cube, each 0.5 by 1 by 1 metres.
    data = td.uniform_grid((3, 2, 2), spacing=(0.5, 1.0, 1.0), unit="m")
    # --8<-- [end:mesh]
    for label, mesh in (("shaped", data), ("polyhedral", td.as_polyhedra(data))):
        analyze(mesh, label)
    traces_and_jumps()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="accepted for validate.py: the results are always checked")
    parser.add_argument("--arch", default="cpu",
                        choices=["cpu", "metal", "cuda", "hip", "level_zero"])
    run(parser.parse_args().arch)
