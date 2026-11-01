# Copyright (c) 2026 Kitware, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Cell averages, face fluxes and Perot reconstruction, checked on two mesh forms."""

import argparse

import numpy as np

import tack
import tack.dataset as td


# --8<-- [start:face-kernel]
@tack.kernel
def face_values(faces, normals, areas, centers, normal_values, volume_values,
                radial_values, vx, vy, vz):
    for f in faces:
        e = faces.entity_id(f)
        un = normals[e].dot(tack.vec3(vx, vy, vz))
        normal_values[e] = un
        volume_values[e] = un * areas[e]
        # A second flow, u(x, y, z) = (x, y, z), has divergence 3.
        radial_values[e] = centers[e].dot(normals[e]) * areas[e]
# --8<-- [end:face-kernel]


def analyze(data, label, check=False):
    # --8<-- [start:cells]
    # One cell average per control volume, retained as a constant in that cell.
    density = td.Field(td.Constant(data), tack.array([1.0, 2.0], tack.f32), unit="kg/m**3")
    data.fields["density"] = density
    cells = td.cell_geometry(data)
    faces = data.topology.faces()
    geometry = td.face_geometry(data)
    centers = td.face_centers(data)
    # --8<-- [end:cells]

    # --8<-- [start:faces]
    normal_values = tack.empty((faces.num_faces,), data.dtype)
    volume_values = tack.empty_like(normal_values)
    radial_values = tack.empty_like(normal_values)
    velocity = (0.3, -1.2, 0.7)
    td.for_each(face_values, data, "faces", geometry.normal.values,
                geometry.area.values, centers.values, normal_values,
                volume_values, radial_values, *velocity)

    face_space = td.Values(data, "faces", oriented=True)
    data.fields["normal_velocity"] = td.Field(face_space, normal_values, unit="m/s")
    data.fields["volume_flux"] = td.Field(face_space, volume_values, unit="m**3/s")
    data.fields["radial_flux"] = td.Field(face_space, radial_values, unit="m**3/s")
    # --8<-- [end:faces]

    # --8<-- [start:balance]
    # Integrated flux through each face: sum first, divide by volume second.
    net = td.net_flux(data, "radial_flux")
    divergence = td.divergence(data, "radial_flux")
    # These two cells have volume 0.5: net outflow 1.5 and divergence 3 each.
    boundary = td.boundary_faces(data)
    # --8<-- [end:balance]

    # --8<-- [start:perot]
    # Perot takes normal components, not components already multiplied by area.
    reconstructed = td.perot(data, "normal_velocity")
    data.fields["velocity"] = reconstructed
    # The result is a Values(data, "cells") field of 3-vectors, with unit m/s.
    # --8<-- [end:perot]

    # --8<-- [start:upwind]
    # Optional transport example: a constant prescribed velocity carries density.
    mass_flux = td.upwind_flux(data, "density", (velocity, "m/s"))
    mass_outflow = td.net_flux(data, mass_flux)
    # mass_flux is integrated through each face (kg/s), not normal velocity.
    # --8<-- [end:upwind]

    # Readbacks are explicit checkpoints, after the device computations above.
    volumes = cells.measure.values.numpy()
    net_values = net.values.numpy()
    div_values = divergence.values.numpy()
    reconstructed_values = reconstructed.values.numpy()
    boundary_outflow = radial_values.numpy()[boundary.ids.numpy()].sum()
    if check:
        np.testing.assert_allclose(volumes, [0.5, 0.5], atol=1e-6)
        np.testing.assert_allclose(cells.centroid.values.numpy(),
                                   [[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]], atol=1e-6)
        np.testing.assert_allclose(net_values, [1.5, 1.5], atol=1e-6)
        np.testing.assert_allclose(div_values, [3.0, 3.0], atol=1e-5)
        np.testing.assert_allclose(net_values.sum(), boundary_outflow, atol=1e-6)
        np.testing.assert_allclose(boundary_outflow, 3.0, atol=1e-6)
        np.testing.assert_allclose(reconstructed_values,
                                   np.tile(velocity, (data.num_cells, 1)), atol=1e-5)
        np.testing.assert_allclose(td.divergence(data, "volume_flux").values.numpy(),
                                   0.0, atol=1e-5)
        # Upwind density is 1 at the internal face and 2 at the right boundary.
        np.testing.assert_allclose(mass_outflow.values.numpy(), [0.0, 0.3], atol=1e-5)
        np.testing.assert_allclose(td.divergence(data, mass_flux).values.numpy(),
                                   [0.0, 0.6], atol=1e-5)
    print(f"{label}: divergence {div_values}, boundary outflow {boundary_outflow:.3f}, "
          f"reconstructed velocity {reconstructed_values[0]}")


def run(arch="cpu", check=False):
    tack.set_default_device(arch)
    # --8<-- [start:mesh]
    # Two hexahedra in a unit cube: each is 0.5 by 1 by 1 metres.
    grid = td.uniform_grid((3, 2, 2), spacing=(0.5, 1.0, 1.0), unit="m")
    # --8<-- [end:mesh]
    for label, data in (("shaped", grid), ("polyhedral", td.as_polyhedra(grid))):
        analyze(data, label, check)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default="cpu",
                        choices=["cpu", "metal", "cuda", "hip", "level_zero"])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    run(args.arch, args.check)
