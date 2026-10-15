# Copyright (c) 2026 Kitware, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Cell and face kernels, a subset, and a checked integrated flux balance."""

import argparse

import numpy as np

import tack
import tack.dataset as td


# --8<-- [start:cell]
@tack.kernel
def corner_mean(cells, temperature, out):
    for c in cells:
        total = 0.0
        for j in range(cells.NUM_POINTS):
            total += temperature.corner_value(c, j)
        out[cells.entity_id(c)] = total / cells.NUM_POINTS
# --8<-- [end:cell]


# --8<-- [start:face]
@tack.kernel
def integrated_flux(faces, centers, normals, areas, out):
    for f in faces:
        e = faces.entity_id(f)
        velocity = centers[e]                   # v(x,y,z) = (x,y,z)
        out[e] = velocity.dot(normals[e]) * areas[e]
# --8<-- [end:face]


def run(arch="cpu", check=False):
    tack.init(arch=getattr(tack, arch))
    data = td.uniform_grid((3, 2, 2), spacing=(0.5, 1.0, 1.0))
    xyz = data.positions()                      # one upload boundary for this example
    values = tack.empty((data.num_points,))
    tack.copyto(values, (xyz[:, 0] + 2 * xyz[:, 1] + 3 * xyz[:, 2]).astype(np.float32))
    data.fields["temperature"] = td.Field(td.H1(data), values)
    # --8<-- [start:subset]
    data.sets["right"] = td.Set(data, "cells", [1])
    means = tack.zeros((data.num_cells,))
    td.for_each(corner_mean, data, "right", data.fields["temperature"], means)
    # Only cell 1 is written. Cell 0 retains its initialized zero.
    # --8<-- [end:subset]
    # --8<-- [start:balance]
    faces = data.topology.faces()               # derive incidence once
    geometry = td.face_geometry(data)
    centers = td.face_centers(data)
    flux_values = tack.empty((faces.num_faces,))
    td.for_each(integrated_flux, data, "faces", centers.values,
                geometry.normal.values, geometry.area.values, flux_values)
    flux = td.Field(td.Values(data, "faces", oriented=True), flux_values)
    net_outflow = td.divergence(data, flux)       # integrated flux, per cell
    volume = td.cell_geometry(data).measure
    # For v = (x,y,z), net_outflow / volume is the divergence 3.
    # --8<-- [end:balance]
    if check:
        np.testing.assert_allclose(means.numpy(), [0, 3.25])
        np.testing.assert_allclose(volume.values.numpy(), [0.5, 0.5])
        np.testing.assert_allclose(net_outflow.values.numpy(), [1.5, 1.5], atol=1e-6)
        boundary = td.boundary_faces(data)
        np.testing.assert_allclose(flux_values.numpy()[boundary.ids.numpy()].sum(), 3.0)
        all_means = tack.empty((data.num_cells,))
        td.for_each(corner_mean, data, "cells", data.fields["temperature"], all_means)
        np.testing.assert_allclose(all_means.numpy(), [2.75, 3.25])
    print(f"Mesh kernels: selected means {means.numpy()}; "
          f"integrated outflow {net_outflow.values.numpy()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default="cpu", choices=["cpu", "metal", "cuda", "hip", "level_zero"])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    run(args.arch, args.check)
