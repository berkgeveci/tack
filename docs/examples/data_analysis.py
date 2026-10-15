# Copyright (c) 2026 Kitware, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Select, cut, sample and analyze an analytic scalar and a rotating flow."""

import argparse

import numpy as np

import tack
import tack.dataset as td
from tack.dataset.arrays import to_host


# --8<-- [start:sample]
@tack.kernel
def sample(xyz, scalar, velocity):
    for p in range(scalar.shape[0]):
        x = xyz.get(p)
        scalar[p] = x.x
        velocity[p] = [-x.y, x.x, 0.0]
# --8<-- [end:sample]


def run(arch="cpu", check=False):
    tack.init(arch=getattr(tack, arch))
    # --8<-- [start:build]
    data = td.uniform_grid((9, 5, 5), origin=(-1, -1, -1), spacing=(0.25, 0.5, 0.5))
    x = tack.empty(tack.f32, shape=(data.num_points,))
    velocity = tack.empty(dtype=tack.vec3(tack.f32), shape=(data.num_points,))
    sample(data.geometry.values, x, velocity)
    data.fields["x"] = td.Field(td.H1(data), x)
    data.fields["velocity"] = td.Field(td.H1(data), velocity)
    # --8<-- [end:build]
    # --8<-- [start:filter]
    whole = td.threshold(data, "x", 0.125, 1.0)                  # all corners in range
    touching = td.threshold(data, "x", 0.125, 1.0, all_points=False)
    plane = td.Plane(origin=(0.125, 0, 0), normal=(1, 0, 0))
    clipped = td.clip(data, plane, fields=["x"])                # keep x >= 0.125
    surface = td.contour(data, "x", 0.125, fields=["velocity"])
    section = td.slice_plane(data, origin=(0.125, 0, 0), normal=(1, 0, 0))
    # --8<-- [end:filter]
    # --8<-- [start:derived]
    speed = td.magnitude(data, "velocity")
    data.fields["speed"] = speed                               # keep the returned field
    velocity_gradient = td.gradients(data, "velocity")
    flow = td.flow_quantities(data, velocity_gradient)
    summary = td.statistics(data, "x")
    # --8<-- [end:derived]
    # --8<-- [start:probe]
    locator = td.CellLocator(data)
    query = np.array([[0.2, 0.3, 0.0], [-0.4, 0.1, 0.0], [2.0, 0.0, 0.0]], np.float32)
    sampled = td.probe(data, query, fields=["x", "velocity"], locator=locator)
    valid = to_host(sampled.fields["valid"].values).astype(bool)
    samples = to_host(sampled.fields["x"].values)
    # Use valid before interpreting samples: zero also denotes a missing sample.
    # --8<-- [end:probe]
    # --8<-- [start:flow]
    seeds = np.array([[0.5, 0, 0], [0, 0.75, 0]], np.float32)
    paths = td.streamlines(data, "velocity", seeds, step_size=0.05, steps=20, locator=locator)
    particles = td.advect(data, "velocity", seeds, step_size=0.05, steps=20, locator=locator)
    # --8<-- [end:flow]

    if check:
        assert data.num_cells == 128
        assert whole.num_cells == 48 and touching.num_cells == 64
        np.testing.assert_allclose(td.cell_geometry(whole).measure.values.sum(), 3.0, atol=1e-5)
        np.testing.assert_allclose(td.cell_geometry(clipped).measure.values.sum(), 3.5, atol=1e-5)
        for cut in (surface, section):
            assert cut.num_cells > 0
            np.testing.assert_allclose(cut.positions()[:, 0], 0.125, atol=1e-6)
            np.testing.assert_allclose(td.cell_geometry(cut).measure.values.sum(), 4.0, atol=1e-5)
        assert set(surface.fields) == {"velocity"} and set(clipped.fields) == {"x"}
        np.testing.assert_allclose(to_host(surface.fields["velocity"].values),
                                   np.c_[-surface.positions()[:, 1],
                                         np.full(surface.num_points, 0.125),
                                         np.zeros(surface.num_points)], atol=1e-6)
        np.testing.assert_allclose(to_host(flow.divergence.values), 0, atol=1e-6)
        np.testing.assert_allclose(to_host(flow.vorticity.values),
                                   np.tile([0, 0, 2], (data.num_cells, 1)), atol=1e-6)
        np.testing.assert_allclose(to_host(flow.q_criterion.values), 1, atol=1e-6)
        np.testing.assert_array_equal(valid, [True, True, False])
        np.testing.assert_allclose(samples[valid], query[valid, 0], atol=1e-6)
        assert summary["n"] == data.num_points
        np.testing.assert_allclose(summary["mean"], 0, atol=1e-6)
        assert isinstance(paths.topology, td.PolylineTopology) and paths.num_cells == 2
        np.testing.assert_allclose(particles.positions()[:, :2],
                                   [[0.5 * np.cos(1), 0.5 * np.sin(1)],
                                    [-0.75 * np.sin(1), 0.75 * np.cos(1)]], atol=2e-5)
        np.testing.assert_allclose(to_host(td.cell_geometry(paths).measure.values),
                                   [0.5, 0.75], atol=1e-4)
    print(f"Data analysis: {whole.num_cells} whole cells; clipped volume "
          f"{td.cell_geometry(clipped).measure.values.sum():.3f}; probe valid {valid}")
    return whole, clipped, surface


def save_figure(result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(10.5, 3.8), layout="constrained")
    titles = ["Threshold: starts at x = 0.25", "Clip: starts at x = 0.125", "Contour: x = 0.125"]
    for k, (data, title) in enumerate(zip(result, titles, strict=True)):
        triangles = td.triangulate(td.external_faces(data)) if k < 2 else data
        xyz = triangles.positions()
        ids = triangles.topology.connectivity.to_numpy().reshape(-1, 3)
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        ax.add_collection3d(Poly3DCollection(xyz[ids], facecolor="#80b5dc", edgecolor="#335c7a",
                                           linewidth=0.35, alpha=0.85))
        ax.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1), xlabel="x", ylabel="y", zlabel="z")
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        ax.set_zticks([-1, 0, 1])
        ax.set_title(title, fontsize=10)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=20, azim=-55)
    fig.savefig(output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default="cpu", choices=["cpu", "metal", "cuda", "hip", "level_zero"])
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", help="Save the filter comparison (requires Matplotlib)")
    args = parser.parse_args()
    result = run(args.arch, args.check)
    if args.output:
        save_figure(result, args.output)
