# Copyright (c) 2026 Kitware, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Two adjacent cells: point, cell and discontinuous data, checked analytically."""

import argparse

import numpy as np

import tack
import tack.dataset as td
from tack.dataset.arrays import to_host


# --8<-- [start:sample]
@tack.kernel
def temperature_at_points(xyz, out):
    for p in range(out.shape[0]):
        x = xyz.get(p)
        out[p] = x.x + 2.0 * x.y + 3.0 * x.z
# --8<-- [end:sample]


def build():
    # --8<-- [start:build]
    data = td.uniform_grid((3, 2, 2), spacing=(0.5, 1.0, 1.0))
    temperature = tack.empty((data.num_points,))
    temperature_at_points(data.geometry.values, temperature)
    data.fields["temperature"] = td.Field(td.H1(data), temperature)

    material = tack.array([1.0, 3.0], dtype=tack.f32)
    data.fields["material"] = td.Field(td.Constant(data), material)
    # --8<-- [end:build]
    return data


def run(arch="cpu", check=False):
    tack.init(arch=getattr(tack, arch))
    data = build()
    # --8<-- [start:spaces]
    cell_temperature = td.to_cells(data, "temperature")
    point_material = td.to_points(data, "material")
    separate_material = td.discontinuous(data, "material")
    gradient = td.gradients(data, "temperature")
    # --8<-- [end:spaces]
    # --8<-- [start:implicit]
    data.fields["constant"] = td.Field(td.H1(data), td.ConstantArray(5.0, data.num_points))
    explicit = td.arrays.materialize(data.fields["constant"].values)
    snapshot = td.arrays.to_host(data.fields["constant"].values)
    # --8<-- [end:implicit]
    # --8<-- [start:traces]
    face_values = td.traces(data, "material")
    material_jump = td.jump(data, "material")
    continuous_jump = td.jump(data, "temperature")
    # --8<-- [end:traces]

    if check:
        xyz = data.positions()
        assert data.num_points == 12 and data.num_cells == 2
        np.testing.assert_allclose(to_host(data.fields["temperature"].values),
                                   xyz[:, 0] + 2 * xyz[:, 1] + 3 * xyz[:, 2])
        np.testing.assert_allclose(to_host(cell_temperature.values), [2.75, 3.25])
        np.testing.assert_allclose(to_host(point_material.values), 1 + 2 * xyz[:, 0])
        np.testing.assert_array_equal(to_host(separate_material.values),
                                      np.repeat([1.0, 3.0], 8))
        np.testing.assert_allclose(to_host(gradient.values), np.tile([1, 2, 3], (2, 1)))
        np.testing.assert_array_equal(explicit.numpy(), np.full(12, 5.0))
        np.testing.assert_array_equal(snapshot, np.full(12, 5.0))
        jumps = to_host(material_jump.values)
        assert np.count_nonzero(jumps) == 1
        np.testing.assert_allclose(np.max(np.abs(jumps)), 2.0)
        np.testing.assert_allclose(to_host(continuous_jump.values), 0, atol=1e-6)
        assert face_values.space is td.SideTraces(data)
    print(f"Data fields: {data.num_points} points, {data.num_cells} cells; "
          f"cell temperatures {to_host(cell_temperature.values)}")
    return data, point_material, separate_material


def save_figure(result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data, continuous, separate = result
    points = data.positions()
    row = np.flatnonzero((points[:, 1] == 0) & (points[:, 2] == 0))
    x = points[row, 0]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharey=True, layout="constrained")
    axes[0].stairs([1, 3], [0, 0.5, 1], baseline=None, color="#3659a8", linewidth=2.5)
    axes[0].set_title("Constant: one value per cell")
    axes[1].plot(x, to_host(continuous.values)[row], "o-", color="#c86a20", linewidth=2.5)
    axes[1].set_title("H1: shared point values")
    # L2 duplicates the shared corners: the two traces at x=0.5 disagree.
    values = to_host(separate.values).reshape(2, 8)
    axes[2].plot([0, 0.5], values[0, :2], "o-", color="#3659a8", linewidth=2.5)
    axes[2].plot([0.5, 1], values[1, :2], "o-", color="#3659a8", linewidth=2.5)
    axes[2].set_title("L2: each cell's own corners")
    for ax in axes:
        ax.axvline(0.5, color="0.75", linestyle="--", linewidth=1)
        ax.set(xlabel="x", xlim=(-0.04, 1.04), ylim=(0.5, 3.5), yticks=[1, 2, 3])
        ax.grid(alpha=0.15)
    axes[0].set_ylabel("Material value")
    fig.savefig(output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default="cpu", choices=["cpu", "metal", "cuda", "hip", "level_zero"])
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", help="Save a comparison of the spaces (requires Matplotlib)")
    args = parser.parse_args()
    result = run(args.arch, args.check)
    if args.output:
        save_figure(result, args.output)
