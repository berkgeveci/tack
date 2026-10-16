# Copyright (c) 2026 Kitware, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Construct L2 and quadratic H1 fields on one mesh; optionally draw their DOFs."""

import argparse
from pathlib import Path

import numpy as np

import tack
import tack.dataset as td
from tack.arrays import to_host


# --8<-- [start:sample]
@tack.func
def quadratic_temperature(x):
    return x.x * x.x + 2.0 * x.y + 3.0 * x.z
# --8<-- [end:sample]


# --8<-- [start:l2-kernel]
@tack.kernel
def initialize_pressure(cells, offsets, values):
    for c in cells:
        cell_id = cells.entity_id(c)
        first = offsets[cell_id]
        for j in range(cells.NUM_POINTS):
            x = cells.corner_position(c, j)
            values[first + j] = x.x + 2.0 * cell_id
# --8<-- [end:l2-kernel]


@tack.kernel
def sample_profiles(cells, temperature, pressure, linearized, xs, ts, ps, ls, count):
    for c in cells:
        for j in range(count):
            pc = tack.vec3(j / (count - 1), 0.5, 0.5)
            k = cells.entity_id(c) * count + j
            xs[k] = cells.position(c, pc).x
            ts[k] = temperature.value(c, pc)
            ps[k] = pressure.value(c, pc)
            ls[k] = linearized.value(c, pc)


def run(arch="cpu", check=False):
    tack.set_default_device(arch)
    # --8<-- [start:spaces]
    # --8<-- [start:mesh]
    # Shortcut mesh setup: two adjacent hexes in the unit cube.
    data = td.uniform_grid((3, 2, 2), spacing=(0.5, 1.0, 1.0))
    # --8<-- [end:mesh]
    # --8<-- [start:h1-build]
    temperature_space = td.H1(data, order=2)
    # The function at all 45 temperature nodes, not just the 12 corners.
    data.fields["temperature"] = td.interpolate(data, quadratic_temperature,
                                                temperature_space)
    # --8<-- [end:h1-build]
    # Where those nodes are, for the checks and the drawing below.
    temperature_nodes = td.project(data, data.geometry, temperature_space)
    # --8<-- [start:l2-build]
    pressure_space = td.L2(data, order=1)
    pressure = tack.empty((pressure_space.size,))
    td.for_each(initialize_pressure, data, "cells", pressure_space.offsets, pressure)
    data.fields["pressure"] = td.Field(pressure_space, pressure)
    # --8<-- [end:l2-build]
    material_space = td.Constant(data)
    data.fields["material"] = td.Field(
        material_space, tack.arrays.ConstantArray(2.0, material_space.size))
    # Geometry keeps its own order-1 H1 space and UniformCoordinates values.
    # --8<-- [end:spaces]
    # --8<-- [start:l2-upload]
    # Each row follows that hex's local corner order: 0, 1, ..., 7.
    per_cell = np.array([
        [0.0, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.0],   # cell 0: p = x
        [2.5, 3.0, 3.0, 2.5, 2.5, 3.0, 3.0, 2.5],   # cell 1: p = x + 2
    ], dtype=np.float32)
    uploaded_pressure = td.Field(pressure_space, tack.array(per_cell.reshape(-1)))
    # --8<-- [end:l2-upload]
    # --8<-- [start:l2-constant]
    cell_space = td.L2(data, order=0)
    cell_pressure = td.Field(cell_space, tack.array([1.0, 3.0], dtype=tack.f32))
    # --8<-- [end:l2-constant]
    # --8<-- [start:evaluate]
    at_centers = td.values_at_centers(data, "temperature")
    gradient = td.gradients(data, "temperature")
    linearized = td.project(data, "temperature", td.H1(data, order=1))
    linear_centers = td.values_at_centers(data, linearized)
    # --8<-- [end:evaluate]
    # --8<-- [start:l2-check]
    pressure_centers = td.values_at_centers(data, "pressure")
    pressure_gradient = td.gradients(data, "pressure")
    pressure_jump = td.jump(data, "pressure")
    # Centers: [0.25, 2.75]; gradients: [1, 0, 0]; interface jump magnitude: 2.
    # --8<-- [end:l2-check]
    count = 101
    xs, ts, ps, ls = (tack.empty((data.num_cells * count,))
                      for _ in range(4))
    td.for_each(sample_profiles, data, "cells", data.fields["temperature"],
                data.fields["pressure"], linearized, xs, ts, ps, ls, count)
    profiles = [to_host(v).reshape(data.num_cells, count) for v in (xs, ts, ps, ls)]
    if check:
        assert data.num_points == 12 and data.num_cells == 2
        assert data.geometry.space is td.H1(data, order=1)
        assert isinstance(data.geometry.values, tack.arrays.UniformCoordinates)
        assert temperature_space.size == 45
        assert pressure_space.size == 16 and material_space.size == 2
        assert all(f.space.topology is data.topology for f in data.fields.values())
        np.testing.assert_array_equal(to_host(pressure_space.offsets), [0, 8, 16])
        np.testing.assert_array_equal(to_host(pressure), to_host(uploaded_pressure.values))
        np.testing.assert_array_equal(to_host(td.values_at_centers(data, cell_pressure).values), [1, 3])
        np.testing.assert_allclose(to_host(at_centers.values), [2.5625, 3.0625], atol=1e-6)
        np.testing.assert_allclose(to_host(gradient.values), [[0.5, 2, 3], [1.5, 2, 3]], atol=1e-6)
        np.testing.assert_allclose(to_host(linear_centers.values), [2.625, 3.125], atol=1e-6)
        np.testing.assert_allclose(to_host(pressure_centers.values), [0.25, 2.75], atol=1e-6)
        np.testing.assert_allclose(to_host(pressure_gradient.values), [[1, 0, 0]] * 2, atol=1e-6)
        jumps = to_host(pressure_jump.values)
        assert np.count_nonzero(jumps) == 1
        np.testing.assert_allclose(np.max(np.abs(jumps)), 2.0)
        np.testing.assert_allclose(to_host(td.jump(data, "temperature").values), 0, atol=1e-6)
        x, t, p, linear = profiles
        np.testing.assert_allclose(t, x * x + 2.5, atol=2e-6)
        np.testing.assert_allclose(p, x + np.array([[0], [2]]), atol=1e-6)
        np.testing.assert_allclose(linear, np.vstack([0.5 * x[0] + 2.5, 1.5 * x[1] + 2]), atol=2e-6)
        # Check node categories independently of the space's DOF numbering.
        nodes = to_host(temperature_nodes.values)
        np.testing.assert_allclose(nodes * [4, 2, 2], np.rint(nodes * [4, 2, 2]), atol=1e-6)
        mids = np.count_nonzero(np.rint(nodes * [4, 2, 2]).astype(int) % 2, axis=1)
        np.testing.assert_array_equal(np.bincount(mids), [12, 20, 11, 2])
        assert np.count_nonzero(np.isclose(nodes[:, 0], 0.5)) == 9
    print(f"Independent spaces: {data.num_points} corners, {temperature_space.size} quadratic H1 DOFs, "
          f"{pressure_space.size} L2 DOFs; varying DG values and jump checked")
    return data, temperature_nodes, profiles


def save_figures(result, directory):
    """Draw nodes from the example and profiles evaluated through its actual bases."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    data, temperature_nodes, profiles = result
    blue, orange, green, purple = "#2463a6", "#c76a19", "#16806a", "#8750a1"
    style = {"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
             "svg.fonttype": "none"}

    def save(fig, name):
        fig.savefig(directory / f"{name}.svg", bbox_inches="tight")
        fig.savefig(directory / f"{name}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.0), layout="constrained")
        h1 = data.positions()
        h2 = to_host(temperature_nodes.values)
        for ax in axes:
            ax.add_patch(Rectangle((0, 0), 0.5, 1, facecolor="#edf3fa", edgecolor="#778391"))
            ax.add_patch(Rectangle((0.5, 0), 0.5, 1, facecolor="#fdf2e7", edgecolor="#778391"))
            ax.axvline(0.5, color="#637080", ls="--", lw=1.2, zorder=2)
            ax.set(xlim=(-0.1, 1.1), ylim=(-0.12, 1.15), aspect="equal", xticks=[0, 0.5, 1],
                   yticks=[0, 1], xlabel="x", ylabel="y")
            ax.text(0.25, 1.07, "cell 0", ha="center", fontsize=10)
            ax.text(0.75, 1.07, "cell 1", ha="center", fontsize=10)
        for ax, nodes, title, description in (
            (axes[0], h1, "H1 order 1", "12 global DOFs in 3D"),
            (axes[1], h2, "H1 order 2", "27 + 27 − 9 = 45 DOFs in 3D"),
        ):
            front = nodes[np.isclose(nodes[:, 2], 0)]
            shared = np.isclose(front[:, 0], 0.5)
            ax.scatter(front[~shared, 0], front[~shared, 1], c=blue, s=60, zorder=3)
            ax.scatter(front[shared, 0], front[shared, 1], c=green, s=85, marker="D", zorder=4)
            ax.set_title(f"{title}\n{description}", fontsize=11)
        for cell, x0 in enumerate((0, 0.5)):
            corners = np.array([[x0, 0], [x0 + 0.5, 0], [x0 + 0.5, 1], [x0, 1]])
            # Separate coincident interface markers only for legibility.
            corners[np.isclose(corners[:, 0], 0.5), 0] += -0.025 if cell == 0 else 0.025
            axes[2].scatter(corners[:, 0], corners[:, 1], c=blue if cell == 0 else orange,
                            marker="o" if cell == 0 else "s", s=65, zorder=4)
        axes[2].set_title("L2 order 1\n8 + 8 = 16 DOFs in 3D", fontsize=11)
        fig.legend(handles=[Line2D([], [], color=green, marker="D", ls="", label="One shared H1 DOF"),
                            Line2D([], [], color=blue, marker="o", ls="", label="Cell 0 L2 DOF"),
                            Line2D([], [], color=orange, marker="s", ls="", label="Cell 1 L2 DOF")],
                   loc="outside lower center", ncol=3, frameon=False)
        fig.suptitle("Front faces (z = 0) of the same two hexahedra", fontsize=14)
        save(fig, "field-dof-layouts")

        # One hex: actual projected nodes from the left cell, rescaled to [0,1]^3.
        nodes = h2[h2[:, 0] <= 0.5 + 1e-6].copy()
        nodes[:, 0] *= 2
        mids = np.count_nonzero(np.rint(nodes * 2).astype(int) % 2, axis=1)
        np.testing.assert_array_equal(np.bincount(mids), [8, 12, 6, 1])
        fig = plt.figure(figsize=(7.5, 5.0), layout="constrained")
        ax = fig.add_subplot(projection="3d")
        for axis in range(3):
            other = [a for a in range(3) if a != axis]
            for a in (0, 1):
                for b in (0, 1):
                    segment = np.zeros((2, 3))
                    segment[:, axis] = [0, 1]
                    segment[:, other[0]], segment[:, other[1]] = a, b
                    ax.plot(*segment.T, color="#9ba5b1", lw=1.1)
        handles = []
        for category, color, marker, label in (
            (0, blue, "o", "8 corner nodes"), (1, green, "s", "12 edge midpoints"),
            (2, orange, "D", "6 face centers"), (3, purple, "*", "1 cell-interior node"),
        ):
            points = nodes[mids == category]
            ax.scatter(*points.T, c=color, marker=marker, s=80 if category < 3 else 200,
                       depthshade=False, edgecolors="white", linewidths=0.5)
            handles.append(Line2D([], [], color=color, marker=marker, ls="", label=label))
        ax.set(xlabel="reference x", ylabel="reference y", zlabel="reference z",
               xticks=[0, 0.5, 1], yticks=[0, 0.5, 1], zticks=[0, 0.5, 1])
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=20, azim=-58)
        ax.set_title("Quadratic H1 on one hex: 8 + 12 + 6 + 1 = 27 nodes", pad=18)
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
        save(fig, "quadratic-hex-nodes")

        x, t, p, linear = profiles
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), layout="constrained")
        axes[0].plot(x.T, p.T, color=blue, lw=2)
        axes[0].plot([0.5, 0.5], [0.5, 2.5], ":", color=orange, lw=1.8)
        axes[0].scatter([0.5, 0.5], [0.5, 2.5], c=[blue, orange], s=55, zorder=3)
        axes[0].annotate("two traces: 0.5 and 2.5\njump magnitude 2", (0.5, 1.5),
                         xytext=(0.06, 1.55), arrowprops={"arrowstyle": "->", "color": orange})
        axes[0].set(title="L2 order 1: vary within cells, jump between them", ylabel="pressure")
        for c in range(2):
            axes[1].plot(x[c], t[c], color=green, lw=2.4, label="H1 order 2" if c == 0 else None)
            axes[1].plot(x[c], linear[c], color=orange, lw=1.8, ls="--",
                         label="Projected to H1 order 1" if c == 0 else None)
        axes[1].set(title="H1 order 2: preserve the quadratic profile", ylabel="temperature")
        axes[1].legend(frameon=False)
        for ax in axes:
            ax.axvline(0.5, color="#9ba5b1", ls="--", lw=1)
            ax.set(xlabel="physical x (y = z = 0.5)", xlim=(0, 1))
            ax.grid(alpha=0.15)
        save(fig, "field-basis-profiles")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default="cpu", choices=["cpu", "metal", "cuda", "hip", "level_zero"])
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--figures", type=Path, help="Write explanatory SVG and PNG figures here")
    args = parser.parse_args()
    result = run(args.arch, args.check)
    if args.figures:
        save_figures(result, args.figures)
