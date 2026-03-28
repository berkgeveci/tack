"""22 -- Marching squares contour with GPU prefix sum.

Port of the Tack contour example demonstrating:
- Cell set abstraction via @tack.data_oriented templates
- Topology-aware cell iteration with point field access
- Multi-pass algorithm with scatter for variable output
- GPU-accelerated exclusive prefix sum (scan) replacing host-side numpy

The marching squares algorithm has variable output per cell: each cell
produces 0, 1, or 2 line segments.  A prefix sum computes scatter offsets
so each cell knows where to write its output in the shared output arrays.

Runs contour on two representations of the same curvilinear mesh:
1. CellSetStructured2D -- connectivity computed from dimensions (zero storage)
2. CellSetExplicitQuads -- connectivity stored in arrays

Both produce identical results, validating the abstraction.

Usage:
  uv run python examples/22_contour.py
  uv run python examples/22_contour.py --arch metal
"""

import numpy as np
import tack
from tack import algorithms

import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument('--arch', default='cpu', choices=['cpu', 'metal', 'cuda', 'hip', 'level_zero'])
_arch = getattr(tack, _parser.parse_args().arch)
tack.init(arch=_arch)

# ================================================================
# MARCHING SQUARES LOOKUP TABLES
# ================================================================
#
# Quad point/edge ordering:
#
#   3 ---e2--- 2
#   |          |
#   e3        e1
#   |          |
#   0 ---e0--- 1
#
# Case ID = 4-bit mask: bit i set if point i is above isovalue.

@tack.data_oriented
class MarchingSquaresTables:
    """Lookup tables for marching squares, passed as a template parameter."""

    def __init__(self):
        self.num_lines = tack.field(dtype=tack.i32, shape=(16,))
        # Flatten (16, 2, 2) -> (64,) for 1D field indexing
        self.segments = tack.field(dtype=tack.i32, shape=(64,))
        # Flatten (4, 2) -> (8,)
        self.edge_verts = tack.field(dtype=tack.i32, shape=(8,))

        self.num_lines.from_numpy(np.array(
            [0, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 0], dtype=np.int32))

        segs = np.full((16, 2, 2), -1, dtype=np.int32)
        segs[1] = [[3, 0], [-1, -1]]
        segs[2] = [[0, 1], [-1, -1]]
        segs[3] = [[3, 1], [-1, -1]]
        segs[4] = [[1, 2], [-1, -1]]
        segs[5] = [[3, 0], [1, 2]]      # saddle
        segs[6] = [[0, 2], [-1, -1]]
        segs[7] = [[3, 2], [-1, -1]]
        segs[8] = [[2, 3], [-1, -1]]
        segs[9] = [[0, 2], [-1, -1]]
        segs[10] = [[0, 1], [2, 3]]     # saddle
        segs[11] = [[1, 2], [-1, -1]]
        segs[12] = [[1, 3], [-1, -1]]
        segs[13] = [[0, 1], [-1, -1]]
        segs[14] = [[0, 3], [-1, -1]]
        self.segments.from_numpy(segs.ravel())

        edge_verts = np.array(
            [[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int32)
        self.edge_verts.from_numpy(edge_verts.ravel())

    @tack.func
    def get_num_lines(self, case_id):
        return self.num_lines[case_id]

    @tack.func
    def get_segment_edge(self, case_id, seg, ep):
        return self.segments[case_id * 4 + seg * 2 + ep]

    @tack.func
    def get_edge_vert(self, edge_id, idx):
        return self.edge_verts[edge_id * 2 + idx]


ms_tables = MarchingSquaresTables()


# ================================================================
# CELL SET ABSTRACTIONS
# ================================================================
# Follows Viskores: cell sets provide get_cell_points(cell_id)
# resolved at JIT time per type via @tack.data_oriented templates.


@tack.data_oriented
class CellSetStructured2D:
    """2D structured quad grid. Connectivity from dimensions (zero storage)."""

    def __init__(self, nx, ny):
        self.nx = nx
        self.ny = ny

    @tack.func
    def get_cell_point(self, cell_id, local_idx):
        cx = cell_id % (self.nx - 1)
        cy = cell_id // (self.nx - 1)
        p0 = cy * self.nx + cx
        # local_idx: 0=p0, 1=p0+1, 2=p0+nx+1, 3=p0+nx
        result = p0
        if local_idx == 1:
            result = p0 + 1
        if local_idx == 2:
            result = p0 + self.nx + 1
        if local_idx == 3:
            result = p0 + self.nx
        return result


@tack.data_oriented
class CellSetExplicitQuads:
    """Unstructured quad grid. Connectivity stored in arrays."""

    def __init__(self, connectivity_np, num_cells, num_points):
        self.connectivity = tack.field(dtype=tack.i32, shape=(len(connectivity_np),))
        self.connectivity.from_numpy(connectivity_np)

    @tack.func
    def get_cell_point(self, cell_id, local_idx):
        s = cell_id * 4
        return self.connectivity[s + local_idx]


# ================================================================
# SHARED WORKLET LOGIC
# ================================================================


@tack.func
def compute_case(p0, p1, p2, p3, scalar_field, isovalue_f):
    case_id = 0
    if scalar_field[p0] > isovalue_f[0]:
        case_id = case_id | 1
    if scalar_field[p1] > isovalue_f[0]:
        case_id = case_id | 2
    if scalar_field[p2] > isovalue_f[0]:
        case_id = case_id | 4
    if scalar_field[p3] > isovalue_f[0]:
        case_id = case_id | 8
    return case_id


# ================================================================
# CONTOUR KERNELS  (one kernel per pass)
# ================================================================


# --- Pass 1: Classify cells (count output lines per cell) ---
@tack.kernel
def classify_cells(cell_set, ms, scalar_field, isovalue_f, num_lines_out):
    for cell_id in range(num_lines_out.shape[0]):
        p0 = cell_set.get_cell_point(cell_id, 0)
        p1 = cell_set.get_cell_point(cell_id, 1)
        p2 = cell_set.get_cell_point(cell_id, 2)
        p3 = cell_set.get_cell_point(cell_id, 3)
        case_id = compute_case(p0, p1, p2, p3, scalar_field, isovalue_f)
        num_lines_out[cell_id] = ms.get_num_lines(case_id)


# --- Pass 2: Generate interpolation data for each contour endpoint ---
@tack.kernel
def generate_contour_edges(
    cell_set, ms, scalar_field, isovalue_f,
    num_lines_f, scatter_offsets,
    interp_pa, interp_pb, interp_w,
):
    for cell_id in range(num_lines_f.shape[0]):
        n = num_lines_f[cell_id]
        if n == 0:
            pass
        else:
            p0 = cell_set.get_cell_point(cell_id, 0)
            p1 = cell_set.get_cell_point(cell_id, 1)
            p2 = cell_set.get_cell_point(cell_id, 2)
            p3 = cell_set.get_cell_point(cell_id, 3)
            case_id = compute_case(p0, p1, p2, p3, scalar_field, isovalue_f)
            base = scatter_offsets[cell_id]

            for seg in range(n):
                # Endpoint 0
                edge_id_0 = ms.get_segment_edge(case_id, seg, 0)
                lp0_0 = ms.get_edge_vert(edge_id_0, 0)
                lp1_0 = ms.get_edge_vert(edge_id_0, 1)
                gp0_0 = cell_set.get_cell_point(cell_id, lp0_0)
                gp1_0 = cell_set.get_cell_point(cell_id, lp1_0)
                v0_0 = scalar_field[gp0_0]
                v1_0 = scalar_field[gp1_0]
                w0 = 0.5
                denom0 = v1_0 - v0_0
                if abs(denom0) > 1e-10:
                    w0 = (isovalue_f[0] - v0_0) / denom0
                w0 = min(max(w0, 0.0), 1.0)
                idx0 = (base + seg) * 2 + 0
                interp_pa[idx0] = gp0_0
                interp_pb[idx0] = gp1_0
                interp_w[idx0] = w0

                # Endpoint 1
                edge_id_1 = ms.get_segment_edge(case_id, seg, 1)
                lp0_1 = ms.get_edge_vert(edge_id_1, 0)
                lp1_1 = ms.get_edge_vert(edge_id_1, 1)
                gp0_1 = cell_set.get_cell_point(cell_id, lp0_1)
                gp1_1 = cell_set.get_cell_point(cell_id, lp1_1)
                v0_1 = scalar_field[gp0_1]
                v1_1 = scalar_field[gp1_1]
                w1 = 0.5
                denom1 = v1_1 - v0_1
                if abs(denom1) > 1e-10:
                    w1 = (isovalue_f[0] - v0_1) / denom1
                w1 = min(max(w1, 0.0), 1.0)
                idx1 = (base + seg) * 2 + 1
                interp_pa[idx1] = gp0_1
                interp_pb[idx1] = gp1_1
                interp_w[idx1] = w1


# --- Pass 3: Interpolate coordinates to contour points ---
@tack.kernel
def interpolate_field_x(
    input_x, input_y,
    interp_pa, interp_pb, interp_w,
    output_x, output_y,
):
    for i in range(output_x.shape[0]):
        pa = interp_pa[i]
        pb = interp_pb[i]
        w = interp_w[i]
        output_x[i] = (1.0 - w) * input_x[pa] + w * input_x[pb]
        output_y[i] = (1.0 - w) * input_y[pa] + w * input_y[pb]


# ================================================================
# CONTOUR FILTER  (orchestrates the passes)
# ================================================================


def run_contour(cell_set, coords_x, coords_y, scalar_field, isovalue,
                num_lines_f, scatter_offsets_f,
                interp_pa, interp_pb, interp_w,
                output_x, output_y):
    """Execute 2D marching squares contour. Returns (total_lines, total_points)."""
    nc = num_lines_f.shape[0]

    # Store isovalue in a 1-element field
    isovalue_f = tack.field(dtype=tack.f32, shape=(1,))
    isovalue_f.from_numpy(np.array([isovalue], dtype=np.float32))

    # Pass 1: classify cells -- count output lines per cell
    classify_cells(cell_set, ms_tables, scalar_field, isovalue_f, num_lines_f)

    # Pass 1.5: GPU exclusive prefix sum to compute scatter offsets
    total_lines = algorithms.exclusive_scan(num_lines_f, scatter_offsets_f, nc)
    total_points = total_lines * 2

    if total_lines == 0:
        return 0, 0

    # Pass 2: generate contour edges
    generate_contour_edges(
        cell_set, ms_tables, scalar_field, isovalue_f,
        num_lines_f, scatter_offsets_f,
        interp_pa, interp_pb, interp_w)

    # Pass 3: interpolate coordinates
    interpolate_field_x(
        coords_x, coords_y,
        interp_pa, interp_pb, interp_w,
        output_x, output_y)

    return total_lines, total_points


# ================================================================
# TEST DATA HELPERS
# ================================================================


def make_curvilinear_grid(nx, ny):
    """Create a warped 2D grid and a radial scalar field."""
    x = np.linspace(-1.0, 1.0, nx, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, ny, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    # Warp
    xw = xx + 0.1 * np.sin(np.pi * yy)
    yw = yy + 0.1 * np.sin(np.pi * xx)
    coords_x = xw.ravel().astype(np.float32)
    coords_y = yw.ravel().astype(np.float32)
    scalar = (xw ** 2 + yw ** 2).ravel().astype(np.float32)
    return coords_x, coords_y, scalar


def structured_to_explicit(nx, ny):
    """Build explicit quad connectivity matching a structured grid."""
    nc = (nx - 1) * (ny - 1)
    conn = np.empty(nc * 4, dtype=np.int32)
    for cj in range(ny - 1):
        for ci in range(nx - 1):
            cid = cj * (nx - 1) + ci
            p0 = cj * nx + ci
            conn[cid * 4: cid * 4 + 4] = [p0, p0 + 1, p0 + nx + 1, p0 + nx]
    return conn


# ================================================================
# DEMO
# ================================================================

if __name__ == "__main__":
    NX, NY = 20, 20
    NUM_CELLS = (NX - 1) * (NY - 1)
    NUM_POINTS = NX * NY
    MAX_LINES = NUM_CELLS * 2
    MAX_POINTS = MAX_LINES * 2
    ISOVALUE = 0.5

    # --- test data ---
    coords_x_np, coords_y_np, scalar_np = make_curvilinear_grid(NX, NY)

    # --- cell sets ---
    structured_cs = CellSetStructured2D(NX, NY)
    conn_np = structured_to_explicit(NX, NY)
    explicit_cs = CellSetExplicitQuads(conn_np, NUM_CELLS, NUM_POINTS)

    # --- allocate fields ---
    coords_x = tack.field(dtype=tack.f32, shape=(NUM_POINTS,))
    coords_y = tack.field(dtype=tack.f32, shape=(NUM_POINTS,))
    scalar_field = tack.field(dtype=tack.f32, shape=(NUM_POINTS,))

    num_lines_f = tack.field(dtype=tack.i32, shape=(NUM_CELLS,))
    scatter_offsets_f = tack.field(dtype=tack.i32, shape=(NUM_CELLS,))
    interp_pa = tack.field(dtype=tack.i32, shape=(MAX_POINTS,))
    interp_pb = tack.field(dtype=tack.i32, shape=(MAX_POINTS,))
    interp_w = tack.field(dtype=tack.f32, shape=(MAX_POINTS,))
    output_x = tack.field(dtype=tack.f32, shape=(MAX_POINTS,))
    output_y = tack.field(dtype=tack.f32, shape=(MAX_POINTS,))

    # --- load data ---
    coords_x.from_numpy(coords_x_np)
    coords_y.from_numpy(coords_y_np)
    scalar_field.from_numpy(scalar_np)

    # --- run on structured grid ---
    tl_s, tp_s = run_contour(
        structured_cs, coords_x, coords_y, scalar_field, ISOVALUE,
        num_lines_f, scatter_offsets_f,
        interp_pa, interp_pb, interp_w,
        output_x, output_y)
    contour_s_x = output_x.to_numpy()[:tp_s]
    contour_s_y = output_y.to_numpy()[:tp_s]

    # --- run on explicit grid ---
    tl_e, tp_e = run_contour(
        explicit_cs, coords_x, coords_y, scalar_field, ISOVALUE,
        num_lines_f, scatter_offsets_f,
        interp_pa, interp_pb, interp_w,
        output_x, output_y)
    contour_e_x = output_x.to_numpy()[:tp_e]
    contour_e_y = output_y.to_numpy()[:tp_e]

    # --- verify ---
    print(f"Structured: {tl_s} lines, {tp_s} points")
    print(f"Explicit:   {tl_e} lines, {tp_e} points")
    assert tl_s == tl_e, f"Line counts differ: {tl_s} vs {tl_e}"
    contour_s = np.stack([contour_s_x, contour_s_y], axis=1)
    contour_e = np.stack([contour_e_x, contour_e_y], axis=1)
    np.testing.assert_allclose(contour_s, contour_e, atol=1e-6)
    print("Results match between structured and explicit cell sets.")
    print(f"GPU prefix sum computed scatter offsets for {NUM_CELLS} cells.")

    # --- visualize ---
    try:
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        for ax, title, cx, cy in [
            (ax1, "CellSetStructured2D", contour_s_x, contour_s_y),
            (ax2, "CellSetExplicitQuads", contour_e_x, contour_e_y),
        ]:
            # mesh edges
            for cj in range(NY - 1):
                for ci in range(NX - 1):
                    p0 = cj * NX + ci
                    quad_x = coords_x_np[[p0, p0+1, p0+NX+1, p0+NX, p0]]
                    quad_y = coords_y_np[[p0, p0+1, p0+NX+1, p0+NX, p0]]
                    ax.plot(quad_x, quad_y, "k-", linewidth=0.3)
            # scalar field
            sc = ax.scatter(
                coords_x_np, coords_y_np,
                c=scalar_np, cmap="coolwarm", s=15, zorder=2, vmin=0, vmax=2)
            # contour
            n_pts = len(cx)
            if n_pts > 0:
                lines = [
                    [(cx[i*2], cy[i*2]), (cx[i*2+1], cy[i*2+1])]
                    for i in range(n_pts // 2)
                ]
                lc = LineCollection(lines, colors="lime", linewidths=2.5, zorder=3)
                ax.add_collection(lc)
            ax.set_title(f"{title}\n{n_pts//2} segments, isovalue={ISOVALUE}")
            ax.set_aspect("equal")
            ax.set_xlim(-1.3, 1.3)
            ax.set_ylim(-1.3, 1.3)

        fig.colorbar(sc, ax=[ax1, ax2], shrink=0.8, label="scalar field (x\u00b2+y\u00b2)")
        fig.suptitle("Tack Marching Squares: same algorithm, two cell set types",
                     fontsize=13, fontweight="bold")
        fig.subplots_adjust(top=0.88)
        import os
        plt.savefig(os.path.join(os.path.dirname(__file__), "..", "results", "contour_result.png"), dpi=150, bbox_inches="tight")
        print("Saved contour_result.png")
        plt.show()
    except ImportError:
        print("matplotlib not available; skipping visualization.")
