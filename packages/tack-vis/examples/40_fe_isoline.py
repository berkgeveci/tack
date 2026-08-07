"""Extract isolines from a high-order DG field using tack.fe.

Demonstrates the three-layer FE abstraction:
- QuadBasis: evaluate Lagrange basis at subdivision points
- ContiguousDofs: access per-element DOFs (DG layout)
- LinearQuadMap: bilinear physical coordinate mapping

Creates a 4×4 quad mesh with a synthetic order-3 DG field
(f(x,y) = sin(π·x)·sin(π·y)), extracts an isoline, and writes VTU.

Usage:
    uv run python examples/40_fe_isoline.py [--arch cpu|metal]
"""

import argparse
import math
import os

import numpy as np

import tack
from tack.fe.accessor import contiguous_from_numpy
from tack.fe.basis import QuadBasis, precompute_basis_matrix_2d
from tack.fe.geometry import linear_quad_map_from_numpy


def make_quad_mesh(nx, ny):
    """Create a regular nx × ny quad mesh on [0,1]^2.

    Returns (vx, vy, connectivity, conn_offsets, elem_indices).
    """
    vx = np.zeros((ny + 1) * (nx + 1))
    vy = np.zeros((ny + 1) * (nx + 1))
    for j in range(ny + 1):
        for i in range(nx + 1):
            vid = j * (nx + 1) + i
            vx[vid] = i / nx
            vy[vid] = j / ny

    connectivity = []
    conn_offsets = [0]
    for j in range(ny):
        for i in range(nx):
            v0 = j * (nx + 1) + i
            v1 = v0 + 1
            v2 = v1 + (nx + 1)
            v3 = v0 + (nx + 1)
            connectivity.extend([v0, v1, v2, v3])
            conn_offsets.append(len(connectivity))

    n_elems = nx * ny
    return (vx, vy,
            np.array(connectivity, dtype=np.int32),
            np.array(conn_offsets, dtype=np.int32),
            np.arange(n_elems, dtype=np.int32))


def make_dg_field(vx, vy, connectivity, conn_offsets, order, func):
    """Create a DG field by evaluating func at GL nodes of each element.

    Returns (dof_values, dof_offsets, gl_nodes_1d).
    """
    n1d = order + 1
    # GL nodes for the given order
    if order == 1:
        gl = np.array([0.0, 1.0])
    elif order == 2:
        gl = np.array([0.0, 0.5, 1.0])
    elif order == 3:
        gl = np.array([0.0, 1.0/3, 2.0/3, 1.0])
    else:
        # Chebyshev-Lobatto nodes as approximation
        gl = 0.5 * (1.0 - np.cos(np.pi * np.arange(n1d) / order))

    n_elems = len(conn_offsets) - 1
    ndof_per_elem = n1d * n1d
    dof_values = np.zeros(n_elems * ndof_per_elem)
    dof_offsets = np.zeros(n_elems + 1, dtype=np.int32)

    for ei in range(n_elems):
        c0 = conn_offsets[ei]
        # Element vertex coordinates
        px = np.array([vx[connectivity[c0 + v]] for v in range(4)])
        py = np.array([vy[connectivity[c0 + v]] for v in range(4)])

        dof_offsets[ei + 1] = dof_offsets[ei] + ndof_per_elem
        dbase = dof_offsets[ei]

        for jj in range(n1d):
            s = gl[jj]
            for ii in range(n1d):
                r = gl[ii]
                # Bilinear map to physical coords
                w0 = (1 - r) * (1 - s)
                w1 = r * (1 - s)
                w2 = r * s
                w3 = (1 - r) * s
                x = w0 * px[0] + w1 * px[1] + w2 * px[2] + w3 * px[3]
                y = w0 * py[0] + w1 * py[1] + w2 * py[2] + w3 * py[3]
                dof_values[dbase + jj * n1d + ii] = func(x, y)

    return dof_values, dof_offsets, gl


def write_vtu_lines(filename, x0, y0, x1, y1):
    """Write line segments as VTK unstructured grid."""
    n_segs = len(x0)
    with open(filename, "w") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="UnstructuredGrid" version="0.1">\n')
        f.write('<UnstructuredGrid>\n')
        f.write(f'<Piece NumberOfPoints="{n_segs*2}" '
                f'NumberOfCells="{n_segs}">\n')
        f.write('<Points>\n')
        f.write('<DataArray type="Float64" NumberOfComponents="3" '
                'format="ascii">\n')
        for i in range(n_segs):
            f.write(f"{x0[i]} {y0[i]} 0.0\n{x1[i]} {y1[i]} 0.0\n")
        f.write('</DataArray>\n</Points>\n')
        f.write('<Cells>\n')
        f.write('<DataArray type="Int32" Name="connectivity" '
                'format="ascii">\n')
        for i in range(n_segs):
            f.write(f"{i*2} {i*2+1}\n")
        f.write('</DataArray>\n')
        f.write('<DataArray type="Int32" Name="offsets" format="ascii">\n')
        for i in range(n_segs):
            f.write(f"{(i+1)*2}\n")
        f.write('</DataArray>\n')
        f.write('<DataArray type="UInt8" Name="types" format="ascii">\n')
        for i in range(n_segs):
            f.write("3\n")
        f.write('</DataArray>\n</Cells>\n')
        f.write('</Piece>\n</UnstructuredGrid>\n</VTKFile>\n')


# ── Tack kernels using tack.fe ───────────────────────────────────────

def define_kernels():

    @tack.kernel
    def eval_field_at_sub_pts(
        accessor: tack.template(),
        basis_matrix, ndof, n_pts,
        elem_indices, field_out,
    ):
        """Evaluate field at all subdivision points per element."""
        for idx in range(elem_indices.shape[0]):
            elem = elem_indices[idx]
            obase = idx * n_pts
            for pt in range(n_pts):
                val = 0.0
                brow = pt * ndof
                for d in range(ndof):
                    val += basis_matrix[brow + d] * accessor.get_dof(elem, d)
                field_out[obase + pt] = val

    @tack.kernel
    def count_segments(field_vals, n_sub, n_elems, isovalue, counts):
        """Count isoline segments per element (fixed subdivision)."""
        nrow = n_sub + 1
        for idx in range(n_elems):
            base = idx * nrow * nrow
            count = 0
            for sj in range(n_sub):
                for si in range(n_sub):
                    v0 = field_vals[base + sj * nrow + si]
                    v1 = field_vals[base + sj * nrow + si + 1]
                    v2 = field_vals[base + (sj+1) * nrow + si + 1]
                    v3 = field_vals[base + (sj+1) * nrow + si]
                    crossings = 0
                    if (v0 >= isovalue) != (v1 >= isovalue):
                        crossings += 1
                    if (v1 >= isovalue) != (v2 >= isovalue):
                        crossings += 1
                    if (v2 >= isovalue) != (v3 >= isovalue):
                        crossings += 1
                    if (v3 >= isovalue) != (v0 >= isovalue):
                        crossings += 1
                    count += crossings // 2
            counts[idx] = count

    @tack.kernel
    def generate_segments(
        field_vals,
        geom: tack.template(),
        elem_indices, seg_offsets,
        n_sub, isovalue,
        out_x0, out_y0, out_x1, out_y1,
    ):
        """Generate isoline segment endpoints in physical coordinates."""
        nrow = n_sub + 1
        inv_n = 1.0 / n_sub
        for idx in range(elem_indices.shape[0]):
            ei = elem_indices[idx]
            fbase = idx * nrow * nrow
            seg_idx = seg_offsets[idx]

            for sj in range(n_sub):
                for si in range(n_sub):
                    v0 = field_vals[fbase + sj * nrow + si]
                    v1 = field_vals[fbase + sj * nrow + si + 1]
                    v2 = field_vals[fbase + (sj+1) * nrow + si + 1]
                    v3 = field_vals[fbase + (sj+1) * nrow + si]

                    r0 = float(si) * inv_n
                    r1 = float(si + 1) * inv_n
                    s0 = float(sj) * inv_n
                    s1 = float(sj + 1) * inv_n

                    cr = tack.shared_like(out_x0, 4)
                    cs = tack.shared_like(out_x0, 4)
                    nc = 0

                    if (v0 >= isovalue) != (v1 >= isovalue):
                        t = (isovalue - v0) / (v1 - v0)
                        cr[nc] = r0 + t * (r1 - r0)
                        cs[nc] = s0
                        nc += 1
                    if (v1 >= isovalue) != (v2 >= isovalue):
                        t = (isovalue - v1) / (v2 - v1)
                        cr[nc] = r1
                        cs[nc] = s0 + t * (s1 - s0)
                        nc += 1
                    if (v2 >= isovalue) != (v3 >= isovalue):
                        t = (isovalue - v2) / (v3 - v2)
                        cr[nc] = r1 + t * (r0 - r1)
                        cs[nc] = s1
                        nc += 1
                    if (v3 >= isovalue) != (v0 >= isovalue):
                        t = (isovalue - v3) / (v0 - v3)
                        cr[nc] = r0
                        cs[nc] = s1 + t * (s0 - s1)
                        nc += 1

                    for p in range(nc // 2):
                        i0 = p * 2
                        i1 = p * 2 + 1
                        out_x0[seg_idx+p] = geom.physical_x(ei, cr[i0], cs[i0])
                        out_y0[seg_idx+p] = geom.physical_y(ei, cr[i0], cs[i0])
                        out_x1[seg_idx+p] = geom.physical_x(ei, cr[i1], cs[i1])
                        out_y1[seg_idx+p] = geom.physical_y(ei, cr[i1], cs[i1])
                        seg_idx += 1

    return eval_field_at_sub_pts, count_segments, generate_segments


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default=os.environ.get("Tack_ARCH", "cpu"))
    parser.add_argument("--order", type=int, default=3)
    parser.add_argument("--nx", type=int, default=8)
    parser.add_argument("--isovalue", type=float, default=0.5)
    parser.add_argument("--nsub", type=int, default=8)
    parser.add_argument("-o", "--output", default="fe_isoline.vtu")
    args = parser.parse_args()

    tack.init(args.arch)
    np_fp = np.float32 if args.arch == "metal" else np.float64
    tack_fp = tack.f32 if args.arch == "metal" else tack.f64

    print(f"Backend: {args.arch}, order: {args.order}, "
          f"mesh: {args.nx}×{args.nx}, isovalue: {args.isovalue}")

    # Create mesh and field
    vx, vy, conn, conn_off, elem_idx = make_quad_mesh(args.nx, args.nx)
    n_elems = len(elem_idx)

    def test_func(x, y):
        return math.sin(math.pi * x) * math.sin(math.pi * y)

    dof_values, dof_offsets, gl_1d = make_dg_field(
        vx, vy, conn, conn_off, args.order, test_func)

    print(f"  {n_elems} elements, {len(dof_values)} DOFs, "
          f"field range [{dof_values.min():.4f}, {dof_values.max():.4f}]")

    # Create tack.fe objects
    basis = QuadBasis(gl_1d, np_fp=np_fp)
    accessor = contiguous_from_numpy(dof_values, dof_offsets, np_fp=np_fp)
    geom = linear_quad_map_from_numpy(vx, vy, conn, conn_off, np_fp=np_fp)

    # Precompute basis matrix at subdivision points
    B = precompute_basis_matrix_2d(basis, args.nsub)
    ndof = basis.n1d * basis.n1d
    n_pts = (args.nsub + 1) ** 2
    f_B = tack.field_like(B.ravel().astype(np_fp))
    f_elem_idx = tack.field_like(elem_idx)

    # Define kernels
    eval_field, count_segs, gen_segs = define_kernels()

    # Step 1: evaluate field at subdivision points
    f_field_vals = tack.field(dtype=tack_fp, shape=(n_elems * n_pts,))
    eval_field(accessor, f_B, ndof, n_pts, f_elem_idx, f_field_vals)

    # Step 2: count segments
    f_counts = tack.field(dtype=tack.i32, shape=(n_elems,))
    count_segs(f_field_vals, args.nsub, n_elems, args.isovalue, f_counts)

    counts = f_counts.to_numpy()
    total_segs = int(counts.sum())
    print(f"  {total_segs} isoline segments")

    if total_segs == 0:
        print("  No isoline found.")
        return

    # Step 3: prefix scan + generate
    offsets = np.zeros(n_elems, dtype=np.int32)
    offsets[1:] = np.cumsum(counts[:-1])
    f_offsets = tack.field_like(offsets)

    f_x0 = tack.field(dtype=tack_fp, shape=(total_segs,))
    f_y0 = tack.field(dtype=tack_fp, shape=(total_segs,))
    f_x1 = tack.field(dtype=tack_fp, shape=(total_segs,))
    f_y1 = tack.field(dtype=tack_fp, shape=(total_segs,))

    gen_segs(f_field_vals, geom, f_elem_idx, f_offsets,
             args.nsub, args.isovalue,
             f_x0, f_y0, f_x1, f_y1)

    # Write output
    write_vtu_lines(args.output,
                    f_x0.to_numpy(), f_y0.to_numpy(),
                    f_x1.to_numpy(), f_y1.to_numpy())
    print(f"  Wrote {total_segs} segments to {args.output}")


if __name__ == "__main__":
    main()
