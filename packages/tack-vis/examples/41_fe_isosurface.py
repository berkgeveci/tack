"""Extract isosurfaces from a high-order DG field on hexes using tack.fe.

Demonstrates 3D FE abstractions:
- HexBasis: tensor-product Lagrange on [0,1]^3
- ContiguousDofs: per-element DOF access
- LinearHexMap: trilinear physical mapping

Creates a 4×4×4 hex mesh with a synthetic order-2 DG field
(f = sin(π·x)·sin(π·y)·sin(π·z)), extracts an isosurface via
adaptive marching cubes, and writes VTU.

Usage:
    uv run python examples/41_fe_isosurface.py [--arch cpu|metal]
"""

import argparse
import math
import os
import sys

import numpy as np

import tack
from tack.fe.accessor import contiguous_from_numpy
from tack.fe.basis import HexBasis, precompute_basis_matrix_3d
from tack.fe.geometry import linear_hex_map_from_numpy

# MC tables
sys.path.insert(0, os.path.dirname(__file__))
try:
    from mc_tables import EDGE_VERTS, NUM_TRIS, TRI_TABLE
except ImportError:
    # Fall back to the MFEM examples copy
    sys.path.insert(
        0, os.path.expanduser("~/Work/mfem/mfem/examples"))
    from mc_tables import EDGE_VERTS, NUM_TRIS, TRI_TABLE


def make_hex_mesh(nx, ny, nz):
    """Create a regular nx×ny×nz hex mesh on [0,1]^3."""
    n_verts = (nx+1) * (ny+1) * (nz+1)
    vx = np.zeros(n_verts)
    vy = np.zeros(n_verts)
    vz = np.zeros(n_verts)
    for k in range(nz+1):
        for j in range(ny+1):
            for i in range(nx+1):
                vid = k*(ny+1)*(nx+1) + j*(nx+1) + i
                vx[vid] = i / nx
                vy[vid] = j / ny
                vz[vid] = k / nz

    connectivity = []
    conn_offsets = [0]
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                v0 = k*(ny+1)*(nx+1) + j*(nx+1) + i
                v1 = v0 + 1
                v2 = v0 + (nx+1) + 1
                v3 = v0 + (nx+1)
                v4 = v0 + (ny+1)*(nx+1)
                v5 = v4 + 1
                v6 = v4 + (nx+1) + 1
                v7 = v4 + (nx+1)
                connectivity.extend([v0, v1, v2, v3, v4, v5, v6, v7])
                conn_offsets.append(len(connectivity))

    n_elems = nx * ny * nz
    return (vx, vy, vz,
            np.array(connectivity, dtype=np.int32),
            np.array(conn_offsets, dtype=np.int32),
            np.arange(n_elems, dtype=np.int32))


def make_dg_field_3d(vx, vy, vz, connectivity, conn_offsets, order, func):
    """Create a DG field on hexes by evaluating func at GL nodes."""
    n1d = order + 1
    if order == 1:
        gl = np.array([0.0, 1.0])
    elif order == 2:
        gl = np.array([0.0, 0.5, 1.0])
    elif order == 3:
        gl = np.array([0.0, 1.0/3, 2.0/3, 1.0])
    else:
        gl = 0.5 * (1.0 - np.cos(np.pi * np.arange(n1d) / order))

    n_elems = len(conn_offsets) - 1
    ndof_per_elem = n1d ** 3
    dof_values = np.zeros(n_elems * ndof_per_elem)
    dof_offsets = np.zeros(n_elems + 1, dtype=np.int32)

    for ei in range(n_elems):
        c0 = conn_offsets[ei]
        px = np.array([vx[connectivity[c0 + v]] for v in range(8)])
        py = np.array([vy[connectivity[c0 + v]] for v in range(8)])
        pz = np.array([vz[connectivity[c0 + v]] for v in range(8)])

        dof_offsets[ei + 1] = dof_offsets[ei] + ndof_per_elem
        dbase = dof_offsets[ei]

        for kk in range(n1d):
            t = gl[kk]
            for jj in range(n1d):
                s = gl[jj]
                for ii in range(n1d):
                    r = gl[ii]
                    w = np.array([
                        (1-r)*(1-s)*(1-t), r*(1-s)*(1-t),
                        r*s*(1-t), (1-r)*s*(1-t),
                        (1-r)*(1-s)*t, r*(1-s)*t,
                        r*s*t, (1-r)*s*t])
                    x = w @ px
                    y = w @ py
                    z = w @ pz
                    dof = kk * n1d * n1d + jj * n1d + ii
                    dof_values[dbase + dof] = func(x, y, z)

    return dof_values, dof_offsets, gl


def write_vtu_triangles(filename, px, py, pz, n_tris):
    """Write triangles as VTK unstructured grid."""
    n_pts = n_tris * 3
    with open(filename, "w") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="UnstructuredGrid" version="0.1">\n')
        f.write('<UnstructuredGrid>\n')
        f.write(f'<Piece NumberOfPoints="{n_pts}" '
                f'NumberOfCells="{n_tris}">\n')
        f.write('<Points>\n')
        f.write('<DataArray type="Float64" NumberOfComponents="3" '
                'format="ascii">\n')
        for i in range(n_pts):
            f.write(f"{px[i]} {py[i]} {pz[i]}\n")
        f.write('</DataArray>\n</Points>\n')
        f.write('<Cells>\n')
        f.write('<DataArray type="Int32" Name="connectivity" '
                'format="ascii">\n')
        for i in range(n_tris):
            f.write(f"{i*3} {i*3+1} {i*3+2}\n")
        f.write('</DataArray>\n')
        f.write('<DataArray type="Int32" Name="offsets" format="ascii">\n')
        for i in range(n_tris):
            f.write(f"{(i+1)*3}\n")
        f.write('</DataArray>\n')
        f.write('<DataArray type="UInt8" Name="types" format="ascii">\n')
        for i in range(n_tris):
            f.write("5\n")
        f.write('</DataArray>\n</Cells>\n')
        f.write('</Piece>\n</UnstructuredGrid>\n</VTKFile>\n')


# ── Tack kernels ─────────────────────────────────────────────────────

def define_kernels():

    @tack.kernel
    def eval_field_at_sub_pts(
        accessor: tack.template(),
        basis_matrix, ndof, n_pts,
        elem_indices, field_out,
    ):
        for idx in range(elem_indices.shape[0]):
            elem = elem_indices[idx]
            obase = idx * n_pts
            for pt in range(n_pts):
                val = 0.0
                brow = pt * ndof
                for d in range(ndof):
                    val += basis_matrix[brow + d] * accessor.get_dof(elem, d)
                field_out[obase + pt] = val

    @tack.func
    def mc_case_index(v0, v1, v2, v3, v4, v5, v6, v7, iso):
        idx = 0
        if v0 < iso: idx = idx | 1
        if v1 < iso: idx = idx | 2
        if v2 < iso: idx = idx | 4
        if v3 < iso: idx = idx | 8
        if v4 < iso: idx = idx | 16
        if v5 < iso: idx = idx | 32
        if v6 < iso: idx = idx | 64
        if v7 < iso: idx = idx | 128
        return idx

    @tack.func
    def get_val(fv, base, nrow, si, sj, sk):
        return fv[base + sk * nrow * nrow + sj * nrow + si]

    @tack.kernel
    def count_tris(field_vals, n_elems, n_sub, isovalue,
                   num_tris_table, counts):
        nrow = n_sub + 1
        for idx in range(n_elems):
            base = idx * nrow * nrow * nrow
            total = 0
            for sk in range(n_sub):
                for sj in range(n_sub):
                    for si in range(n_sub):
                        v0 = get_val(field_vals, base, nrow, si, sj, sk)
                        v1 = get_val(field_vals, base, nrow, si+1, sj, sk)
                        v2 = get_val(field_vals, base, nrow, si+1, sj+1, sk)
                        v3 = get_val(field_vals, base, nrow, si, sj+1, sk)
                        v4 = get_val(field_vals, base, nrow, si, sj, sk+1)
                        v5 = get_val(field_vals, base, nrow, si+1, sj, sk+1)
                        v6 = get_val(field_vals, base, nrow, si+1, sj+1, sk+1)
                        v7 = get_val(field_vals, base, nrow, si, sj+1, sk+1)
                        ci = mc_case_index(v0,v1,v2,v3,v4,v5,v6,v7, isovalue)
                        total += num_tris_table[ci]
            counts[idx] = total

    @tack.func
    def edge_interp_t(fv, base, nrow, si, sj, sk, edge, iso, ev):
        vi0 = ev[edge * 2]
        vi1 = ev[edge * 2 + 1]
        di0 = 0; dj0 = 0; dk0 = 0
        di1 = 0; dj1 = 0; dk1 = 0
        if vi0==1 or vi0==2 or vi0==5 or vi0==6: di0 = 1
        if vi0==2 or vi0==3 or vi0==6 or vi0==7: dj0 = 1
        if vi0 >= 4: dk0 = 1
        if vi1==1 or vi1==2 or vi1==5 or vi1==6: di1 = 1
        if vi1==2 or vi1==3 or vi1==6 or vi1==7: dj1 = 1
        if vi1 >= 4: dk1 = 1
        va = fv[base + (sk+dk0)*nrow*nrow + (sj+dj0)*nrow + si+di0]
        vb = fv[base + (sk+dk1)*nrow*nrow + (sj+dj1)*nrow + si+di1]
        t = 0.0
        if vb != va:
            t = (iso - va) / (vb - va)
        return t

    @tack.func
    def edge_ref(si, sj, sk, edge, t, comp, inv_n, ev):
        vi0 = ev[edge * 2]
        vi1 = ev[edge * 2 + 1]
        d0 = 0; d1 = 0; bc = 0
        if comp == 0:
            bc = si
            if vi0==1 or vi0==2 or vi0==5 or vi0==6: d0 = 1
            if vi1==1 or vi1==2 or vi1==5 or vi1==6: d1 = 1
        elif comp == 1:
            bc = sj
            if vi0==2 or vi0==3 or vi0==6 or vi0==7: d0 = 1
            if vi1==2 or vi1==3 or vi1==6 or vi1==7: d1 = 1
        else:
            bc = sk
            if vi0 >= 4: d0 = 1
            if vi1 >= 4: d1 = 1
        return (float(bc + d0) + t * float(d1 - d0)) * inv_n

    @tack.kernel
    def generate_tris(
        field_vals,
        geom: tack.template(),
        elem_indices, tri_offsets,
        n_sub, isovalue,
        num_tris_table, tri_table, edge_verts,
        out_px, out_py, out_pz,
    ):
        nrow = n_sub + 1
        inv_n = 1.0 / n_sub
        for idx in range(elem_indices.shape[0]):
            ei = elem_indices[idx]
            base = idx * nrow * nrow * nrow
            tri_idx = tri_offsets[idx]

            for sk in range(n_sub):
                for sj in range(n_sub):
                    for si in range(n_sub):
                        v0 = get_val(field_vals, base, nrow, si, sj, sk)
                        v1 = get_val(field_vals, base, nrow, si+1, sj, sk)
                        v2 = get_val(field_vals, base, nrow, si+1, sj+1, sk)
                        v3 = get_val(field_vals, base, nrow, si, sj+1, sk)
                        v4 = get_val(field_vals, base, nrow, si, sj, sk+1)
                        v5 = get_val(field_vals, base, nrow, si+1, sj, sk+1)
                        v6 = get_val(field_vals, base, nrow, si+1, sj+1, sk+1)
                        v7 = get_val(field_vals, base, nrow, si, sj+1, sk+1)
                        ci = mc_case_index(v0,v1,v2,v3,v4,v5,v6,v7, isovalue)
                        nt = num_tris_table[ci]
                        for ti in range(nt):
                            for vi in range(3):
                                edge = tri_table[ci*16 + ti*3 + vi]
                                t = edge_interp_t(
                                    field_vals, base, nrow,
                                    si, sj, sk, edge, isovalue,
                                    edge_verts)
                                rr = edge_ref(si,sj,sk,edge,t,0,inv_n,edge_verts)
                                ss = edge_ref(si,sj,sk,edge,t,1,inv_n,edge_verts)
                                tt = edge_ref(si,sj,sk,edge,t,2,inv_n,edge_verts)
                                out_px[tri_idx*3+vi] = geom.physical_x(ei, rr, ss, tt)
                                out_py[tri_idx*3+vi] = geom.physical_y(ei, rr, ss, tt)
                                out_pz[tri_idx*3+vi] = geom.physical_z(ei, rr, ss, tt)
                            tri_idx += 1

    return eval_field_at_sub_pts, count_tris, generate_tris


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default=os.environ.get("Tack_ARCH", "cpu"))
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument("--nx", type=int, default=4)
    parser.add_argument("--isovalue", type=float, default=0.5)
    parser.add_argument("--nsub", type=int, default=4)
    parser.add_argument("-o", "--output", default="fe_isosurface.vtu")
    args = parser.parse_args()

    tack.init(args.arch)
    np_fp = np.float32 if args.arch == "metal" else np.float64
    tack_fp = tack.f32 if args.arch == "metal" else tack.f64

    print(f"Backend: {args.arch}, order: {args.order}, "
          f"mesh: {args.nx}^3, isovalue: {args.isovalue}")

    # Create mesh and field
    vx, vy, vz, conn, conn_off, elem_idx = make_hex_mesh(
        args.nx, args.nx, args.nx)
    n_elems = len(elem_idx)

    def test_func(x, y, z):
        return math.sin(math.pi * x) * math.sin(math.pi * y) * \
               math.sin(math.pi * z)

    dof_values, dof_offsets, gl_1d = make_dg_field_3d(
        vx, vy, vz, conn, conn_off, args.order, test_func)

    print(f"  {n_elems} elements, {len(dof_values)} DOFs, "
          f"field range [{dof_values.min():.4f}, {dof_values.max():.4f}]")

    # Create tack.fe objects
    basis = HexBasis(gl_1d, np_fp=np_fp)
    accessor = contiguous_from_numpy(dof_values, dof_offsets, np_fp=np_fp)
    geom = linear_hex_map_from_numpy(vx, vy, vz, conn, conn_off, np_fp=np_fp)

    # Precompute basis matrix
    B = precompute_basis_matrix_3d(basis, args.nsub)
    ndof = basis.n1d ** 3
    n_pts = (args.nsub + 1) ** 3
    f_B = tack.field_like(B.ravel().astype(np_fp))
    f_elem_idx = tack.field_like(elem_idx)

    # MC tables
    f_num_tris = tack.field_like(NUM_TRIS)
    f_tri_table = tack.field_like(TRI_TABLE)
    f_edge_verts = tack.field_like(EDGE_VERTS)

    eval_field, count_tris_k, gen_tris = define_kernels()

    # Evaluate field
    f_field_vals = tack.field(dtype=tack_fp, shape=(n_elems * n_pts,))
    eval_field(accessor, f_B, ndof, n_pts, f_elem_idx, f_field_vals)

    # Count
    f_counts = tack.field(dtype=tack.i32, shape=(n_elems,))
    count_tris_k(f_field_vals, n_elems, args.nsub, args.isovalue,
                  f_num_tris, f_counts)

    counts = f_counts.to_numpy()
    total_tris = int(counts.sum())
    print(f"  {total_tris} triangles")

    if total_tris == 0:
        print("  No isosurface found.")
        return

    # Generate
    offsets = np.zeros(n_elems, dtype=np.int32)
    offsets[1:] = np.cumsum(counts[:-1])
    f_offsets = tack.field_like(offsets)

    n_out = total_tris * 3
    f_px = tack.field(dtype=tack_fp, shape=(n_out,))
    f_py = tack.field(dtype=tack_fp, shape=(n_out,))
    f_pz = tack.field(dtype=tack_fp, shape=(n_out,))

    gen_tris(f_field_vals, geom, f_elem_idx, f_offsets,
             args.nsub, args.isovalue,
             f_num_tris, f_tri_table, f_edge_verts,
             f_px, f_py, f_pz)

    px = f_px.to_numpy()
    py = f_py.to_numpy()
    pz = f_pz.to_numpy()

    write_vtu_triangles(args.output, px, py, pz, total_tris)
    print(f"  Wrote {total_tris} triangles to {args.output}")


if __name__ == "__main__":
    main()
