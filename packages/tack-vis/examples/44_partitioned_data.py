"""44 -- Partitioned data: a dataset in pieces, ghost marks, and reductions.

A PartitionedDataSet is a list of DataSets, each a whole dataset with its own
topology, fields and device. split() cuts a structured grid into pieces;
neighbouring pieces share the points between them, and each shared point has
one owner: the other copies are ghosts, marked in piece.ghosts. Work on one
piece takes the piece, in a plain loop. statistics() and entropy() take the
partitioned dataset and count only what each piece owns, so they give the
whole grid's answer.

With several devices of the chosen type (CUDA or HIP GPUs), the pieces are
spread over them and each is reduced where it is. With VTK installed, the
pieces are written as a .vtpd file (VTK's partitioned dataset), ghost marks
included, for ParaView.

Usage:
  uv run python packages/tack-vis/examples/44_partitioned_data.py
  uv run python packages/tack-vis/examples/44_partitioned_data.py --arch cuda --pieces 8
"""

import argparse
import pathlib

import tack
import tack.dataset as td

_parser = argparse.ArgumentParser()
_parser.add_argument('--arch', default='cpu',
                     choices=['cpu', 'metal', 'cuda', 'hip', 'level_zero'])
_parser.add_argument('--pieces', type=int, default=6, help='number of pieces (default 6)')
_parser.add_argument('--extent', type=int, default=20,
                     help='wavelet extent: (2*extent+1)^3 points (default 20)')
_args = _parser.parse_args()
tack.set_default_device(_args.arch)


# ================================================================
# THE WHOLE GRID, AND ITS PIECES
# ================================================================

whole = td.wavelet(_args.extent)
whole.fields["cells"] = td.project(whole, "RTData", td.Constant(whole))
pieces = td.split(whole, _args.pieces)
print(f"{whole.num_points} points and {whole.num_cells} cells in {pieces}")

owned_total = 0
for i, piece in enumerate(pieces):
    marks = piece.ghosts.points.numpy()
    owned = int((marks & td.Ghosts.DUPLICATEPOINT == 0).sum())
    owned_total += owned
    print(f"  piece {i}: {piece.topology.point_dims} points, {owned} owned, "
          f"{piece.num_points - owned} ghosts")
assert owned_total == whole.num_points, "every point should have exactly one owner"


# ================================================================
# SPREAD OVER DEVICES (when there are several)
# ================================================================

kind = tack.get_default_device().type
count = tack.device_count(kind) if kind in ("cuda", "hip", "level_zero") else 1
if count > 1:
    pieces = td.PartitionedDataSet(piece.to_device(f"{kind}:{i % count}")
                                   for i, piece in enumerate(pieces))
    print(f"pieces spread over {count} {kind} devices: "
          f"{[str(piece.device) for piece in pieces]}")


# ================================================================
# WORK ON EACH PIECE: A PLAIN LOOP
# ================================================================

triangles = 0
for piece in pieces:
    piece.fields["grad"] = td.gradients(piece, "RTData")      # on the piece's device
    triangles += td.contour(piece, "RTData", 160.0).num_cells
print(f"gradients on every piece; the 160 isosurface has {triangles} triangles in all")
# split makes no ghost cells, so every cell is in one piece and the pieces'
# isosurfaces together are the whole grid's.
assert triangles == td.contour(whole, "RTData", 160.0).num_cells


# ================================================================
# REDUCTIONS OVER THE PIECES: THE WHOLE'S ANSWER
# ================================================================

for name in ("RTData", "cells"):                    # point data, then cell data
    a, b = td.statistics(whole, name), td.statistics(pieces, name)
    e_whole, e_pieces = td.entropy(whole, name), td.entropy(pieces, name)
    print(f"{name:>6}: n {b.n} (whole {a.n}), mean {b.mean:.6f} (whole {a.mean:.6f}), "
          f"stddev {b.population_stddev:.6f} (whole {a.population_stddev:.6f}), "
          f"entropy {e_pieces:.6f} (whole {e_whole:.6f})")
    assert b.n == a.n
    assert abs(b.mean - a.mean) <= 1e-5 * abs(a.mean)
    assert abs(b.m2 - a.m2) <= 1e-4 * abs(a.m2)
    assert abs(e_pieces - e_whole) <= 1e-9


# ================================================================
# TO VTK (when it is installed)
# ================================================================

try:
    from vtkmodules.vtkIOParallelXML import vtkXMLPartitionedDataSetWriter

    from tack.interop.vtk import dataset_to_vtk
except ImportError:
    print("VTK is not installed: not writing the pieces")
else:
    out = pathlib.Path("results") / "partitioned_wavelet.vtpd"
    out.parent.mkdir(exist_ok=True)
    writer = vtkXMLPartitionedDataSetWriter()
    writer.SetInputData(dataset_to_vtk(td.PartitionedDataSet(
        piece.to_device("cpu") for piece in pieces)))
    writer.SetFileName(str(out))
    writer.Write()
    print(f"wrote {out} ({len(pieces)} pieces, ghost marks as vtkGhostType)")
