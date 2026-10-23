"""44 -- Partitioned data: a dataset in pieces, ghost marks, and reductions.

A PartitionedDataSet is a list of DataSets, each a whole dataset with its own
topology, fields and device. split() cuts a structured grid into pieces, here
with a layer of each neighbour's cells around every piece (ghost cells), and
every cell and point the pieces share has one owner: the other copies are
ghosts, marked in piece.ghosts. Work on one piece takes the piece, in a plain
loop; filters carry the marks, so what a piece makes from its ghost cells is
marked as its neighbour's. statistics() and entropy() take the partitioned
dataset and count only what each piece owns, so they give the whole grid's
answer -- of the grid, and of the isosurfaces the pieces made.

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
whole.fields["RTData_cells"] = td.project(whole, "RTData", td.Constant(whole))
pieces = td.split(whole, _args.pieces, ghost_layers=1)
print(f"{whole.num_points} points and {whole.num_cells} cells in {pieces}")


def owned(marks):
    """How many of a piece's cells or points it owns."""
    return int((marks.numpy() & td.Ghosts.DUPLICATECELL == 0).sum())   # = DUPLICATEPOINT


owned_points = owned_cells = 0
for i, piece in enumerate(pieces):
    cells, points = owned(piece.ghosts.cells), owned(piece.ghosts.points)
    owned_cells += cells
    owned_points += points
    print(f"  piece {i}: {piece.topology.point_dims} points, {points} owned; "
          f"{piece.num_cells - cells} ghost cells")
assert owned_points == whole.num_points, "every point should have exactly one owner"
assert owned_cells == whole.num_cells, "every cell should have exactly one owner"


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

surfaces = []
for piece in pieces:
    piece.fields["grad"] = td.gradients(piece, "RTData")      # on the piece's device
    surfaces.append(td.contour(piece, "RTData", 160.0))
surfaces = td.PartitionedDataSet(surfaces)
triangles = sum(owned(surface.ghosts.cells) for surface in surfaces)
made = sum(surface.num_cells for surface in surfaces)
print(f"gradients on every piece; the 160 isosurface has {triangles} triangles in all, "
      f"and {made - triangles} more from the ghost cells, marked as the neighbours'")
# Each piece owns the triangles of the cells it owns, and each point on a shared
# edge has one owner, so the owned parts together are the whole grid's isosurface.
iso = td.contour(whole, "RTData", 160.0)
assert triangles == iso.num_cells
assert td.statistics(surfaces, "RTData").n == iso.num_points


# ================================================================
# REDUCTIONS OVER THE PIECES: THE WHOLE'S ANSWER
# ================================================================

for name in ("RTData", "RTData_cells"):              # point data, then cell data
    a, b = td.statistics(whole, name), td.statistics(pieces, name)
    e_whole, e_pieces = td.entropy(whole, name), td.entropy(pieces, name)
    print(f"{name:>12}: n {b.n} (whole {a.n}), mean {b.mean:.6f} (whole {a.mean:.6f}), "
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
    vtkXMLPartitionedDataSetWriter(file_name=str(out))(dataset_to_vtk(td.PartitionedDataSet(
        piece.to_device("cpu") for piece in pieces)))
    print(f"wrote {out} ({len(pieces)} pieces, ghost marks as vtkGhostType)")
    # The isosurface without the ghost cells' triangles: what to hand to code that
    # does not know ghost marks.
    out = pathlib.Path("results") / "partitioned_isosurface.vtpd"
    vtkXMLPartitionedDataSetWriter(file_name=str(out))(dataset_to_vtk(td.PartitionedDataSet(
        td.remove_ghosts(surface).to_device("cpu") for surface in surfaces)))
    print(f"wrote {out} (the isosurface's pieces, ghost triangles removed)")
