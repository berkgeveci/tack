"""42 -- A tour of the dataset API prototype (docs/design/dataset-api.md).

One mixed unstructured grid (hexahedra, wedges, pyramids and tetrahedra)
and one rectilinear grid, each run through the same algorithms:

- topology: faces and edges derived from the cells, each face knowing the
  cells on its two sides, each cell its faces and edges;
- spaces: point data (H1), cell data (Constant), a linear DG field (L2),
  and values on faces and edges, all as Fields -- the geometry among them,
  the field named "shape";
- iteration domains: kernels over cells, faces, edges, or a set of faces,
  once per shape, reading fields through views built for that shape.

With --output DIR it writes, for ParaView:
  <grid>.vt[ur]          the grid, with H1 and Constant fields
  <grid>_boundary.vtu    the boundary faces, with face area and the jump of
                         a cell field across each face as cell data
  mixed_dg.vtu           the DG field, one copy of each point per cell, the
                         way MFEM's Catalyst "mesh" channel writes DG data:
                         colour by "dg" to see the jumps between cells
  mixed_shrunk.vtu       the same grid with an L2 geometry: each cell's
                         corners pulled toward its center, over the same
                         topology

Usage:
  python examples/42_dataset_api_tour.py [--arch cpu|metal|cuda|hip|level_zero]
                                         [--output DIR]
"""

import argparse
import os

import numpy as np

import tack

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--arch", default="cpu",
                    choices=["cpu", "metal", "cuda", "hip", "level_zero"])
parser.add_argument("--output", help="directory to write VTK files into")
args = parser.parse_args()
if args.arch == "level_zero":
    from tack.interop.vtk import init_level_zero
    init_level_zero()
else:
    tack.init(arch=getattr(tack, args.arch))

import tack.data as td
from tack.data import algorithms as alg


def mixed_grid():
    """Blocks of 2x2x1 unit cubes from vtkCellTypeSource, merged into one grid:
    hexahedra, wedges and pyramids side by side along x, meeting through quad
    faces, and a block of tetrahedra on its own beside them (its triangles
    would not match a neighbour's quads)."""
    from vtkmodules.vtkFiltersCore import vtkAppendFilter
    from vtkmodules.vtkFiltersSources import vtkCellTypeSource

    pieces = vtkAppendFilter()
    pieces.MergePointsOn()
    for cell_type, (dx, dy) in ((12, (0, 0)), (13, (2, 0)), (14, (4, 0)), (10, (0, 3))):
        source = vtkCellTypeSource()
        source.SetCellType(cell_type)
        source.SetBlocksDimensions(2, 2, 1)
        source.Update()
        block = source.GetOutput()
        points = block.GetPoints()
        for p in range(points.GetNumberOfPoints()):
            x, y, z = points.GetPoint(p)
            points.SetPoint(p, x + dx, y + dy, z)
        pieces.AddInputData(block)
    pieces.Update()
    return pieces.GetOutput()


def show(name, data):
    from tack.interop.vtk import dataset_to_vtk

    faces = data.topology.faces()
    edges = data.topology.edges()
    groups = ", ".join(f"{g.count} {g.shape.__name__}" for g in data.topology.groups())
    print(f"\n{name}: {data.num_points} points, {data.num_cells} cells ({groups})")
    face_groups = ", ".join(f"{g.count} {g.shape.__name__}" for g in faces.groups())
    print(f"  derived: {faces.num_faces} faces ({face_groups}), {edges.num_edges} edges")

    # Geometry is evaluated through the cells' shape functions; an H1 field too.
    x = data.positions()
    data.fields["height"] = td.Field(td.H1(), _scalars(x[:, 2] + 0.25 * x[:, 0]))
    centers = alg.cell_centers(data).values.to_numpy(vectors=True)
    data.fields["cell id"] = td.Field(td.Constant(),
                                      _scalars(np.arange(data.num_cells, dtype=float)))
    at_centers = alg.values_at_centers(data, data.fields["height"]).values.to_numpy()
    print(f"  H1 'height' at the cell centers matches the centers' own height: "
          f"{np.allclose(at_centers, centers[:, 2] + 0.25 * centers[:, 0], atol=1e-5)}")
    # Its gradient: its own basis, and the geometry field's Jacobian.
    gradient = alg.gradients(data, data.fields["height"]).values.to_numpy(vectors=True)
    print(f"  its gradient is (0.25, 0, 1) in every cell: "
          f"{np.allclose(gradient, [0.25, 0, 1], atol=1e-5)}")

    # Fields on faces and edges.
    normals, areas = alg.face_geometry(data)
    lengths = alg.edge_lengths(data)
    print(f"  face areas sum to {areas.values.to_numpy().sum():.4g}, "
          f"edge lengths to {lengths.values.to_numpy().sum():.4g}")

    # Two sides of a face: a cell field's jump, and the outward sum back onto cells.
    jump = alg.jump(data, data.fields["cell id"])
    print(f"  'cell id' jumps across {np.count_nonzero(jump.values.to_numpy())} "
          f"interior faces")
    area_vectors = normals.values.to_numpy(vectors=True) * areas.values.to_numpy()[:, None]
    closed = alg.divergence(data, td.Field(td.Values("faces"), _vectors(area_vectors)))
    print(f"  every cell's outward area vectors sum to zero: "
          f"{np.allclose(closed.values.to_numpy(vectors=True), 0, atol=1e-5)}")

    # A side set, and a kernel run over it alone.
    boundary = alg.boundary_faces(data)
    print(f"  boundary: {boundary.shape[0]} faces, area "
          f"{areas.values.to_numpy()[boundary.to_numpy()].sum():.4g}")
    data.fields["area"] = areas
    data.fields["jump"] = jump
    surface = alg.extract_surface(data)

    # Cell data to points, and the same function in the DG layout.
    data.fields["cell id at points"] = alg.to_points(data, data.fields["cell id"])
    dg = alg.discontinuous(data, data.fields["height"])
    back = alg.to_points(data, dg).values.to_numpy()
    print(f"  'height' as an L2 field: {dg.values.shape[0]} values, one per cell corner; "
          f"averaged back onto the points it is unchanged: "
          f"{np.allclose(back, data.fields['height'].values.to_numpy(), atol=1e-5)}")

    if args.output:
        os.makedirs(args.output, exist_ok=True)
        _write(dataset_to_vtk(data), os.path.join(args.output, name))
        _write(dataset_to_vtk(surface), os.path.join(args.output, f"{name}_boundary"))
    return dg


def explode(data, dg):
    """L2 data as VTK can show it: a copy of each cell's points, so each cell carries
    its own values at its corners. An unstructured topology's L2 values are laid out
    like its connectivity, so this is a gather -- or, for an L2 geometry, nothing:
    its values are already one position per cell corner."""
    from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkUnstructuredGrid

    topology = data.topology
    connectivity = topology.connectivity.to_numpy()
    grid = vtkUnstructuredGrid()
    points = vtkPoints()
    if data.geometry.space == td.L2():
        corners = data.geometry.values.to_numpy(vectors=True)
    else:
        corners = data.positions()[connectivity]
    points.SetData(numpy_to_vtk(corners, deep=1))
    grid.SetPoints(points)
    cells = vtkCellArray()
    cells.SetData(numpy_to_vtkIdTypeArray(topology.offsets.to_numpy().astype(np.int64), deep=1),
                  numpy_to_vtkIdTypeArray(np.arange(len(connectivity), dtype=np.int64),
                                          deep=1))
    grid.SetCells(numpy_to_vtk(topology.types.to_numpy(), deep=1), cells)
    values = numpy_to_vtk(dg.values.to_numpy(), deep=1)
    values.SetName("dg")
    grid.GetPointData().AddArray(values)
    return grid


def _write(grid, stem):
    from vtkmodules.vtkIOXML import vtkXMLRectilinearGridWriter, vtkXMLUnstructuredGridWriter

    rectilinear = grid.IsA("vtkRectilinearGrid")
    writer = vtkXMLRectilinearGridWriter() if rectilinear else vtkXMLUnstructuredGridWriter()
    writer.SetFileName(stem + (".vtr" if rectilinear else ".vtu"))
    writer.SetInputData(grid)
    writer.Write()
    print(f"  wrote {writer.GetFileName()}")


def _scalars(values):
    f = tack.field(tack.f32, shape=(len(values),))
    f.from_numpy(np.asarray(values, np.float32))
    return f


def _vectors(values):
    f = tack.Vector.field(3, tack.f32, shape=(len(values),))
    f.from_numpy(np.asarray(values, np.float32))
    return f


from tack.interop.vtk import vtk_to_dataset

mixed = vtk_to_dataset(mixed_grid())
dg = show("mixed", mixed)

# Make the DG field discontinuous: each cell tilts its own values, so cells
# disagree where they meet, and averaging onto points no longer recovers them.
offsets = mixed.l2_offsets().to_numpy()
values = dg.values.to_numpy()
rng = np.random.default_rng(0)
for c in range(mixed.num_cells):
    values[offsets[c]:offsets[c + 1]] += rng.uniform(-0.3, 0.3)
dg = td.Field(td.L2(), _scalars(values), dg.offsets)
spread = alg.to_points(mixed, dg).values.to_numpy() - mixed.fields["height"].values.to_numpy()
print(f"  each cell shifted by up to 0.3: the point averages now differ by up to "
      f"{np.abs(spread).max():.3f}")
if args.output:
    _write(explode(mixed, dg), os.path.join(args.output, "mixed_dg"))

# The geometry is a field, so it can be discontinuous too: each cell's own
# corners, pulled toward its center. The topology is the same, so the cells
# still know their faces and neighbours; the faces just have no single
# position any more, which is what per-side traces will be for.
centers = alg.cell_centers(mixed).values.to_numpy(vectors=True)
connectivity = mixed.topology.connectivity.to_numpy()
corners = mixed.positions()[connectivity]
for c in range(mixed.num_cells):
    rows = slice(offsets[c], offsets[c + 1])
    corners[rows] = centers[c] + 0.7 * (corners[rows] - centers[c])
shrunk = td.DataSet(mixed.topology, td.Field(td.L2(), _vectors(corners), mixed.l2_offsets()),
                    fields={"height": mixed.fields["height"]})
gradient = alg.gradients(shrunk, shrunk.fields["height"]).values.to_numpy(vectors=True)
print(f"\nshrunk: the mixed grid with an L2 geometry, cells at 70%: "
      f"{shrunk.topology.faces().num_faces} faces, the same; 'height' over the "
      f"smaller cells has gradient (0.25, 0, 1) / 0.7 in every cell: "
      f"{np.allclose(gradient, np.array([0.25, 0, 1]) / 0.7, atol=1e-4)}")
if args.output:
    _write(explode(shrunk, alg.discontinuous(shrunk, shrunk.fields["height"])),
           os.path.join(args.output, "mixed_shrunk"))

show("rectilinear", td.rectilinear_grid(np.linspace(0, 3, 7), [0, 0.5, 1.5, 3], [0, 1, 2]))
