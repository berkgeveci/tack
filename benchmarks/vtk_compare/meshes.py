"""Meshes for the VTK comparison, built once with NumPy and handed to both sides.

A mesh is points and cells of one linear shape (hexahedra or tetrahedra),
with two scalar fields: ``p`` on the points and ``c`` on the cells, both
smooth, so a contour, slice or threshold cuts a realistic share of cells.
Each mesh is given to VTK and to Tack in two forms:

- shape-based: a ``vtkUnstructuredGrid`` of hexahedra or tetrahedra, and a
  Tack ``UnstructuredTopology`` of the same cells;
- polyhedral: every cell as a polyhedron of its faces. VTK gets
  ``SetPolyhedralCells`` with each cell's own outward copy of its faces, as
  its readers produce; Tack gets ``as_polyhedra`` of the shape-based dataset,
  each face stored once. ``as_polyhedra`` is timed on its own.
"""

from dataclasses import dataclass
from itertools import permutations

import numpy as np

VTK_TETRA, VTK_HEXAHEDRON, VTK_POLYHEDRON = 10, 12, 42

# Each shape's faces, outward, in VTK's numbering (vtkHexahedron, vtkTetra).
FACES = {
    VTK_HEXAHEDRON: ((0, 4, 7, 3), (1, 2, 6, 5), (0, 1, 5, 4), (3, 7, 6, 2),
                     (0, 3, 2, 1), (4, 5, 6, 7)),
    VTK_TETRA: ((0, 1, 3), (1, 2, 3), (2, 0, 3), (0, 2, 1)),
}

# Approximate cell counts of the size sweep.
SIZES = {"10k": 10_000, "100k": 100_000, "1M": 1_000_000, "5M": 5_000_000}


@dataclass
class Mesh:
    name: str
    cell_type: int
    points: np.ndarray            # (n, 3) float32
    cells: np.ndarray             # (m, k) int64, VTK's corner order
    p: np.ndarray = None          # on the points, float32
    c: np.ndarray = None          # on the cells, float32

    @property
    def num_cells(self):
        return self.cells.shape[0]

    @property
    def num_points(self):
        return self.points.shape[0]


def _with_fields(mesh):
    """``p``: a wavy height over the mesh's box; ``c``: each cell center's
    position along x across the box, in [0, 1]."""
    lo, hi = mesh.points.min(axis=0), mesh.points.max(axis=0)
    u = (mesh.points - lo) / np.where(hi > lo, hi - lo, 1)
    mesh.p = (0.25 * np.sin(2 * np.pi * u[:, 0]) * np.cos(2 * np.pi * u[:, 1])
              + u[:, 2]).astype(np.float32)
    centers = u[mesh.cells].mean(axis=1)
    mesh.c = centers[:, 0].astype(np.float32)
    return mesh


def _grid_points(n):
    axis = np.linspace(0.0, 1.0, n + 1, dtype=np.float32)
    z, y, x = np.meshgrid(axis, axis, axis, indexing="ij")
    return np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)


def _cube_corners(n):
    """Each cube's eight corner point ids, in VTK hexahedron order."""
    m = n + 1
    k, j, i = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
    base = (i + m * (j + m * k)).ravel()
    offsets = np.array([0, 1, 1 + m, m, m * m, 1 + m * m, 1 + m + m * m, m + m * m])
    return base[:, None] + offsets[None, :]


def hexahedra(cells):
    """About ``cells`` unit-cube hexahedra filling [0, 1]^3."""
    n = max(1, round(cells ** (1 / 3)))
    return _with_fields(Mesh("hex", VTK_HEXAHEDRON, _grid_points(n), _cube_corners(n)))


def _kuhn_tets():
    """Six positive tetrahedra of a cube (Kuhn's split: every path from corner 000 to
    111), as indices into the hexahedron's corners. Neighbouring cubes split their
    shared faces the same way, so the mesh is conforming."""
    corner = {(0, 0, 0): 0, (1, 0, 0): 1, (1, 1, 0): 2, (0, 1, 0): 3,
              (0, 0, 1): 4, (1, 0, 1): 5, (1, 1, 1): 6, (0, 1, 1): 7}
    tets = []
    for order in permutations(range(3)):
        at, path = [0, 0, 0], [(0, 0, 0)]
        for axis in order:
            at[axis] = 1
            path.append(tuple(at))
        x = np.array(path, float)
        if np.linalg.det(x[1:] - x[0]) < 0:
            path[1], path[2] = path[2], path[1]
        tets.append([corner[p] for p in path])
    return np.array(tets)


def tetrahedra(cells):
    """About ``cells`` tetrahedra: cubes filling [0, 1]^3, six tetrahedra each."""
    n = max(1, round((cells / 6) ** (1 / 3)))
    corners = _cube_corners(n)
    tets = corners[:, _kuhn_tets()].reshape(-1, 4)
    return _with_fields(Mesh("tet", VTK_TETRA, _grid_points(n), tets))


def square_bend(path):
    """OpenFOAM's squareBend tutorial mesh (112k hexahedra), read by VTK."""
    from vtkmodules.util.numpy_support import vtk_to_numpy
    from vtkmodules.vtkIOGeometry import vtkOpenFOAMReader

    reader = vtkOpenFOAMReader()
    reader.SetFileName(path)
    reader.Update()
    blocks = reader.GetOutput().NewIterator()
    blocks.InitTraversal()
    grid = blocks.GetCurrentDataObject()
    types = vtk_to_numpy(grid.GetCellTypesArray())
    if not (types == VTK_HEXAHEDRON).all():
        raise ValueError("squareBend is expected to be all hexahedra")
    connectivity = vtk_to_numpy(grid.GetCells().GetConnectivityArray()).astype(np.int64)
    points = vtk_to_numpy(grid.GetPoints().GetData()).astype(np.float32)
    return _with_fields(Mesh("squareBend", VTK_HEXAHEDRON, points,
                             connectivity.reshape(-1, 8)))


# ── VTK inputs ──────────────────────────────────────────────────────

def _cell_array(offsets, connectivity):
    from vtkmodules.util.numpy_support import numpy_to_vtkIdTypeArray
    from vtkmodules.vtkCommonDataModel import vtkCellArray

    cells = vtkCellArray()
    cells.SetData(numpy_to_vtkIdTypeArray(np.ascontiguousarray(offsets, np.int64), deep=True),
                  numpy_to_vtkIdTypeArray(np.ascontiguousarray(connectivity, np.int64),
                                          deep=True))
    return cells


def _points(mesh):
    from vtkmodules.util.numpy_support import numpy_to_vtk
    from vtkmodules.vtkCommonCore import vtkPoints

    points = vtkPoints()
    points.SetData(numpy_to_vtk(mesh.points, deep=True))
    return points


def _add_fields(grid, mesh):
    from vtkmodules.util.numpy_support import numpy_to_vtk

    for data, name, values in ((grid.GetPointData(), "p", mesh.p),
                               (grid.GetCellData(), "c", mesh.c)):
        array = numpy_to_vtk(values, deep=True)
        array.SetName(name)
        data.AddArray(array)
    return grid


def vtk_shape_grid(mesh):
    from vtkmodules.vtkCommonDataModel import vtkUnstructuredGrid

    m, k = mesh.cells.shape
    grid = vtkUnstructuredGrid()
    grid.SetPoints(_points(mesh))
    grid.SetCells(mesh.cell_type, _cell_array(np.arange(0, m * k + 1, k), mesh.cells.ravel()))
    return _add_fields(grid, mesh)


def vtk_polyhedral_grid(mesh):
    """Every cell a ``VTK_POLYHEDRON`` with its own outward copy of its faces."""
    from vtkmodules.util.numpy_support import numpy_to_vtk
    from vtkmodules.vtkCommonDataModel import vtkUnstructuredGrid

    m, k = mesh.cells.shape
    faces = np.array(FACES[mesh.cell_type])
    nf, s = faces.shape
    face_points = mesh.cells[:, faces].reshape(-1)              # (m * nf * s,)
    grid = vtkUnstructuredGrid()
    grid.SetPoints(_points(mesh))
    types = numpy_to_vtk(np.full(m, VTK_POLYHEDRON, np.uint8), deep=True)
    grid.SetPolyhedralCells(
        types,
        _cell_array(np.arange(0, m * k + 1, k), mesh.cells.ravel()),
        _cell_array(np.arange(0, m * nf + 1, nf), np.arange(m * nf)),
        _cell_array(np.arange(0, m * nf * s + 1, s), face_points))
    return _add_fields(grid, mesh)


def vtk_subset(grid, point_fields=(), cell_fields=()):
    """A shallow copy of ``grid`` carrying only the named arrays."""
    copy = type(grid)()
    copy.ShallowCopy(grid)
    for data, keep in ((copy.GetPointData(), point_fields), (copy.GetCellData(), cell_fields)):
        for name in [data.GetArrayName(i) for i in range(data.GetNumberOfArrays())]:
            if name not in keep:
                data.RemoveArray(name)
        if keep:
            data.SetActiveScalars(keep[0])
    return copy


# ── Tack inputs ─────────────────────────────────────────────────────

def _scalars(values):
    import tack

    field = tack.field(tack.f32, shape=(values.shape[0],))
    field.from_numpy(np.ascontiguousarray(values, np.float32))
    return field


class TackMesh:
    """A mesh on the device: its topology arrays, geometry and field values, from
    which datasets are made -- once and kept (a pipeline's caches warm), or anew
    for each run (``fresh``: nothing derived beyond what VTK's grid holds)."""

    def __init__(self, mesh, form):
        import tack
        import tack.data as td

        m, k = mesh.cells.shape
        shape = td.UnstructuredTopology(
            np.full(m, mesh.cell_type, np.uint8), np.arange(0, m * k + 1, k, dtype=np.int32),
            mesh.cells.ravel().astype(np.int32), num_points=mesh.num_points)
        self.positions = tack.Vector.field(3, tack.f32, shape=(mesh.num_points,))
        self.positions.from_numpy(np.ascontiguousarray(mesh.points, np.float32))
        self.values = {"p": ("points", _scalars(mesh.p)), "c": ("cells", _scalars(mesh.c))}
        self.form = form
        self.topology = shape
        if form == "polyhedral":
            self.topology = td.as_polyhedra(td.DataSet(shape, self.positions)).topology
        self.topology.groups()

    def dataset(self, fields=(), fresh=False):
        import tack.data as td

        topology = self.topology
        if fresh:
            t = topology
            if self.form == "shape":
                topology = td.UnstructuredTopology(t.types, t.offsets, t.connectivity,
                                                   num_points=t.num_points)
            else:
                topology = td.PolyhedralTopology(t.face_offsets, t.face_points, t.cell_offsets,
                                                 t.cell_faces, t.cell_face_sides,
                                                 num_points=t.num_points)
            # VTK's grid holds the cell types and, for polyhedra, each cell's
            # points; their Tack equivalents are made here, outside the timing.
            topology.groups()
        data = td.DataSet(topology, self.positions)
        for name in fields:
            on, values = self.values[name]
            data.fields[name] = td.Field(td.H1(data) if on == "points" else td.Constant(data),
                                         values)
        return data
