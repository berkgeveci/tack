"""``tetrahedralize``: every 3D cell as tetrahedra that meet face to face, as VTK's vtkDataSetTriangleFilter makes them (a structured grid's hexahedra in 5, alternating; a hexahedron or voxel of any other topology in 6, a wedge in 3, a pyramid in 2); other cells are left out. The points are the same; cell fields go to each cell's tetrahedra."""

from tack.dataset.algorithms._refine import _split
from tack.runtime.dispatch import on_inputs_device

__all__ = ['tetrahedralize']


@on_inputs_device
def tetrahedralize(data, fields=None):
    """Every 3D cell as tetrahedra that meet face to face: two cells sharing a face
    split it along one diagonal, as VTK's vtkDataSetTriangleFilter does. A
    structured grid's hexahedra take 5 each, the two mirror splits alternating
    with the parity of (i, j, k); any other cell is the cone from its corner of
    smallest point id over the faces without it, each quadrilateral face split
    from its own smallest corner (a hexahedron or voxel in 6, a wedge in 3, a
    pyramid in 2), which fills a convex cell. Other cells are left out. The
    points are the same; cell fields go to each cell's tetrahedra."""
    from tack.dataset.ghosts import visible

    data = visible(data)
    if not getattr(data.topology, "reference_cells", True):
        raise NotImplementedError("tetrahedralize takes cells of the linear shapes")
    return _split(data, "tetrahedra", fields)
