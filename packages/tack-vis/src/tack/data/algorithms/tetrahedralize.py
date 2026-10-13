"""``tetrahedralize``: every 3D cell as tetrahedra, by Viskores' tables (a hexahedron or voxel in 5, a wedge in 3, a pyramid in 2); other cells are left out. The points are the same; cell fields go to each cell's tetrahedra."""

from tack.data.algorithms._refine import _split

__all__ = ['tetrahedralize']


def tetrahedralize(data, fields=None):
    """Every 3D cell as tetrahedra, by Viskores' tables (a hexahedron or voxel in 5,
    a wedge in 3, a pyramid in 2); other cells are left out. The points are the
    same; cell fields go to each cell's tetrahedra."""
    if not getattr(data.topology, "reference_cells", True):
        raise NotImplementedError("tetrahedralize takes cells of the linear shapes")
    return _split(data, "tetrahedra", fields)
