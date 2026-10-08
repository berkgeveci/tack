"""Point coordinates other than one position per point.

A ``DataSet``'s points are a field of 3-vectors, or a coordinates object:

- ``RectilinearCoordinates(x, y, z)``: the points of a grid whose lines
  run along the axes, as ``vtkRectilinearGrid`` stores them. Point
  ``(i, j, k)`` is at ``(x[i], y[j], z[k])``, and the flat point id runs
  x fastest: ``i + nx * (j + ny * k)``. Only the three coordinate arrays
  are stored.

Cell views read positions through ``point(c, j)`` whatever the
coordinates are. Structured cells over rectilinear coordinates find a
point's ``(i, j, k)`` from the cell's own indices, with no division;
other cells recover it from the flat point id.
"""

import numpy as np

import tack
from tack.lang.field import Field

__all__ = ["RectilinearCoordinates"]


def _axis(values, dtype):
    if isinstance(values, Field):
        if values.dtype != dtype or len(values.shape) != 1:
            raise TypeError(f"a coordinate array must be a one-dimensional {dtype.name} field")
        return values
    values = np.ascontiguousarray(values, dtype=dtype.numpy_dtype).reshape(-1)
    if not values.size:
        raise ValueError("a coordinate array needs at least one value")
    field = tack.field(dtype, shape=values.shape)
    field.from_numpy(values)
    return field


class RectilinearCoordinates:
    """The points of a grid whose lines run along the axes.

    ``x``, ``y`` and ``z`` are the coordinates along each axis, as NumPy
    arrays (copied to fields of ``dtype``) or one-dimensional fields. A
    two-dimensional grid has one ``z`` value, a one-dimensional grid one
    ``y`` and one ``z``; they default to ``[0]``.
    """

    def __init__(self, x, y=(0.0,), z=(0.0,), dtype=tack.f32):
        self.dtype = dtype
        self.x = _axis(x, dtype)
        self.y = _axis(y, dtype)
        self.z = _axis(z, dtype)
        self.dims = (self.x.shape[0], self.y.shape[0], self.z.shape[0])

    @property
    def num_points(self):
        return int(np.prod(self.dims))

    @property
    def point_dims(self):
        """The grid's dimensions without the trailing ones, as a ``StructuredCellSet`` takes them."""
        dims = list(self.dims)
        while len(dims) > 1 and dims[-1] == 1:
            dims.pop()
        return tuple(dims)

    def to_numpy(self, vectors=True):
        """Every point's position, ``(num_points, 3)``, x fastest."""
        z, y, x = np.meshgrid(self.z.to_numpy(), self.y.to_numpy(), self.x.to_numpy(),
                              indexing="ij")
        points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        return points if vectors else points.reshape(-1)
