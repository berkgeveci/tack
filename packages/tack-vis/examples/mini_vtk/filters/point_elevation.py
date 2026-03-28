"""Point elevation filter -- compute a scalar from point coordinates."""

import tack
from ..arrays import AOSArray


@tack.kernel
def _elevation_kernel(coords: tack.template(), output, ax, ay, az, n):
    for i in range(n):
        x = coords.get_value(i, 0)
        y = coords.get_value(i, 1)
        z = coords.get_value(i, 2)
        output[i] = ax * x + ay * y + az * z


def point_elevation(dataset, direction=(0.0, 0.0, 1.0), name="elevation"):
    """Add an elevation scalar to point data.

    Computes elevation = dot(coordinates, direction) for each point.
    """
    n = dataset.num_points
    out_field = tack.field(dtype=tack.f32, shape=(n,))
    _elevation_kernel(dataset.coordinates, out_field,
                      direction[0], direction[1], direction[2], n)
    dataset.add_point_array(name, AOSArray(out_field))
    return dataset
