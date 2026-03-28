"""GPU vertex normal computation for triangle meshes.

Computes smooth vertex normals by accumulating face normals via atomics,
then normalizing. All work stays on GPU.

Public API
----------
compute_normals(points, conn, n_pts, n_tris)
    Returns tack.field of interleaved normals (n_pts*3,) f32.
"""

import tack


@tack.kernel
def _accumulate_face_normals(points, conn, normals, n_tris):
    """For each triangle, compute face normal and scatter-add to vertices."""
    for tri in range(n_tris):
        i0 = conn[tri * 3]
        i1 = conn[tri * 3 + 1]
        i2 = conn[tri * 3 + 2]

        # Load vertex positions
        v0x = points[i0 * 3]
        v0y = points[i0 * 3 + 1]
        v0z = points[i0 * 3 + 2]
        v1x = points[i1 * 3]
        v1y = points[i1 * 3 + 1]
        v1z = points[i1 * 3 + 2]
        v2x = points[i2 * 3]
        v2y = points[i2 * 3 + 1]
        v2z = points[i2 * 3 + 2]

        # Edge vectors
        e1x = v1x - v0x
        e1y = v1y - v0y
        e1z = v1z - v0z
        e2x = v2x - v0x
        e2y = v2y - v0y
        e2z = v2z - v0z

        # Cross product (face normal, area-weighted)
        nx = e1y * e2z - e1z * e2y
        ny = e1z * e2x - e1x * e2z
        nz = e1x * e2y - e1y * e2x

        # Scatter-add to all 3 vertices
        tack.atomic_add(normals, i0 * 3, nx)
        tack.atomic_add(normals, i0 * 3 + 1, ny)
        tack.atomic_add(normals, i0 * 3 + 2, nz)
        tack.atomic_add(normals, i1 * 3, nx)
        tack.atomic_add(normals, i1 * 3 + 1, ny)
        tack.atomic_add(normals, i1 * 3 + 2, nz)
        tack.atomic_add(normals, i2 * 3, nx)
        tack.atomic_add(normals, i2 * 3 + 1, ny)
        tack.atomic_add(normals, i2 * 3 + 2, nz)


@tack.kernel
def _normalize(normals, n_pts):
    """Normalize each vertex normal in-place."""
    for i in range(n_pts):
        nx = normals[i * 3]
        ny = normals[i * 3 + 1]
        nz = normals[i * 3 + 2]
        length = (nx * nx + ny * ny + nz * nz) ** 0.5
        if length > 1.0e-10:
            inv = 1.0 / length
            normals[i * 3] = nx * inv
            normals[i * 3 + 1] = ny * inv
            normals[i * 3 + 2] = nz * inv


def compute_normals(points, conn, n_pts, n_tris):
    """Compute smooth vertex normals on GPU.

    Args:
        points: tack.field, interleaved (n_pts*3,) f32
        conn: tack.field, interleaved (n_tris*3,) i32
        n_pts: number of vertices
        n_tris: number of triangles

    Returns:
        tack.field, interleaved normals (n_pts*3,) f32
    """
    normals = tack.field(dtype=tack.f32, shape=(n_pts * 3,))
    normals.fill(0)
    _accumulate_face_normals(points, conn, normals, n_tris)
    _normalize(normals, n_pts)
    return normals
