"""Scene graph: actors, lights, and scene container."""

import numpy as np
import pgc


def compute_normals(points_np, conn_np):
    """Compute per-vertex normals by averaging adjacent face normals.

    Args:
        points_np: (n_verts, 3) float32 vertex positions.
        conn_np: (n_tris, 3) int32 triangle vertex indices.

    Returns:
        (n_verts, 3) float32 normalized vertex normals.
    """
    n_verts = points_np.shape[0]
    normals = np.zeros((n_verts, 3), dtype=np.float64)

    v0 = points_np[conn_np[:, 0]]
    v1 = points_np[conn_np[:, 1]]
    v2 = points_np[conn_np[:, 2]]
    e1 = v1 - v0
    e2 = v2 - v0
    face_normals = np.cross(e1, e2)  # (n_tris, 3), not normalized (area-weighted)

    # Accumulate face normals to each vertex
    np.add.at(normals, conn_np[:, 0], face_normals)
    np.add.at(normals, conn_np[:, 1], face_normals)
    np.add.at(normals, conn_np[:, 2], face_normals)

    # Normalize
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths[lengths == 0] = 1.0
    normals /= lengths

    return normals.astype(np.float32)


class Actor:
    """Triangle mesh with uniform color.

    Args:
        points: pgc.field f32, shape (n_verts * 3,) — interleaved xyz.
        connectivity: pgc.field i32, shape (n_tris * 3,) — triangle indices.
        color: RGB tuple in [0, 1], default mid-grey.
        smooth: if True, compute and use vertex normals for smooth shading.
    """

    def __init__(self, points, connectivity, color=(0.8, 0.8, 0.8),
                 smooth=False):
        self.points = points
        self.connectivity = connectivity
        self.color = tuple(float(c) for c in color)
        self.n_verts = points.shape[0] // 3
        self.n_tris = connectivity.shape[0] // 3
        self.smooth = smooth


class PointLight:
    """Point light source.

    Args:
        position: (x, y, z) world position.
        intensity: scalar brightness, default 1.0.
        color: RGB tuple in [0, 1], default white.
    """

    def __init__(self, position, intensity=1.0, color=(1.0, 1.0, 1.0)):
        self.position = tuple(float(v) for v in position)
        self.intensity = float(intensity)
        self.color = tuple(float(c) for c in color)


class Scene:
    """Container for actors and lights."""

    def __init__(self):
        self.actors = []
        self.lights = []

    def add(self, obj):
        if isinstance(obj, Actor):
            self.actors.append(obj)
        elif isinstance(obj, PointLight):
            self.lights.append(obj)
        else:
            raise TypeError(f"Cannot add {type(obj).__name__} to Scene")

    def _prepare(self):
        """Merge all actors into unified geometry arrays.

        Returns dict with pgc fields (points, conn, tri_colors, normals)
        and metadata for BVH construction.
        """
        all_points = []
        all_conn = []
        all_colors = []
        vert_offset = 0
        any_smooth = False

        for actor in self.actors:
            pts_np = actor.points.to_numpy()
            conn_np = actor.connectivity.to_numpy()
            n_verts = pts_np.shape[0] // 3
            n_tris = conn_np.shape[0] // 3

            all_points.append(pts_np.astype(np.float32))
            all_conn.append(conn_np.astype(np.int32) + vert_offset)

            c = np.array(actor.color, dtype=np.float32)
            all_colors.append(np.tile(c, n_tris))

            if actor.smooth:
                any_smooth = True

            vert_offset += n_verts

        points_flat = np.concatenate(all_points)
        conn_flat = np.concatenate(all_conn)
        colors_flat = np.concatenate(all_colors)

        n_tris = conn_flat.shape[0] // 3
        total_verts = points_flat.shape[0] // 3

        points_field = pgc.field(dtype=pgc.f32, shape=(points_flat.shape[0],))
        points_field.from_numpy(points_flat)

        conn_field = pgc.field(dtype=pgc.i32, shape=(conn_flat.shape[0],))
        conn_field.from_numpy(conn_flat)

        colors_field = pgc.field(dtype=pgc.f32, shape=(colors_flat.shape[0],))
        colors_field.from_numpy(colors_flat)

        # Compute vertex normals if any actor requests smooth shading
        has_normals = 0
        if any_smooth:
            normals_np = compute_normals(points_flat.reshape(-1, 3),
                                         conn_flat.reshape(-1, 3))
            # Zero out normals for flat-shaded actors' vertices
            offset = 0
            for actor in self.actors:
                n_v = actor.points.shape[0] // 3
                if not actor.smooth:
                    normals_np[offset:offset + n_v] = 0.0
                offset += n_v
            normals_field = pgc.field(dtype=pgc.f32,
                                      shape=(normals_np.size,))
            normals_field.from_numpy(normals_np.reshape(-1))
            has_normals = 1
        else:
            # Dummy 1-element field (kernel still needs a field parameter)
            normals_field = pgc.field(dtype=pgc.f32, shape=(3,))

        return {
            'points': points_field,
            'conn': conn_field,
            'tri_colors': colors_field,
            'normals': normals_field,
            'has_normals': has_normals,
            'n_tris': n_tris,
        }
