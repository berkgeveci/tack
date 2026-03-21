"""Scene graph: actors, lights, and scene container."""

import numpy as np
import pgc


class Actor:
    """Triangle mesh with uniform color.

    Args:
        points: pgc.field f32, shape (n_verts * 3,) — interleaved xyz.
        connectivity: pgc.field i32, shape (n_tris * 3,) — triangle indices.
        color: RGB tuple in [0, 1], default mid-grey.
    """

    def __init__(self, points, connectivity, color=(0.8, 0.8, 0.8)):
        self.points = points
        self.connectivity = connectivity
        self.color = tuple(float(c) for c in color)
        self.n_verts = points.shape[0] // 3
        self.n_tris = connectivity.shape[0] // 3


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

        Returns dict with pgc fields (points, conn, tri_colors) and numpy
        arrays (points_np, conn_np) for BVH construction.
        """
        all_points = []
        all_conn = []
        all_colors = []
        vert_offset = 0

        for actor in self.actors:
            pts_np = actor.points.to_numpy()
            conn_np = actor.connectivity.to_numpy()
            n_verts = pts_np.shape[0] // 3
            n_tris = conn_np.shape[0] // 3

            all_points.append(pts_np.astype(np.float32))
            all_conn.append(conn_np.astype(np.int32) + vert_offset)

            c = np.array(actor.color, dtype=np.float32)
            all_colors.append(np.tile(c, n_tris))

            vert_offset += n_verts

        points_flat = np.concatenate(all_points)
        conn_flat = np.concatenate(all_conn)
        colors_flat = np.concatenate(all_colors)

        n_tris = conn_flat.shape[0] // 3

        points_field = pgc.field(dtype=pgc.f32, shape=(points_flat.shape[0],))
        points_field.from_numpy(points_flat)

        conn_field = pgc.field(dtype=pgc.i32, shape=(conn_flat.shape[0],))
        conn_field.from_numpy(conn_flat)

        colors_field = pgc.field(dtype=pgc.f32, shape=(colors_flat.shape[0],))
        colors_field.from_numpy(colors_flat)

        return {
            'points': points_field,
            'conn': conn_field,
            'tri_colors': colors_field,
            'points_np': points_flat.reshape(-1, 3),
            'conn_np': conn_flat.reshape(-1, 3),
            'n_tris': n_tris,
        }
