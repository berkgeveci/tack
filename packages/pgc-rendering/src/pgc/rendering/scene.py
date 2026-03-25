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
    face_normals = np.cross(e1, e2)

    np.add.at(normals, conn_np[:, 0], face_normals)
    np.add.at(normals, conn_np[:, 1], face_normals)
    np.add.at(normals, conn_np[:, 2], face_normals)

    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths[lengths == 0] = 1.0
    normals /= lengths

    return normals.astype(np.float32)


# ================================================================
# GPU KERNELS for scene preparation
# ================================================================

@pgc.kernel
def _accumulate_face_normals(points, conn, normals, n_tris):
    """Accumulate area-weighted face normals to vertices using atomic_add."""
    for t in range(n_tris):
        i0 = conn[t * 3]
        i1 = conn[t * 3 + 1]
        i2 = conn[t * 3 + 2]
        e1x = points[i1 * 3]     - points[i0 * 3]
        e1y = points[i1 * 3 + 1] - points[i0 * 3 + 1]
        e1z = points[i1 * 3 + 2] - points[i0 * 3 + 2]
        e2x = points[i2 * 3]     - points[i0 * 3]
        e2y = points[i2 * 3 + 1] - points[i0 * 3 + 1]
        e2z = points[i2 * 3 + 2] - points[i0 * 3 + 2]
        nx = e1y * e2z - e1z * e2y
        ny = e1z * e2x - e1x * e2z
        nz = e1x * e2y - e1y * e2x
        pgc.atomic_add(normals, i0 * 3,     nx)
        pgc.atomic_add(normals, i0 * 3 + 1, ny)
        pgc.atomic_add(normals, i0 * 3 + 2, nz)
        pgc.atomic_add(normals, i1 * 3,     nx)
        pgc.atomic_add(normals, i1 * 3 + 1, ny)
        pgc.atomic_add(normals, i1 * 3 + 2, nz)
        pgc.atomic_add(normals, i2 * 3,     nx)
        pgc.atomic_add(normals, i2 * 3 + 1, ny)
        pgc.atomic_add(normals, i2 * 3 + 2, nz)


@pgc.kernel
def _normalize_vectors(normals, n_verts):
    """Normalize each 3-component vector in-place."""
    for i in range(n_verts):
        nx = normals[i * 3]
        ny = normals[i * 3 + 1]
        nz = normals[i * 3 + 2]
        length = sqrt(nx * nx + ny * ny + nz * nz) + 1.0e-20
        normals[i * 3]     = nx / length
        normals[i * 3 + 1] = ny / length
        normals[i * 3 + 2] = nz / length


@pgc.kernel
def _zero_normals_range(normals, start, count):
    """Zero out normals for a range of vertices (flat-shaded actors)."""
    for i in range(count):
        normals[(start + i) * 3] = 0.0
        normals[(start + i) * 3 + 1] = 0.0
        normals[(start + i) * 3 + 2] = 0.0


def compute_normals_gpu(points, conn, n_verts, n_tris):
    """Compute per-vertex normals entirely on GPU.

    Returns pgc.field f32 (n_verts * 3,) with normalized vertex normals.
    """
    normals = pgc.field(dtype=pgc.f32, shape=(n_verts * 3,))
    # Zero-initialize (field may contain garbage)
    _zero_normals_range(normals, 0, n_verts)
    _accumulate_face_normals(points, conn, normals, n_tris)
    _normalize_vectors(normals, n_verts)
    return normals


@pgc.kernel
def _copy_points(src, dst, dst_offset, n):
    """Copy n floats from src to dst starting at dst_offset."""
    for i in range(n):
        dst[dst_offset + i] = src[i]


@pgc.kernel
def _copy_conn_offset(src, dst, dst_offset, vert_offset, n):
    """Copy connectivity with vertex offset added."""
    for i in range(n):
        dst[dst_offset + i] = src[i] + vert_offset


@pgc.kernel
def _fill_color(tri_colors, offset, cr, cg, cb, n_tris):
    """Fill per-triangle colors for one actor."""
    for i in range(n_tris):
        tri_colors[offset + i * 3] = cr
        tri_colors[offset + i * 3 + 1] = cg
        tri_colors[offset + i * 3 + 2] = cb


@pgc.kernel
def _fill_material_id(mat_ids, offset, mat_id, n_tris):
    """Fill per-triangle material IDs for one actor."""
    for i in range(n_tris):
        mat_ids[offset + i] = mat_id


class Material:
    """Surface material controlling shading behavior.

    Args:
        mat_type: ``Material.MATTE`` (Lambertian diffuse, default),
            ``Material.SPECULAR`` (perfect mirror reflection), or
            ``Material.TRANSPARENT`` (glass with refraction).
        ior: Index of refraction for transparent materials (default 1.5).
    """

    MATTE = 0
    SPECULAR = 1
    TRANSPARENT = 2

    def __init__(self, mat_type=0, ior=1.5):
        self.mat_type = int(mat_type)
        self.ior = float(ior)

    def __eq__(self, other):
        return (isinstance(other, Material) and
                self.mat_type == other.mat_type and
                self.ior == other.ior)

    def __hash__(self):
        return hash((self.mat_type, self.ior))


class Actor:
    """Triangle mesh with per-vertex, scalar-mapped, or uniform color.

    Args:
        points: pgc.field f32, shape (n_verts * 3,) — interleaved xyz.
        connectivity: pgc.field i32, shape (n_tris * 3,) — triangle indices.
        color: RGB tuple in [0, 1], default mid-grey. Used when neither
            point_colors nor scalars is provided.
        point_colors: per-vertex colors. Can be:
            - pgc.field f32, shape (n_verts * 3,) — RGB in [0, 1]
            - numpy uint8 array, shape (n_verts, 3) or (n_verts * 3,) —
              converted to f32 [0, 1] and uploaded.
            - None — uses uniform ``color`` or scalar coloring.
        scalars: per-vertex scalar field for color-table mapping. Can be:
            - pgc.field f32, shape (n_verts,)
            - numpy float32 array, shape (n_verts,)
            - None — no scalar coloring.
            Requires ``color_table`` to be set.  Takes precedence over
            ``color`` but not over ``point_colors``.
        color_table: :class:`~pgc.rendering.ColorTable` instance for
            mapping ``scalars`` to RGB colors.  Required when ``scalars``
            is provided.
        smooth: if True, compute and use vertex normals for smooth shading.
        normals: pre-computed per-vertex normals. Can be:
            - pgc.field f32, shape (n_verts * 3,) — interleaved xyz
            - numpy float32 array, shape (n_verts, 3) or (n_verts * 3,)
            - None — uses ``smooth`` to decide.
            Takes precedence over ``smooth`` when provided.
        material: :class:`Material` instance.  Default is
            ``Material(Material.MATTE)``.
        render_mode: ``"solid"`` (default, path traced), ``"wireframe"``,
            or ``"points"``.
    """

    def __init__(self, points, connectivity, color=(0.8, 0.8, 0.8),
                 point_colors=None, scalars=None, color_table=None,
                 smooth=False, normals=None, material=None,
                 render_mode="solid"):
        self.points = points
        self.connectivity = connectivity
        self.color = tuple(float(c) for c in color)
        self.n_verts = points.shape[0] // 3
        self.n_tris = connectivity.shape[0] // 3
        self.smooth = smooth

        # Handle pre-computed normals
        if normals is None:
            self.normals = None
        elif isinstance(normals, np.ndarray):
            n = normals.reshape(-1).astype(np.float32)
            self.normals = pgc.field(dtype=pgc.f32, shape=(n.shape[0],))
            self.normals.from_numpy(n)
        else:
            self.normals = normals
        self.material = material if material is not None else Material()
        if render_mode not in ("solid", "wireframe", "points"):
            raise ValueError(
                f"render_mode must be 'solid', 'wireframe', or 'points', "
                f"got '{render_mode}'")
        self.render_mode = render_mode

        # Handle point_colors input
        if point_colors is None:
            self.point_colors = None
        elif isinstance(point_colors, np.ndarray):
            # Convert numpy array (uint8 or float) to pgc.field f32
            pc = point_colors.reshape(-1).astype(np.float32)
            if point_colors.dtype == np.uint8:
                pc = pc / 255.0
            self.point_colors = pgc.field(dtype=pgc.f32, shape=(pc.shape[0],))
            self.point_colors.from_numpy(pc)
        else:
            # Assume pgc.field f32
            self.point_colors = point_colors

        # Handle scalar field + color table
        if scalars is not None and color_table is None:
            raise ValueError("scalars requires a color_table")
        self.color_table = color_table
        if scalars is None:
            self.scalars = None
        elif isinstance(scalars, np.ndarray):
            s = scalars.reshape(-1).astype(np.float32)
            self.scalars = pgc.field(dtype=pgc.f32, shape=(s.shape[0],))
            self.scalars.from_numpy(s)
        else:
            # Assume pgc.field f32
            self.scalars = scalars


class PointLight:
    """Point light source.

    Multiple lights can be added to a scene.  Each light contributes
    independently with its own shadow ray during rendering.

    Args:
        position: (x, y, z) world-space position.
        intensity: Scalar brightness.
        color: (r, g, b) light color in [0, 1].  Default white.
    """

    def __init__(self, position, intensity=1.0, color=(1.0, 1.0, 1.0)):
        self.position = tuple(float(v) for v in position)
        self.intensity = float(intensity)
        self.color = tuple(float(c) for c in color)


class DirectionalLight:
    """Directional (infinite distance) light source.

    Args:
        direction: (x, y, z) light direction (points toward the light).
        intensity: Scalar brightness.
        color: (r, g, b) light color in [0, 1].
    """

    def __init__(self, direction, intensity=1.0, color=(1.0, 1.0, 1.0)):
        d = np.array(direction, dtype=np.float64)
        d = d / (np.linalg.norm(d) + 1e-20)
        self.direction = tuple(float(v) for v in d)
        self.intensity = float(intensity)
        self.color = tuple(float(c) for c in color)


class Scene:
    """Container for actors, volumes, and lights."""

    def __init__(self):
        self.actors = []
        self.volumes = []
        self.lights = []
        self._version = 0
        self._cached_geom = None
        self._cached_bvh = None
        self._cache_version = -1

    def add(self, obj):
        from pgc.rendering.volume import Volume
        if isinstance(obj, Actor):
            self.actors.append(obj)
            self._version += 1
        elif isinstance(obj, Volume):
            self.volumes.append(obj)
            self._version += 1
        elif isinstance(obj, (PointLight, DirectionalLight)):
            self.lights.append(obj)
        else:
            raise TypeError(f"Cannot add {type(obj).__name__} to Scene")

    def _prepare(self):
        """Merge all actors into unified geometry arrays.

        Single-actor: zero-copy for points and connectivity.
        Multi-actor: GPU kernels concatenate fields (no host round-trip
        for geometry).  Normals still require host for compute_normals().
        """
        any_has_normals = any(
            a.normals is not None or a.smooth for a in self.actors)

        # --- Single actor: zero copy for geometry ---
        if len(self.actors) == 1:
            actor = self.actors[0]
            n_tris = actor.n_tris

            colors_field = pgc.field(dtype=pgc.f32, shape=(n_tris * 3,))
            _fill_color(colors_field, 0,
                        actor.color[0], actor.color[1], actor.color[2],
                        n_tris)

            has_normals = 0
            normals_field = pgc.field(dtype=pgc.f32, shape=(3,))
            if actor.normals is not None:
                normals_field = actor.normals
                has_normals = 1
            elif actor.smooth:
                normals_field = compute_normals_gpu(
                    actor.points, actor.connectivity,
                    actor.n_verts, n_tris)
                has_normals = 1

            has_point_colors = 0
            pc_field = pgc.field(dtype=pgc.f32, shape=(3,))
            if actor.point_colors is not None:
                pc_field = actor.point_colors
                has_point_colors = 1
            elif actor.scalars is not None:
                pc_field = actor.color_table.map_scalars(
                    actor.scalars, actor.n_verts)
                has_point_colors = 1

            # Material
            mat_ids = pgc.field(dtype=pgc.i32, shape=(n_tris,))
            _fill_material_id(mat_ids, 0, 0, n_tris)
            mat_np = np.array([float(actor.material.mat_type),
                               actor.material.ior, 0.0, 0.0],
                              dtype=np.float32)
            mat_table = pgc.field(dtype=pgc.f32, shape=(4,))
            mat_table.from_numpy(mat_np)

            return {
                'points': actor.points,
                'conn': actor.connectivity,
                'tri_colors': colors_field,
                'point_colors': pc_field,
                'has_point_colors': has_point_colors,
                'normals': normals_field,
                'has_normals': has_normals,
                'mat_ids': mat_ids,
                'mat_table': mat_table,
                'n_tris': n_tris,
            }

        # --- Multi-actor: GPU concat ---
        total_verts = sum(a.n_verts for a in self.actors)
        total_tris = sum(a.n_tris for a in self.actors)

        points_field = pgc.field(dtype=pgc.f32, shape=(total_verts * 3,))
        conn_field = pgc.field(dtype=pgc.i32, shape=(total_tris * 3,))
        colors_field = pgc.field(dtype=pgc.f32, shape=(total_tris * 3,))

        vert_offset = 0
        pts_offset = 0
        conn_offset = 0
        color_offset = 0

        for actor in self.actors:
            n_v = actor.n_verts
            n_t = actor.n_tris

            _copy_points(actor.points, points_field, pts_offset, n_v * 3)
            _copy_conn_offset(actor.connectivity, conn_field,
                              conn_offset, vert_offset, n_t * 3)
            _fill_color(colors_field, color_offset,
                        actor.color[0], actor.color[1], actor.color[2], n_t)

            vert_offset += n_v
            pts_offset += n_v * 3
            conn_offset += n_t * 3
            color_offset += n_t * 3

        # Normals: pre-computed normals take precedence, then smooth, then flat
        has_normals = 0
        normals_field = pgc.field(dtype=pgc.f32, shape=(3,))
        if any_has_normals:
            # Check if all actors provide pre-computed normals
            all_precomputed = all(a.normals is not None for a in self.actors)
            if all_precomputed:
                # Concatenate pre-computed normals
                normals_field = pgc.field(dtype=pgc.f32,
                                          shape=(total_verts * 3,))
                v_off = 0
                for actor in self.actors:
                    _copy_points(actor.normals, normals_field,
                                 v_off * 3, actor.n_verts * 3)
                    v_off += actor.n_verts
            else:
                # Compute normals on GPU, then overwrite with pre-computed
                normals_field = compute_normals_gpu(
                    points_field, conn_field, total_verts, total_tris)
                offset = 0
                for actor in self.actors:
                    n_v = actor.n_verts
                    if actor.normals is not None:
                        _copy_points(actor.normals, normals_field,
                                     offset * 3, n_v * 3)
                    elif not actor.smooth:
                        _zero_normals_range(normals_field, offset, n_v)
                    offset += n_v
            has_normals = 1

        # Per-vertex colors (explicit point_colors or scalar-mapped)
        def _actor_has_vertex_colors(a):
            return a.point_colors is not None or a.scalars is not None

        any_point_colors = any(_actor_has_vertex_colors(a) for a in self.actors)
        has_point_colors = 0
        pc_field = pgc.field(dtype=pgc.f32, shape=(3,))
        if any_point_colors:
            pc_field = pgc.field(dtype=pgc.f32, shape=(total_verts * 3,))
            v_offset = 0
            for actor in self.actors:
                n_v = actor.n_verts
                if actor.point_colors is not None:
                    _copy_points(actor.point_colors, pc_field,
                                 v_offset * 3, n_v * 3)
                elif actor.scalars is not None:
                    mapped = actor.color_table.map_scalars(
                        actor.scalars, n_v)
                    _copy_points(mapped, pc_field, v_offset * 3, n_v * 3)
                else:
                    # Fill with uniform actor color for actors without
                    # per-vertex colors
                    _fill_color(pc_field, v_offset * 3,
                                actor.color[0], actor.color[1],
                                actor.color[2], n_v)
                v_offset += n_v
            has_point_colors = 1

        # Materials: build material table and per-triangle IDs
        mat_ids_field = pgc.field(dtype=pgc.i32, shape=(total_tris,))
        unique_mats = []
        mat_index_map = {}
        tri_offset = 0
        for actor in self.actors:
            m = actor.material
            if m not in mat_index_map:
                mat_index_map[m] = len(unique_mats)
                unique_mats.append(m)
            mid = mat_index_map[m]
            _fill_material_id(mat_ids_field, tri_offset, mid, actor.n_tris)
            tri_offset += actor.n_tris
        n_mats = len(unique_mats)
        mat_np = np.zeros(n_mats * 4, dtype=np.float32)
        for i, m in enumerate(unique_mats):
            mat_np[i * 4] = float(m.mat_type)
            mat_np[i * 4 + 1] = float(m.ior)
        mat_table = pgc.field(dtype=pgc.f32, shape=(n_mats * 4,))
        mat_table.from_numpy(mat_np)

        return {
            'points': points_field,
            'conn': conn_field,
            'tri_colors': colors_field,
            'point_colors': pc_field,
            'has_point_colors': has_point_colors,
            'normals': normals_field,
            'has_normals': has_normals,
            'mat_ids': mat_ids_field,
            'mat_table': mat_table,
            'n_tris': total_tris,
        }
