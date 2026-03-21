"""Linear BVH construction (Karras 2012) — GPU-accelerated.

Builds a bounding volume hierarchy from triangle geometry using Morton codes
and the Karras parallel tree construction algorithm.  GPU kernels handle
all stages: centroid/AABB computation, Morton codes, radix sort, tree
construction, and AABB propagation.  The only host round-trip is reading
centroid bounds for Morton code normalization.

Node layout
-----------
Nodes are numbered 0..2*n_tris-2.  Nodes 0..n_inner-1 are inner nodes;
nodes n_inner..2*n_tris-2 are leaves.  Each leaf corresponds to one
triangle: leaf node ``n_inner + k`` maps to ``tri_ids[k]``, the original
triangle index in the merged connectivity array.

Fields for GPU traversal
------------------------
node_aabb : f32 (n_nodes * 6,)
    Per-node AABB: [xmin, ymin, zmin, xmax, ymax, zmax].
node_children : i32 (n_inner * 2,)
    Inner-node children: [left, right] node indices.
tri_ids : i32 (n_tris,)
    Sorted triangle indices (leaf order).
"""

import numpy as np
import pgc


STACK_DEPTH = 24


# ================================================================
# DEVICE HELPERS
# ================================================================

@pgc.func
def _expand_bits(v):
    """Expand 10-bit integer to 30 bits by inserting 2 zeros after each bit."""
    v = (v | (v << 16)) & 50331903
    v = (v | (v << 8))  & 50393103
    v = (v | (v << 4))  & 51146947
    v = (v | (v << 2))  & 153391689
    return v


@pgc.func
def _clz32(x):
    """Count leading zeros of a non-negative 32-bit integer."""
    n = 0
    if x == 0:
        n = 32
    if x != 0:
        if (x >> 16) == 0:
            n = n + 16
            x = x << 16
        if (x >> 24) == 0:
            n = n + 8
            x = x << 8
        if (x >> 28) == 0:
            n = n + 4
            x = x << 4
        if (x >> 30) == 0:
            n = n + 2
            x = x << 2
        if (x >> 31) == 0:
            n = n + 1
    return n


@pgc.func
def _delta(codes, n, i, j):
    """Longest common prefix length between sorted Morton codes[i] and codes[j]."""
    result = -1
    if j >= 0 and j < n:
        xi = codes[i] ^ codes[j]
        if xi == 0:
            result = 32 + _clz32(i ^ j)
        if xi != 0:
            result = _clz32(xi)
    return result


# ================================================================
# GPU KERNELS
# ================================================================

@pgc.kernel
def _compute_centroids_and_aabbs(points, conn, centroids, tri_aabbs, n_tris):
    """Compute triangle centroids and AABBs from vertex data."""
    for t in range(n_tris):
        i0 = conn[t * 3]
        i1 = conn[t * 3 + 1]
        i2 = conn[t * 3 + 2]
        v0x = points[i0 * 3]
        v0y = points[i0 * 3 + 1]
        v0z = points[i0 * 3 + 2]
        v1x = points[i1 * 3]
        v1y = points[i1 * 3 + 1]
        v1z = points[i1 * 3 + 2]
        v2x = points[i2 * 3]
        v2y = points[i2 * 3 + 1]
        v2z = points[i2 * 3 + 2]

        centroids[t * 3]     = (v0x + v1x + v2x) / 3.0
        centroids[t * 3 + 1] = (v0y + v1y + v2y) / 3.0
        centroids[t * 3 + 2] = (v0z + v1z + v2z) / 3.0

        eps = 0.000001
        tri_aabbs[t * 6]     = min(v0x, min(v1x, v2x)) - eps
        tri_aabbs[t * 6 + 1] = min(v0y, min(v1y, v2y)) - eps
        tri_aabbs[t * 6 + 2] = min(v0z, min(v1z, v2z)) - eps
        tri_aabbs[t * 6 + 3] = max(v0x, max(v1x, v2x)) + eps
        tri_aabbs[t * 6 + 4] = max(v0y, max(v1y, v2y)) + eps
        tri_aabbs[t * 6 + 5] = max(v0z, max(v1z, v2z)) + eps


@pgc.kernel
def _compute_morton_codes(centroids, codes,
                          scene_min_x, scene_min_y, scene_min_z,
                          inv_ext_x, inv_ext_y, inv_ext_z, n_tris):
    """Compute 30-bit Morton codes from normalized centroids."""
    for t in range(n_tris):
        nx = (centroids[t * 3]     - scene_min_x) * inv_ext_x
        ny = (centroids[t * 3 + 1] - scene_min_y) * inv_ext_y
        nz = (centroids[t * 3 + 2] - scene_min_z) * inv_ext_z
        ix = int(nx * 1024.0)
        iy = int(ny * 1024.0)
        iz = int(nz * 1024.0)
        if ix < 0: ix = 0
        if ix > 1023: ix = 1023
        if iy < 0: iy = 0
        if iy > 1023: iy = 1023
        if iz < 0: iz = 0
        if iz > 1023: iz = 1023
        codes[t] = (_expand_bits(iz) << 2) | (_expand_bits(iy) << 1) | _expand_bits(ix)


# ================================================================
# GPU BITONIC SORT
# ================================================================

@pgc.kernel
def _bitonic_step(keys, vals, j, k, n):
    """One step of bitonic sort: compare-and-swap pairs at distance j."""
    for i in range(n):
        l = i ^ j
        if l > i:
            swap = 0
            ki = keys[i]
            kl = keys[l]
            if (i & k) == 0:
                if ki > kl:
                    swap = 1
            if (i & k) != 0:
                if ki < kl:
                    swap = 1
            if swap == 1:
                keys[i] = kl
                keys[l] = ki
                vi = vals[i]
                vals[i] = vals[l]
                vals[l] = vi


@pgc.kernel
def _copy_and_pad(src_keys, src_vals, dst_keys, dst_vals,
                  n_real, pad_val, n_padded):
    """Copy real data and pad with max values."""
    for i in range(n_padded):
        if i < n_real:
            dst_keys[i] = src_keys[i]
            dst_vals[i] = src_vals[i]
        if i >= n_real:
            dst_keys[i] = pad_val
            dst_vals[i] = 0


def _gpu_sort(codes, tri_ids, n):
    """Sort codes and tri_ids using GPU bitonic sort.

    Pads to next power of 2, sorts in-place, returns padded fields
    (first n elements are the sorted result).
    ~210 kernel launches for 1M elements.
    """
    n_padded = 1
    while n_padded < n:
        n_padded *= 2

    keys = pgc.field(dtype=pgc.i32, shape=(n_padded,))
    vals = pgc.field(dtype=pgc.i32, shape=(n_padded,))

    pad_val = 1073741823  # 0x3FFFFFFF — max 30-bit value
    _copy_and_pad(codes, tri_ids, keys, vals, n, pad_val, n_padded)

    # Bitonic sort: O(log^2(n)) steps
    k = 2
    while k <= n_padded:
        j = k // 2
        while j >= 1:
            _bitonic_step(keys, vals, j, k, n_padded)
            j = j // 2
        k = k * 2

    return keys, vals


@pgc.kernel
def _reduce_bounds(centroids, out_min, out_max, n_tris):
    """Compute min/max of centroids using atomic_min/atomic_max.

    out_min and out_max are single-element fields (3 floats each),
    pre-initialized to +inf/-inf.  Each thread updates them atomically.

    Since pgc.atomic_min/max works on i32, we encode floats as sortable
    ints (works for positive floats).
    """
    for i in range(n_tris):
        cx = centroids[i * 3]
        cy = centroids[i * 3 + 1]
        cz = centroids[i * 3 + 2]
        pgc.atomic_min(out_min, 0, cx)
        pgc.atomic_min(out_min, 1, cy)
        pgc.atomic_min(out_min, 2, cz)
        pgc.atomic_max(out_max, 0, cx)
        pgc.atomic_max(out_max, 1, cy)
        pgc.atomic_max(out_max, 2, cz)


@pgc.kernel
def _iota(field, n):
    """Fill field with 0, 1, 2, ..., n-1."""
    for i in range(n):
        field[i] = i


@pgc.kernel
def _init_ready(ready, n_inner, n_nodes):
    """Initialize ready flags: 0 for inner nodes, 1 for leaves."""
    for i in range(n_nodes):
        if i < n_inner:
            ready[i] = 0
        if i >= n_inner:
            ready[i] = 1


@pgc.kernel
def _fill_i32(field, val, n):
    """Fill i32 field with a constant value."""
    for i in range(n):
        field[i] = val


@pgc.kernel
def _reorder_leaf_aabbs(tri_aabbs, node_aabb, sorted_ids, n_inner, n_tris):
    """Copy triangle AABBs into leaf node positions in sorted order."""
    for k in range(n_tris):
        src = sorted_ids[k]
        dst = n_inner + k
        node_aabb[dst * 6]     = tri_aabbs[src * 6]
        node_aabb[dst * 6 + 1] = tri_aabbs[src * 6 + 1]
        node_aabb[dst * 6 + 2] = tri_aabbs[src * 6 + 2]
        node_aabb[dst * 6 + 3] = tri_aabbs[src * 6 + 3]
        node_aabb[dst * 6 + 4] = tri_aabbs[src * 6 + 4]
        node_aabb[dst * 6 + 5] = tri_aabbs[src * 6 + 5]


@pgc.kernel
def _build_karras(codes, node_children, n_inner, n_tris):
    """Karras 2012 parallel tree construction.

    Each inner node i independently determines its children by examining
    the common prefix structure of sorted Morton codes.
    """
    for i in range(n_inner):
        # Determine direction of the range
        d_right = _delta(codes, n_tris, i, i + 1)
        d_left = _delta(codes, n_tris, i, i - 1)
        d = 1
        if d_right <= d_left:
            d = -1

        # Upper bound for range length
        delta_min = _delta(codes, n_tris, i, i - d)
        l_max = 2
        d_test = _delta(codes, n_tris, i, i + l_max * d)
        while d_test > delta_min:
            l_max = l_max * 2
            d_test = _delta(codes, n_tris, i, i + l_max * d)

        # Binary search for actual range end
        l = 0
        step = l_max >> 1
        while step > 0:
            rj = i + (l + step) * d
            if rj >= 0 and rj < n_tris:
                d_test = _delta(codes, n_tris, i, rj)
                if d_test > delta_min:
                    l = l + step
            step = step >> 1
        j = i + l * d

        # Find split position
        delta_node = _delta(codes, n_tris, i, j)
        s = 0
        step = 1
        while step <= l:
            step = step * 2
        step = step >> 1
        while step > 0:
            if s + step <= l:
                sj = i + (s + step) * d
                if sj >= 0 and sj < n_tris:
                    ds_test = _delta(codes, n_tris, i, sj)
                    if ds_test > delta_node:
                        s = s + step
            step = step >> 1

        gamma = i + s * d
        if d < 0:
            gamma = gamma - 1

        # Assign children
        left_idx = gamma
        if min(i, j) == gamma:
            left_idx = gamma + n_inner
        right_idx = gamma + 1
        if max(i, j) == gamma + 1:
            right_idx = gamma + 1 + n_inner

        node_children[i * 2] = left_idx
        node_children[i * 2 + 1] = right_idx


@pgc.kernel
def _compute_parents(node_children, parent, n_inner):
    """Build parent array from children array."""
    for i in range(n_inner):
        left = node_children[i * 2]
        right = node_children[i * 2 + 1]
        parent[left] = i
        parent[right] = i


@pgc.kernel
def _propagate_pass(node_aabb, node_children, ready, n_inner, remaining):
    """One pass of bottom-up AABB propagation.

    Processes inner nodes whose both children are ready.  Sets ready[node]=1
    and decrements remaining[0] for each newly completed node.
    """
    for i in range(n_inner):
        if ready[i] == 0:
            l = node_children[i * 2]
            r = node_children[i * 2 + 1]
            if ready[l] == 1 and ready[r] == 1:
                lx0 = node_aabb[l * 6]
                ly0 = node_aabb[l * 6 + 1]
                lz0 = node_aabb[l * 6 + 2]
                lx1 = node_aabb[l * 6 + 3]
                ly1 = node_aabb[l * 6 + 4]
                lz1 = node_aabb[l * 6 + 5]
                rx0 = node_aabb[r * 6]
                ry0 = node_aabb[r * 6 + 1]
                rz0 = node_aabb[r * 6 + 2]
                rx1 = node_aabb[r * 6 + 3]
                ry1 = node_aabb[r * 6 + 4]
                rz1 = node_aabb[r * 6 + 5]
                node_aabb[i * 6]     = min(lx0, rx0)
                node_aabb[i * 6 + 1] = min(ly0, ry0)
                node_aabb[i * 6 + 2] = min(lz0, rz0)
                node_aabb[i * 6 + 3] = max(lx1, rx1)
                node_aabb[i * 6 + 4] = max(ly1, ry1)
                node_aabb[i * 6 + 5] = max(lz1, rz1)
                ready[i] = 1
                pgc.atomic_add(remaining, 0, -1)


# ================================================================
# PUBLIC API
# ================================================================

class BVH:
    """Linear BVH for GPU ray traversal.

    After ``build()``, the following pgc fields are available:

    - ``node_aabb``:  f32 (n_nodes * 6,)
    - ``node_children``: i32 (n_inner * 2,)
    - ``tri_ids``: i32 (n_tris,)
    """

    def __init__(self):
        self.node_aabb = None
        self.node_children = None
        self.tri_ids = None
        self.n_inner = 0
        self.n_tris = 0

    def build(self, points, conn, n_tris, gpu_sort=True):
        """Build BVH from pgc fields.

        Args:
            points: pgc.field f32, shape (n_verts * 3,) — interleaved xyz.
            conn: pgc.field i32, shape (n_tris * 3,) — triangle indices.
            n_tris: number of triangles.
            gpu_sort: if True, use GPU bitonic sort; if False, use numpy.
        """
        self.n_tris = n_tris

        if n_tris == 0:
            return

        n_inner = n_tris - 1
        self.n_inner = n_inner
        n_nodes = 2 * n_tris - 1

        import time as _time

        # --- Step 1: Compute centroids and triangle AABBs (GPU) ---
        _t0 = _time.perf_counter()
        centroids = pgc.field(dtype=pgc.f32, shape=(n_tris * 3,))
        tri_aabbs = pgc.field(dtype=pgc.f32, shape=(n_tris * 6,))
        _compute_centroids_and_aabbs(points, conn, centroids, tri_aabbs, n_tris)
        _t1 = _time.perf_counter()

        # --- Step 2: Morton codes (GPU) + sort (GPU) ---
        # GPU reduction for centroid bounds (only 6 floats come to host)
        out_min = pgc.field(dtype=pgc.f32, shape=(3,))
        out_max = pgc.field(dtype=pgc.f32, shape=(3,))
        out_min.from_numpy(np.array([1e30, 1e30, 1e30], dtype=np.float32))
        out_max.from_numpy(np.array([-1e30, -1e30, -1e30], dtype=np.float32))
        _reduce_bounds(centroids, out_min, out_max, n_tris)
        scene_min = out_min.to_numpy()
        scene_max = out_max.to_numpy()
        extent = scene_max - scene_min
        extent[extent == 0] = 1.0
        inv_ext = 1.0 / extent

        codes = pgc.field(dtype=pgc.i32, shape=(n_tris,))
        _compute_morton_codes(centroids, codes,
                              float(scene_min[0]), float(scene_min[1]),
                              float(scene_min[2]),
                              float(inv_ext[0]), float(inv_ext[1]),
                              float(inv_ext[2]), n_tris)

        # Initialize triangle IDs on GPU
        tri_ids = pgc.field(dtype=pgc.i32, shape=(n_tris,))
        _iota(tri_ids, n_tris)

        # Sort by Morton code
        if gpu_sort:
            sorted_codes, tri_ids = _gpu_sort(codes, tri_ids, n_tris)
        else:
            codes_np = codes.to_numpy().astype(np.uint32)
            sorted_ids = np.argsort(codes_np).astype(np.int32)
            tri_ids = pgc.field(dtype=pgc.i32, shape=(n_tris,))
            tri_ids.from_numpy(sorted_ids)
            sorted_codes = pgc.field(dtype=pgc.i32, shape=(n_tris,))
            sorted_codes.from_numpy(codes_np[sorted_ids].astype(np.int32))
        self.tri_ids = tri_ids
        _t2 = _time.perf_counter()

        if n_tris == 1:
            self.n_inner = 0
            self.node_aabb = pgc.field(dtype=pgc.f32, shape=(6,))
            aabb_np = tri_aabbs.to_numpy()[:6]
            self.node_aabb.from_numpy(aabb_np)
            self.node_children = pgc.field(dtype=pgc.i32, shape=(2,))
            self.node_children.from_numpy(np.zeros(2, dtype=np.int32))
            return

        # --- Step 3: Build Karras tree (GPU) ---
        self.node_children = pgc.field(dtype=pgc.i32, shape=(n_inner * 2,))
        _build_karras(sorted_codes, self.node_children, n_inner, n_tris)
        _t3 = _time.perf_counter()

        # --- Step 4: Reorder leaf AABBs (GPU) ---
        self.node_aabb = pgc.field(dtype=pgc.f32, shape=(n_nodes * 6,))
        _reorder_leaf_aabbs(tri_aabbs, self.node_aabb, tri_ids, n_inner, n_tris)
        _t4 = _time.perf_counter()

        # --- Step 5: Propagate AABBs bottom-up (GPU, iterative) ---
        ready = pgc.field(dtype=pgc.i32, shape=(n_nodes,))
        _init_ready(ready, n_inner, n_nodes)

        remaining = pgc.field(dtype=pgc.i32, shape=(1,))
        remaining.from_numpy(np.array([n_inner], dtype=np.int32))

        # Iterate until all inner nodes are done
        while int(remaining.to_numpy()[0]) > 0:
            _propagate_pass(self.node_aabb, self.node_children, ready,
                            n_inner, remaining)
        _t5 = _time.perf_counter()

        print(f"    [bvh] centroids={_t1-_t0:.3f}s  morton+radix={_t2-_t1:.3f}s  "
              f"karras={_t3-_t2:.3f}s  reorder={_t4-_t3:.3f}s  propagate={_t5-_t4:.3f}s")
