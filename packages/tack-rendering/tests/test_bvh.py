"""The BVH, checked against what it is supposed to be.

`bvh.py` had no test of its own. It was exercised only through rendering,
and that is the worst way to test an acceleration structure: a subtly
wrong BVH still renders a plausible image. Miss a triangle in a corner of
the tree and you get a slightly wrong picture that no pixel assertion is
tuned to notice.

So these check the structure directly — every node's box contains its
children, every triangle is referenced exactly once, the root encloses the
whole mesh — and then check traversal the only way that really settles it:
against brute force over every triangle.

Two builds exist, GPU bitonic sort and a numpy path. Both are tested, and
tested against each other.
"""

import numpy as np
import pytest

import tack
from tack.rendering.bvh import BVH


def _upload(points_np, conn_np):
    points = tack.field(dtype=tack.f32, shape=(points_np.size,))
    points.from_numpy(points_np.ravel().astype(np.float32))
    conn = tack.field(dtype=tack.i32, shape=(conn_np.size,))
    conn.from_numpy(conn_np.ravel().astype(np.int32))
    return points, conn


def _random_mesh(n_tris, seed=0, scale=1.0):
    """Independent triangles — no shared vertices, so indices stay simple."""
    rng = np.random.default_rng(seed)
    points = rng.random((n_tris * 3, 3)).astype(np.float32) * scale
    conn = np.arange(n_tris * 3, dtype=np.int32).reshape(n_tris, 3)
    return points, conn


def _build(points_np, conn_np, gpu_sort=True):
    points, conn = _upload(points_np, conn_np)
    bvh = BVH()
    bvh.build(points, conn, len(conn_np), gpu_sort=gpu_sort)
    return bvh


def _tri_order(bvh):
    """The triangle permutation — tri_ids is padded, so slice it."""
    return bvh.tri_ids.to_numpy()[:bvh.n_tris]


def _boxes(bvh):
    """node_aabb as (n_nodes, 6) → (min_xyz, max_xyz) per node."""
    a = bvh.node_aabb.to_numpy().reshape(-1, 6)
    return a[:, :3], a[:, 3:]


# ── Structure ────────────────────────────────────────────────────────

def test_every_triangle_appears_exactly_once(backend):
    """A permutation, not a selection. Dropping one loses geometry silently."""
    n_tris = 64
    bvh = _build(*_random_mesh(n_tris))
    ids = np.sort(_tri_order(bvh))
    np.testing.assert_array_equal(ids, np.arange(n_tris))


def test_root_box_encloses_the_whole_mesh(backend):
    n_tris = 32
    points, conn = _random_mesh(n_tris, seed=3)
    bvh = _build(points, conn)
    lo, hi = _boxes(bvh)

    # Root is node 0 of the inner nodes.
    assert (lo[0] <= points.min(axis=0) + 1e-5).all()
    assert (hi[0] >= points.max(axis=0) - 1e-5).all()


def test_parent_boxes_contain_their_children(backend):
    """The invariant traversal depends on: skipping a node skips its subtree."""
    bvh = _build(*_random_mesh(64, seed=1))
    lo, hi = _boxes(bvh)
    children = bvh.node_children.to_numpy().reshape(-1, 2)

    for parent in range(bvh.n_inner):
        for child in children[parent]:
            # Leaves are encoded as indices at or past the inner-node count.
            idx = int(child)
            assert 0 <= idx < len(lo), f"child index {idx} out of range"
            assert (lo[parent] <= lo[idx] + 1e-4).all(), \
                f"node {parent} does not contain child {idx}"
            assert (hi[parent] >= hi[idx] - 1e-4).all(), \
                f"node {parent} does not contain child {idx}"


def test_leaf_boxes_contain_their_triangles(backend):
    """The bottom of the tree has to be right too, not just the shape of it."""
    n_tris = 48
    points, conn = _random_mesh(n_tris, seed=5)
    bvh = _build(points, conn)
    lo, hi = _boxes(bvh)
    order = _tri_order(bvh)

    for slot, tri in enumerate(order):
        node = bvh.n_inner + slot
        verts = points[conn[tri]]
        assert (lo[node] <= verts.min(axis=0) + 1e-4).all()
        assert (hi[node] >= verts.max(axis=0) - 1e-4).all()


def test_node_count_is_the_binary_tree_identity(backend):
    """n leaves ⇒ n-1 inner nodes. Anything else means a malformed tree."""
    for n_tris in (2, 3, 8, 33, 64):
        bvh = _build(*_random_mesh(n_tris, seed=n_tris))
        assert bvh.n_inner == n_tris - 1, f"n_tris={n_tris}"


def test_child_indices_are_in_range(backend):
    bvh = _build(*_random_mesh(40, seed=7))
    children = bvh.node_children.to_numpy().reshape(-1, 2)[:bvh.n_inner]
    n_nodes = bvh.n_inner + bvh.n_tris
    assert children.min() >= 0
    assert children.max() < n_nodes


def test_every_node_is_reachable_from_the_root(backend):
    """An orphaned subtree is geometry the tracer can never find."""
    bvh = _build(*_random_mesh(64, seed=11))
    children = bvh.node_children.to_numpy().reshape(-1, 2)

    seen, stack = set(), [0]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        if node < bvh.n_inner:
            stack.extend(int(c) for c in children[node])

    assert len(seen) == bvh.n_inner + bvh.n_tris


# ── Traversal, against brute force ───────────────────────────────────

def _brute_force_hit(origin, direction, points, conn):
    """Closest Möller–Trumbore hit over every triangle. The reference."""
    best_t, best_tri = np.inf, -1
    for i, tri in enumerate(conn):
        v0, v1, v2 = points[tri].astype(np.float64)
        e1, e2 = v1 - v0, v2 - v0
        pvec = np.cross(direction, e2)
        det = e1 @ pvec
        if abs(det) < 1e-12:
            continue
        inv = 1.0 / det
        tvec = origin - v0
        u = (tvec @ pvec) * inv
        if u < 0.0 or u > 1.0:
            continue
        qvec = np.cross(tvec, e1)
        v = (direction @ qvec) * inv
        if v < 0.0 or u + v > 1.0:
            continue
        t = (e2 @ qvec) * inv
        if 1e-6 < t < best_t:
            best_t, best_tri = t, i
    return best_t, best_tri


def _boxes_hit_by(origin, direction, lo, hi):
    """Which node boxes a ray crosses — slab test, the traversal predicate."""
    inv = 1.0 / np.where(np.abs(direction) < 1e-12, 1e-12, direction)
    t1 = (lo - origin) * inv
    t2 = (hi - origin) * inv
    tmin = np.minimum(t1, t2).max(axis=1)
    tmax = np.maximum(t1, t2).min(axis=1)
    return (tmax >= np.maximum(tmin, 0.0))


def test_the_tree_never_hides_a_hit(backend):
    """The claim an acceleration structure has to make.

    For every ray that brute force says hits a triangle, the boxes along
    the path from the root down to that triangle's leaf must all be hit —
    otherwise traversal prunes the subtree and the hit is lost.
    """
    n_tris = 40
    points, conn = _random_mesh(n_tris, seed=13)
    bvh = _build(points, conn)
    lo, hi = _boxes(bvh)
    children = bvh.node_children.to_numpy().reshape(-1, 2)
    order = list(_tri_order(bvh))

    parent = {}
    for p in range(bvh.n_inner):
        for c in children[p]:
            parent[int(c)] = p

    rng = np.random.default_rng(99)
    checked = 0
    for _ in range(60):
        origin = rng.random(3).astype(np.float64) * 2.0 - 0.5
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)

        t, tri = _brute_force_hit(origin, direction, points, conn)
        if tri < 0:
            continue
        checked += 1

        node = bvh.n_inner + order.index(tri)
        crossed = _boxes_hit_by(origin, direction, lo, hi)
        while True:
            assert crossed[node], (
                f"ray misses the box of node {node} but hits triangle {tri} — "
                f"traversal would prune the subtree containing the hit")
            if node == 0:
                break
            node = parent[node]

    assert checked >= 5, f"only {checked} rays hit anything; test is vacuous"


# ── The two sort paths agree ─────────────────────────────────────────

def test_gpu_and_numpy_sort_build_the_same_tree(backend):
    """Two implementations of the same step must not diverge."""
    points, conn = _random_mesh(48, seed=17)
    gpu = _build(points, conn, gpu_sort=True)
    cpu = _build(points, conn, gpu_sort=False)

    np.testing.assert_array_equal(_tri_order(gpu), _tri_order(cpu))
    np.testing.assert_allclose(gpu.node_aabb.to_numpy(),
                               cpu.node_aabb.to_numpy(), rtol=1e-5)
    np.testing.assert_array_equal(gpu.node_children.to_numpy(),
                                  cpu.node_children.to_numpy())


@pytest.mark.parametrize("n_tris", [2, 3, 5, 16, 17, 64, 65])
def test_sizes_that_are_not_powers_of_two(backend, n_tris):
    """The bitonic sort pads to a power of two.

    tri_ids is the padded field, so the permutation is its first n_tris
    entries — the padding sits past them and must not intrude.
    """
    bvh = _build(*_random_mesh(n_tris, seed=n_tris + 100))
    ids = np.sort(_tri_order(bvh))
    np.testing.assert_array_equal(ids, np.arange(n_tris))


# ── Degenerate input ─────────────────────────────────────────────────

def test_empty_mesh_builds_nothing(backend):
    points = tack.field(dtype=tack.f32, shape=(3,))
    conn = tack.field(dtype=tack.i32, shape=(3,))
    bvh = BVH()
    bvh.build(points, conn, 0)
    assert bvh.n_tris == 0


def test_single_triangle(backend):
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    conn = np.array([[0, 1, 2]], dtype=np.int32)
    bvh = _build(points, conn)
    assert bvh.n_tris == 1
    np.testing.assert_array_equal(_tri_order(bvh), [0])


def test_coincident_triangles(backend):
    """Identical Morton codes are the classic BVH build hazard."""
    one = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    points = np.tile(one, (8, 1))
    conn = np.arange(24, dtype=np.int32).reshape(8, 3)
    bvh = _build(points, conn)
    ids = np.sort(_tri_order(bvh))
    np.testing.assert_array_equal(ids, np.arange(8))


def test_degenerate_flat_mesh(backend):
    """All triangles coplanar — one axis of the bounding box has zero extent."""
    rng = np.random.default_rng(23)
    points = rng.random((30, 3)).astype(np.float32)
    points[:, 2] = 0.0
    conn = np.arange(30, dtype=np.int32).reshape(10, 3)
    bvh = _build(points, conn)
    ids = np.sort(_tri_order(bvh))
    np.testing.assert_array_equal(ids, np.arange(10))
