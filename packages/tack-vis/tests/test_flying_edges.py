"""Golden-value tests for flying edges isosurface extraction.

Flying edges is the largest and most intricate algorithm in tack-vis, and
its output is hard to eyeball.  These tests pin it against two analytic
surfaces where the right answer is known in closed form:

  * a plane through a uniform grid, where linear interpolation is exact,
    so both the point count and the triangle count are exact integers; and
  * a sphere, where the surface must be watertight, consistently wound,
    of genus 0, and enclose 4/3·π·r³.

The sphere checks are what catch a merged-point regression: an isosurface
that emits duplicate or misindexed points still looks fine rendered, but
stops being a closed manifold immediately.
"""

import numpy as np
import pytest
from vis_helpers import (
    directed_edge_counts,
    enclosed_volume,
    euler_characteristic,
    make_grid,
    node_coords,
    plane_field,
    sphere_field,
    undirected_edge_counts,
    upload,
)

import tack
from tack.algorithms.flying_edges import (
    _NUM_TRIS,
    _TRI_TABLE,
    MCTables,
    flying_edges,
    flying_edges_multiblock,
)

RADIUS = 1.0
SPHERE_VOLUME = 4.0 / 3.0 * np.pi * RADIUS ** 3


# ================================================================
# Marching-cubes tables (host-side, no backend needed)
# ================================================================

def test_num_tris_matches_tri_table():
    """_NUM_TRIS must agree with the triangles actually listed per case."""
    for case in range(256):
        row = _TRI_TABLE[case * 16:(case + 1) * 16]
        counted = sum(1 for t in range(0, 16, 3) if row[t] >= 0)
        assert _NUM_TRIS[case] == counted, f"case {case}"


def test_tri_table_triples_are_complete():
    """A triangle is three edges or none — never a partial triple."""
    for case in range(256):
        row = _TRI_TABLE[case * 16:(case + 1) * 16]
        for t in range(0, 16, 3):
            triple = row[t:t + 3]
            assert (triple >= 0).all() or (triple < 0).all(), \
                f"case {case} triple at {t}: {triple}"


def test_tri_table_edges_in_range():
    """Every edge reference is one of the 12 cube edges."""
    valid = _TRI_TABLE[_TRI_TABLE >= 0]
    assert valid.min() >= 0 and valid.max() <= 11


def test_only_the_uniform_cases_emit_nothing():
    """All-below (0) and all-above (255) are the only empty cases."""
    assert _NUM_TRIS[0] == 0
    assert _NUM_TRIS[255] == 0
    assert (_NUM_TRIS == 0).sum() == 2


def test_triangle_count_never_exceeds_five():
    """A cube case can produce at most 5 triangles."""
    assert _NUM_TRIS.min() >= 0
    assert _NUM_TRIS.max() == 5


def test_mc_tables_upload(backend):
    """MCTables mirrors the numpy tables into device fields."""
    tables = MCTables()
    np.testing.assert_array_equal(tables.tri_table.to_numpy(), _TRI_TABLE)
    np.testing.assert_array_equal(tables.num_tris.to_numpy(), _NUM_TRIS)


# ================================================================
# Plane — exact counts, exact geometry
# ================================================================

def _plane_case(n=4):
    """n³ grid over [0,1]³ cut by a plane inside the k=1 cell layer."""
    grid = make_grid(n, lo=0.0, hi=1.0)
    isovalue = 0.375  # strictly between z=0.25 and z=0.5 for n=4
    return grid, plane_field(grid), isovalue


def test_plane_point_count_is_merged(backend):
    """A planar cut crosses one z-edge per node column: (n+1)² points.

    Unmerged marching cubes would emit 3 points per triangle — 6n² — so
    this number is the whole point of the flying-edges edge-ownership pass.
    """
    n = 4
    grid, scalar, isovalue = _plane_case(n)
    result = flying_edges(scalar, grid, isovalue)
    assert result["total_points"] == (n + 1) ** 2


def test_plane_triangle_count(backend):
    """Each cell in the cut layer contributes exactly two triangles."""
    n = 4
    grid, scalar, isovalue = _plane_case(n)
    result = flying_edges(scalar, grid, isovalue)
    assert result["total_tris"] == 2 * n * n


def test_plane_points_lie_on_the_plane(backend):
    """Linear interpolation of a linear field is exact."""
    grid, scalar, isovalue = _plane_case()
    result = flying_edges(scalar, grid, isovalue)
    z = result["points"][:, 2]
    np.testing.assert_allclose(z, isovalue, atol=1e-5)


def test_plane_spans_the_grid(backend):
    """The cut covers the full x/y extent of the grid."""
    grid, scalar, isovalue = _plane_case()
    result = flying_edges(scalar, grid, isovalue)
    p = result["points"]
    np.testing.assert_allclose(p[:, 0].min(), 0.0, atol=1e-6)
    np.testing.assert_allclose(p[:, 0].max(), 1.0, atol=1e-6)
    np.testing.assert_allclose(p[:, 1].min(), 0.0, atol=1e-6)
    np.testing.assert_allclose(p[:, 1].max(), 1.0, atol=1e-6)


def test_plane_boundary_is_the_grid_perimeter(backend):
    """An open patch: interior edges shared by two triangles, rim by one."""
    n = 4
    grid, scalar, isovalue = _plane_case(n)
    result = flying_edges(scalar, grid, isovalue)
    counts = undirected_edge_counts(result["conn"])
    assert set(counts.values()) <= {1, 2}
    boundary = sum(1 for c in counts.values() if c == 1)
    assert boundary == 4 * n


# ================================================================
# Sphere — topology and convergence
# ================================================================

def test_sphere_is_watertight(backend):
    """Every edge is shared by exactly two triangles — a closed manifold."""
    grid = make_grid(24)
    result = flying_edges(sphere_field(grid, RADIUS), grid, 0.0)
    counts = undirected_edge_counts(result["conn"])
    assert set(counts.values()) == {2}


def test_sphere_winding_is_consistent(backend):
    """Each directed edge appears once, so neighbours agree on orientation."""
    grid = make_grid(24)
    result = flying_edges(sphere_field(grid, RADIUS), grid, 0.0)
    counts = directed_edge_counts(result["conn"])
    assert set(counts.values()) == {1}


def test_sphere_is_genus_zero(backend):
    """V − E + F == 2."""
    grid = make_grid(24)
    result = flying_edges(sphere_field(grid, RADIUS), grid, 0.0)
    assert euler_characteristic(result) == 2


def test_sphere_has_no_orphan_points(backend):
    """Every emitted point is referenced by the connectivity."""
    grid = make_grid(24)
    result = flying_edges(sphere_field(grid, RADIUS), grid, 0.0)
    assert np.unique(result["conn"]).size == result["total_points"]


def test_sphere_conn_indices_in_range(backend):
    grid = make_grid(24)
    result = flying_edges(sphere_field(grid, RADIUS), grid, 0.0)
    conn = result["conn"]
    assert conn.min() >= 0
    assert conn.max() < result["total_points"]


def test_sphere_points_lie_on_the_isosurface(backend):
    """Points sit on the sphere to within the interpolation error bound.

    On an edge of length h the linear interpolant of a quadratic field is
    off by at most h²/4 in field value, so |x²+y²+z² − r²| ≤ h²/4 up to
    float rounding.  A merged-point indexing bug blows straight past this.
    """
    n = 24
    grid = make_grid(n)
    result = flying_edges(sphere_field(grid, RADIUS), grid, 0.0)
    p = result["points"].astype(np.float64)
    residual = np.abs((p ** 2).sum(axis=1) - RADIUS ** 2)
    h = grid.dx
    assert residual.max() <= 1.1 * h * h / 4


def test_sphere_encloses_the_right_volume(backend):
    """Enclosed volume matches 4/3·π·r³ and is positive (outward winding)."""
    grid = make_grid(24)
    result = flying_edges(sphere_field(grid, RADIUS), grid, 0.0)
    volume = enclosed_volume(result)
    assert volume > 0
    assert abs(volume - SPHERE_VOLUME) / SPHERE_VOLUME < 0.02


def test_sphere_volume_converges_second_order(backend):
    """Halving the spacing cuts the volume error by roughly four."""
    errors = []
    for n in (12, 24, 48):
        grid = make_grid(n)
        result = flying_edges(sphere_field(grid, RADIUS), grid, 0.0)
        errors.append(abs(enclosed_volume(result) - SPHERE_VOLUME))
    assert errors[1] < errors[0] / 2.5
    assert errors[2] < errors[1] / 2.5


def test_sphere_winding_follows_the_field_sign(backend):
    """Negating the field flips the surface orientation, not its shape."""
    grid = make_grid(24)
    inside = flying_edges(sphere_field(grid, RADIUS, True), grid, 0.0)
    outside = flying_edges(sphere_field(grid, RADIUS, False), grid, 0.0)
    assert enclosed_volume(inside) > 0
    assert enclosed_volume(outside) < 0
    np.testing.assert_allclose(enclosed_volume(inside),
                               -enclosed_volume(outside), rtol=1e-6)


def test_sphere_radius_scales_with_isovalue(backend):
    """Raising the isovalue on r²−|p|² shrinks the extracted sphere."""
    grid = make_grid(24)
    scalar = sphere_field(grid, RADIUS)
    for target in (0.6, 0.8, 1.0):
        # r²−|p|² = iso  ⇔  |p| = sqrt(r² − iso)
        result = flying_edges(scalar, grid, RADIUS ** 2 - target ** 2)
        radii = np.linalg.norm(result["points"].astype(np.float64), axis=1)
        np.testing.assert_allclose(radii.mean(), target, rtol=0.01)


# ================================================================
# Grid placement
# ================================================================

def test_origin_and_spacing_are_honored(backend):
    """A shifted, anisotropically spaced grid places points where it says."""
    nx, ny, nz = 20, 24, 16
    grid = type(make_grid(2))(nx, ny, nz, 2.0, -5.0, 10.0, 0.1, 0.25, 0.125)
    c = node_coords(grid)
    center = np.array([3.0, -2.0, 11.0])
    r = 0.6
    scalar = upload(r ** 2 - ((c - center) ** 2).sum(axis=1))
    result = flying_edges(scalar, grid, 0.0)
    p = result["points"].astype(np.float64)
    np.testing.assert_allclose(p.min(axis=0), center - r, atol=0.05)
    np.testing.assert_allclose(p.max(axis=0), center + r, atol=0.05)


# ================================================================
# Degenerate inputs
# ================================================================

def test_isovalue_above_range_returns_none(backend):
    grid = make_grid(8)
    assert flying_edges(sphere_field(grid, RADIUS), grid, 1e6) is None


def test_isovalue_below_range_returns_none(backend):
    grid = make_grid(8)
    assert flying_edges(sphere_field(grid, RADIUS), grid, -1e6) is None


def test_no_blocks_returns_none(backend):
    assert flying_edges_multiblock([], 0.0) is None


# ================================================================
# Cell masking
# ================================================================

def test_masking_every_cell_returns_none(backend):
    grid = make_grid(16)
    mask = tack.field(dtype=tack.i32, shape=(grid.nx * grid.ny * grid.nz,))
    mask.fill(1)
    result = flying_edges_multiblock(
        [{"scalar": sphere_field(grid, RADIUS), "grid": grid, "mask": mask}], 0.0)
    assert result is None


def test_masking_half_the_cells_halves_the_triangles(backend):
    """The mask gates triangle emission; the sphere is symmetric in k."""
    n = 16
    grid = make_grid(n)
    scalar = sphere_field(grid, RADIUS)
    full = flying_edges(scalar, grid, 0.0)

    n_cells = grid.nx * grid.ny * grid.nz
    mask_np = np.zeros(n_cells, dtype=np.int32)
    mask_np[:n_cells // 2] = 1  # lower half in k
    mask = upload(mask_np, dtype=tack.i32, np_dtype=np.int32)

    masked = flying_edges_multiblock(
        [{"scalar": scalar, "grid": grid, "mask": mask}], 0.0)
    assert masked["total_tris"] == full["total_tris"] // 2


def test_masking_does_not_change_point_count(backend):
    """Points are owned by edges, which the mask does not gate.

    Documents the current contract: a masked run keeps the unmasked point
    array so point IDs stay valid across blocks.
    """
    n = 16
    grid = make_grid(n)
    scalar = sphere_field(grid, RADIUS)
    full = flying_edges(scalar, grid, 0.0)

    n_cells = grid.nx * grid.ny * grid.nz
    mask_np = np.zeros(n_cells, dtype=np.int32)
    mask_np[:n_cells // 2] = 1
    mask = upload(mask_np, dtype=tack.i32, np_dtype=np.int32)
    masked = flying_edges_multiblock(
        [{"scalar": scalar, "grid": grid, "mask": mask}], 0.0)

    assert masked["total_points"] == full["total_points"]


# ================================================================
# Multi-block
# ================================================================

def _split_sphere_blocks(n=24):
    """The [-1.5,1.5]³ sphere domain as two blocks stacked in z.

    The blocks share their interface node plane, as adjacent AMR/VTK
    blocks do, so the two surfaces meet without a gap.
    """
    h = 3.0 / n
    half = n // 2
    grid_type = type(make_grid(2))
    lower = grid_type(n, n, half, -1.5, -1.5, -1.5, h, h, h)
    upper = grid_type(n, n, half, -1.5, -1.5, -1.5 + half * h, h, h, h)
    return [{"scalar": sphere_field(g, RADIUS), "grid": g}
            for g in (lower, upper)]


def test_multiblock_totals_are_the_sum_of_blocks(backend):
    blocks = _split_sphere_blocks()
    singles = [flying_edges(b["scalar"], b["grid"], 0.0) for b in blocks]
    merged = flying_edges_multiblock(blocks, 0.0)

    assert merged["total_points"] == sum(s["total_points"] for s in singles)
    assert merged["total_tris"] == sum(s["total_tris"] for s in singles)
    assert merged["block_point_counts"] == [s["total_points"] for s in singles]
    assert merged["block_tri_counts"] == [s["total_tris"] for s in singles]


def test_multiblock_connectivity_is_offset_per_block(backend):
    """Each block's triangles index only into that block's point range."""
    blocks = _split_sphere_blocks()
    merged = flying_edges_multiblock(blocks, 0.0)

    conn = merged["conn"]
    assert conn.min() >= 0
    assert conn.max() < merged["total_points"]

    start_tri = 0
    start_pt = 0
    for n_pts, n_tris in zip(merged["block_point_counts"],
                             merged["block_tri_counts"]):
        block_conn = conn[start_tri:start_tri + n_tris]
        assert block_conn.min() >= start_pt
        assert block_conn.max() < start_pt + n_pts
        start_tri += n_tris
        start_pt += n_pts


def test_multiblock_reproduces_the_single_block_surface(backend):
    """Splitting the domain changes point IDs, not the surface itself."""
    n = 24
    whole = make_grid(n)
    single = flying_edges(sphere_field(whole, RADIUS), whole, 0.0)
    merged = flying_edges_multiblock(_split_sphere_blocks(n), 0.0)

    assert merged["total_tris"] == single["total_tris"]
    np.testing.assert_allclose(enclosed_volume(merged),
                               enclosed_volume(single), rtol=1e-6)


def test_multiblock_seam_points_are_duplicated_not_dropped(backend):
    """Blocks are meshed independently, so the shared plane is emitted twice."""
    n = 24
    whole = make_grid(n)
    single = flying_edges(sphere_field(whole, RADIUS), whole, 0.0)
    merged = flying_edges_multiblock(_split_sphere_blocks(n), 0.0)
    assert merged["total_points"] > single["total_points"]


def test_single_block_matches_flying_edges(backend):
    """flying_edges() is flying_edges_multiblock() with one block."""
    grid = make_grid(16)
    scalar = sphere_field(grid, RADIUS)
    a = flying_edges(scalar, grid, 0.0)
    b = flying_edges_multiblock([{"scalar": scalar, "grid": grid}], 0.0)
    np.testing.assert_array_equal(a["points"], b["points"])
    np.testing.assert_array_equal(a["conn"], b["conn"])


def test_empty_block_is_skipped(backend):
    """A block with no crossings contributes nothing but does not break."""
    n = 16
    grid_type = type(make_grid(2))
    h = 3.0 / n
    live = make_grid(n)
    far = grid_type(n, n, n, 100.0, 100.0, 100.0, h, h, h)

    blocks = [{"scalar": sphere_field(live, RADIUS), "grid": live},
              {"scalar": upload(np.full((n + 1) ** 3, -1.0)), "grid": far}]
    merged = flying_edges_multiblock(blocks, 0.0)

    alone = flying_edges(blocks[0]["scalar"], live, 0.0)
    assert merged["block_tri_counts"][1] == 0
    assert merged["total_tris"] == alone["total_tris"]


# ================================================================
# Output fields
# ================================================================

def test_result_fields_match_the_numpy_arrays(backend):
    """points_field/conn_field are the same data, kept on device for interop."""
    grid = make_grid(16)
    result = flying_edges(sphere_field(grid, RADIUS), grid, 0.0)
    np.testing.assert_array_equal(
        result["points_field"].to_numpy().reshape(-1, 3), result["points"])
    np.testing.assert_array_equal(
        result["conn_field"].to_numpy().reshape(-1, 3), result["conn"])
    assert result["points"].shape == (result["total_points"], 3)
    assert result["conn"].shape == (result["total_tris"], 3)


@pytest.mark.parametrize("n", [2, 3, 5])
def test_tiny_grids(backend, n):
    """Small odd-sized grids exercise the row bookkeeping at the edges."""
    grid = make_grid(n)
    result = flying_edges(sphere_field(grid, 1.0), grid, 0.0)
    if result is None:
        return
    counts = undirected_edge_counts(result["conn"])
    assert set(counts.values()) == {2}
