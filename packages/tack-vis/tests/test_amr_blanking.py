"""Tests for tack.algorithms.amr_blanking.

build_masks marks two independent things in one i32 per cell: ghost cells
(bit 0) copied from an AMReX ghost array, and coarse cells covered by a
finer level (bit 3).  The coarsening arithmetic and the ghost-offset
mapping from AMR-box coordinates to grid-local indices are where this
goes wrong, so the tests drive both directly.
"""

import numpy as np

from tack.algorithms.amr_blanking import (
    DUPLICATECELL,
    REFINEDCELL,
    build_masks,
)


def _block(lo, hi, ng=0, ghost_array=None):
    """A block covering the AMR box [lo, hi] with ng ghost cells per side."""
    dims = [hi[d] - lo[d] + 1 + 2 * ng for d in range(3)]
    block = {
        "lo": lo, "hi": hi, "ng": ng,
        "nx_cells": dims[0], "ny_cells": dims[1], "nz_cells": dims[2],
    }
    if ghost_array is not None:
        block["ghost_array"] = ghost_array
    return block


def _as_grid(mask_field, block):
    """Mask as a (nz, ny, nx) array for indexing by grid-local coords."""
    return mask_field.to_numpy().reshape(
        block["nz_cells"], block["ny_cells"], block["nx_cells"])


def test_single_level_is_unmasked(backend):
    """With no finer level and no ghosts, nothing is marked."""
    block = _block((0, 0, 0), (3, 3, 3))
    masks = build_masks([[block]], [])
    assert len(masks) == 1 and len(masks[0]) == 1
    assert not masks[0][0].to_numpy().any()


def test_mask_shape_matches_the_ghosted_grid(backend):
    block = _block((0, 0, 0), (3, 3, 3), ng=2)
    masks = build_masks([[block]], [])
    assert masks[0][0].to_numpy().size == 8 * 8 * 8


def test_ghost_array_bits_are_merged(backend):
    """DUPLICATECELL bits from the input ghost array survive into the mask."""
    block_shape = (4, 4, 4)
    n = int(np.prod(block_shape))
    ghost = np.zeros(n, dtype=np.uint8)
    ghost[[0, 5, 63]] = DUPLICATECELL
    block = _block((0, 0, 0), (3, 3, 3), ghost_array=ghost)

    mask = build_masks([[block]], [])[0][0].to_numpy()
    assert (mask & DUPLICATECELL).astype(bool).tolist() == \
        (ghost & DUPLICATECELL).astype(bool).tolist()


def test_unrelated_ghost_bits_are_ignored(backend):
    """Only the DUPLICATECELL bit is copied; other VTK flags are not."""
    ghost = np.zeros(64, dtype=np.uint8)
    ghost[7] = 0b0100  # some other vtkGhostType flag
    block = _block((0, 0, 0), (3, 3, 3), ghost_array=ghost)
    assert not build_masks([[block]], [])[0][0].to_numpy().any()


def test_fully_covered_coarse_block_is_all_refined(backend):
    """A child covering the whole parent box blanks every parent cell."""
    coarse = _block((0, 0, 0), (3, 3, 3))
    fine = _block((0, 0, 0), (7, 7, 7))       # ratio 2 over the same region
    masks = build_masks([[coarse], [fine]], [2])

    coarse_mask = masks[0][0].to_numpy()
    assert (coarse_mask == REFINEDCELL).all()
    # The finest level is never blanked — nothing refines it.
    assert not masks[1][0].to_numpy().any()


def test_partial_coverage_blanks_only_the_overlap(backend):
    """A child over half the parent blanks exactly that half."""
    coarse = _block((0, 0, 0), (3, 3, 3))
    fine = _block((0, 0, 0), (3, 7, 7))       # x: cells 0..1 after coarsening
    masks = build_masks([[coarse], [fine]], [2])

    grid = _as_grid(masks[0][0], coarse)
    assert (grid[:, :, 0:2] == REFINEDCELL).all()
    assert (grid[:, :, 2:4] == 0).all()


def test_refinement_ratio_is_applied(backend):
    """Ratio 4 coarsens the child box four times as much as ratio 1."""
    coarse = _block((0, 0, 0), (7, 7, 7))
    fine = _block((0, 0, 0), (7, 7, 7))
    grid = _as_grid(build_masks([[coarse], [fine]], [4])[0][0], coarse)
    # child 0..7 at ratio 4 coarsens to 0..1 on every axis
    assert (grid[0:2, 0:2, 0:2] == REFINEDCELL).all()
    assert (grid == REFINEDCELL).sum() == 8


def test_disjoint_child_marks_nothing(backend):
    """A child that misses the parent box entirely leaves it alone."""
    coarse = _block((0, 0, 0), (3, 3, 3))
    fine = _block((100, 100, 100), (107, 107, 107))
    assert not build_masks([[coarse], [fine]], [2])[0][0].to_numpy().any()


def test_multiple_children_accumulate(backend):
    """Two children covering different halves blank the union."""
    coarse = _block((0, 0, 0), (3, 3, 3))
    left = _block((0, 0, 0), (3, 7, 7))       # coarse x 0..1
    right = _block((4, 0, 0), (7, 7, 7))      # coarse x 2..3
    masks = build_masks([[coarse], [left, right]], [2])
    assert (masks[0][0].to_numpy() == REFINEDCELL).all()


def test_ghost_offset_shifts_the_blanked_region(backend):
    """Blanking is placed in grid-local coords, past the ghost layer."""
    ng = 2
    coarse = _block((0, 0, 0), (3, 3, 3), ng=ng)
    fine = _block((0, 0, 0), (1, 7, 7))       # coarse x cell 0 only
    grid = _as_grid(build_masks([[coarse], [fine]], [2])[0][0], coarse)

    # Valid cell (0,0,0) lives at grid-local (ng, ng, ng).
    assert grid[ng, ng, ng] == REFINEDCELL
    # The ghost layer itself is outside the AMR box and stays clear.
    assert grid[0, 0, 0] == 0
    assert (grid[:, :, ng + 1:] == 0).all()


def test_ghost_and_refined_bits_coexist(backend):
    """A cell can be both a ghost and covered; the bits OR together."""
    ng = 0
    n = 4 * 4 * 4
    ghost = np.zeros(n, dtype=np.uint8)
    ghost[:] = DUPLICATECELL
    coarse = _block((0, 0, 0), (3, 3, 3), ng=ng, ghost_array=ghost)
    fine = _block((0, 0, 0), (7, 7, 7))
    mask = build_masks([[coarse], [fine]], [2])[0][0].to_numpy()
    assert (mask == (DUPLICATECELL | REFINEDCELL)).all()


def test_negative_box_coordinates_coarsen_downward(backend):
    """VTK's coarsening rounds toward −∞ so negative boxes still nest."""
    coarse = _block((-4, 0, 0), (-1, 3, 3))
    fine = _block((-8, 0, 0), (-5, 7, 7))     # coarsens to -4..-3
    grid = _as_grid(build_masks([[coarse], [fine]], [2])[0][0], coarse)
    assert (grid[:, :, 0:2] == REFINEDCELL).all()
    assert (grid[:, :, 2:] == 0).all()


def test_three_levels(backend):
    """Each level is blanked only by the level immediately below it."""
    l0 = _block((0, 0, 0), (3, 3, 3))
    l1 = _block((0, 0, 0), (3, 7, 7))         # covers coarse x 0..1
    l2 = _block((0, 0, 0), (3, 15, 15))       # covers l1 x 0..1
    masks = build_masks([[l0], [l1], [l2]], [2, 2])

    g0 = _as_grid(masks[0][0], l0)
    assert (g0[:, :, 0:2] == REFINEDCELL).all()
    assert (g0[:, :, 2:] == 0).all()

    g1 = _as_grid(masks[1][0], l1)
    assert (g1[:, :, 0:2] == REFINEDCELL).all()
    assert (g1[:, :, 2:] == 0).all()

    assert not masks[2][0].to_numpy().any()


def test_per_axis_ghost_widths(backend):
    """ng may be a per-axis tuple, not just a scalar."""
    ng = (1, 2, 3)
    block = {
        "lo": (0, 0, 0), "hi": (3, 3, 3), "ng": ng,
        "nx_cells": 4 + 2 * ng[0],
        "ny_cells": 4 + 2 * ng[1],
        "nz_cells": 4 + 2 * ng[2],
    }
    fine = _block((0, 0, 0), (1, 1, 1))       # coarse cell (0,0,0) only
    grid = _as_grid(build_masks([[block], [fine]], [2])[0][0], block)
    assert grid[ng[2], ng[1], ng[0]] == REFINEDCELL
    assert (grid == REFINEDCELL).sum() == 1
