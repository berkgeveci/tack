"""AMR cell blanking and ghost marking for overlapping AMR grids.

Builds a per-cell mask array that combines:
  - Ghost cell flags (DUPLICATECELL = 1): cells in ghost layers
  - AMR blanking flags (REFINEDCELL = 8): coarse cells covered by finer level

The mask is computed following VTK's vtkAMRUtilities::BlankGridsAtLevel
algorithm: blanking is determined using valid-region AMR boxes (no ghosts),
then mapped onto the full ghosted grid.

Public API
----------
build_mask(blocks_by_level, refinement_ratios)
    Build masks for all blocks across all levels.
"""

import numpy as np
import tack

# VTK ghost type flags
DUPLICATECELL = 1   # ghost cell
REFINEDCELL = 8     # covered by finer level


def build_masks(blocks_by_level, refinement_ratios):
    """Build ghost + blanking masks for all blocks in an AMR hierarchy.

    Args:
        blocks_by_level: list of lists. blocks_by_level[level] is a list of
            block dicts, each with:
            - 'lo': (i, j, k) AMR box lower corner (valid region, global coords)
            - 'hi': (i, j, k) AMR box upper corner (valid region, inclusive)
            - 'nx_cells': total cell count in x including ghosts
            - 'ny_cells': total cell count in y including ghosts
            - 'nz_cells': total cell count in z including ghosts
            - 'ng': number of ghost cells per side (or per-side tuple)
            - 'ghost_array': numpy uint8 array of ghost flags (optional,
              from AMReX vtkGhostType field). If provided, its DUPLICATECELL
              bits are merged into the mask.
        refinement_ratios: list of int. refinement_ratios[level] is the
            ratio between level and level+1.

    Returns:
        list of lists of tack.field (i32), same structure as blocks_by_level.
        Each field has shape (nx_cells * ny_cells * nz_cells,) with mask values.
    """
    n_levels = len(blocks_by_level)
    masks = []

    for level in range(n_levels):
        level_masks = []
        for b, block in enumerate(blocks_by_level[level]):
            nx_c = block['nx_cells']
            ny_c = block['ny_cells']
            nz_c = block['nz_cells']
            n_cells = nx_c * ny_c * nz_c
            mask = np.zeros(n_cells, dtype=np.int32)

            # --- Ghost marking ---
            # If a ghost_array is provided (from AMReX), merge DUPLICATECELL bits
            if 'ghost_array' in block:
                ghost = block['ghost_array']
                for idx in range(n_cells):
                    if ghost[idx] & DUPLICATECELL:
                        mask[idx] |= DUPLICATECELL

            # --- AMR blanking ---
            # Mark cells covered by finer-level blocks
            if level < n_levels - 1 and level < len(refinement_ratios):
                ratio = refinement_ratios[level]
                box_lo = np.array(block['lo'], dtype=np.int64)
                box_hi = np.array(block['hi'], dtype=np.int64)

                ng = block.get('ng', 0)
                if isinstance(ng, int):
                    ng_lo = np.array([ng, ng, ng], dtype=np.int64)
                else:
                    ng_lo = np.array(ng, dtype=np.int64)

                # The grid extent in AMR-box-relative coords:
                # extent_lo = -ng (ghost cells before valid region)
                # Cell (ix, iy, iz) in AMR global coords maps to
                # grid-local index (ix - box_lo[0], iy - box_lo[1], iz - box_lo[2])
                # which then maps to linear index accounting for ghost offset.

                for child in blocks_by_level[level + 1]:
                    child_lo = np.array(child['lo'], dtype=np.int64)
                    child_hi = np.array(child['hi'], dtype=np.int64)

                    # Coarsen child box to this level's coordinates
                    coarse_lo = np.empty(3, dtype=np.int64)
                    coarse_hi = np.empty(3, dtype=np.int64)
                    for d in range(3):
                        # VTK coarsen: floor division for positive, special for negative
                        lo = child_lo[d]
                        hi = child_hi[d]
                        coarse_lo[d] = lo // ratio if lo >= 0 else -(abs(lo + 1) // ratio) - 1
                        coarse_hi[d] = hi // ratio if hi >= 0 else -(abs(hi + 1) // ratio) - 1

                    # Intersect with parent box
                    int_lo = np.maximum(coarse_lo, box_lo)
                    int_hi = np.minimum(coarse_hi, box_hi)

                    if np.any(int_lo > int_hi):
                        continue  # no intersection

                    # Mark cells in intersection as REFINEDCELL
                    for iz in range(int(int_lo[2]), int(int_hi[2]) + 1):
                        for iy in range(int(int_lo[1]), int(int_hi[1]) + 1):
                            for ix in range(int(int_lo[0]), int(int_hi[0]) + 1):
                                # Convert AMR global coords to grid-local coords
                                # Grid local = (ix - box_lo) + ng_lo
                                # (ng_lo accounts for ghost cells before valid region)
                                li = int(ix - box_lo[0] + ng_lo[0])
                                lj = int(iy - box_lo[1] + ng_lo[1])
                                lk = int(iz - box_lo[2] + ng_lo[2])
                                if (0 <= li < nx_c and 0 <= lj < ny_c
                                        and 0 <= lk < nz_c):
                                    cell_id = lk * ny_c * nx_c + lj * nx_c + li
                                    mask[cell_id] |= REFINEDCELL

            mask_field = tack.field(dtype=tack.i32, shape=(n_cells,))
            mask_field.from_numpy(mask)
            level_masks.append(mask_field)

        masks.append(level_masks)

    return masks
