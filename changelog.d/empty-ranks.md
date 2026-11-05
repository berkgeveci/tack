### Fixed

- Over MPI, the algorithms over pieces take ranks that hold no pieces, as a reader
  that gives some ranks nothing makes them. Each rank must make the same calls
  that communicate, so a rank must not decide from its own pieces alone what the
  others decide from theirs:
  - `probe` of a `PartitionedDataSet` or an AMR hierarchy, `streamlines` and
    `advect` over pieces, `voxelize` of pieces or a hierarchy, and `Advection`
    (and the path filters) over pieces refused or failed on a rank without
    pieces ("needs pieces on this rank to start from", "a rank with no pieces has
    no piece to start its seeds from"), leaving the others waiting. Such a rank
    now takes part through a piece of nothing (`_stand_in`): its points and seeds
    go to the pieces that hold them, and the answers come back.
  - `update_ghosts`, `extract_surface` of pieces, AMR `ghost_cells` and
    `average_down`, `integrate` over pieces, and the VTK `Statistics`, `Entropy`
    and `Integrate` algorithms chose their fields (or a shortcut) from this rank's
    pieces; a rank without any now gets the others' choice.
  - `merge_pieces` onto a root rank without pieces refused; the merge is made on
    the first rank that has pieces and moved to the root.
  - `test_empty_ranks.py` runs every algorithm over pieces, AMR hierarchies,
    octrees, advection and the VTK algorithms over pieces on 3 ranks, with the
    pieces on one, two or none of them, against the same pieces in one process.
