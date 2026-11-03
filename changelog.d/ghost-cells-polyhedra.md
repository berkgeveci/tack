### Added

- `ghost_cells` takes polyhedral pieces (OpenFOAM's processor pieces read as
  polyhedra): each ghost is sent with its faces wound out of it and the
  piece's faces are matched again, so a face a ghost shares with the piece's
  own cell is one face; `update_ghosts` and `generate_global_ids` run on
  them. The split reading's two pieces are ghosted together once the shaped
  piece is made polyhedral (`as_polyhedra`), and then reduce exactly.
