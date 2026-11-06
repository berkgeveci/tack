### Fixed

- Level Zero: every kernel requires sub-groups of 16
  (`intel_reqd_sub_group_size(16)`), so IGC never compiles one SIMD32. On a
  Max 1100 (IGC 2.7.11, compute runtime 25.05) IGC's SIMD32 code for five
  tack-vis kernels was wrong: crossings of high-order contours and slices
  (in f32, and with 64-bit ids), a simplex subdivision's lattice points, and
  AMR `ghost_cells`' marks. Their SIMD16 code, fully optimized, is right.
  Compute runtime 26.40 does not fix it.
