### Changed

- CUDA writes the multiply-add contractions the language contract permits as
  explicit fused multiply-adds and compiles with `--fmad=false`. NVRTC's own
  fusion depended on the code around an expression, so two kernels inlining
  the same function could differ in the last bit (streamlines with and without
  the locator's previous-cell try, on curved wedges).
