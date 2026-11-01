### Changed

- The CUDA, HIP, OpenCL and Metal code generators find the types of the
  variables an `if` declares in one walk of its branches. Kernels with many
  nested branches, which inlined device functions produce, generated their
  source in time that grew with the number of variables times the branches'
  size at every level: a spread of tack-vis algorithms' 105 kernels now
  generate in a third of the time, to the same source.
