### Changed

- The CPU backend keeps far less memory per compiled kernel: compiled code shares
  JIT engines (32 modules to one) instead of an engine each, the LLVM module is
  dropped once its code is made, and a compiled variant keeps only the launch
  bounds of its IR, not the body. A small kernel's variant went from about
  965 KB to 75 KB, a high-order contour and clip from 51 MB to about 21 MB, and
  the quick test tier's workers peak at 1.5 GB instead of 4.9.
