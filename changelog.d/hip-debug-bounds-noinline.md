### Fixed

- Debug mode on HIP: the bounds check helper `__tack_in_bounds__` is
  `__noinline__`. hipRTC (ROCm 7.1.1) at `-O2` and above miscompiled it
  inlined, and five `test_high_order.py` cases reported in-bounds accesses out
  of bounds, with 0.5f's bits in an index's high half. CUDA's stays inline.
