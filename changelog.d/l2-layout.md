### Added

- `L2(..., layout=)`: how a tensor basis' values are stored, `"blocks"` (each
  cell's together: MFEM's and VTK cell grids' layout) or `"cells"`
  (cell-fastest, so neighbouring GPU threads read neighbouring values).
  `"auto"`, the default, is `"cells"` for a tensor basis of order 3 or more on
  a GPU and `"blocks"` elsewhere: on an RTX 4060 Ti cell-fastest reads 2.3x and
  writes 3.6x faster at order 6, on the CPU blocks reads 1.5-1.8x and writes
  about 2x faster. `to_device` transposes when the layout changes; the MFEM
  interop reads and writes either; `space.blocks(values)` and
  `space.indices()` read a field per cell whatever its layout.
  `TACK_L2_LAYOUT=cells` (or `blocks`) overrides `"auto"` for a whole run.

### Fixed

- A filter carrying an `L2` field of a tensor basis kept its order but not its
  nodes, so a Gauss-Legendre field came out labelled Gauss-Lobatto, another
  basis, and read wrongly.
