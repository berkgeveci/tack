### Fixed

- Slicing or contouring serendipity cells (quad8, hex20, wedge15) that carry a
  64-bit integer point field failed: the field's full values asked a device for
  what a backend has. Found running VTK's `cutter.py` on Tack (`cuttertest.vtu`'s
  node numbers).
- `extract_extent` of a flat curvilinear grid refused VTK's three-axis extent and
  rates (VTK always writes three; a grid's flat axis is the one whose pair is
  `(0, 0)`). Found running VTK's `NetCDFCF2DBounds.py` on Tack.
