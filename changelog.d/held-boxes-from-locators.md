### Fixed

- Streamlines, advection and probes over pieces bounded each piece by its
  positions' box taken on the host on every call: a second of each call over
  the AhmedML mesh's 21M-cell piece, half the split reading's streamline time.
  The box is now the piece's locator's, found once when the locator is made;
  the split reading's 2,000 streamlines take 0.75 s, as the whole reading's
  do, where they took 1.71.
