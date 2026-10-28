### Fixed

- A CPU fan-out caps its chunks at about 20 µs of estimated work only on a
  short range, where the pool's own chunks are at most 256 elements. On any
  range the cap cut long dispatches into thousands of interleaved chunks, and
  on 28 threads the workers faulted the same huge pages of new outputs at
  once: threshold of 20M hexahedra took 650 ms against 200, point to cell
  120 against 30, and the ONERA M6's cell locator build 5.7 s against 2.0
  (jura, EPYC 7453). Short ranges of costly elements, such as a few
  streamlines, still go one element a claim.
