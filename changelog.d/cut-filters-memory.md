### Changed

- `clip` scans its five kinds of counts one at a time through one buffer and
  lets the cells' counts and offsets go before merging points; `contour` lets
  its cells' counts and offsets, and the per-corner arrays it copied when
  dropping repeated triangles, go before its merge. Clip's peak on 5M
  tetrahedra fell from 609 to 304 MiB of arrays, contour's on Damavand from
  3.47 to 3.19 GiB; the outputs are the same.
