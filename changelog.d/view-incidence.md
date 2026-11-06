### Changed

- **Breaking:** a kernel over shape-based cells that reads their faces or
  edges (`face_id`, `face_side`, `face_orientation`, `face_position`,
  `edge_id`, `edge_sign`) asks for them:
  `for_each(kernel, data, "cells", ..., incidence="faces")`, or `"edges"`, or
  both, and `domain_view(domain, group, incidence=...)` the same. They are
  derived if they are not yet. A cell view had them whenever the faces or
  edges had been derived, by this filter or any before it, so the same kernel
  over the same cells compiled again once another filter had derived them,
  and the disk cache kept gaining entries from one run of a program to the
  next as the order of its filters changed (about 25 a pass of the quick
  tests, from `contour`, `project` and `subdivide`). A view now has only what
  it asks for; a curved geometry still reads the incidence it needs.
