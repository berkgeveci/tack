### Changed

- A `PolygonalTopology` matches its edges when something first needs them
  (`faces()`, the facet arrays, `num_faces`), not when it is made, and refuses
  an edge of three or more polygons, or an inconsistent winding, there. Its
  polygons work without the matching: areas, centres, integrals, probes,
  quality and writing take any loops.

### Fixed

- `contour` of a polyhedral mesh raised where the iso-surface met an edge
  three or more times: a point within half an ulp of the level is a vertex
  on some of its edges and not others, so several faces trace one iso-edge.
  The AhmedML mesh's Q = 1, 2 and 5 contours (2.0M, 1.5M and 0.8M polygons)
  are made now; only their faces are refused.
