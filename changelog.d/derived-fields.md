### Added

- Derived fields: a name a dataset stores no field under, at the location it is read,
  may stand for a field the dataset makes from itself (`register_derived_field(name,
  compute, at)`), made the first time it is asked for and kept on the geometry.
  `extract_surface` of the same geometry keeps its surface geometry too, so a
  flux over many steps derives the normals once. One name is one quantity: `normals` is a surface's normals at its
  points and cells and every mesh's face normals at its faces; `measures` each
  cell's volume, area or length and each face's area; `centroids` each cell's.
  Every field argument reads them (`named_field`), and so do field expressions,
  so `integrate(surface, ["dot(U, normals)"])` is a flux with no filter first.
