### Fixed

- Reading polyhedra from VTK, and any cells with
  `vtk_to_dataset(..., polyhedral=True)`, is about ten times faster:
  `PolyhedralTopology.from_cell_faces` matches the faces cells share in
  whole-array steps instead of a Python loop over them, and the reader takes a
  cell type's faces from its class once rather than calling `GetCell` per cell
  (512,000 polyhedra: 7.2 s to 0.7 s). The topology is the same, array for array.
- Triangle strips read about four times faster: their cell data is copied to
  the triangles by a NumPy gather per array, not a `vtkIdList` filled one id
  at a time (2M triangles: 319 ms to 74 ms).
