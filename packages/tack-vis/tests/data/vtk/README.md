# VTK test meshes

Copies of files from VTK's `Testing/Data`, used by `test_polyhedra.py` to
check polyhedra read from real meshes. They are VTK's, under VTK's BSD-3
license in `LICENSE-VTK.txt`. Each was fetched from VTK's data store by
the SHA-512 recorded in VTK's source tree (`Testing/Data/<name>.sha512`,
VTK master `48939248a1`), and matches it:

| File | Bytes | SHA-512 (first 16) |
|---|---|---|
| `onePolyhedron.vtu` | 3906 | `4c7a9a787ffe862e` |
| `polyhedron2pieces.vtu` | 2483 | `c1a46dfda1265e15` |
| `polyhedron_mesh.vtu` | 1626 | `27dbbe338630d688` |
| `concavePolyhedron.vtu` | 1451 | `d4baefc7e1d3761e` |
| `sliceOfPolyhedron.vtu` | 674907 | `6039d9bacc52e01e` |
| `vtkHDF/polyhedron.vtu` | 4103 | `6817750d1538cb5d` |
| `nonWatertightPolyhedron.vtu` | 6694 | `3098af8357a96ca8` |
| `EngineSector.cgns` | 1088363 | `1f69b4276708003a` |
| `Example_nface_n.cgns` | 110592 | `13fc52ebc5d64a0d` |
| `Example_ngon_pe.cgns` | 68561 | `ffbec6b7754068cc` |
