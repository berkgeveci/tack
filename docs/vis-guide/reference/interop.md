# Dataset interoperability

Use [Connect Simulation Data](../connect-simulation-data.md) for complete workflows.
This catalog records format mappings and geometric restrictions.

## Reading and writing other formats

**VTK unstructured grids.** Linear cells read as the shapes above. A grid with
polyhedra reads as a `PolyhedralTopology`, every cell a polyhedron of its faces,
since one topology holds either shapes or polyhedra for now (a topology holding
both is planned). For a large mesh that mixes them, read it with
`vtk_to_dataset(grid, polyhedral="split")`: a `PartitionedDataSet` of two
pieces over the same points, the shaped cells as themselves and the polyhedra
as polyhedra, each point owned by one of them so reductions count it once. A
CFD mesh of 21M hexahedra and 508K polyhedra reads in under 5 seconds this way
(with `copy=False`); read as polyhedra throughout, 27 seconds and 19 GB.

Cells whose corners are in the reverse of VTK's order -- inside out, with negative
volumes and inward faces, as some converters write prisms -- are found on every
read and named in a warning; `vtk_to_dataset(grid, orient=True)` turns them
outward. Wedges are the usual case: VTK's documentation described the wedge's
corner order the reverse of how its code (faces, parametric coordinates, Verdict's
volume) has always used it, until March 2026, so writers that followed the
documentation wrote every wedge inside out. Read as polyhedra, such cells make the faces they share refused, so a
mesh with them needs `orient=True` there.
Higher-order cells read as `H1` fields of the grid's largest order, in any mix
of shapes, dimensions and orders:

| VTK cells | Read as |
|---|---|
| Lagrange (68-73: curve, triangle, quadrilateral, tetrahedron, hexahedron, wedge) | equispaced `H1` of their order |
| Bezier (75-80) | `H1(basis="bernstein")` |
| quadratic edge, triangle, tetrahedron (21, 22, 24) | `H1` of order 2 |
| quadratic quad, hexahedron, wedge (23, 25, 26: quad8, hex20, wedge15) | serendipity, or the full order-2 space beside other cells |
| biquadratic quad, triquadratic hexahedron, biquadratic-quadratic wedge (28, 29, 32) | `H1` of order 2 |
| quadratic-linear quad and wedge, biquadratic-quadratic hexahedron (30, 31, 33) | `H1` of order 2 (exact: their spaces lie in it) |
| biquadratic triangle (34), cubic line (35) | `H1` of order 3 (exact) |
| quadratic, triquadratic, Lagrange and Bezier pyramids (27, 37, 74, 81), quadratic polygons (36) | refused |

`dataset_to_vtk(..., higher_order=...)` writes Lagrange, Bezier or VTK's
classic quadratic cells.

**VTK cell grids** (`vtkCellGrid`, `.dg` files). These need a VTK newer than
9.7.1.

| Cell grid basis | Read as |
|---|---|
| `HGRAD C1`, shared | `H1` order 1 (copied) |
| `HGRAD C1`, per cell | `L2` order 1 (copied) |
| `constant` | `Constant` |
| `HGRAD C2`, `I2`, `A` (any order), `G` (Gauss points), Bezier | resampled by VTK onto Tack's equispaced space of that order: `H1` shared, `L2` per cell (exact) |
| `HCURL`, `HDIV`, and pyramids' bases other than `C1` | left out, with a warning |

`dataset_to_vtk(data, cell_grid=True)` writes `C1` at order 1 and VTK's `A`
basis above it.

**MFEM.** `mfem_to_dataset` reads H1 and L2 collections of any order (Gauss–Lobatto,
equispaced, Gauss–Legendre and positive bases), MFEM's older collections,
nonconforming (AMR) meshes, periodic meshes (an `L2` geometry), and curved
meshes. `mfem_grid_function` writes back.

**Nonconforming meshes and T-junctions.** A nonconforming mesh's hanging
vertices are points of the finer cells, and its fields come with every
value MFEM keeps per element, the constrained ones included, so the values on
the two sides of a T-junction agree: contours, slices, clips, probes,
gradients and subdivision are as on a conforming mesh. What Tack does not yet
know is that a coarse cell's edge or face and the finer ones covering it are
the same interface. `faces()` gives each of them one side, as if it were on
the boundary, so `boundary_faces`, `extract_surface`, `jump`, `upwind_flux`,
`net_flux` and other face-based algorithms treat T-junctions as boundary. On
MFEM's ex6 star after refinement, 1280 of the 1450 one-sided edges are
T-junctions. An octree's leaves (`OctreeDataSet.leaves()`) are the same.

See the [Data Model reference](../reference/data-model.md) for each function.
