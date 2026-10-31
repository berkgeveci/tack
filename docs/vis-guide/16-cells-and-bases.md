# Cell types and bases

Every cell, every space and every basis Tack supports today, in one place, and
what is planned. [Function Space Fundamentals](00-function-spaces.md) explains
what a space and a basis are, and [Fields and Spaces](03-fields-and-spaces.md)
how to make and convert fields in them.

## Cell shapes

A topology's cells are shapes from `tack.dataset.shapes`. Most have a
*reference element*, a fixed parametric cell that every cell of the shape is a
map of; bases are defined on it. Tack's shapes, their corner orders and their
parametric coordinates are VTK's classic linear cells'.

| Shape | Dimension | Corners | Reference element | VTK type |
|---|---|---|---|---|
| `Vertex` | 0 | 1 | a point | `VTK_VERTEX` (1) |
| `Line` (segment) | 1 | 2 | `[0, 1]` | `VTK_LINE` (3) |
| `Triangle` | 2 | 3 | the unit triangle | `VTK_TRIANGLE` (5) |
| `Quad` | 2 | 4 | `[0, 1]²`, corners counterclockwise | `VTK_QUAD` (9) |
| `Pixel` | 2 | 4 | `[0, 1]²`, corners x fastest (an axis-aligned quad) | `VTK_PIXEL` (8) |
| `Tetra` | 3 | 4 | the unit tetrahedron | `VTK_TETRA` (10) |
| `Hexahedron` | 3 | 8 | `[0, 1]³` | `VTK_HEXAHEDRON` (12) |
| `Voxel` | 3 | 8 | `[0, 1]³`, corners x fastest | `VTK_VOXEL` (11) |
| `Wedge` | 3 | 6 | the unit triangle times `[0, 1]` | `VTK_WEDGE` (13) |
| `Pyramid` | 3 | 5 | `[0, 1]³` with its top collapsed to the apex | `VTK_PYRAMID` (14) |
| `PolyLine` | 1 | any | none | `VTK_POLY_LINE` (4) |
| `Polygon` | 2 | any | none | `VTK_POLYGON` (7) |
| `Polyhedron` | 3 | any | none | `VTK_POLYHEDRON` (42) |

The topologies that hold them:

| Topology | Its cells |
|---|---|
| `StructuredTopology` | a grid's lines, quads or hexahedra, numbered by index (images, rectilinear and curvilinear grids) |
| `UnstructuredTopology` | any mix of the shapes with a reference element, of any dimensions, in one topology |
| `PolylineTopology` | polylines (lines are two-point polylines) |
| `PolygonalTopology` | polygons (triangles and quads are polygons too) |
| `PolyhedralTopology` | polyhedra, each a list of faces |

Polylines, polygons and polyhedra have no reference element. A field on them is
linear on pieces: on a polyline's segments, on the triangles that fan a polygon
from its points' mean, and on the sub-tetrahedra that fan a polyhedron's faces
from their means and the cell's. So linear fields are exact there, and
`integrate`, `probe`, `gradients` and the cutting filters all work on these
pieces.

## Spaces

| Space | Values | Basis |
|---|---|---|
| `H1(data, order=p)` | shared between the cells around each node | a continuous polynomial of order `p` in each cell |
| `L2(data, order=p)` | each cell's own | a polynomial of order `p` in each cell, discontinuous between cells |
| `Constant(data)` | one per cell | none: the value everywhere in the cell |
| `Values(data, on)` | one per point, cell, face or edge | none: data attached to entities |
| `SideTraces(data)` | per face, per side, per face corner | two-sided face values |

## What each shape supports

**yes**: supported; **—**: refused, with a message saying why.

| Shape | `H1` order 1 | `H1` order 2 | `H1` order 3 and up | `L2` order 0 | `L2` order 1 | `L2` order 2 and up | Curved geometry |
|---|---|---|---|---|---|---|---|
| Vertex | yes | — | — | yes | yes | — | — |
| Line | yes | yes | yes | yes | yes | yes | yes |
| Triangle | yes | yes | yes | yes | yes | yes | yes |
| Quad, Pixel | yes | yes | yes | yes | yes | yes | yes |
| Tetra | yes | yes | yes | yes | yes | yes | yes |
| Hexahedron, Voxel | yes | yes | yes | yes | yes | yes | yes |
| Wedge | yes | yes | yes | yes | yes | yes | yes |
| Pyramid | yes | — | — | yes | yes | — | — |
| PolyLine, Polygon, Polyhedron | yes (linear on pieces) | — | — | `Constant` | — | — | — |

There is no fixed upper order (the tests go to order 6). An `L2` order may
also vary per cell (orders 0 and 1, `L2(data, order=array)`).

A **curved geometry** is a geometry field above order 1, in `H1`, or in `L2` for
cells that do not meet (a mesh displaced by a DG field, a periodic mesh). Cells
are then located, probed, measured, integrated, cut and drawn as they are, not
as their corners make them. A field's order may differ from the geometry's:
its nodes are placed where the curved geometry puts them, bit for bit the same
in every piece.

## Bases

A space above order 1 names its **basis**, the family of functions its values
are the coefficients of:

| Basis | What its values are | Spaces | Shapes |
|---|---|---|---|
| `"gauss_lobatto"` | values at Gauss–Lobatto nodes (Lagrange); MFEM's default | `H1`, `L2` | lines, quads, pixels, hexahedra, voxels |
| `"equispaced"` | values at equispaced nodes (Lagrange); VTK's Lagrange cells and gmsh's | `H1`, `L2` | every shape with a basis |
| `"gauss_legendre"` | values at Gauss–Legendre nodes, inside the cell (Lagrange); MFEM's default `L2` | `L2`, from order 1 | lines, quads, pixels, hexahedra, voxels |
| `"bernstein"` | Bernstein coefficients (VTK's Bezier cells, MFEM's positive bases) | `H1`, `L2`, from order 2 | every shape with a basis |
| `"serendipity"` | order 2 without face centres and interiors: quad8, hex20, wedge15 (the full space on triangles and tetrahedra) | `H1`, order 2 | triangles, quads, pixels, tetrahedra, hexahedra, voxels, wedges |

When no basis is given above order 2, it is equispaced where the mesh has
triangles, tetrahedra or wedges, which have no other nodes, and Gauss–Lobatto
otherwise. At order 2 and below, Gauss–Lobatto and equispaced nodes are the
same.

Two layouts and one numbering go with a space:

- **`L2`'s layout:** a tensor basis' values are stored either each cell's
  together (`layout="blocks"`, MFEM's and VTK cell grids' layout) or
  cell-fastest (`layout="cells"`, faster on GPUs from order 3). `"auto"`
  chooses by device.
- **Element-to-DOF tables:** `H1(data, order=p, dofs=table)` numbers values by
  a row per cell, as MFEM and many solvers do, instead of by entity.

## Mixing shapes

An `UnstructuredTopology` may mix every shape with a reference element, and
every space above applies to the mix: a mesh of hexahedra and wedges holds one
`H1` field of order 3. The filters that cut cells into
linear sub-cells (`subdivide`, `contour`, `slice`, `clip`, `threshold`) take any
mix of one dimension. Each family is refined alone, and the parts are joined
where they share lattice points. On such a mix, crossings are placed on the
linear sub-cells; exact crossings need cells of one family.

## Reading and writing other formats

**VTK unstructured grids.** Linear cells read as the shapes above.
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

See the [Data Model reference](reference/data-model.md) for each function.

## Planned

For 0.8:

- **T-junctions as interfaces:** `faces()` pairing a coarse face with the
  finer faces that cover it, from MFEM's nonconforming mesh or an octree's
  leaves, so face-based algorithms see their two sides.

- **Higher-order pyramids:** VTK's quadratic (13 nodes), triquadratic (19),
  Lagrange and Bezier pyramids, and cell grids' `C2` and `pyramid19`. Their
  bases are rational, so they come with:
- **Rational Bezier cells:** VTK's Bezier cells with weights other than 1.
- **`H(curl)` and `H(div)`** at order 1, with Piola maps, on every shape but
  the pyramid. These are cell grids' `HCURL` and `HDIV`, now left out.
- **More bases natively:**
  - the Gauss-point bases on triangles, tetrahedra and wedges (now resampled
    into equispaced ones);
  - serendipity per cell (`L2`);
  - the enriched `F2` cells (15-node tetrahedron, 21-node wedge);
  - modal bases;
  - an order per axis, as VTK's quadratic-linear cells have (now embedded in
    the full order).
- **Adaptive subdivision** for the cutting filters, per cell from the cut's
  error.

Not planned: higher-order polygons and polyhedra (VTK's quadratic polygon among
them).
