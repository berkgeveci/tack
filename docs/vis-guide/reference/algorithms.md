# Algorithms

Every algorithm of `tack.dataset`, by what it does, with its parameters. Each
group opens with a table: what each algorithm gives, what it takes (a
`DataSet`; pieces, a `PartitionedDataSet`, over MPI ranks or not; an
`AMRDataSet`; an `OctreeDataSet`), and its nearest counterpart in VTK and in
Viskores. Below the table, each algorithm's signature and description.

```python
import tack.dataset as td

hot = td.threshold(data, "temperature", 300.0, 400.0)
grad = td.gradients(data, "pressure", at="points")
s = td.statistics(data, "pressure")
```

Every algorithm is in `tack.dataset.algorithms`, one module each, and
`tack.dataset` re-exports them all. They share conventions, set out in the
Dataset API reference's [Algorithms and filters](dataset-api.md#algorithms-and-filters):

- the dataset comes first, and a field argument is a `Field` or the name of one
  of the dataset's fields;
- an algorithm that makes a new dataset takes `fields=`: which fields to carry
  (all by default, a name, a list of names, or `[]` for none), carried by the
  rules in [how fields carry through filters](dataset-api.md#algorithms-and-filters);
- an algorithm that computes a field returns a `Field`, several fields a named
  tuple of them, a summary Python numbers, a selection a `Set`;
- nothing changes its input, except where an entry says "in place";
- every algorithm runs, and allocates, on its dataset's device; algorithms over
  pieces run each piece on its own device;
- hidden cells are never processed, and ghost cells and points are left out of
  summaries;
- a parameter that is a length, a time or a field's value (a radius, a step
  size, a threshold's range) also takes a quantity (`"10 cm"`, `(10, "cm")`),
  converted to the dataset's or the field's unit; a plain number is taken as
  it is ([Units](dataset-api.md#units)).

Which algorithms work on which topologies (polyhedra, polygons, polylines) is
in the Dataset API reference's [support table](dataset-api.md#algorithms-and-filters).
The classes the algorithms take and give -- datasets, fields, spaces,
topologies, sets, implicit functions, locators -- are in
[Data Model](data-model.md).

## Contents

- [Cells kept whole](#cells-kept-whole): New datasets of some of the input's cells, unchanged, with only the points they use.
- [Cutting](#cutting): New datasets of pieces of cells, with points made on their edges.
- [Surfaces and edges](#surfaces-and-edges): A mesh's faces and edges, as sets or datasets on the same points.
- [Connectivity and winding](#connectivity-and-winding): Which cells belong together, and which are wound wrong.
- [Changing cell types](#changing-cell-types): The same points and region, in other cells.
- [Moving the geometry](#moving-the-geometry): New positions for the same topology.
- [Measures and quality](#measures-and-quality): Fields describing the mesh itself.
- [Fields from fields](#fields-from-fields): New fields computed from a dataset's fields.
- [Finite volumes](#finite-volumes): Values on faces and the cells around them.
- [Summaries](#summaries): Python numbers about a field, over every piece and rank.
- [Probing, sampling and particles](#probing-sampling-and-particles): Values at other points, and paths through a velocity field.
- [Point clouds](#point-clouds): Points as vertex cells: neighbours found through a `PointLocator`.
- [Pieces and ghosts](#pieces-and-ghosts): A `PartitionedDataSet`'s pieces, over MPI ranks or not.
- [Adaptive mesh refinement](#adaptive-mesh-refinement): Algorithms only an `AMRDataSet` takes; `ghost_cells`, `probe`, `statistics` and the rest above take one too.

## Cells kept whole

New datasets of some of the input's cells, unchanged, with only the points they use.

| Algorithm | Gives | Takes | VTK | Viskores |
|---|---|---|---|---|
| [`threshold`](#tack.dataset.algorithms.threshold.threshold) | cells whose field lies in a range | dataset | vtkThreshold | Threshold |
| [`extract_cells`](#tack.dataset.algorithms.extract_cells.extract_cells) | cells by id, array or cell set | dataset | vtkExtractCells | — |
| [`extract_geometry`](#tack.dataset.algorithms.extract_geometry.extract_geometry) | cells inside (or outside, or cut by) an implicit function | dataset | vtkExtractGeometry | ExtractGeometry |
| [`extract_extent`](#tack.dataset.algorithms.extract_extent.extract_extent) | a box of a structured grid, every n-th point | dataset | vtkExtractGrid | ExtractStructured |
| [`mask`](#tack.dataset.algorithms.mask.mask) | every n-th cell | dataset | — | Mask |

::: tack.dataset.algorithms.threshold.threshold
    options: {show_root_full_path: false}

::: tack.dataset.algorithms.extract_cells.extract_cells
    options: {show_root_full_path: false}

::: tack.dataset.algorithms.extract_geometry.extract_geometry
    options: {show_root_full_path: false}

::: tack.dataset.algorithms.extract_extent.extract_extent
    options: {show_root_full_path: false}

::: tack.dataset.algorithms.mask.mask
    options: {show_root_full_path: false}

## Cutting

New datasets of pieces of cells, with points made on their edges.

| Algorithm | Gives | Takes | VTK | Viskores |
|---|---|---|---|---|
| [`contour`](#tack.dataset.algorithms.contour.contour) | isosurfaces, isolines or points of a field, one level or several | dataset | vtkContourFilter | Contour |
| [`slice`](#tack.dataset.algorithms.slice.slice) | where an implicit function is zero (or a level) | dataset | vtkCutter | Slice |
| [`slice_plane`](#tack.dataset.algorithms.slice.slice_plane) | a plane's cut, or parallel planes | dataset | vtkCutter | Slice |
| [`clip`](#tack.dataset.algorithms.clip.clip) | the part above (or below) a field's level or inside an implicit function | dataset | vtkTableBasedClipDataSet | ClipWithField, ClipWithImplicitFunction |

::: tack.dataset.algorithms.contour.contour
    options: {show_root_full_path: false}

::: tack.dataset.algorithms.slice.slice
    options: {show_root_full_path: false}

::: tack.dataset.algorithms.slice.slice_plane
    options: {show_root_full_path: false}

::: tack.dataset.algorithms.clip.clip
    options: {show_root_full_path: false}

## Surfaces and edges

A mesh's faces and edges, as sets or datasets on the same points.

| Algorithm | Gives | Takes | VTK | Viskores |
|---|---|---|---|---|
| [`boundary_faces`](#tack.dataset.algorithms.boundary_faces.boundary_faces) | the boundary, as a face set | dataset | — | — |
| [`extract_surface`](#tack.dataset.algorithms.extract_surface.extract_surface) | faces (the boundary by default) as cells | dataset | vtkDataSetSurfaceFilter | ExternalFaces |
| [`extract_edges`](#tack.dataset.algorithms.extract_edges.extract_edges) | edges as line cells | dataset | vtkExtractEdges | — |
| [`feature_edges`](#tack.dataset.algorithms.feature_edges.feature_edges) | a surface's boundary, feature, non-manifold and manifold edges | dataset | vtkFeatureEdges | — |
| [`surface_normals`](#tack.dataset.algorithms.surface_normals.surface_normals) | cell and point normals of a surface, wound consistently | dataset | vtkPolyDataNormals | SurfaceNormals |

::: tack.dataset.algorithms.boundary_faces.boundary_faces
    options: {show_root_full_path: false}

::: tack.dataset.algorithms.extract_surface.extract_surface
    options: {show_root_full_path: false}

::: tack.dataset.algorithms.extract_edges.extract_edges
    options: {show_root_full_path: false}

::: tack.dataset.algorithms.feature_edges.feature_edges
    options: {show_root_full_path: false}

::: tack.dataset.algorithms.feature_edges.EdgeKind
    options: {show_root_full_path: false}

::: tack.dataset.algorithms.surface_normals.surface_normals
    options: {show_root_full_path: false}

## Connectivity and winding

Which cells belong together, and which are wound wrong.

| Algorithm | Gives | Takes | VTK | Viskores |
|---|---|---|---|---|
| [`connected_components`](#tack.dataset.algorithms.connected_components.connected_components) | regions of cells connected through points (or within a radius) | dataset, pieces | vtkConnectivityFilter, vtkEuclideanClusterExtraction | CellSetConnectivity |
| [`check_winding`](#tack.dataset.algorithms.check_winding.check_winding) | cells whose faces do not wind consistently, as a cell set | dataset | — | — |

::: tack.dataset.algorithms.connected_components.connected_components
    options: {show_root_full_path: false}

::: tack.dataset.algorithms.check_winding.check_winding
    options: {show_root_full_path: false}

## Changing cell types

The same points and region, in other cells.

| Algorithm | Gives | Takes | VTK | Viskores |
|---|---|---|---|---|
| [`tetrahedralize`](#tack.dataset.algorithms.tetrahedralize.tetrahedralize) | 3D cells as tetrahedra that meet face to face | dataset | vtkDataSetTriangleFilter | Tetrahedralize |
| [`triangulate`](#tack.dataset.algorithms.triangulate.triangulate) | 2D cells as triangles | dataset | vtkTriangleFilter | Triangulate |
| [`as_polyhedra`](#tack.dataset.algorithms.as_polyhedra.as_polyhedra) | a shape-based mesh as a polyhedral one | dataset | — | — |
| [`as_polygons`](#tack.dataset.algorithms.as_polygons.as_polygons) | 2D cells as a polygonal mesh | dataset | — | — |
| [`shrink`](#tack.dataset.algorithms.shrink.shrink) | every cell shrunk toward its centroid | dataset | vtkShrinkFilter | Shrink |

::: tack.dataset.algorithms.tetrahedralize.tetrahedralize
    options: {show_root_full_path: false}

::: tack.dataset.algorithms.triangulate.triangulate
    options: {show_root_full_path: false}

::: tack.dataset.algorithms.as_polyhedra.as_polyhedra
    options: {show_root_full_path: false}

::: tack.dataset.algorithms.as_polygons.as_polygons
    options: {show_root_full_path: false}

::: tack.dataset.algorithms.shrink.shrink
    options: {show_root_full_path: false}

## Moving the geometry

New positions for the same topology.

| Algorithm | Gives | Takes | VTK | Viskores |
|---|---|---|---|---|
| [`transform`](#tack.dataset.algorithms.transform.transform) | an affine map (4x4 or 3x4 matrix) | dataset | vtkTransformFilter | PointTransform |
| [`warp`](#tack.dataset.algorithms.warp.warp) | positions moved along a vector field or vector | dataset | vtkWarpVector, vtkWarpScalar | Warp |
| [`cylindrical`](#tack.dataset.algorithms.cylindrical.cylindrical) | Cartesian to cylindrical coordinates, or back | dataset | — | CylindricalCoordinateTransform |
| [`spherical`](#tack.dataset.algorithms.spherical.spherical) | Cartesian to spherical coordinates, or back | dataset | — | SphericalCoordinateTransform |

::: tack.dataset.algorithms.transform.transform
    options: {show_root_full_path: false}

::: tack.dataset.algorithms.warp.warp
    options: {show_root_full_path: false}

::: tack.dataset.algorithms.cylindrical.cylindrical
    options: {show_root_full_path: false}

::: tack.dataset.algorithms.spherical.spherical
    options: {show_root_full_path: false}

## Measures and quality

Fields describing the mesh itself.

| Algorithm | Gives | Takes | VTK | Viskores |
|---|---|---|---|---|
| [`cell_geometry`](#tack.dataset.algorithms.cell_geometry.cell_geometry) | each cell's volume (area, length) and centroid | dataset | vtkCellSizeFilter | CellMeasures |
| [`face_geometry`](#tack.dataset.algorithms.face_geometry.face_geometry) | each face's normal and area | dataset | — | — |
| [`cell_centers`](#tack.dataset.algorithms.cell_centers.cell_centers) | each cell's parametric center (or cell point) | dataset | vtkCellCenters | — |
| [`face_centers`](#tack.dataset.algorithms.face_centers.face_centers) | each face's area centroid | dataset | — | — |
| [`edge_lengths`](#tack.dataset.algorithms.edge_lengths.edge_lengths) | each edge's length | dataset | — | — |
| [`cell_quality`](#tack.dataset.algorithms.cell_quality.cell_quality) | scaled Jacobian, Jacobian ratio, condition number, edge ratio, edge lengths | dataset | vtkMeshQuality | MeshQuality |
| [`point_elevation`](#tack.dataset.algorithms.point_elevation.point_elevation) | each position along a line, mapped to a range | dataset | vtkElevationFilter | PointElevation |
| [`implicit_values`](#tack.dataset.algorithms.implicit_values.implicit_values) | an implicit function at the geometry's values | dataset | vtkSampleImplicitFunctionFilter | — |
| [`cell_ids`](#tack.dataset.algorithms.cell_ids.cell_ids) | each cell's id (provenance through filters) | dataset | vtkGenerateIds | — |
| [`point_ids`](#tack.dataset.algorithms.point_ids.point_ids) | each point's id (provenance through filters) | dataset | vtkGenerateIds | — |

::: tack.dataset.algorithms.cell_geometry.cell_geometry
    options: {show_root_full_path: false}

::: tack.dataset.algorithms.cell_geometry.CellGeometry
    options: {show_root_full_path: false}

::: tack.dataset.algorithms.face_geometry.face_geometry
    options: {show_root_full_path: false}

::: tack.dataset.algorithms.face_geometry.FaceGeometry
    options: {show_root_full_path: false}

::: tack.dataset.algorithms.cell_centers.cell_centers
    options: {show_root_full_path: false}

::: tack.dataset.algorithms.face_centers.face_centers
    options: {show_root_full_path: false}

::: tack.dataset.algorithms.edge_lengths.edge_lengths
    options: {show_root_full_path: false}

::: tack.dataset.algorithms.cell_quality.cell_quality
    options: {show_root_full_path: false}

::: tack.dataset.algorithms.point_elevation.point_elevation
    options: {show_root_full_path: false}

::: tack.dataset.algorithms.implicit_values.implicit_values
    options: {show_root_full_path: false}

::: tack.dataset.algorithms.cell_ids.cell_ids
    options: {show_root_full_path: false}

::: tack.dataset.algorithms.point_ids.point_ids
    options: {show_root_full_path: false}

## Fields from fields

New fields computed from a dataset's fields.

| Algorithm | Gives | Takes | VTK | Viskores |
|---|---|---|---|---|
| [`gradients`](#tack.dataset.algorithms.gradients.gradients) | the gradient of a scalar or vector field, at cells or points | dataset | vtkGradientFilter | Gradient |
| [`flow_quantities`](#tack.dataset.algorithms.flow_quantities.flow_quantities) | divergence, vorticity and Q-criterion from a velocity gradient | dataset | vtkGradientFilter | Gradient |
| [`project`](#tack.dataset.algorithms.project.project) | a field moved into another space (cells to points, points to cells, ...) | dataset | vtkCellDataToPointData, vtkPointDataToCellData | PointAverage, CellAverage |
| [`values_at_centers`](#tack.dataset.algorithms.values_at_centers.values_at_centers) | a field at each cell's parametric center | dataset | — | — |
| [`interpolate`](#tack.dataset.algorithms.interpolate.interpolate) | a `@tack.func` of position at a space's nodes | dataset | — | — |
| [`magnitude`](#tack.dataset.algorithms.magnitude.magnitude) | each vector's length | dataset | — | VectorMagnitude |
| [`dot`](#tack.dataset.algorithms.dot.dot) | the dot product of two vector fields | dataset | — | DotProduct |
| [`cross`](#tack.dataset.algorithms.cross.cross) | the cross product of two 3-vector fields | dataset | — | CrossProduct |
| [`composite`](#tack.dataset.algorithms.composite.composite) | scalar fields as the components of a vector field | dataset | — | CompositeVectors |
| [`log_values`](#tack.dataset.algorithms.log_values.log_values) | the logarithm of a field, in any base | dataset | — | LogValues |
| [`convert`](#tack.dataset.algorithms.convert.convert) | a field, or the geometry, in another unit | dataset | — | — |

::: tack.dataset.algorithms.gradients.gradients
    options: {show_root_full_path: false}

::: tack.dataset.algorithms.flow_quantities.flow_quantities
    options: {show_root_full_path: false}

::: tack.dataset.algorithms.flow_quantities.FlowQuantities
    options: {show_root_full_path: false}

::: tack.dataset.algorithms.project.project
    options: {show_root_full_path: false}

::: tack.dataset.algorithms.values_at_centers.values_at_centers
    options: {show_root_full_path: false}

::: tack.dataset.algorithms.interpolate.interpolate
    options: {show_root_full_path: false}

::: tack.dataset.algorithms.magnitude.magnitude
    options: {show_root_full_path: false}

::: tack.dataset.algorithms.dot.dot
    options: {show_root_full_path: false}

::: tack.dataset.algorithms.cross.cross
    options: {show_root_full_path: false}

::: tack.dataset.algorithms.composite.composite
    options: {show_root_full_path: false}

::: tack.dataset.algorithms.log_values.log_values
    options: {show_root_full_path: false}

::: tack.dataset.algorithms.convert.convert
    options: {show_root_full_path: false}

## Finite volumes

Values on faces and the cells around them.

| Algorithm | Gives | Takes | VTK | Viskores |
|---|---|---|---|---|
| [`jump`](#tack.dataset.algorithms.jump.jump) | a field's difference across each face | dataset | — | — |
| [`upwind_flux`](#tack.dataset.algorithms.upwind_flux.upwind_flux) | a field's upwind flux through each face | dataset | — | — |
| [`net_flux`](#tack.dataset.algorithms.net_flux.net_flux) | each cell's outward sum of face fluxes | dataset | — | — |
| [`divergence`](#tack.dataset.algorithms.divergence.divergence) | each cell's finite-volume divergence of face fluxes | dataset | — | — |
| [`perot`](#tack.dataset.algorithms.perot.perot) | cell vectors from face-normal components (C-grid models) | dataset | — | — |

::: tack.dataset.algorithms.jump.jump
    options: {show_root_full_path: false}

::: tack.dataset.algorithms.upwind_flux.upwind_flux
    options: {show_root_full_path: false}

::: tack.dataset.algorithms.net_flux.net_flux
    options: {show_root_full_path: false}

::: tack.dataset.algorithms.divergence.divergence
    options: {show_root_full_path: false}

::: tack.dataset.algorithms.perot.perot
    options: {show_root_full_path: false}

## Summaries

Python numbers about a field, over every piece and rank.

| Algorithm | Gives | Takes | VTK | Viskores |
|---|---|---|---|---|
| [`statistics`](#tack.dataset.algorithms.statistics.statistics) | count, extrema, mean, moments, variance, skewness, kurtosis; weighted | dataset, pieces, AMR, octree | vtkDescriptiveStatistics | Statistics |
| [`entropy`](#tack.dataset.algorithms.entropy.entropy) | Shannon entropy of a field's histogram | dataset, pieces, AMR, octree | — | Entropy |

::: tack.dataset.algorithms.statistics.statistics
    options: {show_root_full_path: false}

::: tack.dataset.algorithms.statistics.Statistics
    options: {show_root_full_path: false}

::: tack.dataset.algorithms.entropy.entropy
    options: {show_root_full_path: false}

## Probing, sampling and particles

Values at other points, and paths through a velocity field.

| Algorithm | Gives | Takes | VTK | Viskores |
|---|---|---|---|---|
| [`probe`](#tack.dataset.algorithms.probe.probe) | fields at given points or another dataset's points | dataset, pieces, AMR, octree | vtkProbeFilter, vtkAMRResampleFilter, vtkHyperTreeGridProbeFilter | Probe |
| [`voxelize`](#tack.dataset.algorithms.voxelize.voxelize) | the share of each voxel the data covers, and fields' means over it | dataset, pieces, AMR, octree | — | — |
| [`advect`](#tack.dataset.algorithms.advect.advect) | particles moved through a velocity field by fixed steps | dataset, pieces | — | ParticleAdvection |
| [`streamlines`](#tack.dataset.algorithms.streamlines.streamlines) | the particles' paths, as polylines | dataset, pieces | vtkStreamTracer | Streamline |

::: tack.dataset.algorithms.probe.probe
    options: {show_root_full_path: false}

::: tack.dataset.algorithms.voxelize.voxelize
    options: {show_root_full_path: false}

::: tack.dataset.algorithms.advect.advect
    options: {show_root_full_path: false}

::: tack.dataset.algorithms._advection.ParticleStatus
    options: {show_root_full_path: false}

::: tack.dataset.algorithms.streamlines.streamlines
    options: {show_root_full_path: false}

## Point clouds

Points as vertex cells: neighbours found through a `PointLocator`.

| Algorithm | Gives | Takes | VTK | Viskores |
|---|---|---|---|---|
| [`extract_points`](#tack.dataset.algorithms.extract_points.extract_points) | points inside an implicit function | dataset | vtkExtractPoints | ExtractPoints |
| [`threshold_points`](#tack.dataset.algorithms.threshold_points.threshold_points) | points whose field lies in a range | dataset | vtkThresholdPoints | ThresholdPoints |
| [`mask_points`](#tack.dataset.algorithms.mask_points.mask_points) | every n-th point | dataset | vtkMaskPoints | MaskPoints |
| [`point_normals`](#tack.dataset.algorithms.point_normals.point_normals) | normals from each point's neighbourhood (PCA) | dataset | vtkPCANormalEstimation | — |
| [`outliers`](#tack.dataset.algorithms.outliers.outliers) | points marked by a radius or a statistical test | dataset, pieces | vtkRadiusOutlierRemoval, vtkStatisticalOutlierRemoval | — |
| [`voxel_downsample`](#tack.dataset.algorithms.voxel_downsample.voxel_downsample) | one point per occupied voxel | dataset, pieces | vtkVoxelGrid | — |
| [`deposit`](#tack.dataset.algorithms.deposit.deposit) | points counted, summed or averaged onto an image (NGP, CIC) | dataset, pieces | — | ParticleDensityNearestGridPoint, ParticleDensityCloudInCell |
| [`sph_density`](#tack.dataset.algorithms.sph.sph_density) | SPH densities of particles | dataset | — | — |
| [`sph_interpolate`](#tack.dataset.algorithms.sph.sph_interpolate) | particles' fields at other points by SPH kernel sums | dataset | vtkSPHInterpolator | — |
| [`halo_catalog`](#tack.dataset.algorithms.halo_catalog.halo_catalog) | a halo catalogue from friends-of-friends labels | dataset, pieces | — | cosmotools |

::: tack.dataset.algorithms.extract_points.extract_points
    options: {show_root_full_path: false}

::: tack.dataset.algorithms.threshold_points.threshold_points
    options: {show_root_full_path: false}

::: tack.dataset.algorithms.mask_points.mask_points
    options: {show_root_full_path: false}

::: tack.dataset.algorithms.point_normals.point_normals
    options: {show_root_full_path: false}

::: tack.dataset.algorithms.outliers.outliers
    options: {show_root_full_path: false}

::: tack.dataset.algorithms.voxel_downsample.voxel_downsample
    options: {show_root_full_path: false}

::: tack.dataset.algorithms.deposit.deposit
    options: {show_root_full_path: false}

::: tack.dataset.algorithms.sph.sph_density
    options: {show_root_full_path: false}

::: tack.dataset.algorithms.sph.sph_interpolate
    options: {show_root_full_path: false}

::: tack.dataset.algorithms.halo_catalog.halo_catalog
    options: {show_root_full_path: false}

## Pieces and ghosts

A `PartitionedDataSet`'s pieces, over MPI ranks or not.

| Algorithm | Gives | Takes | VTK | Viskores |
|---|---|---|---|---|
| [`split`](#tack.dataset.algorithms.split.split) | a structured grid cut into pieces, with ghost layers | structured dataset | — | — |
| [`ghost_cells`](#tack.dataset.algorithms.ghost_cells.ghost_cells) | pieces with layers of their neighbours' cells | pieces, AMR | vtkGhostCellsGenerator | — |
| [`ghost_points`](#tack.dataset.algorithms.ghost_points.ghost_points) | point-cloud pieces with copies of their neighbours' points | pieces | — | — |
| [`update_ghosts`](#tack.dataset.algorithms.ghost_cells.update_ghosts) | ghost values refreshed from their owners, in place | pieces, AMR | — | — |
| [`remove_ghosts`](#tack.dataset.algorithms.remove_ghosts.remove_ghosts) | a piece without its ghost cells | dataset (a piece) | vtkRemoveGhosts | GhostCellRemove |
| [`generate_global_ids`](#tack.dataset.algorithms.generate_global_ids.generate_global_ids) | points and cells numbered across every piece | pieces | vtkGenerateGlobalIds | — |
| [`merge_pieces`](#tack.dataset.algorithms.merge_pieces.merge_pieces) | the pieces as one dataset, on one rank | pieces | vtkMergeBlocks | MergeDataSets |

::: tack.dataset.algorithms.split.split
    options: {show_root_full_path: false}

::: tack.dataset.algorithms.ghost_cells.ghost_cells
    options: {show_root_full_path: false}

::: tack.dataset.algorithms.ghost_points.ghost_points
    options: {show_root_full_path: false}

::: tack.dataset.algorithms.ghost_cells.update_ghosts
    options: {show_root_full_path: false}

::: tack.dataset.algorithms.remove_ghosts.remove_ghosts
    options: {show_root_full_path: false}

::: tack.dataset.algorithms.generate_global_ids.generate_global_ids
    options: {show_root_full_path: false}

::: tack.dataset.algorithms.merge_pieces.merge_pieces
    options: {show_root_full_path: false}

## Adaptive mesh refinement

Algorithms only an `AMRDataSet` takes; `ghost_cells`, `probe`, `statistics` and the rest above take one too.

| Algorithm | Gives | Takes | VTK | Viskores |
|---|---|---|---|---|
| [`average_down`](#tack.dataset.algorithms.amr_ghosts.average_down) | finer levels' data averaged into the coarse cells they cover, in place | AMR | — | — |

::: tack.dataset.algorithms.amr_ghosts.average_down
    options: {show_root_full_path: false}
