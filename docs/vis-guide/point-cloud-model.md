# Point clouds and particles

A point cloud is a `DataSet` with one implicit vertex cell per point.
Positions and fields are real arrays; the one-vertex incidence needs no stored
connectivity. LiDAR returns, SPH particles, N-body particles and tracers can
all use this representation.

## Positions and per-point fields

`td.point_cloud(positions, fields=...)` assembles a cloud. Its fields associate
one value with each point. Points do not imply a surface or volume mesh
between them. Reconstructing either requires another choice.

The same [origins, units and timestamps](12-units.md) apply. For survey
coordinates the constructor can choose an f64 origin and keep nearby offsets
in f32. Per-point value width and storage remain independent of the coordinates.

## Neighborhoods are spatial queries

A point cloud has no mesh neighbors through shared edges. `PointLocator`
queries radius or nearest-distance neighborhoods. It is derived search
state, not new mesh incidence. After moving positions, rebuild the required
search state before relying on it.

Neighborhoods support outlier tests, thinning, normals and connected regions.
Radius and neighbor count define the operation. SPH additionally uses mass,
density, smoothing lengths and a kernel; it is not interpolation on vertex
cells. Periodic queries change distances, not field associations.

## Points in pieces

Survey tiles and simulation ranks can each hold subsets. Neighbor-dependent
work needs points from nearby pieces. [Ghost points](09-pieces-across.md#point-clouds-in-pieces)
supply them; ownership prevents duplicated summaries.

Continue with [Process Point Clouds and Particles](08-point-clouds.md) for
complete LiDAR, SPH and halo recipes, or [Sample and Trace Fields](06-pipelines.md)
when points are queries or seeds in another dataset's field.
