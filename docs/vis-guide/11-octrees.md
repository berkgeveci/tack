# Octrees

Tree-based AMR codes -- RAMSES and its kind, and CEA's codes, which ParaView
reads as VTK's HyperTreeGrid -- refine cell by cell: each cell is split in two
along each axis wherever more resolution is needed, into a tree of cells. An
`OctreeDataSet` holds a grid of such trees, laid out for GPUs as a *linear
octree*: every node is three numbers -- its tree, its level, and its Morton
key, its integer coordinates in the tree interleaved bit by bit -- and the
nodes are sorted so each subtree is one contiguous run. Whether a node is a
leaf, and its position and size, come from the keys; nothing about position is
stored.

This first pass holds cell data (one value per node, refined ones included, as
the codes store them) and is about **statistics**. Nodes that are not leaves are
`REFINEDCELL`, so reductions count each region once, over the leaves. The
leaves are also a dataset of cells, for every filter, and an octree can be
probed.


## Roots, nodes and leaves

The root grid supplies the domain and geometry. A node identifies a region
at one refinement level. A refined parent and its children can all have
stored values, but they do not represent disjoint volume. A leaf is a node
with no children; default reductions use leaves and skip hidden values.

```mermaid
flowchart LR
    R[Root grid] --> N[Tree nodes: level and Morton key]
    N --> P[Refined parent: excluded from leaf reductions]
    N --> L[Leaves: current physical resolution]
    L --> D[leaves(): DataSet of voxels or pixels]
```

Unlike [block AMR](10-amr.md), refinement is encoded by tree nodes rather than
overlapping levels of boxes. The tree representation and its leaf mesh offer
different access paths to the same leaf quantities.

## From codes and VTK

- `OctreeDataSet.from_cells(roots, level, position, fields)` takes each cell's
  level and a position in it, as a RAMSES reader or yt gives them (whole octs),
  over a grid of root cells (an image or a rectilinear grid); it adds the
  ancestors no cell gives, their values NaN.
- `vtk_to_dataset` takes a `vtkHyperTreeGrid` (branching factor 2) and
  `dataset_to_vtk` gives one back, a mask as `HIDDENCELL`.

## Leaves as cells, and probes

`octree.leaves()` is the leaves as an unstructured `DataSet` of voxels (pixels
in 2D) with every field as cell data: cell `c` is the `c`-th leaf in the
nodes' order (`np.flatnonzero(octree.is_leaf)[c]`), and hidden leaves are
`HIDDENCELL`. Corners that coincide are one point, found by sorting their
indices on the finest level's lattice, so the cells are connected; corners on
a finer neighbour's face stay hanging, as the tree has them. Every filter that
reads cell data works on it: `threshold`, `cell_centers`, `extract_cells`,
`probe`.

`probe(octree, points)` gives each point the values of the leaf holding it,
found by one binary search over the keys per point: the point's key on the
finest level is compared with each node's key carried there, and in pre-order
the last node at or below it is the leaf. Values are constant over a leaf, as
VTK's `vtkHyperTreeGridProbeFilter` gives them; `valid` is 0 outside, on
hidden leaves and off a 2D octree's plane. A probe onto an image resamples the
octree. Over MPI ranks, each point goes to the rank whose trees hold it.


## Interpreting summaries

A sample mean gives a small leaf the same weight as a large leaf. A
volume-weighted mean answers a different physical question. A density's
volume-weighted sum is mass. The [adaptive-data recipe](adaptive-data.md#weights)
checks both interpretations and shows how to select a level explicitly.

## Not here yet

Point data, coarse values from the leaves (HyperTreeGrid's `EvaluateCoarse`),
and contours (the dual grid) come later.


Continue with [Analyze Adaptive Data](adaptive-data.md#construct-and-summarize-an-octree)
for construction, probes and leaf conversion.

??? info "Related detailed sections"
    These topics are developed in their own chapters.

    <span id="weights" data-guide-forward="../adaptive-data/#weights"></span>
    [Weights](adaptive-data.md#weights).
