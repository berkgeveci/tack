# Visualization examples

Sometimes a whole program teaches faster than a page of explanation. Each
program below answers one question, and the tables list them by that
question, so you can pick the one closest to your problem, run it, and change
it. Every program runs on the CPU and on the GPU backends, and checks its own
result.

Run any of the programs in `docs/examples/` from a checkout:

```bash
uv run python docs/examples/data_analysis.py --arch cpu --check
```

`--arch` takes `metal`, `cuda`, `hip` or `level_zero` too, and `--check` holds
the result to an analytic answer or to a built-in algorithm.

## Worked examples

Short programs with exact answers, independent of any simulation code. They
need NumPy and a Tack backend.

| Question | Program | Page |
|---|---|---|
| How much volume and area do threshold, clip and contour keep, and do sampled values match the field? | [data_analysis.py](../examples/data_analysis.py) | [Extract](extract.md) |
| How do point and cell fields project into each other, and is a gradient of an affine field exact? | [data_fields.py](../examples/data_fields.py) | [Fields](../data/fields.md) |
| How do I build `L2` and order-2 `H1` fields, and what is a DG jump? | [data_spaces.py](../examples/data_spaces.py) | [Higher order](../data/higher-order.md) |
| Do face fluxes balance, and does upwinding conserve mass on shaped and polyhedral cells? | [data_finite_volume.py](../examples/data_finite_volume.py) | [Finite-volume data and fluxes](finite-volume.md) |
| How do I write kernels over cells, faces and a space's nodes? | [vis3_mesh_algorithm.py](../examples/vis3_mesh_algorithm.py) | [Write a mesh algorithm](write-a-mesh-algorithm.md) |
| Does a face kernel's flux add up to the right divergence? | [data_mesh_kernels.py](../examples/data_mesh_kernels.py) | [Write a mesh algorithm](write-a-mesh-algorithm.md) |
| Can a filter written with public names give a built-in's result, every part of it? | [data_filters.py](../examples/data_filters.py) | [Write a filter](write-a-filter.md) |
| What do filters do to a dataset in pieces? | [data_partitioned.py](../examples/data_partitioned.py) | [Filters over pieces](over-pieces.md) |
| What do filters do with higher-order fields? | [data_high_order.py](../examples/data_high_order.py) | [Higher order](../data/higher-order.md) |
| How do I share a VTK image's memory with Tack and update it in a kernel? It needs a VTK built with DLPack support. | [vtk_interop.py](../examples/vtk_interop.py) | [Share arrays](../core/share-arrays.md) |

## Programs with VTK and MFEM

Longer programs in `packages/tack-vis/examples/`, each a whole workflow. The
VTK and MFEM ones need the `interop` extra.

| Question | Program |
|---|---|
| How do Tack's filters sit in a VTK pipeline, between a reader, a writer and a renderer? | `47_vtk_pipeline.py` |
| How does a `vtkPartitionedDataSet` go through ghost cells, a contour and a merge? | `48_vtk_pieces.py` |
| How do Tack's streamlines feed VTK's tubes and rendering? | `49_vtk_streamlines.py` |
| How does an Exodus file's collection go through Tack's contours and streamlines? | `50_vtk_exodus.py` |
| How does an OpenFOAM case's multiblock go through gradients, a slice and integrals over its patches? | `51_vtk_openfoam.py` |
| How do pathlines follow a time series, driving the pipeline and in situ? | `52_vtk_pathlines.py` |
| How do Tack's filters take an AMR file, the hierarchy whole and box by box? | `53_vtk_amr.py` |
| How do arrays pass between Tack and VTK without copies, Tack, then VTK, then Tack again? | `39_vtk_interop.py` |
| How does an MFEM solution on a curved, quadratic mesh come into Tack, and does it match MFEM? | `43_mfem_poisson.py` |
| What does the dataset API look like on one mixed grid of hexahedra, wedges, pyramids and tetrahedra? | `42_dataset_api_tour.py` |
| How are a dataset's pieces, ghost marks and reductions used together? | `44_partitioned_data.py` |
| How do gradients, vorticity, particles and streamlines work on one velocity field? | `45_flow.py` |
| How do implicit functions, clip, extract, slice and probe cut and sample a dataset? | `46_cut_and_sample.py` |
| How is a contour taken across the patches of an AMR hierarchy? | `31_multiblock_contour.py` |

## The gallery

The [Gallery](../vis-guide/gallery.md) shows what Tack's filters make of each
kind of cell: polyhedra, tetrahedra, order-4 fields on straight cells, and
order-3 fields on curved cells. Tack makes everything drawn, and VTK only
draws the polygons and lines. Each picture is a script in `docs/gallery/`.

| Question | Section |
|---|---|
| Does a contour follow a higher-order field inside its cells, and on polyhedra? | [Contour](../vis-guide/gallery.md#contour) |
| What does a slice through curved cells look like? | [Slice](../vis-guide/gallery.md#slice) |
| How does clip cut cells of each kind, and keep whole cells whole? | [Clip](../vis-guide/gallery.md#clip) |
| Which cells does threshold keep, and how does that differ from clip? | [Threshold](../vis-guide/gallery.md#threshold) |
| What are a dataset's surface and its feature edges? | [Surfaces and feature edges](../vis-guide/gallery.md#surfaces-and-feature-edges) |
| How good are a mesh's cells, by Verdict's measures? | [Cell quality](../vis-guide/gallery.md#cell-quality) |
| What does a gradient look like on each kind of cell? | [Gradients](../vis-guide/gallery.md#gradients) |
| How does probing sample a field at world positions? | [Probe](../vis-guide/gallery.md#probe) |
| How do curved and higher-order cells become linear cells for drawing? | [Subdivide](../vis-guide/gallery.md#subdivide) |
| How do pieces and their ghost cells fit together? | [Pieces and ghost cells](../vis-guide/gallery.md#pieces-and-ghost-cells) |
| How do streamlines, glyphs and pathlines show a flow? | [Flow](../vis-guide/gallery.md#flow) |

## The MFEM gallery

The [MFEM gallery](../vis-guide/mfem-gallery.md) solves MFEM's examples with
PyMFEM on MFEM's own meshes, and draws what Tack's filters make of them.

| Question | Section |
|---|---|
| Can Tack contour and slice a Poisson solution of order 3 on curved tetrahedra and hexahedra? | [Poisson on curved cells](../vis-guide/mfem-gallery.md#poisson-on-curved-cells) |
| How does a beam look warped by its displacement, in `H1` and in DG? | [Elasticity](../vis-guide/mfem-gallery.md#elasticity) |
| Does a nonconforming mesh from adaptive refinement draw without cracks, and a curved surface with its solution? | [Adaptive refinement and curved surfaces](../vis-guide/mfem-gallery.md#adaptive-refinement-and-curved-surfaces) |
| How does a DG field on a periodic mesh cross the seam? | [DG advection on a periodic mesh](../vis-guide/mfem-gallery.md#dg-advection-on-a-periodic-mesh) |

## Real data

Tack's filters have been run on production simulations' output, read with
VTK's readers, and checked against VTK's output: the ONERA M6 wing (46.9M
tetrahedra and wedges), an aneurysm's blood flow (8.7M tetrahedra), the LANL
wildfire (18.7M curvilinear cells), and 512 pieces of one simulation's output.
[Performance](../performance.md#real-data-against-vtk) gives each filter's
time against VTK, and how it was measured.
[Streamlines on the GPU](../performance.md#streamlines-on-the-gpu-against-viskores)
compares Tack's streamlines with Viskores' on the M6 wing and the aneurysm.

## Where to go next

- [Write a mesh algorithm](write-a-mesh-algorithm.md), to start from one of these
  programs and write your own kernels.
- [VTK](vtk.md), to read your own data and draw the results.
