# Worked Examples

These programs develop the dataset API with analytic answers, independently of
a simulation solver. They run with NumPy and a Tack backend; figures are optional.

| Example | Numerical check | Chapter |
|---|---|---|
| [data_spaces.py](../examples/data_spaces.py) | Construct L2 and quadratic H1 fields; node diagrams, a DG jump and basis profiles | [Field construction](create-fields.md#construct-an-l2-field) |
| [data_meshes.py](../examples/data_meshes.py) | Grid counts, cube volume, polygon area and chain length | [Build Meshes](02-meshes.md) |
| [data_fields.py](../examples/data_fields.py) | Point/cell projection, a material jump and exact affine gradients | [Fields and Spaces](03-fields-and-spaces.md) |
| [data_analysis.py](../examples/data_analysis.py) | Threshold/clip volumes, contour area, sampled values and analytic rotation | [Filter and Analyze](04-filters.md) |
| [data_mesh_kernels.py](../examples/data_mesh_kernels.py) | A selected-cell output and an oriented face flux balance | [Write Mesh Algorithms](05-mesh-kernels.md) |
| [data_finite_volume.py](../examples/data_finite_volume.py) | Flux/divergence balance, Perot velocity and upwind mass flux on shaped and polyhedral cells | [Finite-volume Data and Fluxes](18-finite-volume.md) |
| [data_filters.py](../examples/data_filters.py) | Five built-in filters rewritten with public names, each held to the built-in | [Write a Filter](15-write-a-filter.md) |

[![Point, cell and discontinuous fields](../assets/data-spaces.png)](03-fields-and-spaces.md)

[![Whole-cell selection versus clipping and contouring](../assets/data-filters.png)](04-filters.md)

```bash
uv run python docs/examples/data_analysis.py --arch cpu --check
uv run --with matplotlib python docs/examples/data_analysis.py --output filters.png
```

The [pipeline chapter](06-pipelines.md) connects these operations to
probing, streamline integration and external tools.

## Choose a reading route

Begin with [the four model chapters](introduction.md#a-first-reading), or
use [advanced routes](introduction.md#advanced-routes) for FE/DG input,
solver storage and custom filters. [Analyze Adaptive Data](adaptive-data.md)
contains the checked AMR and octree examples. The
[polyhedra chapter](18-polyhedra.md#a-concave-cell-checked) checks a concave
cell against analytic results. Higher-order examples remain in
[data_high_order.py](../examples/data_high_order.py).
