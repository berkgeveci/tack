# Worked Examples

These programs develop the dataset API with analytic answers, independently of
a simulation solver. They run with NumPy and a Tack backend; figures are optional.

| Example | Numerical check | Chapter |
|---|---|---|
| [data_spaces.py](../examples/data_spaces.py) | Construct L2 and quadratic H1 fields; node diagrams, a DG jump and basis profiles | [Field construction](03-fields-and-spaces.md#construct-an-l2-field) |
| [data_meshes.py](../examples/data_meshes.py) | Grid counts, cube volume, polygon area and chain length | [Build Meshes](02-meshes.md) |
| [data_fields.py](../examples/data_fields.py) | Point/cell projection, a material jump and exact affine gradients | [Fields and Spaces](03-fields-and-spaces.md) |
| [data_analysis.py](../examples/data_analysis.py) | Threshold/clip volumes, contour area, sampled values and analytic rotation | [Filter and Analyze](04-filters.md) |
| [data_mesh_kernels.py](../examples/data_mesh_kernels.py) | A selected-cell output and an oriented face flux balance | [Write Mesh Algorithms](05-mesh-kernels.md) |

[![Point, cell and discontinuous fields](../assets/data-spaces.png)](03-fields-and-spaces.md)

[![Whole-cell selection versus clipping and contouring](../assets/data-filters.png)](04-filters.md)

```bash
uv run python docs/examples/data_analysis.py --arch cpu --check
uv run --with matplotlib python docs/examples/data_analysis.py --output filters.png
```

The [pipeline chapter](06-pipelines.md) connects these operations to
probing, streamline integration and external tools.
