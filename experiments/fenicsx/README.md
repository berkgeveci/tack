# FEniCSx interoperability investigation

This experiment imports serial DOLFINx geometry and Basix P fields into Tack.
It remains outside the installed API. It adds no FEniCSx dependency or changes
to Tack's core field model. Run it in the `fenicsx-tack` conda environment:

```bash
conda activate fenicsx-tack
python experiments/fenicsx/verify.py --report /tmp/fenicsx-validation.json
```

The original `python experiments/fenicsx/interop.py` entry point also runs the
checks. Options select shapes, variants, orders and component counts; see
`--help`. Validated with DOLFINx/Basix 0.11.0, Python 3.12.16 and the installed
Tack checkout at `b2190066`, using CPU float64 on Apple Silicon.

## Coverage and results

The matrix covers H1 orders 1–4 and DG orders 0–4, scalar and blocked
two-component fields, and eight requested Basix variants: equispaced,
GLL warped/Isaac/centroid, Chebyshev warped, GL warped, Legendre and Bernstein.
Only source-defined combinations are tested as imports; for example, Basix
does not provide continuous Legendre elements or Bernstein elements on hexes.

| Shape | Direct sharing | Lazy conversion | Passed |
|---|---:|---:|---:|
| Interval | 96 | 16 | 112 |
| Triangle | 62 | 50 | 112 |
| Quadrilateral | 88 | 16 | 104 |
| Tetrahedron | 62 | 50 | 112 |
| Hexahedron | 88 | 16 | 104 |
| Prism / wedge | 46 | 22 | 68 |
| Pyramid | 26 | 2 | 28 |
| **Total** | **468** | **172** | **640** |

All 640 import cases passed. Basix declined 320 requested combinations, and
DOLFINx could not construct 20 function spaces; 28 source-defined combinations
were explicitly refused for unavailable Tack
pyramid spaces. None failed unexpectedly. Counts include low-order variants
that reduce to the same basis. The full run comprises 1,008 requested cases.

Each imported field uses random source coefficients and seven interior sample
points per cell. Values are compared against DOLFINx `Function.eval`;
world-gradient checks use field-view parametric derivatives and the experiment's
Jacobian kernel, compared against Basix derivative tabulation and the source
geometry Jacobian. Largest observed absolute errors were `9.0e-13` for values
and `2.6e-12` for gradients. Checks repeat after changing the source vector to
verify live borrowing. Memory checks verify shared geometry/coefficient storage.

Meshes have multiple adjacent cells; tetrahedra and triangles exercise varied
cell orientations, and two pyramids share a base with reversed reference axes.
Geometry is degree 1. Higher-order geometry and distributed meshes remain out
of scope.

## Solved-problem analysis validation

[ANALYSIS.md](ANALYSIS.md) records 14 end-to-end cases backed by 16 variational
solves, followed by the public integration, probe, gradient, contour, slice,
flow-diagnostic and streamline APIs. This includes native Bernstein and
lazy-converted H1/modal-DG fields. Run `python experiments/fenicsx/analysis.py`.

These tests found and repaired an orientation defect: valid DOLFINx prisms can
have negative reference orientation, whose signed Tack volumes canceled.
The importer now normalizes 3D cell orientation and applies the same reference
reflection to field addressing. Geometry and coefficient storage remain shared;
the check uses temporary work arrays and allocates orientation/index metadata.
The full 640-case representation matrix passed again after the repair, and
positive-cell/domain-measure checks were added for every shape.

## Two explicit import modes

```python
data = import_mesh(dolfinx_mesh)
field = import_function(data, function)  # compatible basis required
field = import_function(data, function, conversion="lazy")
print(field._dolfinx_import)  # experimental inspection metadata
```

**Direct:** geometry and coefficients are borrowed. Integer connectivity,
permutations and DOF tables allocate. Basis functions are compared by tabulation
to find a coefficient permutation; this handles both nodal bases and native
Bernstein coefficients. An element's interpolation points are not assumed to
be coefficient locations. In particular, Basix Bernstein elements use more
interpolation/quadrature points than coefficients.

DOLFINx interleaves vertex and higher-order DOFs. An adapter numbering makes
corner indices match Tack's point IDs; `arrays.Permutation` reads the original
vector without reordering its storage. Different fields use independent maps
on the same topology. DG fields use the existing L2 block layout. Blocked
vectors borrow vector elements rather than treating each component as a node.

**Lazy conversion:** when bases differ, the explicit option represents each
target nodal coefficient as a weighted sum of source coefficients. The weights
come from source basis tabulation at target nodes, with reconstruction checked
at independent reference points. H1 retains shared target DOFs; DG remains
cell-local. Source coefficients stay borrowed and updates remain visible.
This supports Legendre modal DG fields and incompatible nodal variants without
allocating a converted coefficient vector. It does not add a native Legendre
basis to Tack.

The cost is a dense dot product per target coefficient read, potentially
quadratic work in the local basis size for field evaluation. Materializing
transformed coefficients could make repeated evaluation faster but would need
explicit allocation and refresh. No performance recommendation follows from
these correctness checks.

The lazy storage currently uses Tack's internal `_Implicit`/`_metadata` hooks.
A production adapter needs a supported extension mechanism or equivalent
public composition; the experiment does not present these hooks as public API.

## Shape and basis limits

- Equispaced and matching GLL/GL nodal variants share coefficients directly.
  Different simplex/wedge node variants use the explicit conversion path.
- Basix Bernstein elements on intervals, triangles and tetrahedra match Tack's
  Bernstein basis up to ordering, preserving their coefficient interpretation.
- Legendre modal DG fields convert exactly on intervals, triangles, quads,
  tetrahedra, hexes and prisms. Pyramid constants convert; order-1 Legendre
  pyramids have eight modes, while Tack's order-1 pyramid has five functions.
  Those are different spaces and are refused, not approximated.
- Pyramids use different reference conventions: Tack's collapsed `(r,s,t)`
  corresponds to Basix `((1-t)*r, (1-t)*s, t)`. Order-1 nodal fields are covered;
  higher-order pyramids are refused until Tack has a matching basis.
- In installed DOLFINx 0.11.0, constructing H1 prism spaces at orders 3–4
  fails for the tested equispaced and GLL variants with
  `Multiple index maps of this dimension. Call index_maps instead.`
  The same error occurs for H1 pyramids at orders 3–4. These 20 cases fail
  before the adapter runs. Prism H1 orders 1–2 and DG orders 0–4 are covered.
  `reproduce_prism_dofmap.py` is a minimal source-only reproducer; its expected
  outcome in this environment is the runtime error above.
- Non-permutation orientation transformations, moment elements, H(curl)/H(div),
  mixed spaces, constraints, periodic vertex identifications, unused geometry
  points, complex coefficients and MPI ownership/ghosts require further work.

## Ownership and devices

Source owners are retained. The solver must not write during Tack reads.
Revision/invalidation rules for cached results and immutable publication remain
design work; live storage and a published snapshot need distinct contracts.

The validated borrowing is CPU float64. It does not establish zero-copy GPU
sharing. Metal cannot directly consume these float64 arrays; conversion and
device-transfer behavior must be explicit and assessed separately.
