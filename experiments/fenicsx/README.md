# FEniCSx interoperability investigation

This experiment tests a native DOLFINx-to-Tack import using the existing field
model. It is not a public adapter. Run it in the `fenicsx-tack` conda environment:

```bash
conda activate fenicsx-tack
python experiments/fenicsx/interop.py
```

Validated with DOLFINx/Basix 0.11.0, Python 3.12.16 and Tack at `b2190066` on
Apple Silicon. Tack remains installed editable from `tack-berk`; this experiment
does not change the installed package or add a DOLFINx dependency to Tack.

## What passed

A serial 2x2 quadrilateral mesh with straight geometry, using CPU float64:

- Continuous H1 Lagrange fields, orders 1 through 4.
- Discontinuous Lagrange fields, orders 0 through 3.
- Scalars and blocked two-component vectors: 16 combinations in total.
- Random coefficients, evaluated at seven interior points per cell against
  DOLFINx's `Function.eval`; the largest observed error was below `1e-14`.
- Changes to the source coefficient vector, made after completed reads, are
  reflected in subsequent Tack evaluations without rebuilding the field.
- DLPack-based memory checks confirm borrowed geometry and coefficient storage.

## How it works

DOLFINx's quadrilateral corner order differs from Tack's cyclic quad order.
The topology's corner connectivity is reordered, and reference-node positions
identify the permutation from Basix's local DOFs to Tack's local basis order.

For H1, DOLFINx interleaves vertex, edge and interior DOFs globally. Tack's
element-to-DOF tables require each corner's coefficient index to equal its
topological point ID. The prototype creates an adapter numbering that meets
that rule, then uses `arrays.Permutation` to read the original DOLFINx vector.
The coefficients themselves are never reordered or copied. Each field can have
its own adapter numbering on the same topology.

For DG, a permutation maps each cell's basis order to its original coefficient
blocks. The resulting values use Tack's existing L2 block layout. Vector
coefficients are borrowed as vector elements rather than flattened scalar DOFs.

Integer connectivity, index permutations and DOF tables are allocated. Geometry
positions and solution coefficients are shared, with explicit source-owner
references retained. This is zero-copy numerical storage, not a claim that
import allocates nothing.

## Limits and next steps

- Single-process, straight quads, real float64 CPU storage only. Mesh and
  function must match. The prototype accepts matching Lagrange nodal bases and
  refuses other element families or mismatched node locations.
- Add triangles/tetrahedra, hexes, curved geometry and deliberately reversed
  entity orientations; test values and derivatives, not only sampled pictures.
- Basix's default simplex variants above order 2 may have nodes different from
  Tack's equispaced simplex bases. Detect compatibility and use an explicit
  exact coefficient conversion where the polynomial spaces match.
- Mixed spaces, unblocked vector elements, moment/modal coefficients,
  H(curl)/H(div), constraints and distributed owned/ghost data need separate
  handling. A shared array does not establish correct ghost values.
- The source must stay alive and cannot be modified during Tack reads. Live
  solver views need a revision/invalidation contract before use as published
  immutable datasets or with cached derived results. A copied snapshot is a
  different ownership mode.
- The CPU borrowing result does not establish zero-copy GPU interoperability.
  The current float64 arrays cannot be used directly by Metal; any conversion
  or device transfer must be explicit. Assess compatible GPU storage separately.

Before exposing a public `dolfinx_to_dataset` API, agree on these ownership and
capability rules, then move the import into an optional interop module with
tests against DOLFINx on the supported matrix.
