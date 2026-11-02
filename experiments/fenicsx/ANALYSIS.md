# Solved problems and public Tack analysis

Run in the `fenicsx-tack` conda environment:

```bash
python experiments/fenicsx/analysis.py --report /tmp/fenicsx-analysis.json
```

`--case NAME` selects cases (comma-separated); `--help` lists the options.
The suite passed **14 end-to-end cases backed by 16 variational linear solves**
on DOLFINx/Basix 0.11.0 and Tack's existing public algorithms, using serial CPU
float64 and degree-1 geometry. This is separate from the 640-case representation
matrix in [README.md](README.md).

## Problems actually solved

All Poisson problems use a finite-element stiffness matrix, load assembly,
Dirichlet boundary conditions and PETSc LU. The harness checks the algebraic
residual and independently integrates the solution's L2 error.

| Cases | Problem and representation |
|---|---|
| 5 | Quadratic manufactured Poisson solution on triangles, quads, tetrahedra, hexes and prisms, with H1 order 2 |
| 1 | The triangle problem solved in a native Bernstein H1 basis |
| 1 | The triangle problem solved at order 3 in GLL warped nodes, imported by lazy conversion |
| 3 | Non-polynomial sine Poisson solution on successively refined triangle meshes |
| 1 | Vector Poisson with an affine rotational/helical velocity on hexes |
| 1 | Vector Poisson with a cubic rotational velocity on triangles, imported by lazy conversion |
| 2 | Solved Poisson fields projected into Legendre modal DG spaces on triangles and prisms using mass-matrix solves |

The DG cases validate variational L2 projection and analysis of its modal
coefficients; they are not a DG discretization of the Poisson operator.

For the quadratic cases, `u = sum((x_i - 1/2)^2)` and `-Delta u = -2d`, with
exact Dirichlet data. The exact volume integral on the unit box is `d/12`.
For the sine cases, `u = product(sin(pi*x_i))` and `-Delta u = d*pi^2*u`.
The latter is not exactly representable by the chosen finite-element space.

For the vector cases, write `a=x-1/2`, `b=y-1/2`. The affine velocity is
`v=(-b,a,0.1)` in 3D. The cubic 2D velocity, stored as three components, is
`v=(-b*(1+a^2+b^2), a*(1+a^2+b^2), 0)`. Both solve manufactured vector Poisson
problems. Their analytic circular/helical trajectories and flow diagnostics
provide independent references; no flow solver such as Navier–Stokes is implied.

## Public algorithms exercised

- **`td.integrate`:** domain measure and scalar/vector integrals against
  DOLFINx quadrature and analytic integrals where available.
- **`td.probe`:** values at 96 interior points and one exterior point, against
  DOLFINx `Function.eval` with its own bounding-box/cell-collision search;
  validity flags and zero output outside are checked too.
- **`td.gradients`:** cell gradients against compiled DOLFINx UFL expressions;
  quadratic problems also check point gradients against the analytic gradient.
- **`td.contour`:** circle/sphere contours of the solved quadratic field.
  Generated vertices are evaluated independently by DOLFINx, and contour
  length/area is checked for convergence toward the analytic measure.
- **`td.slice_plane`:** unit-box plane cuts, checking coordinates, area,
  carried field values against DOLFINx, and integral convergence.
- **`td.flow_quantities`:** divergence, vorticity and Q-criterion against
  analytic results. For the cubic field, with `s=a^2+b^2`, these are
  `0`, `(0,0,2+4s)`, and `(1+s)*(1+3s)`.
- **`td.streamlines`:** two seeds, 100 RK4 steps, comparing complete paths
  and times against analytic trajectories. Both bin search and neighbor
  walking are exercised, including the lazy-converted cubic vector field.

## Numerical results

Largest observed errors across the 14 cases:

| Check | Error |
|---|---:|
| Relative algebraic residual | `1.84e-14` |
| Tack integral versus DOLFINx | `1.34e-15` |
| Probe versus DOLFINx | `5.56e-16` |
| Public gradient versus UFL | `1.02e-14` |
| Contour level residual evaluated by DOLFINx | `4.31e-16` |
| Slice field values versus DOLFINx | `8.89e-16` |
| RK4 trajectory versus analytic path | `1.94e-11` |

The sine Poisson problem's L2 errors were:

| Mesh subdivisions per axis | L2 error | Observed order |
|---|---:|---:|
| 4 | `4.3276e-3` | — |
| 8 | `5.4806e-4` | 2.98 |
| 16 | `6.8739e-5` | 3.00 |

Contour vertices satisfy the solved FE level accurately, while polygonal
length/area retains a reconstruction error. For example, the hex sphere's area
error fell from `4.9351e-2` to `1.1229e-2` at subdivision levels 4 and 8.
Likewise, a slice carries correct point samples of a quadratic field but its
linear output interpolant does not integrate that quadratic exactly.

For a plane at `x=0.43`, the tetrahedral slice's integral errors were
`1.4324e-3`, `8.6550e-4`, `1.4830e-4` at levels 4, 8, 16. Hex errors were
`5.2083e-3`, `1.3021e-3`, `3.2552e-4`; prism errors were `3.2833e-3`,
`1.0958e-3`, `2.3646e-4`. The harness checks sustained convergence; it does not
assume a fixed reduction factor for every plane/subcell intersection pattern.

## Importer defect found and repaired

DOLFINx's unit prism mesh has cells of both reference orientations. The original
prototype preserved those orientations. Pointwise values and gradients were
correct, but `cell_geometry`'s signed volumes canceled: two unit-box prisms
reported `+0.5` and `-0.5`, so `td.integrate(...).measure` returned zero.

The importer now checks geometry Jacobian signs, reflects negatively oriented
3D cells, and composes the same reference reflection into each field's DOF
permutation or lazy basis matrix. Coordinates and coefficients remain borrowed.
The sign check uses temporary work arrays, and orientation/index maps allocate.
Detected degenerate or folded geometry is refused.

The repaired prism analysis returns unit volume and the correct field integral.
All 640 representation cases were rerun and passed. The representation harness
now also checks positive cell measures and analytic domain measure for all seven
mesh shapes, guarding against defects that value comparisons alone miss.
An additional 34-case regression deliberately reverses the reference orientation
of tetrahedra, hexes, prisms and pyramids and checks geometry, values and gradients:

```bash
python experiments/fenicsx/verify.py --reflect-3d \
  --shapes tetrahedron,hexahedron,prism,pyramid \
  --variants equispaced,gll_warped,legendre,bernstein --orders 1,3 --components 1
```

## Source-library and scope limits

DOLFINx's convenience `locate_entities_boundary` rejects prism meshes' mixed
facet types. The prism solves use its public geometric DOF locator, appropriate
for these nodal Lagrange boundary DOFs. This is separate from the documented
order-3/4 prism H1 function-space construction failure.

These solves cover five cell shapes, H1 and projected modal DG fields, and
native Bernstein and lazy-converted representations. They do not establish
every representation-matrix combination as a solved PDE pipeline. Higher-order
geometry, MPI ownership/ghosts, GPU transfers, non-permutation moment/Piola
elements and live-source cache invalidation remain follow-ups.

Detailed results are saved in
`$CONDA_PREFIX/share/fenicsx-tack/fenicsx-analysis-validation.json`.
