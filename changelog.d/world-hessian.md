### Added

- `world_hessian(cells, u, c, pc)`: a field's second derivatives in the world
  at a parametric point of a solid cell, for kernels. By the chain rule
  through the geometry: the field view's new `parametric_hessian` less the
  geometry's new `geometry_curvature` along the world gradient (zero where the
  map is affine), taken to world axes by the inverse Jacobian. Analytic from
  the shapes' new `shape_hessian` and `quadratic_hessian` at orders 1 and 2,
  and from the bases at any order on tensor shapes (Lagrange and Bernstein)
  and simplices (equispaced and Bernstein); wedges above order 2 difference
  their parametric gradients.
