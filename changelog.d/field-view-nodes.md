### Added

- Every `H1`, `L2` and `Constant` field view on cells with a reference element
  numbers its own nodes: `NODES` per cell, node `k` at `node_parametric(k)`,
  its value `dof(c, k)` stored at `dof_index(c, k)`. One kernel computes a
  field at another's nodes on every shape, space, order and basis
  ([Work at a field's nodes](docs/vis-guide/05-mesh-kernels.md#work-at-a-fields-nodes)).
  Order 1 had no `dof_index` and no node count of its own; a loop over the
  shape's `NUM_QUADRATIC`, the quadratic element's nodes, wrote nothing on
  segments, whose order-2 view numbers its nodes in a tensor basis.
