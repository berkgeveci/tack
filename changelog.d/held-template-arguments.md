### Fixed

- A template method may pass one template it holds to a method of another
  (`self.values.scaled(self.mesh, i)`): the callee is resolved when that call is
  reached. It raised a `KeyError` for the callee's resolved name.
