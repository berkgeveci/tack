### Added

- A dataset's fields may share a name in different spaces, as VTK's point and
  cell arrays do: `data.fields["p", td.Constant]` (a kind of space, or the
  space itself) picks one; algorithms take `("p", td.H1)` wherever they take a
  field's name, and `fields="p"` on a filter keeps every field of that name.
  `data.fields.named(name)` and `data.fields.entries()` list them.

### Changed

- **Breaking:** `data.fields` is a `Fields` mapping, not a `dict`. A name two
  fields share is refused rather than read as one of them (`KeyError`, from
  `get` and `pop` too); its keys are the names, and `(name, space)` for a
  shared name, so `dict(data.fields)` keeps every field. A field of another
  topology is refused as it is put in, not when an algorithm reads it.
- **Breaking:** `vtk_to_dataset` reads a point array and a cell array of one
  name as two fields of that name, no longer renaming the cell one to
  `"<name> (cells)"` with a warning; `dataset_to_vtk` writes both under the
  name. Writing two point (or two cell) fields of one name is refused.
