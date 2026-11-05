### Changed

- Template methods take only the synthetic parameters (a template's arrays and
  runtime scalars) they read, themselves or through the methods they call. Every
  method took its whole tree's, and each inlined call copied them all: a view
  holding other views carried theirs into every call.
- A template's methods are resolved when the kernel reaches them, not all of
  them for every kernel. A method the kernel never calls costs nothing and is
  not required to resolve.
