### Fixed

- Python crashed in its garbage collector when it freed, in a reference cycle (a
  traceback's frames, say), a VTK dataset one of Tack's VTK algorithms had read: the
  `DeleteEvent` observer that dropped the dataset's cache entry, referenced only by
  the dataset, was cleared as garbage before the deletion called it (a VTK bug, in
  9.7.1 and master). The cache entry is now an attribute of the dataset's Python
  wrapper, which VTK keeps while the dataset lives and frees with it: no observer.
  An entry is converted again when the dataset's modification time or time step
  changes; after changing a dataset in place, call `Modified()`, as VTK's own
  filters need.
