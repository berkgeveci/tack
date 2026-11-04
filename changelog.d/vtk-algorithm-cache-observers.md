### Fixed

- Python crashed in its garbage collector when it freed, in a reference cycle (a
  traceback's frames, say), a VTK dataset one of Tack's VTK algorithms had read: the
  observer that drops the dataset's cache entry, referenced only by the dataset, was
  cleared as garbage before the deletion called it (a VTK bug, in 9.7.1 and master).
  The observers are kept reachable until they fire, and only note the deletion; the
  cache drops the entry at its next use.
