"""tack — alias for pgc (Portable GPU Compute).

This package re-exports the entire pgc namespace so you can write:

    import tack
    tack.init(arch=tack.cpu)
    x = tack.field(dtype=tack.f32, shape=(1024,))

    @tack.kernel
    def add(x, y, out):
        for i in range(x.shape[0]):
            out[i] = x[i] + y[i]

When ready to fully rename, just find-replace 'tack' → new name.
"""

import sys as _sys

# Re-export everything from pgc top-level
from pgc import *  # noqa: F401,F403

# Eagerly import key sub-packages so tack.algorithms etc. work
import pgc.algorithms
import pgc.lang
import pgc.codegen
import pgc.runtime
try:
    import pgc.rendering
except ImportError:
    pass
try:
    import pgc.interop
except ImportError:
    pass
try:
    import pgc.data
except ImportError:
    pass

# Alias all loaded pgc.* modules as tack.*
for _name in list(_sys.modules):
    if _name.startswith('pgc.'):
        _sys.modules['tack' + _name[3:]] = _sys.modules[_name]
