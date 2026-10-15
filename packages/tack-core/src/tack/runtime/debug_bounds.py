"""Debug mode at dispatch: the hidden arguments, and the IndexError after a launch.

A kernel lowered in debug mode takes two hidden arrays after its own
parameters (``lang/ir_bounds.py``): the errors record, which its bounds
checks fill on the first violation, and the extents of every parameter,
which the checks compare indices against. ``hidden_args`` appends them
for a call; ``launched`` notes the launch; ``check`` reads the record
afterwards and raises ``IndexError`` naming the access.

Each thread has its own errors array per backend, so launches from two
threads do not report each other's violations. A launch is checked when
the kernel returns, or, if the launch came another way, before the next
debug launch on the thread.
"""

import math
import threading
import weakref
from typing import NamedTuple

import numpy as np

from tack.lang import ir_bounds
from tack.lang.types import i32, i64

_local = threading.local()


class _Launch(NamedTuple):
    errors: object          # the errors Array
    sites: tuple            # the kernel's bounds_sites


def _arrays(backend) -> dict:
    """This thread's hidden arrays on ``backend``: ``errors`` and extents by size."""
    per_backend = getattr(_local, "arrays", None)
    if per_backend is None:
        per_backend = _local.arrays = weakref.WeakKeyDictionary()
    found = per_backend.get(backend)
    if found is None:
        found = per_backend[backend] = {}
    return found


def hidden_args(backend, template, args) -> list:
    """``args`` and the hidden errors and extents arrays a debug kernel takes."""
    from tack.lang.array import Array, copyto, zeros

    own = len(template.params) - len(ir_bounds.HIDDEN_PARAMS)
    if len(args) != own:
        raise TypeError(f"Kernel '{template.name}' expects {own} arguments, got {len(args)}")
    table = np.zeros(ir_bounds.EXTENTS_PER_PARAM * len(template.params), np.int64)
    for position, arg in enumerate(args):
        if isinstance(arg, Array):
            shape = tuple(getattr(arg, "_logical_shape", None) or arg.shape)
            slot = ir_bounds.EXTENTS_PER_PARAM * position
            table[slot] = shape[0] if shape else 1
            table[slot + 1] = math.prod(shape)
            table[slot + 2] = arg.scalar_size
    arrays = _arrays(backend)
    errors = arrays.get("errors")
    if errors is None:
        errors = arrays["errors"] = zeros((ir_bounds.ERRORS_SIZE,), i32)
    extents = arrays.get(table.size)
    if extents is None:
        extents = arrays[table.size] = zeros((table.size,), i64)
    copyto(extents, table)
    return [*args, errors, extents]


def launched(args, template):
    """Note a debug launch, whose errors array ``check`` reads."""
    _local.pending = _Launch(args[-len(ir_bounds.HIDDEN_PARAMS)], template.bounds_sites)


def check():
    """Raise ``IndexError`` for this thread's last debug launch if it went out of
    bounds; nothing if it did not, or if it was checked already."""
    launch = getattr(_local, "pending", None)
    if launch is None:
        return
    _local.pending = None
    record = launch.errors.numpy()
    if record[ir_bounds.COUNT] == 0:
        return
    launch.errors.fill(0)
    raise IndexError(describe(launch.sites, record))


def _int64(low, high) -> int:
    return int(np.uint32(low)) | (int(high) << 32)


def describe(sites, record) -> str:
    """The message for an errors record: the first violation, and how many."""
    site = sites[int(record[ir_bounds.SITE])]
    dim = int(record[ir_bounds.DIM])
    index = _int64(record[ir_bounds.INDEX_LO], record[ir_bounds.INDEX_HI])
    extent = _int64(record[ir_bounds.EXTENT_LO], record[ir_bounds.EXTENT_HI])
    where = f"Kernel '{site.kernel}'"
    if site.function is not None:
        where = f"Device function '{site.function}' (inlined into kernel '{site.kernel}')"
    if site.file is not None:
        position = f" at {site.file}:{site.line}, column {site.column}"
    elif site.line is not None:
        position = f" at line {site.line}, column {site.column}"
    else:
        position = ""
    which = f"index {index} of dimension {dim}" if site.dimensions > 1 else f"index {index}"
    count = int(np.uint32(record[ir_bounds.COUNT]))
    more = (f" {count - 1} more out-of-bounds access{'es' if count > 2 else ''} in the "
            f"launch {'were' if count > 2 else 'was'} skipped too." if count > 1 else "")
    return (f"{where}: {site.text}{position}: {which} is outside [0, {extent}). "
            f"The access was skipped (a load gives 0).{more}")

