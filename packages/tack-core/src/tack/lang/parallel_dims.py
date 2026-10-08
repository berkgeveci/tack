"""Multi-dimensional parallel loops: lowering helpers and launch geometry.

A kernel's parallel loop over ``tack.ndrange`` of two or three dimensions
keeps its dimensions (``IRParallelFor.dims``/``extents``) instead of
recovering them from one flat index by division. Each backend binds them
from a launch of the same shape: CPU workers walk their chunk row by row,
GPUs launch a 2D or 3D grid. Integer division by a runtime value has no
hardware instruction on GPUs, and recovering three indices from a flat
one cost a 200³ structured-cell kernel 6.3 ms on an M1 Max against 0.9 ms
without it.

``flatten`` turns such a loop back into the flat form: workgroup kernels
keep it, because their collectives assume one-dimensional 256-thread
groups. ``launch_geometry`` picks a GPU launch and says when the device's
grid limits force the flat fallback the generated kernels carry.
"""

from tack.lang import ir
from tack.lang.ir_traversal import clone_ir


def parallel_loop(ir_func):
    """The kernel's top-level parallel loop."""
    for stmt in ir_func.body:
        if isinstance(stmt, ir.IRParallelFor):
            return stmt
    raise RuntimeError(f"Kernel '{ir_func.name}' has no parallel for-loop")


def loop_dims(ir_func):
    """The parallel loop's dimension variables, slowest first, or None for a flat loop."""
    return parallel_loop(ir_func).dims


def decomposition(index, dims, extents):
    """Statements binding ``dims`` from the flat ``index`` (slowest first), by division."""
    stmts = []
    remaining = index
    for k in range(len(dims) - 1, 0, -1):
        stmts.append(ir.IRAssign(target=dims[k], value=ir.IRBinOp(
            op="%", left=remaining, right=extents[k])))
        remaining = ir.IRBinOp(op="//", left=remaining, right=extents[k])
    stmts.append(ir.IRAssign(target=dims[0], value=remaining))
    return stmts[::-1]


def flatten(loop):
    """Make a multi-dimensional parallel loop divide its flat index, in place."""
    if not loop.dims:
        return
    # Copies: the body's extents get resolved and compiled in, while the
    # loop's bound (their product) stays for the host to evaluate.
    extents = [clone_ir(e) for e in loop.extents]
    loop.body = decomposition(ir.IRName(loop.var), loop.dims, extents) + loop.body
    loop.dims = None
    loop.extents = None


def _ceil_pow2(n):
    return 1 << max(int(n) - 1, 0).bit_length()


def launch_geometry(extents, *, max_grid, max_block, block_size=256):
    """A GPU launch for ``extents`` (slowest first, two or three of them).

    Returns ``(grid, block, flat)``: x/y/z triples with x the fastest
    dimension, and whether the kernel must run its flat fallback instead,
    as a one-dimensional launch of ``block_size`` threads per group over
    the product. The block takes up to ``block_size`` threads, as many along
    x as the extent uses (rounded to a power of two), then y, then z, so a
    narrow fastest dimension does not leave most of a block idle.
    ``max_grid`` and ``max_block`` are the device's per-dimension limits.
    """
    sizes = list(extents[::-1]) + [1] * (3 - len(extents))
    block = []
    room = block_size
    for size, limit in zip(sizes, max_block):
        b = max(1, min(room, _ceil_pow2(size), limit))
        block.append(b)
        room //= b
    grid = [-(-size // b) for size, b in zip(sizes, block)]
    if all(g <= limit for g, limit in zip(grid, max_grid)):
        return tuple(grid), tuple(block), False
    total = 1
    for size in sizes:
        total *= size
    return (-(-total // block_size), 1, 1), (block_size, 1, 1), True
