"""09 -- Shared memory and thread synchronization.

On GPUs, threads within a workgroup (threadgroup on Metal, thread block
on CUDA/HIP) can communicate through fast shared memory.

Key APIs:
  pgc.shared(dtype, size)       -- allocate threadgroup-local memory
  pgc.shared_like(field, size)  -- allocate shared memory matching a field's dtype
  pgc.thread_id()               -- local thread index within the workgroup
  pgc.barrier()                 -- synchronize all threads in the workgroup

Usage:
  uv run python examples/09_shared_memory.py
"""

import numpy as np
import pgc

import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument('--arch', default='cpu', choices=['cpu', 'metal', 'cuda', 'hip', 'level_zero'])
_arch = getattr(pgc, _parser.parse_args().arch)
pgc.init(arch=_arch)

n = 256
x = pgc.field(dtype=pgc.f32, shape=(n,))
out = pgc.field(dtype=pgc.f32, shape=(n,))

x.from_numpy(np.arange(n, dtype=np.float32))


@pgc.kernel
def double_via_shared(x, out):
    """Load data into shared memory, transform, then write back.

    This pattern is the foundation of many GPU algorithms:
    1. Cooperatively load data into fast shared memory
    2. Synchronize (barrier)
    3. Each thread reads from shared memory to compute its output
    """
    smem = pgc.shared(pgc.f32, 256)
    for i in range(x.shape[0]):
        tid = pgc.thread_id()
        # Load into shared memory
        smem[tid] = x[i] * 2.0
        # Ensure all threads have finished writing
        pgc.barrier()
        # Read back (on GPU, could read a neighbor's value)
        out[i] = smem[tid]


double_via_shared(x, out)

expected = np.arange(n, dtype=np.float32) * 2.0
assert np.allclose(out.to_numpy(), expected)
print(f"Shared memory double: first 10 = {out.to_numpy()[:10]}")
print("Shared memory + barrier: OK")


# --- shared_like: inherit dtype from a field ---
# When the shared memory dtype should match a field parameter, use
# pgc.shared_like instead of hardcoding the type. This makes kernels
# generic over field dtype.

x_i32 = pgc.field(dtype=pgc.i32, shape=(n,))
out_i32 = pgc.field(dtype=pgc.i32, shape=(n,))
x_i32.from_numpy(np.arange(n, dtype=np.int32))


@pgc.kernel
def double_via_shared_like(x, out):
    """Same pattern, but shared memory dtype matches the field automatically."""
    smem = pgc.shared_like(x, 256)
    for i in range(x.shape[0]):
        tid = pgc.thread_id()
        smem[tid] = x[i] * 2
        pgc.barrier()
        out[i] = smem[tid]


# Works with i32 fields — shared memory is automatically int
double_via_shared_like(x_i32, out_i32)
assert np.array_equal(out_i32.to_numpy(), np.arange(n, dtype=np.int32) * 2)
print("shared_like with i32: OK")

# Works with f32 fields too — same kernel, shared memory is automatically float
double_via_shared_like(x, out)
assert np.allclose(out.to_numpy(), expected)
print("shared_like with f32: OK")
