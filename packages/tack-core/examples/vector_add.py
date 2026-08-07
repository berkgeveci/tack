"""Vector addition -- simplest Tack example, now JIT-compiled."""

import time

import numpy as np

import tack

tack.init(arch=tack.cpu)

n = 1_000_000
x = tack.field(dtype=tack.f32, shape=(n,))
y = tack.field(dtype=tack.f32, shape=(n,))
out = tack.field(dtype=tack.f32, shape=(n,))

x.from_numpy(np.arange(n, dtype=np.float32))
y.from_numpy(np.ones(n, dtype=np.float32) * 2.0)


@tack.kernel
def vector_add(x, y, out):
    for i in range(x.shape[0]):
        out[i] = x[i] + y[i]


# Warm up (includes JIT compilation)
t0 = time.perf_counter()
vector_add(x, y, out)
t1 = time.perf_counter()
print(f"First call (includes JIT): {(t1-t0)*1000:.1f} ms")

# Verify
result = out.to_numpy()
expected = np.arange(n, dtype=np.float32) + 2.0
assert np.allclose(result, expected)
print(f"Correctness: OK ({n:,} elements)")

# Benchmark (cached)
times = []
for _ in range(10):
    t0 = time.perf_counter()
    vector_add(x, y, out)
    t1 = time.perf_counter()
    times.append(t1 - t0)

avg = sum(times) / len(times)
print(f"Avg execution:  {avg*1000:.2f} ms ({n/avg/1e6:.0f} M elements/sec)")

# Compare with numpy
times_np = []
np_x = x.to_numpy()
np_y = y.to_numpy()
for _ in range(10):
    t0 = time.perf_counter()
    np_out = np_x + np_y
    t1 = time.perf_counter()
    times_np.append(t1 - t0)

avg_np = sum(times_np) / len(times_np)
print(f"NumPy baseline: {avg_np*1000:.2f} ms ({n/avg_np/1e6:.0f} M elements/sec)")
