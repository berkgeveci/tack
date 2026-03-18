"""Vector addition on Metal GPU -- same kernel as CPU, different backend."""

import time
import numpy as np
import pgc

pgc.init(arch=pgc.metal)

n = 1_000_000
x = pgc.field(dtype=pgc.f32, shape=(n,))
y = pgc.field(dtype=pgc.f32, shape=(n,))
out = pgc.field(dtype=pgc.f32, shape=(n,))

x.from_numpy(np.arange(n, dtype=np.float32))
y.from_numpy(np.ones(n, dtype=np.float32) * 2.0)


@pgc.kernel
def vector_add(x, y, out):
    for i in range(x.shape[0]):
        out[i] = x[i] + y[i]


# Warm up (includes MSL compilation + Metal pipeline creation)
t0 = time.perf_counter()
vector_add(x, y, out)
t1 = time.perf_counter()
print(f"First call (includes compilation): {(t1-t0)*1000:.1f} ms")

# Verify
result = out.to_numpy()
expected = np.arange(n, dtype=np.float32) + 2.0
assert np.allclose(result, expected)
print(f"Correctness: OK ({n:,} elements)")

# Benchmark (cached pipeline)
times = []
for _ in range(10):
    t0 = time.perf_counter()
    vector_add(x, y, out)
    t1 = time.perf_counter()
    times.append(t1 - t0)

avg = sum(times) / len(times)
print(f"Avg execution:  {avg*1000:.2f} ms ({n/avg/1e6:.0f} M elements/sec)")
