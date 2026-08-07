"""CPU vs HIP performance comparison."""
import time

import numpy as np

import tack

DATA_SIZES = {
    "16KB":   16 * 1024 // 4,
    "256KB": 256 * 1024 // 4,
    "4MB":     4 * 1024 * 1024 // 4,
    "64MB":   64 * 1024 * 1024 // 4,
}

# Kernels
@tack.kernel
def k_saxpy(x, y, z):
    for i in range(x.shape[0]):
        z[i] = 17.0 * x[i] + y[i]

@tack.kernel
def k_memcpy(src, dst):
    for i in range(src.shape[0]):
        dst[i] = src[i]

@tack.kernel
def k_fill(dst):
    for i in range(dst.shape[0]):
        dst[i] = 42.0

@tack.kernel
def k_stencil1d(src, dst):
    for i in range(dst.shape[0]):
        dst[i] = 0.5 * (src[i] + src[i + 1])

@tack.kernel
def k_sqrt(x, out):
    for i in range(x.shape[0]):
        out[i] = sqrt(x[i])

@tack.kernel
def k_sin(x, out):
    for i in range(x.shape[0]):
        out[i] = sin(x[i])

@tack.kernel
def k_exp(x, out):
    for i in range(x.shape[0]):
        out[i] = exp(x[i])

REDUCE_BLOCK = 256

@tack.kernel
def k_reduce(data, out):
    for block_idx in range(out.shape[0]):
        s = 0.0
        for j in range(256):
            s = s + data[block_idx * 256 + j]
        out[block_idx] = s


def bench(fn, warmup=3, trials=10):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    times.sort()
    return times[len(times) // 2]  # median


def make_fields(n, count):
    fields = []
    for _ in range(count):
        f = tack.field(dtype=tack.f32, shape=(n,))
        f.from_numpy(np.abs(np.random.randn(n).astype(np.float32)) + 0.1)
        fields.append(f)
    return fields


def run_bench(name, kernel_fn, n, n_arrays):
    fields = make_fields(n if name != "stencil1d" else n + 1, 1) if name in ("fill",) else None

    if name == "saxpy":
        x, y, z = make_fields(n, 3)
        fn = lambda: kernel_fn(x, y, z)
    elif name == "memcpy":
        src, dst = make_fields(n, 2)
        fn = lambda: kernel_fn(src, dst)
    elif name == "fill":
        dst, = make_fields(n, 1)
        fn = lambda: kernel_fn(dst)
    elif name == "stencil1d":
        src = tack.field(dtype=tack.f32, shape=(n + 1,))
        src.from_numpy(np.random.randn(n + 1).astype(np.float32))
        dst = tack.field(dtype=tack.f32, shape=(n,))
        fn = lambda: kernel_fn(src, dst)
    elif name in ("sqrt", "sin", "exp"):
        x, out = make_fields(n, 2)
        fn = lambda: kernel_fn(x, out)
    elif name == "reduce":
        n = (n // REDUCE_BLOCK) * REDUCE_BLOCK
        data = tack.field(dtype=tack.f32, shape=(n,))
        data.from_numpy(np.random.randn(n).astype(np.float32))
        out = tack.field(dtype=tack.f32, shape=(n // REDUCE_BLOCK,))
        fn = lambda: kernel_fn(data, out)
    else:
        raise ValueError(name)

    trials = 20 if n <= 1_000_000 else 10 if n <= 4_000_000 else 5
    return bench(fn, warmup=3, trials=trials)


def main():
    np.random.seed(42)
    import os
    print(f"CPU cores: {os.cpu_count()}")

    benchmarks = [
        ("saxpy",     k_saxpy,     3),
        ("memcpy",    k_memcpy,    2),
        ("fill",      k_fill,      1),
        ("stencil1d", k_stencil1d, 2),
        ("sqrt",      k_sqrt,      2),
        ("sin",       k_sin,       2),
        ("exp",       k_exp,       2),
        ("reduce",    k_reduce,    1),
    ]

    # Collect results
    results = {}
    for backend in ["cpu", "hip"]:
        tack.init(arch=backend)
        print(f"\n--- Running on {backend.upper()} ---")
        for bench_name, kernel, n_arrays in benchmarks:
            for size_tag, n in DATA_SIZES.items():
                t = run_bench(bench_name, kernel, n, n_arrays)
                results[(bench_name, size_tag, backend)] = t
                bw = n * 4 * n_arrays / t / 1e9
                print(f"  {bench_name:>12s}  {size_tag:>6s}  {t*1000:>10.3f} ms  {bw:>8.1f} GB/s")

    # Summary table
    print(f"\n{'='*90}")
    print("  SUMMARY: CPU vs HIP (median time in ms, speedup = CPU/HIP)")
    print(f"{'='*90}")
    print(f"  {'Benchmark':>12s}  {'Size':>6s}  {'CPU (ms)':>10s}  {'HIP (ms)':>10s}  {'Speedup':>8s}")
    print(f"  {'-'*12}  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*8}")

    for bench_name, _, _ in benchmarks:
        for size_tag in DATA_SIZES:
            cpu_t = results.get((bench_name, size_tag, "cpu"))
            hip_t = results.get((bench_name, size_tag, "hip"))
            if cpu_t and hip_t:
                speedup = cpu_t / hip_t
                print(f"  {bench_name:>12s}  {size_tag:>6s}  {cpu_t*1000:>10.3f}  {hip_t*1000:>10.3f}  {speedup:>7.1f}x")

    # Volrender comparison (already measured externally, just print note)
    print(f"\n{'='*90}")
    print("  VOLUME RENDER: see --bench output from packages/tack-rendering/examples/tack_volrender.py")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
