"""Tack Microbenchmark Suite — Taichi-style systematic performance sweep.

Sweeps across:
  - Data sizes: 16KB, 256KB, 4MB, 64MB
  - Backends: CPU JIT, Metal GPU, NumPy
  - Kernels: saxpy, stencil1d, fill, math ops, reduction, memcpy

Measures median time (ms) and throughput (GB/s or GFlop/s).
"""

import argparse
import json
import time
import sys

import numpy as np
import tack

# ─────────────────────────────────────────────────────────────
# Infrastructure
# ─────────────────────────────────────────────────────────────

DATA_SIZES = {
    "16KB":   16 * 1024,
    "256KB": 256 * 1024,
    "4MB":     4 * 1024 * 1024,
    "64MB":   64 * 1024 * 1024,
}

BYTES_PER_F32 = 4


def n_elements(size_bytes):
    return size_bytes // BYTES_PER_F32


def scaled_repeats(size_bytes, base=20):
    """Fewer repeats for larger data to keep total runtime reasonable."""
    if size_bytes >= 64 * 1024 * 1024:
        return max(base // 4, 5)
    if size_bytes >= 4 * 1024 * 1024:
        return max(base // 2, 10)
    return base


def bench(fn, warmup=3, trials=20):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return sorted(times)


def median(times):
    n = len(times)
    return times[n // 2]


def fmt_ms(sec):
    return f"{sec * 1000:>9.3f}"


def fmt_bw(size_bytes, sec, n_arrays=2):
    """Bandwidth in GB/s (reads+writes)."""
    if sec <= 0:
        return "     inf"
    gb = size_bytes * n_arrays / 1e9
    return f"{gb / sec:>8.1f}"


def fmt_tp(n, sec):
    """Throughput in M elements/sec."""
    if sec <= 0:
        return "     inf"
    return f"{n / sec / 1e6:>8.0f}"


def print_header(title):
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")
    print(f"  {'Size':>6s}  {'n':>10s}  {'Backend':>10s}  {'Median ms':>10s}  "
          f"{'GB/s':>8s}  {'M elem/s':>10s}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*10}")


def print_row(size_tag, n, backend, times, n_arrays=2):
    med = median(times)
    size_bytes = n * BYTES_PER_F32
    print(f"  {size_tag:>6s}  {n:>10,}  {backend:>10s}  {fmt_ms(med)}  "
          f"{fmt_bw(size_bytes, med, n_arrays)}  {fmt_tp(n, med)}")


# ─────────────────────────────────────────────────────────────
# Kernel definitions
# ─────────────────────────────────────────────────────────────

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


@tack.kernel
def k_abs(x, out):
    for i in range(x.shape[0]):
        out[i] = abs(x[i])


REDUCE_BLOCK = 256


@tack.kernel
def k_reduce(data, out):
    for block_idx in range(out.shape[0]):
        s = 0.0
        for j in range(256):
            s = s + data[block_idx * 256 + j]
        out[block_idx] = s


# ─────────────────────────────────────────────────────────────
# Benchmark runners
# ─────────────────────────────────────────────────────────────

def make_fields(n, count):
    fields = []
    for _ in range(count):
        f = tack.field(dtype=tack.f32, shape=(n,))
        f.from_numpy(np.random.randn(n).astype(np.float32).clip(-10, 10))
        fields.append(f)
    return fields


def bench_saxpy(sizes):
    print_header("SAXPY: z = 17*x + y  (3 arrays: 2 read, 1 write)")
    results = []

    for tag, size_bytes in sizes.items():
        n = n_elements(size_bytes)
        repeats = scaled_repeats(size_bytes)

        for backend in ["cpu", "metal"]:
            tack.init(arch=backend)
            x, y, z = make_fields(n, 3)
            k_saxpy(x, y, z)  # warmup/compile
            times = bench(lambda: k_saxpy(x, y, z), warmup=3, trials=repeats)
            print_row(tag, n, backend, times, n_arrays=3)
            results.append({"bench": "saxpy", "size": tag, "backend": backend,
                            "median_ms": median(times) * 1000, "n": n})

        # NumPy
        np_x = np.random.randn(n).astype(np.float32)
        np_y = np.random.randn(n).astype(np.float32)
        def np_saxpy():
            return 17.0 * np_x + np_y
        times = bench(np_saxpy, warmup=3, trials=repeats)
        print_row(tag, n, "numpy", times, n_arrays=3)
        results.append({"bench": "saxpy", "size": tag, "backend": "numpy",
                        "median_ms": median(times) * 1000, "n": n})

    return results


def bench_memcpy(sizes):
    print_header("MEMCPY: dst = src  (2 arrays: 1 read, 1 write)")
    results = []

    for tag, size_bytes in sizes.items():
        n = n_elements(size_bytes)
        repeats = scaled_repeats(size_bytes)

        for backend in ["cpu", "metal"]:
            tack.init(arch=backend)
            src, dst = make_fields(n, 2)
            k_memcpy(src, dst)
            times = bench(lambda: k_memcpy(src, dst), warmup=3, trials=repeats)
            print_row(tag, n, backend, times, n_arrays=2)
            results.append({"bench": "memcpy", "size": tag, "backend": backend,
                            "median_ms": median(times) * 1000, "n": n})

        np_src = np.random.randn(n).astype(np.float32)
        np_dst = np.empty(n, dtype=np.float32)
        times = bench(lambda: np.copyto(np_dst, np_src), warmup=3, trials=repeats)
        print_row(tag, n, "numpy", times, n_arrays=2)
        results.append({"bench": "memcpy", "size": tag, "backend": "numpy",
                        "median_ms": median(times) * 1000, "n": n})

    return results


def bench_fill(sizes):
    print_header("FILL: dst = 42.0  (1 array: write only)")
    results = []

    for tag, size_bytes in sizes.items():
        n = n_elements(size_bytes)
        repeats = scaled_repeats(size_bytes)

        for backend in ["cpu", "metal"]:
            tack.init(arch=backend)
            dst, = make_fields(n, 1)
            k_fill(dst)
            times = bench(lambda: k_fill(dst), warmup=3, trials=repeats)
            print_row(tag, n, backend, times, n_arrays=1)
            results.append({"bench": "fill", "size": tag, "backend": backend,
                            "median_ms": median(times) * 1000, "n": n})

        np_dst = np.empty(n, dtype=np.float32)
        times = bench(lambda: np_dst.fill(42.0), warmup=3, trials=repeats)
        print_row(tag, n, "numpy", times, n_arrays=1)
        results.append({"bench": "fill", "size": tag, "backend": "numpy",
                        "median_ms": median(times) * 1000, "n": n})

    return results


def bench_stencil(sizes):
    print_header("STENCIL 1D: dst[i] = 0.5*(src[i-1]+src[i+1])  (2 arrays)")
    results = []

    for tag, size_bytes in sizes.items():
        n = n_elements(size_bytes)
        repeats = scaled_repeats(size_bytes)

        for backend in ["cpu", "metal"]:
            tack.init(arch=backend)
            # src needs n+1 elements so src[i+1] is valid when i = n-1
            src = tack.field(dtype=tack.f32, shape=(n + 1,))
            src.from_numpy(np.random.randn(n + 1).astype(np.float32).clip(-10, 10))
            dst = tack.field(dtype=tack.f32, shape=(n,))
            dst.from_numpy(np.zeros(n, dtype=np.float32))
            k_stencil1d(src, dst)
            times = bench(lambda: k_stencil1d(src, dst), warmup=3, trials=repeats)
            print_row(tag, n, backend, times, n_arrays=2)
            results.append({"bench": "stencil1d", "size": tag, "backend": backend,
                            "median_ms": median(times) * 1000, "n": n})

        np_src = np.random.randn(n + 1).astype(np.float32)
        np_dst = np.empty(n, dtype=np.float32)
        def np_stencil():
            np_dst[:] = 0.5 * (np_src[:-1] + np_src[1:])
        times = bench(np_stencil, warmup=3, trials=repeats)
        print_row(tag, n, "numpy", times, n_arrays=2)
        results.append({"bench": "stencil1d", "size": tag, "backend": "numpy",
                        "median_ms": median(times) * 1000, "n": n})

    return results


def bench_math(sizes):
    math_kernels = [
        ("sqrt", k_sqrt, np.sqrt),
        ("sin",  k_sin,  np.sin),
        ("exp",  k_exp,  np.exp),
        ("abs",  k_abs,  np.abs),
    ]

    results = []
    for name, kernel, np_fn in math_kernels:
        print_header(f"MATH: {name}(x)  (2 arrays: 1 read, 1 write)")

        for tag, size_bytes in sizes.items():
            n = n_elements(size_bytes)
            repeats = scaled_repeats(size_bytes)

            for backend in ["cpu", "metal"]:
                tack.init(arch=backend)
                x, out = make_fields(n, 2)
                # Ensure positive values for sqrt/exp
                x.from_numpy(np.abs(np.random.randn(n).astype(np.float32)) + 0.1)
                kernel(x, out)
                times = bench(lambda: kernel(x, out), warmup=3, trials=repeats)
                print_row(tag, n, backend, times, n_arrays=2)
                results.append({"bench": f"math_{name}", "size": tag, "backend": backend,
                                "median_ms": median(times) * 1000, "n": n})

            np_x = np.abs(np.random.randn(n).astype(np.float32)) + 0.1
            np_out = np.empty(n, dtype=np.float32)
            times = bench(lambda: np_fn(np_x, out=np_out), warmup=3, trials=repeats)
            print_row(tag, n, "numpy", times, n_arrays=2)
            results.append({"bench": f"math_{name}", "size": tag, "backend": "numpy",
                            "median_ms": median(times) * 1000, "n": n})

    return results


def bench_reduce(sizes):
    print_header("REDUCE: partial block sums (block=256)")
    results = []

    for tag, size_bytes in sizes.items():
        n = n_elements(size_bytes)
        # Round down to multiple of REDUCE_BLOCK
        n = (n // REDUCE_BLOCK) * REDUCE_BLOCK
        if n == 0:
            continue
        n_blocks = n // REDUCE_BLOCK
        repeats = scaled_repeats(size_bytes)

        for backend in ["cpu", "metal"]:
            tack.init(arch=backend)
            data = tack.field(dtype=tack.f32, shape=(n,))
            out = tack.field(dtype=tack.f32, shape=(n_blocks,))
            data.from_numpy(np.random.randn(n).astype(np.float32))
            k_reduce(data, out)
            times = bench(lambda: k_reduce(data, out), warmup=3, trials=repeats)
            print_row(tag, n, backend, times, n_arrays=1)
            results.append({"bench": "reduce", "size": tag, "backend": backend,
                            "median_ms": median(times) * 1000, "n": n})

        np_data = np.random.randn(n).astype(np.float32).reshape(-1, REDUCE_BLOCK)
        times = bench(lambda: np_data.sum(axis=1), warmup=3, trials=repeats)
        print_row(tag, n, "numpy", times, n_arrays=1)
        results.append({"bench": "reduce", "size": tag, "backend": "numpy",
                        "median_ms": median(times) * 1000, "n": n})

    return results


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Tack Microbenchmark Suite")
    parser.add_argument("--output", "-o", help="Save results to JSON file")
    parser.add_argument("--bench", "-b", nargs="+",
                        choices=["saxpy", "memcpy", "fill", "stencil", "math", "reduce", "all"],
                        default=["all"], help="Which benchmarks to run")
    args = parser.parse_args()

    np.random.seed(42)

    # Print system info
    import Metal as MetalFramework
    device = MetalFramework.MTLCreateSystemDefaultDevice()
    print(f"Metal device: {device.name()}")
    print(f"CPU cores: {__import__('os').cpu_count()}")
    print(f"Data sizes: {', '.join(DATA_SIZES.keys())}")

    benches = args.bench
    if "all" in benches:
        benches = ["saxpy", "memcpy", "fill", "stencil", "math", "reduce"]

    all_results = []
    runners = {
        "saxpy": bench_saxpy,
        "memcpy": bench_memcpy,
        "fill": bench_fill,
        "stencil": bench_stencil,
        "math": bench_math,
        "reduce": bench_reduce,
    }

    for name in benches:
        all_results.extend(runners[name](DATA_SIZES))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
