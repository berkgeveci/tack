"""Thorough performance benchmark — CPU JIT vs Metal GPU vs NumPy.

Tests across:
  - Data sizes: 1K → 10M elements
  - Kernel complexity: simple add, SAXPY, multi-op, math-heavy, conditional
  - Measures: compilation time, per-call latency, throughput (elements/sec)
"""

import time
import numpy as np
import pgc


# ---------------------------------------------------------------------------
# Benchmark infrastructure
# ---------------------------------------------------------------------------

def bench(fn, warmup=3, trials=20):
    """Benchmark a callable. Returns list of times in seconds."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return times


def stats(times):
    """Return (median, min, max) from a list of times."""
    s = sorted(times)
    n = len(s)
    return s[n // 2], s[0], s[-1]


def fmt_throughput(n, median_sec):
    """Format throughput as M elements/sec."""
    if median_sec <= 0:
        return "inf"
    return f"{n / median_sec / 1e6:,.0f}"


def print_row(label, n, times):
    med, lo, hi = stats(times)
    tp = fmt_throughput(n, med)
    print(f"  {label:<12s}  median {med*1e6:>8.0f} µs  "
          f"(min {lo*1e6:.0f}, max {hi*1e6:.0f})  "
          f"{tp:>8s} M elem/s")


# ---------------------------------------------------------------------------
# Kernel definitions (defined once, used by both backends)
# ---------------------------------------------------------------------------

@pgc.kernel
def k_add(x, y, out):
    for i in range(x.shape[0]):
        out[i] = x[i] + y[i]


@pgc.kernel
def k_saxpy(x, y, out):
    for i in range(x.shape[0]):
        out[i] = 2.0 * x[i] + y[i]


@pgc.kernel
def k_multi(a, b, c, out):
    for i in range(a.shape[0]):
        out[i] = (a[i] - b[i]) * c[i] / 2.0 + 1.0


@pgc.kernel
def k_math(x, out):
    for i in range(x.shape[0]):
        out[i] = sqrt(x[i] * x[i] + 1.0)


@pgc.kernel
def k_cond(x, out):
    for i in range(x.shape[0]):
        if x[i] > 0.0:
            out[i] = x[i] * 2.0
        else:
            out[i] = -x[i] * 0.5


# ---------------------------------------------------------------------------
# NumPy reference implementations
# ---------------------------------------------------------------------------

def np_add(x, y):
    return x + y

def np_saxpy(x, y):
    return 2.0 * x + y

def np_multi(a, b, c):
    return (a - b) * c / 2.0 + 1.0

def np_math(x):
    return np.sqrt(x * x + 1.0)

def np_cond(x):
    return np.where(x > 0.0, x * 2.0, -x * 0.5)


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

SIZES = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]

KERNELS = [
    ("add",   k_add,   lambda n: 3, np_add,   lambda n: 2),
    ("saxpy", k_saxpy, lambda n: 3, np_saxpy, lambda n: 2),
    ("multi", k_multi, lambda n: 4, np_multi, lambda n: 3),
    ("math",  k_math,  lambda n: 2, np_math,  lambda n: 1),
    ("cond",  k_cond,  lambda n: 2, np_cond,  lambda n: 1),
]


def make_fields(n, count):
    """Create `count` f32 fields of size n, filled with test data."""
    fields = []
    for j in range(count):
        f = pgc.field(dtype=pgc.f32, shape=(n,))
        # Use different data per field to avoid trivial patterns
        f.from_numpy(np.random.randn(n).astype(np.float32) + (j + 1.0))
        fields.append(f)
    return fields


def _available_backends():
    """Detect which GPU backends are available."""
    backends = [("CPU JIT", pgc.cpu)]
    try:
        import Metal
        if Metal.MTLCreateSystemDefaultDevice() is not None:
            backends.append(("Metal GPU", pgc.metal))
    except ImportError:
        pass
    try:
        from cuda.bindings import driver
        driver.cuInit(0)
        err, dev = driver.cuDeviceGet(0)
        if err == driver.CUresult.CUDA_SUCCESS:
            backends.append(("CUDA GPU", pgc.cuda))
    except (ImportError, Exception):
        pass
    return backends


BENCH_BACKENDS = _available_backends()


def run_kernel_bench(name, kernel, nfields_fn, np_fn, np_nargs_fn, n):
    """Benchmark one kernel at one size across all available backends + NumPy."""
    nfields = nfields_fn(n)

    for label, arch in BENCH_BACKENDS:
        pgc.init(arch=arch)
        fields = make_fields(n, nfields)
        kernel(*fields)  # warmup (includes compilation)
        times = bench(lambda: kernel(*fields))
        print_row(label, n, times)

    # --- NumPy ---
    pgc.init(arch=pgc.cpu)
    ref_fields = make_fields(n, nfields)
    np_args = [ref_fields[j].to_numpy() for j in range(np_nargs_fn(n))]
    np_times = bench(lambda: np_fn(*np_args))
    print_row("NumPy", n, np_times)


def bench_compilation():
    """Measure first-call (compilation) time for each backend."""
    print("=" * 72)
    print("COMPILATION TIME (first call, n=10,000)")
    print("=" * 72)

    n = 10_000

    for name, kernel_fn, nfields_fn, _, _ in KERNELS:
        print(f"\n  Kernel: {name}")

        for label, arch in BENCH_BACKENDS:
            pgc.init(arch=arch)
            kernel_fn._compiled = {}
            fields = make_fields(n, nfields_fn(n))
            t0 = time.perf_counter()
            kernel_fn(*fields)
            t1 = time.perf_counter()
            print(f"    {label + ' compile:':<22s} {(t1-t0)*1000:>8.1f} ms")


def main():
    np.random.seed(42)

    # Print system info
    import os
    print(f"CPU cores: {os.cpu_count()}")
    for label, arch in BENCH_BACKENDS:
        if arch == pgc.metal:
            try:
                import Metal as MetalFramework
                device = MetalFramework.MTLCreateSystemDefaultDevice()
                print(f"Metal device: {device.name()}")
            except ImportError:
                pass
        elif arch == pgc.cuda:
            try:
                from cuda.bindings import driver
                driver.cuInit(0)
                err, dev = driver.cuDeviceGet(0)
                err, name_bytes = driver.cuDeviceGetName(256, dev)
                name_str = name_bytes.split(b'\x00')[0].decode()
                print(f"CUDA device: {name_str}")
            except Exception:
                pass
    print()

    # Compilation benchmarks
    bench_compilation()

    # Per-kernel, per-size benchmarks
    for name, kernel, nfields_fn, np_fn, np_nargs_fn in KERNELS:
        print()
        print("=" * 72)
        print(f"KERNEL: {name}")
        print("=" * 72)

        for n in SIZES:
            print(f"\n  n = {n:>12,}")
            run_kernel_bench(name, kernel, nfields_fn, np_fn, np_nargs_fn, n)


if __name__ == "__main__":
    main()
