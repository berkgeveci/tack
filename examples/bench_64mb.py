"""64 MB benchmark — CPU JIT vs CUDA GPU vs NumPy, formatted table."""

import time
import numpy as np
import pgc


# 64 MB of f32 = 16,777,216 elements
N = 64 * 1024 * 1024 // 4


def bench(fn, warmup=5, trials=30):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return sorted(times)[len(times) // 2]


# ── Kernel definitions ──────────────────────────────────────────

@pgc.kernel
def k_saxpy(x, y, out):
    for i in range(x.shape[0]):
        out[i] = 2.0 * x[i] + y[i]

@pgc.kernel
def k_memcpy(src, dst):
    for i in range(src.shape[0]):
        dst[i] = src[i]

@pgc.kernel
def k_fill(out):
    for i in range(out.shape[0]):
        out[i] = 42.0

@pgc.kernel
def k_stencil(src, dst):
    for i in range(src.shape[0]):
        if i > 0:
            if i < 16777215:
                dst[i] = 0.25 * src[i - 1] + 0.5 * src[i] + 0.25 * src[i + 1]

@pgc.kernel
def k_sqrt(x, out):
    for i in range(x.shape[0]):
        out[i] = sqrt(x[i])

@pgc.kernel
def k_sin(x, out):
    for i in range(x.shape[0]):
        out[i] = sin(x[i])

@pgc.kernel
def k_exp(x, out):
    for i in range(x.shape[0]):
        out[i] = exp(x[i])

@pgc.kernel
def k_abs(x, out):
    for i in range(x.shape[0]):
        out[i] = abs(x[i])

REDUCE_BLOCK = 256
N_BLOCKS = N // REDUCE_BLOCK

@pgc.kernel
def k_reduce(data, out):
    for block_idx in range(out.shape[0]):
        s = 0.0
        for j in range(256):
            idx = block_idx * 256 + j
            s = s + data[idx]
        out[block_idx] = s


# ── NumPy references ────────────────────────────────────────────

def np_saxpy(x, y):     return 2.0 * x + y
def np_memcpy(src):      return src.copy()
def np_fill(n):          return np.full(n, 42.0, dtype=np.float32)
def np_stencil(src):
    dst = src.copy()
    dst[1:-1] = 0.25 * src[:-2] + 0.5 * src[1:-1] + 0.25 * src[2:]
    return dst
def np_sqrt(x):          return np.sqrt(x)
def np_sin(x):           return np.sin(x)
def np_exp(x):           return np.exp(x)
def np_abs(x):           return np.abs(x)
def np_reduce(x):        return x.reshape(-1, 256).sum(axis=1)


# ── Benchmark runner ────────────────────────────────────────────

def throughput(median_sec):
    return N / median_sec / 1e6  # M elements/sec


def run_kernel(name, pgc_fn, pgc_setup, np_fn, np_setup):
    """Benchmark one kernel across CPU, CUDA, NumPy. Returns (cpu, cuda, numpy) M/s."""
    results = {}

    for label, arch in [("cpu", pgc.cpu), ("cuda", pgc.cuda)]:
        pgc.init(arch=arch)
        fields = pgc_setup()
        pgc_fn(*fields)  # warmup / compile
        med = bench(lambda: pgc_fn(*fields))
        results[label] = throughput(med)

    np_args = np_setup()
    np_fn(*np_args)  # warmup
    med = bench(lambda: np_fn(*np_args))
    results["numpy"] = throughput(med)

    return results


def main():
    np.random.seed(42)

    print(f"64 MB benchmark  ({N:,} f32 elements)")
    print(f"CPU cores: {__import__('os').cpu_count()}")
    try:
        from cuda.bindings import driver
        driver.cuInit(0)
        err, dev = driver.cuDeviceGet(0)
        err, name_bytes = driver.cuDeviceGetName(256, dev)
        print(f"CUDA device: {name_bytes.split(b'\\x00')[0].decode()}")
    except Exception:
        pass
    print()

    # ── Define benchmarks ──
    def setup_2in_1out():
        x = pgc.field(dtype=pgc.f32, shape=(N,))
        y = pgc.field(dtype=pgc.f32, shape=(N,))
        out = pgc.field(dtype=pgc.f32, shape=(N,))
        x.from_numpy(np.random.randn(N).astype(np.float32).clip(0.1, 10.0))
        y.from_numpy(np.random.randn(N).astype(np.float32))
        return (x, y, out)

    def setup_1in_1out():
        x = pgc.field(dtype=pgc.f32, shape=(N,))
        out = pgc.field(dtype=pgc.f32, shape=(N,))
        x.from_numpy(np.abs(np.random.randn(N).astype(np.float32)) + 0.1)
        return (x, out)

    def setup_copy():
        src = pgc.field(dtype=pgc.f32, shape=(N,))
        dst = pgc.field(dtype=pgc.f32, shape=(N,))
        src.from_numpy(np.random.randn(N).astype(np.float32))
        return (src, dst)

    def setup_fill():
        out = pgc.field(dtype=pgc.f32, shape=(N,))
        return (out,)

    def setup_reduce():
        data = pgc.field(dtype=pgc.f32, shape=(N,))
        out = pgc.field(dtype=pgc.f32, shape=(N_BLOCKS,))
        data.from_numpy(np.random.randn(N).astype(np.float32))
        return (data, out)

    # numpy setup (host arrays, reused)
    np_x = np.abs(np.random.randn(N).astype(np.float32)) + 0.1
    np_y = np.random.randn(N).astype(np.float32)
    np_src = np.random.randn(N).astype(np.float32)

    benchmarks = [
        ("SAXPY",   k_saxpy,   setup_2in_1out, np_saxpy,   lambda: (np_x, np_y)),
        ("MEMCPY",  k_memcpy,  setup_copy,     np_memcpy,  lambda: (np_src,)),
        ("FILL",    k_fill,    setup_fill,     lambda n: np.full(n, 42.0, dtype=np.float32), lambda: (N,)),
        ("STENCIL", k_stencil, setup_copy,     np_stencil, lambda: (np_src,)),
        ("sqrt",    k_sqrt,    setup_1in_1out, np_sqrt,    lambda: (np_x,)),
        ("sin",     k_sin,     setup_1in_1out, np_sin,     lambda: (np_x,)),
        ("exp",     k_exp,     setup_1in_1out, np_exp,     lambda: (np_x,)),
        ("abs",     k_abs,     setup_1in_1out, np_abs,     lambda: (np_x,)),
        ("reduce",  k_reduce,  setup_reduce,   np_reduce,  lambda: (np_src,)),
    ]

    rows = []
    for name, pgc_fn, pgc_setup, np_fn, np_setup in benchmarks:
        r = run_kernel(name, pgc_fn, pgc_setup, np_fn, np_setup)
        rows.append((name, r))

    # ── Print table ──
    hdr  = "  Kernel  | CPU JIT   | CUDA GPU   |  NumPy   | CUDA vs  | CUDA vs   "
    hdr2 = "          |   (M/s)   |   (M/s)    |  (M/s)   |    CPU   |   NumPy   "
    sep  = "----------+-----------+------------+----------+----------+-----------"

    print(sep)
    print(hdr)
    print(hdr2)
    print(sep)

    for name, r in rows:
        cpu  = r["cpu"]
        cuda = r["cuda"]
        npy  = r["numpy"]
        vs_cpu = cuda / cpu if cpu > 0 else 0
        vs_npy = cuda / npy if npy > 0 else 0

        def fmt_ratio(x):
            if x >= 10:
                return f"{x:.0f}x"
            return f"{x:.1f}x"

        print(f"  {name:<7s} | {cpu:>9,.0f} | {cuda:>10,.0f} | {npy:>8,.0f} | {fmt_ratio(vs_cpu):>8s} | {fmt_ratio(vs_npy):>9s} ")
        print(sep)

    print()


if __name__ == "__main__":
    main()
