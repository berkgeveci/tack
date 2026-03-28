"""Validation suite -- runs Taichi-style examples on both CPU and Metal backends.

Tests:
  1. Vector add       -- simplest kernel, validates basic pipeline
  2. SAXPY            -- scalar * vector + vector, fused ops
  3. Reduction        -- sum of array (multi-pass)
  4. Mandelbrot       -- nested loops, complex math, 2D output
  5. N-body           -- multiple fields, distance calculations
  6. Jacobi iteration -- stencil pattern, read/write fields
  7. Matrix multiply  -- 2D indexing, accumulation
"""

import time
import numpy as np
import tack


def _available_backends():
    """Detect which backends are available on this machine."""
    backends = ["cpu"]
    try:
        import Metal
        if Metal.MTLCreateSystemDefaultDevice() is not None:
            backends.append("metal")
    except ImportError:
        pass
    try:
        from cuda.bindings import driver
        driver.cuInit(0)
        err, dev = driver.cuDeviceGet(0)
        if err == driver.CUresult.CUDA_SUCCESS:
            backends.append("cuda")
    except (ImportError, Exception):
        pass
    return backends


BACKENDS = _available_backends()


def run_on_all(name, setup_fn, kernel_fn, verify_fn):
    """Run a validation test on all available backends, verify correctness."""
    print(f"\n{'-' * 60}")
    print(f"  {name}")
    print(f"{'-' * 60}")

    for backend in BACKENDS:
        tack.init(arch=backend)
        fields = setup_fn()
        t0 = time.perf_counter()
        kernel_fn(*fields)
        t1 = time.perf_counter()
        ok = verify_fn(*fields)
        status = "OK" if ok else "FAIL"
        print(f"  {backend:>5s}:  {(t1-t0)*1000:>8.2f} ms  [{status}]")
        if not ok:
            raise AssertionError(f"{name} failed on {backend}")


# -------------------------------------------------------------
# 1. Vector add
# -------------------------------------------------------------

@tack.kernel
def vector_add(x, y, out):
    for i in range(x.shape[0]):
        out[i] = x[i] + y[i]


def va_setup():
    n = 100_000
    x = tack.field(dtype=tack.f32, shape=(n,))
    y = tack.field(dtype=tack.f32, shape=(n,))
    out = tack.field(dtype=tack.f32, shape=(n,))
    x.from_numpy(np.arange(n, dtype=np.float32))
    y.from_numpy(np.ones(n, dtype=np.float32) * 2.0)
    return (x, y, out)


def va_verify(x, y, out):
    result = out.to_numpy()
    expected = x.to_numpy() + y.to_numpy()
    return np.allclose(result, expected)


# -------------------------------------------------------------
# 2. SAXPY
# -------------------------------------------------------------

@tack.kernel
def saxpy(x, y, out):
    for i in range(x.shape[0]):
        out[i] = 2.5 * x[i] + y[i]


def saxpy_setup():
    n = 100_000
    x = tack.field(dtype=tack.f32, shape=(n,))
    y = tack.field(dtype=tack.f32, shape=(n,))
    out = tack.field(dtype=tack.f32, shape=(n,))
    x.from_numpy(np.random.randn(n).astype(np.float32))
    y.from_numpy(np.random.randn(n).astype(np.float32))
    return (x, y, out)


def saxpy_verify(x, y, out):
    result = out.to_numpy()
    expected = 2.5 * x.to_numpy() + y.to_numpy()
    return np.allclose(result, expected, rtol=1e-4, atol=1e-6)


# -------------------------------------------------------------
# 3. Reduction (parallel partial sums + CPU final reduce)
# -------------------------------------------------------------

REDUCE_BLOCK = 256


@tack.kernel
def partial_sum(data, out):
    for block_idx in range(out.shape[0]):
        s = 0.0
        for j in range(256):
            idx = block_idx * 256 + j
            s = s + data[idx]
        out[block_idx] = s


def reduce_setup():
    n = REDUCE_BLOCK * 1024  # 262,144
    data = tack.field(dtype=tack.f32, shape=(n,))
    partial = tack.field(dtype=tack.f32, shape=(1024,))
    data.from_numpy(np.ones(n, dtype=np.float32))
    return (data, partial)


def reduce_verify(data, partial):
    # Each partial sum should be REDUCE_BLOCK (256 ones)
    partials = partial.to_numpy()
    total = float(np.sum(partials))
    expected = float(data.shape[0])  # all ones
    return abs(total - expected) < 1.0


# -------------------------------------------------------------
# 4. Mandelbrot
# -------------------------------------------------------------

WIDTH = 800
HEIGHT = 600
MAX_ITER = 100


@tack.kernel
def mandelbrot(pixels):
    for i in range(pixels.shape[0]):
        # Convert linear index to 2D
        px = i % 800
        py = i // 800

        # Map to complex plane [-2, 1] x [-1.5, 1.5]
        x0 = -2.0 + px * 3.0 / 800.0
        y0 = -1.5 + py * 3.0 / 600.0

        x = 0.0
        y = 0.0
        count = 0.0

        j = 0
        while j < 100:
            x_new = x * x - y * y + x0
            y = 2.0 * x * y + y0
            x = x_new
            if x * x + y * y > 4.0:
                j = 100
            count = count + 1.0
            j = j + 1

        pixels[i] = count


def mandelbrot_setup():
    pixels = tack.field(dtype=tack.f32, shape=(WIDTH * HEIGHT,))
    return (pixels,)


def mandelbrot_verify(pixels):
    data = pixels.to_numpy().reshape(HEIGHT, WIDTH)
    # Center of Mandelbrot set (0,0 maps to px=533,py=300) should have max iterations
    center_px = int(2.0 / 3.0 * WIDTH)  # x0=0 -> px=533
    center_py = int(1.5 / 3.0 * HEIGHT)  # y0=0 -> py=300
    center_val = data[center_py, center_px]
    if center_val < MAX_ITER:
        return False

    # Point clearly outside the set: x0=1.5, y0=1.5
    outside_px = int((1.5 + 2.0) / 3.0 * WIDTH)
    outside_py = int((1.5 + 1.5) / 3.0 * HEIGHT)
    if outside_px < WIDTH and outside_py < HEIGHT:
        outside_val = data[outside_py, outside_px]
        if outside_val >= MAX_ITER:
            return False

    # Check that the output has a reasonable distribution
    # (not all zeros, not all MAX_ITER)
    in_set = np.sum(data >= MAX_ITER)
    escaped = np.sum(data < MAX_ITER)
    if in_set < 1000 or escaped < 1000:
        return False

    return True


# -------------------------------------------------------------
# 5. N-body (gravitational force calculation)
# -------------------------------------------------------------

N_BODIES = 512
SOFTENING = 0.01


@tack.kernel
def nbody_forces(px, py, pz, mass, fx, fy, fz):
    for i in range(px.shape[0]):
        ax = 0.0
        ay = 0.0
        az = 0.0
        for j in range(512):
            dx = px[j] - px[i]
            dy = py[j] - py[i]
            dz = pz[j] - pz[i]
            dist_sq = dx * dx + dy * dy + dz * dz + 0.01
            inv_dist = 1.0 / sqrt(dist_sq)
            inv_dist3 = inv_dist * inv_dist * inv_dist
            ax = ax + mass[j] * dx * inv_dist3
            ay = ay + mass[j] * dy * inv_dist3
            az = az + mass[j] * dz * inv_dist3
        fx[i] = ax
        fy[i] = ay
        fz[i] = az


def nbody_setup():
    np.random.seed(42)
    n = N_BODIES
    px = tack.field(dtype=tack.f32, shape=(n,))
    py = tack.field(dtype=tack.f32, shape=(n,))
    pz = tack.field(dtype=tack.f32, shape=(n,))
    mass = tack.field(dtype=tack.f32, shape=(n,))
    fx = tack.field(dtype=tack.f32, shape=(n,))
    fy = tack.field(dtype=tack.f32, shape=(n,))
    fz = tack.field(dtype=tack.f32, shape=(n,))

    px.from_numpy(np.random.randn(n).astype(np.float32))
    py.from_numpy(np.random.randn(n).astype(np.float32))
    pz.from_numpy(np.random.randn(n).astype(np.float32))
    mass.from_numpy(np.ones(n, dtype=np.float32))
    return (px, py, pz, mass, fx, fy, fz)


def nbody_verify(px, py, pz, mass, fx, fy, fz):
    # Verify against numpy reference (vectorized, no Python loops)
    x = px.to_numpy().astype(np.float64)
    y = py.to_numpy().astype(np.float64)
    z = pz.to_numpy().astype(np.float64)
    m = mass.to_numpy().astype(np.float64)
    n = len(m)

    # dx[i,j] = x[j] - x[i]
    dx = x[np.newaxis, :] - x[:, np.newaxis]
    dy = y[np.newaxis, :] - y[:, np.newaxis]
    dz = z[np.newaxis, :] - z[:, np.newaxis]
    dist_sq = dx * dx + dy * dy + dz * dz + SOFTENING
    inv_dist3 = dist_sq ** (-1.5)

    ref_fx = np.sum(m[np.newaxis, :] * dx * inv_dist3, axis=1)
    ref_fy = np.sum(m[np.newaxis, :] * dy * inv_dist3, axis=1)
    ref_fz = np.sum(m[np.newaxis, :] * dz * inv_dist3, axis=1)

    return (np.allclose(fx.to_numpy(), ref_fx.astype(np.float32), rtol=1e-3, atol=1e-3) and
            np.allclose(fy.to_numpy(), ref_fy.astype(np.float32), rtol=1e-3, atol=1e-3) and
            np.allclose(fz.to_numpy(), ref_fz.astype(np.float32), rtol=1e-3, atol=1e-3))


# -------------------------------------------------------------
# 6. Jacobi iteration (1D heat equation)
# -------------------------------------------------------------

JACOBI_N = 1024
JACOBI_STEPS = 100


@tack.kernel
def jacobi_step(src, dst):
    for i in range(src.shape[0]):
        if i > 0:
            if i < 1023:
                dst[i] = 0.5 * (src[i - 1] + src[i + 1])


def jacobi_setup():
    src = tack.field(dtype=tack.f32, shape=(JACOBI_N,))
    dst = tack.field(dtype=tack.f32, shape=(JACOBI_N,))
    # Initial: hot at center
    init = np.zeros(JACOBI_N, dtype=np.float32)
    init[JACOBI_N // 2] = 100.0
    src.from_numpy(init)
    dst.from_numpy(init)
    return (src, dst)


def jacobi_verify(src, dst):
    # Re-run reference in numpy
    ref = np.zeros(JACOBI_N, dtype=np.float32)
    ref[JACOBI_N // 2] = 100.0
    ref2 = ref.copy()
    for _ in range(JACOBI_STEPS):
        ref2[1:-1] = 0.5 * (ref[:-2] + ref[2:])
        ref, ref2 = ref2, ref
    return np.allclose(src.to_numpy(), ref, atol=1e-3)


def run_jacobi(src, dst):
    for _ in range(JACOBI_STEPS):
        jacobi_step(src, dst)
        # Swap: copy dst -> src for next iteration
        src.from_numpy(dst.to_numpy())


# -------------------------------------------------------------
# 7. Matrix multiply
# -------------------------------------------------------------

MAT_N = 64


@tack.kernel
def matmul(a, b, c):
    for i in range(64):
        for j in range(64):
            s = 0.0
            for k in range(64):
                s = s + a[i * 64 + k] * b[k * 64 + j]
            c[i * 64 + j] = s


def matmul_setup():
    np.random.seed(123)
    n = MAT_N
    a = tack.field(dtype=tack.f32, shape=(n * n,))
    b = tack.field(dtype=tack.f32, shape=(n * n,))
    c = tack.field(dtype=tack.f32, shape=(n * n,))
    a.from_numpy(np.random.randn(n * n).astype(np.float32))
    b.from_numpy(np.random.randn(n * n).astype(np.float32))
    return (a, b, c)


def matmul_verify(a, b, c):
    n = MAT_N
    np_a = a.to_numpy().reshape(n, n)
    np_b = b.to_numpy().reshape(n, n)
    np_c = c.to_numpy().reshape(n, n)
    expected = np_a @ np_b
    return np.allclose(np_c, expected, rtol=1e-3, atol=1e-3)


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------

def main():
    np.random.seed(42)
    print("Tack Validation Suite")
    print("=" * 60)

    run_on_all("1. Vector Add", va_setup, vector_add, va_verify)
    run_on_all("2. SAXPY", saxpy_setup, saxpy, saxpy_verify)
    run_on_all("3. Reduction (partial sums)", reduce_setup, partial_sum, reduce_verify)
    run_on_all("4. Mandelbrot (800x600)", mandelbrot_setup, mandelbrot, mandelbrot_verify)
    run_on_all("5. N-body (512 bodies)", nbody_setup, nbody_forces, nbody_verify)

    # Jacobi needs special handling (multi-step iteration)
    print(f"\n{'-' * 60}")
    print(f"  6. Jacobi Iteration (1D, {JACOBI_STEPS} steps)")
    print(f"{'-' * 60}")
    for backend in BACKENDS:
        tack.init(arch=backend)
        src, dst = jacobi_setup()
        t0 = time.perf_counter()
        run_jacobi(src, dst)
        t1 = time.perf_counter()
        ok = jacobi_verify(src, dst)
        status = "OK" if ok else "FAIL"
        print(f"  {backend:>5s}:  {(t1-t0)*1000:>8.2f} ms  [{status}]")
        if not ok:
            raise AssertionError(f"Jacobi failed on {backend}")

    run_on_all("7. Matrix Multiply (64x64)", matmul_setup, matmul, matmul_verify)

    print(f"\n{'=' * 60}")
    print("  All validations passed!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
