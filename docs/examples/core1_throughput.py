# Copyright (c) 2026 Kitware, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Time one kernel, out = alpha * x + y, on a device, against NumPy.

    uv run python docs/examples/core1_throughput.py --arch cpu
    uv run python docs/examples/core1_throughput.py --arch metal

Prints the first call's time, which includes compiling the kernel unless an
earlier run left it in the disk cache, and the median of warm calls, as
milliseconds and as gigabytes moved per second (each element reads x and y
and writes out: 12 bytes).
"""

import argparse
from time import perf_counter

import numpy as np

import tack


# --8<-- [start:kernel]
@tack.kernel
def saxpy(x, y, out, alpha):
    for i in range(x.shape[0]):
        out[i] = alpha * x[i] + y[i]
# --8<-- [end:kernel]


def median_ms(call, repeats):
    times = []
    for _ in range(repeats):
        start = perf_counter()
        call()
        times.append(perf_counter() - start)
    return float(np.median(times)) * 1e3


def run(arch="cpu", n=1 << 24, repeats=30):
    tack.set_default_device(arch)
    host_x = np.random.default_rng(0).random(n, dtype=np.float32)
    host_y = np.ones(n, dtype=np.float32)
    x = tack.array(host_x)
    y = tack.array(host_y)
    out = tack.empty((n,))

    # --8<-- [start:time]
    start = perf_counter()
    saxpy(x, y, out, 2.0)
    tack.sync()                        # a GPU call returns once it is queued
    first_ms = (perf_counter() - start) * 1e3

    for _ in range(5):                 # let the CPU's threading settle
        saxpy(x, y, out, 2.0)

    def warm():
        saxpy(x, y, out, 2.0)
        tack.sync()

    warm_ms = median_ms(warm, repeats)
    # --8<-- [end:time]

    np.testing.assert_allclose(out.numpy(), 2.0 * host_x + host_y, rtol=1e-6)
    numpy_ms = median_ms(lambda: 2.0 * host_x + host_y, repeats)
    gigabytes = 12 * n / 1e9
    print(f"{n:,} elements on {tack.get_default_device()}")
    print(f"  first call    {first_ms:8.2f} ms")
    print(f"  warm call     {warm_ms:8.2f} ms   {gigabytes / (warm_ms / 1e3):6.1f} GB/s")
    print(f"  NumPy         {numpy_ms:8.2f} ms   {gigabytes / (numpy_ms / 1e3):6.1f} GB/s")
    return warm_ms, numpy_ms


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arch", default="cpu", choices=["cpu", "metal", "cuda", "hip", "level_zero"])
    parser.add_argument("--size", type=int, default=1 << 24, help="elements per array")
    args = parser.parse_args()
    run(args.arch, args.size)
