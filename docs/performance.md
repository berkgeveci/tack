# Tack Performance Results

64 MB of f32 data (16,777,216 elements), median of 30 trials after warmup.

Reproduce: `uv run python examples/bench_64mb.py`

## Mac — Apple Silicon (Metal)

Machine: Apple M-series, unified memory (~200 GB/s bandwidth)

```
  Kernel  | CPU JIT   | Metal GPU  |  NumPy   | Metal vs | Metal vs
          |   (M/s)   |   (M/s)    |  (M/s)   |    CPU   |   NumPy
----------+-----------+------------+----------+----------+-----------
  SAXPY   |     8,311 |     18,823 |    3,900 |     2.3x |      4.8x
  MEMCPY  |    13,418 |     26,134 |   11,414 |     1.9x |      2.3x
  FILL    |    17,592 |     41,327 |   15,930 |     2.3x |      2.6x
  STENCIL |    10,048 |     22,946 |    3,261 |     2.3x |      7.0x
  sqrt    |     8,936 |     24,618 |    6,255 |     2.8x |      3.9x
  sin     |       890 |     26,013 |      697 |      29x |       37x
  exp     |     2,914 |     24,346 |      595 |     8.4x |       41x
  abs     |     9,142 |     27,208 |    8,676 |     3.0x |      3.1x
  reduce  |     1,960 |     39,662 |    5,226 |      20x |      7.6x
```

## Linux — Xeon E5-2650 + RTX 4060 Ti

Machine: 2× Intel Xeon E5-2650 (Sandy Bridge, 2012), 32 threads, DDR3 (~4 GB/s effective).
GPU: NVIDIA GeForce RTX 4060 Ti 16GB, CUDA 13.1, 288 GB/s GDDR6.

```
  Kernel  | CPU JIT   | CUDA GPU   |  NumPy   | CUDA vs  | CUDA vs
          |   (M/s)   |   (M/s)    |  (M/s)   |    CPU   |   NumPy
----------+-----------+------------+----------+----------+-----------
  SAXPY   |     1,139 |     19,973 |      226 |      18x |       88x
  MEMCPY  |     1,875 |     29,420 |      299 |      16x |       98x
  FILL    |     2,084 |     62,367 |      403 |      30x |      155x
  STENCIL |     1,476 |     29,358 |       56 |      20x |      520x
  sqrt    |     1,385 |     29,258 |      277 |      21x |      105x
  sin     |       647 |     29,357 |       77 |      45x |      382x
  exp     |       858 |     29,407 |      112 |      34x |      262x
  abs     |     1,509 |     29,399 |      372 |      19x |       79x
  reduce  |       967 |     26,033 |    1,838 |      27x |       14x
```

## Linux — AMD Instinct MI300X (HIP/ROCm)

Machine: 20-core CPU, AMD Instinct MI300X, ROCm 7.1, HBM3 (~5.3 TB/s bandwidth).

Reproduce: `uv run python benchmarks/bench_cpu_vs_hip.py`

```
  Kernel  | CPU JIT   | HIP GPU    |  Speedup
          |   (ms)    |   (ms)     |  HIP/CPU
----------+-----------+------------+----------
  SAXPY   |     2.516 |      0.079 |     31.7x
  MEMCPY  |     1.642 |      0.064 |     25.8x
  FILL    |     1.206 |      0.060 |     20.2x
  STENCIL |     1.644 |      0.072 |     22.7x
  sqrt    |     1.638 |      0.061 |     26.9x
  sin     |     9.627 |      0.067 |    144.2x
  exp     |     4.951 |      0.068 |     72.5x
  reduce  |     1.636 |      0.062 |     26.5x
```

Data size: 64 MB of f32 (16,777,216 elements), median of 5 trials after warmup.

### Volume Render (800x800, Enzo 64^3 AMR, 718 grids, 1.2M cells)

```
  Backend |  Steady-state  |  Speedup
----------+----------------+----------
  CPU     |     251.3 ms   |     1.0x
  HIP     |       2.4 ms   |   104.7x
```

## Notes

- **Metal GPU vs CUDA GPU** are in the same ballpark (~19-29 G elem/s) despite
  very different architectures. Both are memory-bandwidth limited for simple
  kernels, compute-limited for transcendentals.

- **CPU/NumPy gap** between Mac and Linux is mostly **memory bandwidth**: Apple
  Silicon unified memory delivers ~100-200 GB/s vs ~4 GB/s effective on this
  DDR3 Xeon box. The CPU itself (Sandy Bridge 2012 @ 2.0 GHz, AVX1 only) also
  contributes to the gap.

- **FILL** hits 62 G elem/s on CUDA (write-only, no reads) — close to the
  4060 Ti's theoretical 288 GB/s (62G × 4 bytes = 248 GB/s).

- **sin/exp** show the largest CUDA-vs-CPU speedup (34-45x) because CUDA's
  Special Function Units (SFUs) compute transcendentals in hardware.

- **reduce** favors Metal (39G vs 26G) — likely benefiting from unified memory
  avoiding a writeback path for partial sums.
