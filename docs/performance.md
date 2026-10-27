# Performance

Tack's visualization filters against VTK on the CPU and Viskores on the GPU,
on synthetic meshes and on real simulation data. Every Tack output is checked
against VTK's before its time counts: counts, sums of positions and values,
areas and volumes. All times are in milliseconds; lower is better.

## Against VTK and Viskores

Unstructured grids of hexahedra and of tetrahedra, at 5M and 20M cells,
timed four ways on one machine:

- **VTK** on the CPU: VTK master built with TBB, 28 threads.
- **Tack on the CPU**: 28 threads, in the same process as VTK.
- **Viskores** 1.2.0 on CUDA, through VTK's Viskores filters.
- **Tack on CUDA**, in the same process as Viskores and on the same device
  arrays.

**Fresh** times include deriving the connectivity Tack needs (faces, point
links) from a mesh it has not seen. VTK and Viskores rebuild theirs on every
call, so fresh is the like-for-like comparison, and the ratio columns use it.
The other Tack columns are warm: the topology keeps what it derived, and a
second filter on the same mesh reuses it.

=== "20M cells"

    | mesh | filter | VTK, 28 threads | Tack CPU | Tack CPU, fresh | VTK / Tack fresh | Viskores, CUDA | Tack CUDA | Tack CUDA, fresh | Viskores / Tack fresh |
    |---|---|---|---|---|---|---|---|---|---|
    | hexahedra | contour ² | 98.3 | 87.9 | 85.5 | 1.15× | 14.1 | 9.53 | 9.46 | 1.49× |
    | hexahedra | slice ² | 118 | 95.9 | 94.0 | 1.26× | 33.7 | 10.1 | 10.0 | 3.4× |
    | hexahedra | threshold | 696 | 204 | 198 | 3.5× | 42.3 | 7.75 | 7.88 | 5.4× |
    | hexahedra | external faces ¹ | 758 | 446 | 455 | 1.67× | 49.2 | 27.8 | 30.1 | 1.63× |
    | hexahedra | point to cell | 69.0 | 28.9 | 28.6 | 2.4× | 3.35 | 1.73 | 1.71 | 1.96× |
    | hexahedra | cell to point | 524 | 36.0 | 224 | 2.3× | 21.0 | 1.81 | 10.1 | 2.1× |
    | hexahedra | point gradient | – | – | – | – | 30.0 | 13.7 | 23.6 | 1.27× |
    | hexahedra | cell gradient | 642 | 49.9 | 49.5 | 13.0× | – | – | – | – |
    | hexahedra | cell volume | 4772 | 194 | 186 | 25.7× | – | – | – | – |
    | hexahedra | cell centers | 792 | 40.2 | 38.5 | 20.6× | – | – | – | – |
    | tetrahedra | contour | 66.5 | 57.6 | 53.6 | 1.24× | 13.7 | 5.95 | 5.95 | 2.3× |
    | tetrahedra | slice | 74.7 | 60.8 | 58.3 | 1.28× | 27.5 | 5.75 | 5.73 | 4.8× |
    | tetrahedra | threshold | 369 | 126 | 121 | 3.0× | 26.0 | 5.43 | 5.33 | 4.9× |
    | tetrahedra | external faces ¹ | 456 | 240 | 238 | 1.92× | 44.2 | 27.0 | 27.0 | 1.64× |
    | tetrahedra | point to cell | 57.4 | 21.6 | 19.9 | 2.9× | 2.89 | 1.28 | 1.25 | 2.3× |
    | tetrahedra | cell to point | 176 | 20.3 | 106 | 1.66× | 10.3 | 1.30 | 10.8 | 0.95× |
    | tetrahedra | point gradient | – | – | – | – | 15.5 | 5.90 | 15.2 | 1.02× |
    | tetrahedra | cell gradient | 567 | 33.9 | 31.9 | 17.8× | – | – | – | – |
    | tetrahedra | cell volume | 2398 | 97.6 | 96.5 | 24.8× | – | – | – | – |
    | tetrahedra | cell centers | 841 | 28.0 | 26.1 | 32.2× | – | – | – | – |

=== "5M cells"

    | mesh | filter | VTK, 28 threads | Tack CPU | Tack CPU, fresh | VTK / Tack fresh | Viskores, CUDA | Tack CUDA | Tack CUDA, fresh | Viskores / Tack fresh |
    |---|---|---|---|---|---|---|---|---|---|
    | hexahedra | contour ² | 28.6 | 45.6 | 38.9 | 0.74× | 8.14 | 7.84 | 7.87 | 1.03× |
    | hexahedra | slice ² | 35.4 | 34.0 | 31.8 | 1.11× | 14.1 | 7.55 | 7.50 | 1.88× |
    | hexahedra | threshold | 174 | 76.7 | 66.5 | 2.6× | 16.4 | 3.62 | 3.73 | 4.4× |
    | hexahedra | external faces ¹ | 184 | 124 | 123 | 1.50× | 19.5 | 9.22 | 9.75 | 2.0× |
    | hexahedra | point to cell | 13.6 | 7.40 | 7.39 | 1.84× | 1.90 | 0.97 | 0.98 | 1.94× |
    | hexahedra | cell to point | 89.2 | 10.1 | 68.3 | 1.31× | 7.22 | 0.93 | 4.23 | 1.71× |
    | hexahedra | point gradient | – | – | – | – | 9.32 | 3.73 | 7.45 | 1.25× |
    | hexahedra | cell gradient | 175 | 17.3 | 18.1 | 9.7× | – | – | – | – |
    | hexahedra | cell volume | 1170 | 48.7 | 46.4 | 25.2× | – | – | – | – |
    | hexahedra | cell centers | 173 | 14.6 | 17.9 | 9.7× | – | – | – | – |
    | tetrahedra | contour | 25.4 | 20.0 | 20.4 | 1.25× | 7.59 | 4.86 | 4.85 | 1.56× |
    | tetrahedra | slice | 22.7 | 19.5 | 19.9 | 1.14× | 13.1 | 4.87 | 4.86 | 2.7× |
    | tetrahedra | threshold | 77.4 | 29.6 | 29.0 | 2.7× | 11.5 | 3.30 | 3.33 | 3.5× |
    | tetrahedra | external faces ¹ | 111 | 70.6 | 71.8 | 1.55× | 17.5 | 9.11 | 9.08 | 1.93× |
    | tetrahedra | point to cell | 11.3 | 5.50 | 5.04 | 2.2× | 1.74 | 0.80 | 0.83 | 2.1× |
    | tetrahedra | cell to point | 48.5 | 8.73 | 36.6 | 1.33× | 4.46 | 0.80 | 4.11 | 1.09× |
    | tetrahedra | point gradient | – | – | – | – | 5.72 | 2.09 | 5.32 | 1.08× |
    | tetrahedra | cell gradient | 166 | 12.7 | 11.3 | 14.7× | – | – | – | – |
    | tetrahedra | cell volume | 588 | 28.5 | 29.9 | 19.7× | – | – | – | – |
    | tetrahedra | cell centers | 179 | 10.9 | 8.86 | 20.2× | – | – | – | – |

¹ Warm external faces were as slow as fresh in this run: the surface was
recomputed on every call. Since then (#167) the topology keeps it, and a
repeated call only carries the fields.

² Contour and slice of hexahedra on CUDA were 1.3–1.5× slower here than a
week earlier (hex 5M contour 5.4 → 7.8 ms), which is being measured.

- **On the GPU**, Tack is ahead of Viskores on every filter fresh except
  tetrahedra's cell to point at 20M (0.95×), up to 5.4× on threshold and
  4.8× on slice. Warm, it is ahead everywhere, by 1.04–12×.
- **On the CPU**, Tack is ahead of VTK everywhere but contour of hexahedra at
  5M (0.74×): by 9.7–32× on cell gradients, volumes and centers, and 1.1–3.5×
  on contour, slice, threshold, external faces and the conversions between
  point and cell data.
- Viskores' default memory, `cudaMallocManaged`, is 4.7–25× slower than the
  `cudaMalloc` it ran with here (`NO_VISKORES_MANAGED_MEMORY=1`), from page
  faults on new buffers. No transfer between the host and the device is in
  Viskores' times: it was handed Tack's device arrays and kept its output on
  the device, and a trace of each run confirmed no copies beyond sizes and
  counts.

Machine: jura, AMD EPYC 7453 (28 cores), NVIDIA RTX 6000 Ada (driver
615.71.09), idle. Tack `main` at `291fcfe` (2026-10-26). Each time is the
mean of two passes, the second in reverse order. Harness:
`benchmarks/vtk_compare` and `benchmarks/viskores_compare`.

## Real data, against VTK

Four datasets from production simulations, read with VTK's readers. Tack's
CPU against VTK on 8 threads each, on an Apple M1 Max. Every check agrees
with VTK's output.

| dataset | step | VTK | Tack | VTK / Tack |
|---|---|---|---|---|
| ONERA M6 wing, 46.9M cells (tetrahedra and wedges) | density isosurface | 258 | 81 | 3.2× |
| | span station (slice) | 120 | 120 | 1.0× |
| | pressure gradients | 1289 | 88 | 14.6× |
| | low-density cells (threshold) | 107 | 83 | 1.3× |
| | outer surfaces of a fresh mesh | 1122 | 1178 | 0.95× |
| | streamlines, 17 seeds, by length ³ | 3900 | 1000 | 3.9× |
| aneurysm blood flow, 8.7M tetrahedra | isosurface, slice, gradients | | | 2.7×, 2.0×, 17× |
| | streamlines, 25 seeds ³ | 3.6 | 1.7 | 2.1× |
| LANL wildfire, 18.7M curvilinear cells | plume, slice, gradients, fuel cells, surface | | | 4.3×, 1.9×, 33×, 4.0×, 5.9× |
| 512 pieces of a simulation's output, 5.5M voxels | global ids | 10,400 | 9,000 | 1.2× |
| | a layer of ghost cells | 27,300 | 28,700 | 0.95× |

³ Tack's streamlines try the cell a particle was last found in before
searching (#166). VTK's default interpolator finds cells through the nearest
point, which ends 7 of the M6's 17 paths early, inside the mesh. With VTK's
cell-locator interpolator, every path runs its full length, within 3.8e-6 of
Tack's, in 22.9 s.

## Kernel throughput (historical)

??? note "Earlier microbenchmarks, from before October 2026"

    These measurements predate safe floating-point math on CUDA and Metal,
    the CPU threading policy's repairs, the CPU's disjoint-array variants,
    queued GPU launches and the CPU thread pool, all in October 2026. Their
    reproduce scripts time a call without `tack.sync()`, so their GPU columns
    would now time the launch rather than the kernel. They are kept as a
    record and have not been edited.

    64 MB of f32 data (16,777,216 elements), median of 30 trials after warmup.

    Reproduce: `uv run python benchmarks/bench_64mb.py`

    #### Mac — Apple Silicon (Metal)

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

    #### Linux — Xeon E5-2650 + RTX 4060 Ti

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

    #### Linux — AMD Instinct MI300X (HIP/ROCm)

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

    #### Volume Render (800x800, Enzo 64^3 AMR, 718 grids, 1.2M cells)

    ```
      Backend |  Steady-state  |  Speedup
    ----------+----------------+----------
      CPU     |     251.3 ms   |     1.0x
      HIP     |       2.4 ms   |   104.7x
    ```

    #### Notes

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
