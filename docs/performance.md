# Performance

Tack's visualization filters against VTK and Viskores, on the CPU and the
GPU, on synthetic meshes and on real simulation data. Every Tack output is checked
against VTK's before its time counts: counts, sums of positions and values,
areas and volumes.

Each entry below is how many times faster Tack is: the other system's time
over Tack's, so above 1× Tack is faster. Tack's time is its **fresh** time,
which includes deriving the connectivity it needs (faces, point links) from a
mesh it has not seen, since VTK and Viskores rebuild theirs on every call.
The running times, fresh and warm, are [at the bottom](#running-times).

## Linux: AMD EPYC 7453 and NVIDIA RTX 6000 Ada

Unstructured grids of 20M hexahedra and of 20M tetrahedra. VTK on 28 CPU
threads (TBB), Tack's CPU on 28 threads, Viskores 1.2.0 and Tack on CUDA.

| mesh | filter | VTK / Tack CPU | Viskores CUDA / Tack CUDA | VTK / Tack CUDA |
|---|---|---|---|---|
| hexahedra | contour | 1.15× | 1.49× | 10.4× |
| hexahedra | slice | 1.26× | 3.4× | 11.8× |
| hexahedra | threshold | 3.5× | 5.4× | 88.3× |
| hexahedra | external faces | 1.66× | 1.63× | 25.2× |
| hexahedra | point to cell | 2.4× | 1.96× | 40.4× |
| hexahedra | cell to point | 2.3× | 2.1× | 51.9× |
| hexahedra | point gradient | – | 1.27× | – |
| hexahedra | cell gradient | 13.0× | – | – |
| hexahedra | cell volume | 25.7× | – | – |
| hexahedra | cell centers | 20.6× | – | – |
| tetrahedra | contour | 1.24× | 2.3× | 11.2× |
| tetrahedra | slice | 1.28× | 4.8× | 13.0× |
| tetrahedra | threshold | 3.0× | 4.9× | 69.3× |
| tetrahedra | external faces | 1.92× | 1.64× | 16.9× |
| tetrahedra | point to cell | 2.9× | 2.3× | 45.9× |
| tetrahedra | cell to point | 1.66× | 0.95× | 16.3× |
| tetrahedra | point gradient | – | 1.02× | – |
| tetrahedra | cell gradient | 17.8× | – | – |
| tetrahedra | cell volume | 24.9× | – | – |
| tetrahedra | cell centers | 32.2× | – | – |

- **On the GPU**, Tack is ahead of Viskores on every filter but tetrahedra's
  cell to point (0.95×), up to 5.4× on threshold and 4.8× on slice. Warm,
  it is ahead on every filter, by 1.04–12×.
- **On the CPU**, Tack is ahead of VTK on every filter, by 13–32× on cell
  gradients, volumes and centers. At 5M cells, contour of hexahedra is the one
  filter where VTK is faster (0.74×).
- **Tack on the GPU against VTK on the CPU**: 10–88×.

Contour and slice of hexahedra on CUDA were 1.3–1.5× slower in this run than
a week earlier (5M hexahedra's contour 5.4 → 7.8 ms), which is being looked
into.

## macOS: Apple M1 Max

Unstructured grids of 20M hexahedra and of 20M tetrahedra. VTK on 8 CPU
threads (TBB), Tack's CPU on 8 threads, Viskores 1.2.0 on 8 TBB threads,
and Tack on the M1 Max's GPU through Metal. Viskores has no Metal device, so
on the Mac it runs on the CPU, and its column against Tack on Metal compares
the best each has here.

| mesh | filter | VTK / Tack CPU | Viskores TBB / Tack CPU | VTK / Tack Metal | Viskores TBB / Tack Metal |
|---|---|---|---|---|---|
| hexahedra | contour | 1.04× | 0.87× | 1.89× | 1.48× |
| hexahedra | slice | 1.05× | 18.3× | 1.85× | 35.7× |
| hexahedra | threshold | 3.6× | 5.2× | 6.3× | 9.4× |
| hexahedra | external faces | 1.29× | 1.01× | 2.4× | 1.87× |
| hexahedra | point to cell | 4.7× | 11.8× | 10.8× | 26.6× |
| hexahedra | cell to point | 0.84× | 8.1× | 1.63× | 15.7× |
| hexahedra | point gradient | – | 5.4× | – | 24.8× |
| hexahedra | cell gradient | 10.4× | – | 90.4× | – |
| hexahedra | cell volume | 8.4× | – | 142.3× | – |
| hexahedra | cell centers | 27.1× | – | 65.4× | – |
| tetrahedra | contour | 1.44× | 1.15× | 1.87× | 1.57× |
| tetrahedra | slice | 1.37× | 21.2× | 1.51× | 23.5× |
| tetrahedra | threshold | 3.8× | 2.6× | 6.3× | 4.2× |
| tetrahedra | external faces | 1.08× | 0.61× | 2.6× | 1.43× |
| tetrahedra | point to cell | 8.2× | 12.7× | 14.4× | 23.2× |
| tetrahedra | cell to point | 0.94× | 9.1× | 0.71× | 7.2× |
| tetrahedra | point gradient | – | 10.8× | – | 11.1× |
| tetrahedra | cell gradient | 19.3× | – | 154.2× | – |
| tetrahedra | cell volume | 7.5× | – | 108.4× | – |
| tetrahedra | cell centers | 47.8× | – | 96.1× | – |

- **Tack's CPU against VTK**: ahead on every filter but cell to point
  (0.84× and 0.94×, where Tack's fresh time is mostly deriving the point
  links; warm, it is 7.4–7.7× ahead), and even on contour of hexahedra.
- **Tack's CPU against Viskores**: ahead but on contour of hexahedra (0.87×)
  and external faces of tetrahedra (0.61×). Viskores' slice and cell to
  point on TBB are 18–21× and 8–9× behind Tack's, far more than on CUDA;
  neither falls back to VTK.
- **Tack on Metal**: against VTK on the CPU, 1.5–154× on every filter but
  tetrahedra's cell to point (0.71×); against Viskores on the CPU, 1.4–36×.

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

## How these were measured

- **One process per comparison.** Tack runs in the same process as VTK (the
  CPU) or Viskores (the GPU), on the same mesh, and its output is checked
  against the other's before any timing.
- **Viskores through VTK**: its filters are VTK's Viskores filters
  (`vtkmContour` and the others). On the GPU it was handed Tack's device
  arrays and kept its output on the device, and a trace of every run
  confirmed no transfers beyond sizes and counts. It ran with `cudaMalloc`
  (`NO_VISKORES_MANAGED_MEMORY=1`): its default, `cudaMallocManaged`, was
  4.7–25× slower from page faults on new buffers.
- **Warm and fresh.** A warm Tack run reuses the connectivity its topology
  keeps from an earlier filter on the same mesh; a fresh one derives it.
- **Timing.** Repeated runs for a fixed budget, the median taken; Tack's
  timing ends with `tack.sync()`, since GPU launches are queued. On the
  Linux machine, two passes, the second in reverse order, and their mean;
  on the Mac, one.
- **Harness**: `benchmarks/vtk_compare` (VTK and Tack) and
  `benchmarks/viskores_compare` (Viskores and Tack), on the meshes of
  `benchmarks/vtk_compare/meshes.py`.

## Running times

All in milliseconds. The Linux machine: jura, idle; NVIDIA driver 615.71.09;
VTK master built with TBB; Viskores 1.2.0 (CUDA arch 89, 64-bit ids,
double precision); Tack `main` at `291fcfe` (2026-10-26).

| mesh | cells | filter | VTK | Tack CPU | Tack CPU, fresh | Viskores CUDA | Tack CUDA | Tack CUDA, fresh |
|---|---|---|---|---|---|---|---|---|
| hexahedra | 5M | contour | 28.6 | 45.6 | 38.9 | 8.14 | 7.84 | 7.87 |
| hexahedra | 5M | slice | 35.4 | 34.0 | 31.8 | 14.1 | 7.55 | 7.50 |
| hexahedra | 5M | threshold | 174 | 76.7 | 66.5 | 16.4 | 3.62 | 3.73 |
| hexahedra | 5M | external faces | 184 | 124 | 123 | 19.5 | 9.22 | 9.75 |
| hexahedra | 5M | point to cell | 13.6 | 7.40 | 7.39 | 1.90 | 0.97 | 0.98 |
| hexahedra | 5M | cell to point | 89.2 | 10.1 | 68.3 | 7.22 | 0.93 | 4.23 |
| hexahedra | 5M | point gradient | – | – | – | 9.32 | 3.73 | 7.45 |
| hexahedra | 5M | cell gradient | 175 | 17.3 | 18.1 | – | – | – |
| hexahedra | 5M | cell volume | 1170 | 48.7 | 46.4 | – | – | – |
| hexahedra | 5M | cell centers | 173 | 14.6 | 17.9 | – | – | – |
| tetrahedra | 5M | contour | 25.4 | 20.0 | 20.4 | 7.59 | 4.86 | 4.85 |
| tetrahedra | 5M | slice | 22.7 | 19.5 | 19.9 | 13.1 | 4.87 | 4.86 |
| tetrahedra | 5M | threshold | 77.4 | 29.6 | 29.0 | 11.5 | 3.30 | 3.33 |
| tetrahedra | 5M | external faces | 111 | 70.6 | 71.8 | 17.5 | 9.11 | 9.08 |
| tetrahedra | 5M | point to cell | 11.3 | 5.50 | 5.04 | 1.74 | 0.80 | 0.83 |
| tetrahedra | 5M | cell to point | 48.5 | 8.73 | 36.6 | 4.46 | 0.80 | 4.11 |
| tetrahedra | 5M | point gradient | – | – | – | 5.72 | 2.09 | 5.32 |
| tetrahedra | 5M | cell gradient | 166 | 12.7 | 11.3 | – | – | – |
| tetrahedra | 5M | cell volume | 588 | 28.5 | 29.9 | – | – | – |
| tetrahedra | 5M | cell centers | 179 | 10.9 | 8.86 | – | – | – |
| hexahedra | 20M | contour | 98.3 | 87.9 | 85.5 | 14.1 | 9.53 | 9.46 |
| hexahedra | 20M | slice | 118 | 95.9 | 94.0 | 33.7 | 10.1 | 10.0 |
| hexahedra | 20M | threshold | 696 | 204 | 198 | 42.3 | 7.75 | 7.88 |
| hexahedra | 20M | external faces | 758 | 446 | 455 | 49.2 | 27.8 | 30.1 |
| hexahedra | 20M | point to cell | 69.0 | 28.9 | 28.6 | 3.35 | 1.73 | 1.71 |
| hexahedra | 20M | cell to point | 524 | 36.0 | 224 | 21.0 | 1.81 | 10.1 |
| hexahedra | 20M | point gradient | – | – | – | 30.0 | 13.7 | 23.6 |
| hexahedra | 20M | cell gradient | 642 | 49.9 | 49.5 | – | – | – |
| hexahedra | 20M | cell volume | 4772 | 194 | 186 | – | – | – |
| hexahedra | 20M | cell centers | 792 | 40.2 | 38.5 | – | – | – |
| tetrahedra | 20M | contour | 66.5 | 57.6 | 53.6 | 13.7 | 5.95 | 5.95 |
| tetrahedra | 20M | slice | 74.7 | 60.8 | 58.3 | 27.5 | 5.75 | 5.73 |
| tetrahedra | 20M | threshold | 369 | 126 | 121 | 26.0 | 5.43 | 5.33 |
| tetrahedra | 20M | external faces | 456 | 240 | 238 | 44.2 | 27.0 | 27.0 |
| tetrahedra | 20M | point to cell | 57.4 | 21.6 | 19.9 | 2.89 | 1.28 | 1.25 |
| tetrahedra | 20M | cell to point | 176 | 20.3 | 106 | 10.3 | 1.30 | 10.8 |
| tetrahedra | 20M | point gradient | – | – | – | 15.5 | 5.90 | 15.2 |
| tetrahedra | 20M | cell gradient | 567 | 33.9 | 31.9 | – | – | – |
| tetrahedra | 20M | cell volume | 2398 | 97.6 | 96.5 | – | – | – |
| tetrahedra | 20M | cell centers | 841 | 28.0 | 26.1 | – | – | – |

Warm external faces were as slow as fresh in this run: the surface was
recomputed on every call. Since then (#167) the topology keeps it, and a
repeated call only carries the fields.

The Mac: Apple M1 Max (8 performance and 2 efficiency cores, 64 GB), macOS
26; VTK master (`5d3a65f159`) built with TBB, and the Viskores 1.2.0 it
bundles, with its TBB device (`VISKORES_NUM_THREADS=8`); Tack `main` at
`533c5da0` (2026-10-27), with #167 (warm external faces only carry
the fields). One pass. Off the GPU, Viskores' times include VTK's conversion
of its arrays to Viskores and of the result back, as every VTK filter that
runs Viskores pays.

| mesh | cells | filter | VTK | Tack CPU | Tack CPU, fresh | Viskores TBB | Tack Metal | Tack Metal, fresh |
|---|---|---|---|---|---|---|---|---|
| hexahedra | 5M | contour | 13.0 | 18.6 | 20.5 | 12.3 | 19.9 | 18.8 |
| hexahedra | 5M | slice | 13.1 | 20.3 | 18.6 | 254 | 20.2 | 16.6 |
| hexahedra | 5M | threshold | 41.2 | 13.5 | 13.1 | 51.1 | 16.6 | 12.8 |
| hexahedra | 5M | external faces | 65.7 | 0.15 | 52.3 | 53.9 | 0.16 | 32.7 |
| hexahedra | 5M | point to cell | 9.40 | 2.43 | 2.35 | 23.9 | 1.87 | 1.71 |
| hexahedra | 5M | cell to point | 26.1 | 3.32 | 31.6 | 218 | 6.17 | 17.1 |
| hexahedra | 5M | point gradient | – | 125 | 151 | 800 | 23.2 | 37.9 |
| hexahedra | 5M | cell gradient | 133 | 11.8 | 11.3 | – | 4.05 | 2.67 |
| hexahedra | 5M | cell volume | 421 | 51.1 | 51.1 | – | 7.47 | 5.79 |
| hexahedra | 5M | cell centers | 76.8 | 3.25 | 3.22 | – | 2.82 | 2.19 |
| tetrahedra | 5M | contour | 12.3 | 9.19 | 9.09 | 10.4 | 18.6 | 17.0 |
| tetrahedra | 5M | slice | 10.5 | 9.72 | 10.3 | 142 | 20.4 | 17.1 |
| tetrahedra | 5M | threshold | 28.5 | 9.20 | 9.93 | 19.3 | 12.4 | 9.58 |
| tetrahedra | 5M | external faces | 56.5 | 0.16 | 57.7 | 32.4 | 0.13 | 26.7 |
| tetrahedra | 5M | point to cell | 8.93 | 1.32 | 1.34 | 13.9 | 1.43 | 1.36 |
| tetrahedra | 5M | cell to point | 13.3 | 1.89 | 14.3 | 127 | 6.67 | 20.8 |
| tetrahedra | 5M | point gradient | – | 11.5 | 23.8 | 251 | 15.5 | 26.6 |
| tetrahedra | 5M | cell gradient | 135 | 7.11 | 6.75 | – | 2.71 | 2.32 |
| tetrahedra | 5M | cell volume | 187 | 24.8 | 25.2 | – | 5.78 | 3.80 |
| tetrahedra | 5M | cell centers | 71.0 | 1.86 | 1.81 | – | 1.67 | 1.48 |
| hexahedra | 20M | contour | 45.8 | 43.4 | 44.2 | 37.3 | 28.9 | 24.3 |
| hexahedra | 20M | slice | 47.8 | 45.6 | 45.7 | 945 | 30.9 | 25.8 |
| hexahedra | 20M | threshold | 157 | 43.5 | 43.7 | 233 | 27.2 | 25.1 |
| hexahedra | 20M | external faces | 262 | 0.16 | 202 | 206 | 0.16 | 109 |
| hexahedra | 20M | point to cell | 38.6 | 8.22 | 8.18 | 95.0 | 4.38 | 3.56 |
| hexahedra | 20M | cell to point | 98.7 | 12.9 | 117 | 958 | 13.3 | 60.7 |
| hexahedra | 20M | point gradient | – | 489 | 615 | 3307 | 80.0 | 133 |
| hexahedra | 20M | cell gradient | 484 | 44.4 | 46.7 | – | 9.73 | 5.35 |
| hexahedra | 20M | cell volume | 1692 | 203 | 200 | – | 13.3 | 11.9 |
| hexahedra | 20M | cell centers | 304 | 11.7 | 11.2 | – | 7.15 | 4.65 |
| tetrahedra | 20M | contour | 38.7 | 28.5 | 26.8 | 31.7 | 28.8 | 20.7 |
| tetrahedra | 20M | slice | 36.7 | 27.1 | 26.8 | 571 | 31.7 | 24.3 |
| tetrahedra | 20M | threshold | 106 | 29.0 | 27.7 | 72.2 | 24.4 | 16.9 |
| tetrahedra | 20M | external faces | 223 | 0.16 | 206 | 124 | 0.15 | 85.8 |
| tetrahedra | 20M | point to cell | 35.4 | 4.37 | 4.34 | 56.1 | 2.79 | 2.46 |
| tetrahedra | 20M | cell to point | 48.0 | 6.46 | 51.1 | 488 | 13.8 | 68.0 |
| tetrahedra | 20M | point gradient | – | 43.7 | 89.8 | 970 | 33.5 | 87.5 |
| tetrahedra | 20M | cell gradient | 512 | 26.2 | 26.6 | – | 6.85 | 3.32 |
| tetrahedra | 20M | cell volume | 742 | 98.3 | 98.4 | – | 12.5 | 6.84 |
| tetrahedra | 20M | cell centers | 280 | 6.00 | 5.87 | – | 3.84 | 2.92 |

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
