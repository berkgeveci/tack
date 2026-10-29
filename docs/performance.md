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
| hexahedra | contour | 1.02× | 1.49× | 12.3× |
| hexahedra | slice | 1.19× | 3.4× | 16.4× |
| hexahedra | threshold | 3.16× | 5.4× | 84.8× |
| hexahedra | external faces | 1.38× | 1.63× | 23.3× |
| hexahedra | point to cell | 2.23× | 1.96× | 39.2× |
| hexahedra | cell to point | 1.84× | 2.1× | 48.8× |
| hexahedra | point gradient | 5.56× | 1.27× | 181× |
| hexahedra | cell gradient | 13.1× | – | 283× |
| hexahedra | cell volume | 30.3× | – | 2,269× |
| hexahedra | cell centers | 19.4× | – | 377× |
| tetrahedra | contour | 1.26× | 2.3× | 12.4× |
| tetrahedra | slice | 1.10× | 4.8× | 11.2× |
| tetrahedra | threshold | 3.37× | 4.9× | 72.7× |
| tetrahedra | external faces | 1.66× | 1.64× | 16.2× |
| tetrahedra | point to cell | 2.57× | 2.3× | 40.1× |
| tetrahedra | cell to point | 1.59× | 0.95× | 16.2× |
| tetrahedra | point gradient | 7.76× | 1.02× | 127× |
| tetrahedra | cell gradient | 16.6× | – | 351× |
| tetrahedra | cell volume | 23.6× | – | 1,406× |
| tetrahedra | cell centers | 28.4× | – | 502× |

- **On the GPU**, Tack is ahead of Viskores on every filter but tetrahedra's
  cell to point (0.95×), up to 5.4× on threshold and 4.8× on slice. Warm,
  it is ahead on every filter, by 1.04–12×.
- **On the CPU**, Tack is ahead of VTK on every filter at 20M cells, by
  13–30× on cell gradients, volumes and centers and 5.6–7.8× on point
  gradients; contour of hexahedra is even (1.02×). At 5M cells, contour of
  hexahedra is the one filter where VTK is faster (0.55×).
- **Tack on the GPU against VTK on the CPU**: 11–2,269×.

**Measured again for 0.6.1.** Tack 0.6.0 capped every CPU fan-out's chunks
at 20 µs of work, which on 28 threads made every CPU filter 2–5× slower than
this page reported (threshold of 20M hexahedra 712 ms against 199; behind
VTK on six filters of each mesh); 0.6.1 fixes it. The VTK and Tack columns
are jura's run of 2026-10-28, two passes, the best of each: Tack `main` at
`a9031fe2` (0.6.0 and the pull requests after it) with the fix, on the CPU
and CUDA. The Viskores column is the run of 2026-10-26 (Tack `main` at
`291fcfe`), which this one did not repeat (see [running times](#running-times)).

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

## Streamlines on the GPU, against Viskores

Viskores' streamlines on CUDA against Tack's, on the ONERA M6 wing (46.9M
tetrahedra and wedges, graded toward the wing in a far field a hundred
chords wide) and the aneurysm (8.7M tetrahedra), by seed count. Viskores
builds its cell locator, a two-level grid, on every call; the column shows
how much of its time that is. Tack's topology keeps its locator, so a later
call on the same mesh reuses it. Tack is shown both ways: building a new
locator each call, as Viskores does, and with the locator kept.

| mesh | seeds | Viskores (its locator) | Tack, new locator | Tack, locator kept | Viskores / Tack, new locator |
|---|---|---|---|---|---|
| M6 wing | 1 | 3,526 (3,059) | 107 | 15.2 | 33× |
| | 17 | 4,143 (3,019) | 188 | 81.5 | 22× |
| | 1,024 | 4,767 (3,041) | 238 | 146 | 20× |
| | 16,384 | 4,705 (3,043) | 197 | 106 | 24× |
| aneurysm | 1 | 1,714 (1,545) | 31.7 | 3.6 | 54× |
| | 25 | 2,113 (1,539) | 55.2 | 22.6 | 38× |
| | 2,000 | 2,887 (1,569) | 88.9 | 60.9 | 32× |
| | 20,000 | 6,812 (1,548) | 561 | 578 | 12× |

- **The paths agree.** On the M6, every path is the same length in both,
  and they lie within 1.6e-6 of each other at 17 seeds and within 3.5e-6
  at 16,384. On the aneurysm the agreement is within 2.1e-6 over the first
  quarter at 25 seeds. At 20,000 seeds, paths through its recirculation
  separate further: 19,730 of 20,000 are the same length (Viskores computes
  in double precision, Tack in the geometry's single precision).
- **Before Tack's two-level grid (#175)** it scanned the M6's crowded bins
  near the wing: 28–105 s on this GPU.

Machine: jura (NVIDIA RTX 6000 Ada). Viskores 1.2.0 (CUDA, double
precision), run through a C++ driver of its `Streamline` filter. Tack
`main` at `2bbaf7e` with #184, the default two-level locator (2026-10-28).
Times are warm, in ms; the first call compiles.

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
| | streamlines, 17 seeds, by length ³ | 3900 | 3.6 | 1,083× |
| aneurysm blood flow, 8.7M tetrahedra | isosurface, slice, gradients | | | 2.7×, 2.0×, 17× |
| | streamlines, 25 seeds ³ | 3.6 | 1.1 | 3.3× |
| LANL wildfire, 18.7M curvilinear cells | plume, slice, gradients, fuel cells, surface | | | 4.3×, 1.9×, 33×, 4.0×, 5.9× |
| 512 pieces of a simulation's output, 5.5M voxels | global ids | 10,400 | 9,000 | 1.2× |
| | a layer of ghost cells | 27,300 | 28,700 | 0.95× |

³ Measured 2026-10-27, after the two-level grid (#175). Tack's
streamlines try the cell a particle was last found in, then search its
two-level cell grid, so a location near the M6's wing no longer scans
thousands of cells (it was 1.0 s with one level of bins). VTK's default
interpolator finds cells through the nearest point, which ends 7 of the
M6's 17 paths early, inside the mesh. With VTK's cell-locator interpolator,
every path runs its full length, within 3.8e-6 of Tack's, in 22.9 s.

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
  Linux machine, two passes: the VTK and Tack columns take the better of
  the two, the Viskores column their mean (the second pass in reverse
  order); on the Mac, one.
- **Harness**: `benchmarks/vtk_compare` (VTK and Tack) and
  `benchmarks/viskores_compare` (Viskores and Tack), on the meshes of
  `benchmarks/vtk_compare/meshes.py`. Point gradients against VTK
  (`vtkGradientFilter`) are a pair added to `vtk_compare` after 0.6.0, on
  `main`.

## Running times

All in milliseconds. The Linux machine: jura, idle; NVIDIA driver 615.71.09;
VTK master (`cc6c421507`) built with TBB; Viskores 1.2.0 (CUDA arch 89,
64-bit ids, double precision). VTK and Tack: Tack `main` at `a9031fe2` with
0.6.1's fix (2026-10-28), the best of two passes. Viskores: the run of
2026-10-26 (Tack `main` at `291fcfe`), whose Tack CUDA times were within 3%
of these but for contour and slice of hexahedra, 1.2–1.7× slower then (5M
contour 7.87 ms against 4.77 here), when merging their crossings sorted by
radix; the Viskores column's ratios for those two understate Tack's lead.

| mesh | cells | filter | VTK | Tack CPU | Tack CPU, fresh | Viskores CUDA | Tack CUDA | Tack CUDA, fresh |
|---|---|---|---|---|---|---|---|---|
| hexahedra | 5M | contour | 26.8 | 53.4 | 49.1 | 8.14 | 4.91 | 4.77 |
| hexahedra | 5M | slice | 35.0 | 34.4 | 31.7 | 14.1 | 5.01 | 5.00 |
| hexahedra | 5M | threshold | 143 | 72.7 | 62.2 | 16.4 | 3.53 | 3.55 |
| hexahedra | 5M | external faces | 171 | 0.35 | 131 | 19.5 | 0.38 | 9.72 |
| hexahedra | 5M | point to cell | 14.9 | 7.20 | 7.73 | 1.90 | 0.87 | 0.87 |
| hexahedra | 5M | cell to point | 88.9 | 9.09 | 58.3 | 7.22 | 1.27 | 3.81 |
| hexahedra | 5M | point gradient | 1184 | 145 | 192 | 9.32 | 3.78 | 6.65 |
| hexahedra | 5M | cell gradient | 167 | 21.1 | 19.2 | – | 1.05 | 1.07 |
| hexahedra | 5M | cell volume | 1166 | 48.9 | 43.9 | – | 0.99 | 0.89 |
| hexahedra | 5M | cell centers | 200 | 18.3 | 17.4 | – | 0.84 | 0.96 |
| tetrahedra | 5M | contour | 23.5 | 22.0 | 19.6 | 7.59 | 4.59 | 4.60 |
| tetrahedra | 5M | slice | 24.9 | 22.0 | 19.9 | 13.1 | 4.89 | 4.91 |
| tetrahedra | 5M | threshold | 82.6 | 30.9 | 29.7 | 11.5 | 3.32 | 3.29 |
| tetrahedra | 5M | external faces | 99.5 | 0.38 | 81.3 | 17.5 | 0.40 | 9.15 |
| tetrahedra | 5M | point to cell | 12.8 | 5.35 | 4.96 | 1.74 | 0.76 | 0.80 |
| tetrahedra | 5M | cell to point | 49.1 | 7.74 | 37.0 | 4.46 | 1.14 | 3.96 |
| tetrahedra | 5M | point gradient | 512 | 34.4 | 58.6 | 5.72 | 2.05 | 4.88 |
| tetrahedra | 5M | cell gradient | 152 | 8.48 | 14.8 | – | 0.85 | 0.88 |
| tetrahedra | 5M | cell volume | 529 | 30.4 | 30.3 | – | 0.81 | 0.82 |
| tetrahedra | 5M | cell centers | 156 | 6.79 | 13.0 | – | 0.75 | 0.78 |
| hexahedra | 20M | contour | 94.1 | 91.5 | 92.5 | 14.1 | 7.22 | 7.65 |
| hexahedra | 20M | slice | 122 | 109 | 102 | 33.7 | 7.50 | 7.43 |
| hexahedra | 20M | threshold | 630 | 200 | 199 | 42.3 | 7.41 | 7.43 |
| hexahedra | 20M | external faces | 721 | 0.39 | 522 | 49.2 | 0.39 | 30.9 |
| hexahedra | 20M | point to cell | 67.6 | 31.4 | 30.4 | 3.35 | 1.66 | 1.73 |
| hexahedra | 20M | cell to point | 488 | 47.3 | 264 | 21.0 | 2.24 | 10.0 |
| hexahedra | 20M | point gradient | 4228 | 481 | 760 | 30.0 | 14.1 | 23.3 |
| hexahedra | 20M | cell gradient | 625 | 47.8 | 47.8 | – | 2.19 | 2.21 |
| hexahedra | 20M | cell volume | 4834 | 158 | 160 | – | 2.14 | 2.13 |
| hexahedra | 20M | cell centers | 762 | 39.6 | 39.2 | – | 2.00 | 2.02 |
| tetrahedra | 20M | contour | 68.5 | 59.1 | 54.4 | 13.7 | 5.52 | 5.55 |
| tetrahedra | 20M | slice | 65.5 | 59.6 | 59.5 | 27.5 | 5.79 | 5.86 |
| tetrahedra | 20M | threshold | 363 | 119 | 108 | 26.0 | 5.01 | 4.98 |
| tetrahedra | 20M | external faces | 448 | 0.39 | 269 | 44.2 | 0.41 | 27.7 |
| tetrahedra | 20M | point to cell | 50.4 | 20.9 | 19.6 | 2.89 | 1.25 | 1.26 |
| tetrahedra | 20M | cell to point | 170 | 21.2 | 107 | 10.3 | 1.61 | 10.5 |
| tetrahedra | 20M | point gradient | 1869 | 117 | 241 | 15.5 | 5.85 | 14.7 |
| tetrahedra | 20M | cell gradient | 544 | 34.6 | 32.7 | – | 1.53 | 1.55 |
| tetrahedra | 20M | cell volume | 2164 | 96.7 | 91.5 | – | 1.55 | 1.54 |
| tetrahedra | 20M | cell centers | 720 | 26.1 | 25.4 | – | 1.41 | 1.43 |

Warm external faces are well under a millisecond: the topology keeps the
surface (#167), and a repeated call only carries the fields.

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
