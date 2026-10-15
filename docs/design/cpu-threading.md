# CPU Threading Policy

Every CPU dispatch makes one decision before it runs: execute the loop range
on the calling thread, or split it into chunks and fan them out over a
persistent thread pool. This page explains how that decision is made, why,
and what is and is not promised about it.

The whole policy lives in `packages/tack-core/src/tack/runtime/cpu.py`:
per-kernel state on `CompiledKernel`, decision and measurement code on
`CPUBackend`, entered through `CPUBackend._dispatch`. The
[runtime chapter](../developers-guide/06-runtime.md) and the
[backends guide](../users-guide/08-backends.md#cpu) carry short versions.

!!! summary "In one paragraph"
    The backend compares two *measured* costs: how long this kernel takes per
    element when run serially, and how long it costs to wake the pool. Both are
    estimated continuously from real dispatches and folded into a per-kernel
    integer threshold (`parallel_min_elems`), so the decision is one integer
    comparison. Because each estimate can only be refreshed by one kind of
    dispatch, the policy periodically runs the *other* kind on purpose, so a
    wrong decision in either direction is eventually re-measured. Every path
    executes each loop index exactly once, so the decision affects speed only.

## The problem

### What a fan-out costs

`CPUBackend._parallel_execute` divides `[start, end)` among the backend's
`num_threads` threads (see "A thread count per launch" below for the
alternative that was tried).
Arguments are marshalled once per dispatch (`CompiledKernel.bind`); only
the loop bounds differ between chunks.

The chunks run through one of two pools. The default is the pool in
`cpu_pool.py`: a small program in LLVM IR, built with `llvmlite.ir` and
JIT'd beside the kernels, whose `N - 1` detached pthreads each wait on
their own mailbox, a launch count (spinning for a while, then sleeping:
on a futex on Linux, on `os_sync_wait_on_address` on macOS 14.4 and
later, and elsewhere in `nanosleep` polls that back off to a
millisecond). macOS used to poll, and every launch then waited for its
slowest sleeper: the policy measured a 1 ms fan-out, kept mid-size
launches serial, and the pool made the dataset filters 1.3–2.6× slower
than the executor on an M1 Max; blocking on an address, its fan-out
measured 73 µs. A fan-out is one ctypes call:
`pool_run` posts to workers `0 .. k - 2`, wakes those of them that sleep,
runs as worker `k - 1` itself and leaves the rest asleep, so a launch of
two on a pool of sixteen costs two wake-ups (4.4 µs hot on a 2 GHz Xeon,
against 11 when every worker was woken) and the `k` threads of every
launch are the same `k`. Workers are not pinned to cores. Pinning was
tried and dropped (2026-10-14): on a two-socket Xeon it was parity on 8
threads of 8 cores and a loss on 16 (every worker pinned: a 1M contour
33 ms against 19; one per core: still behind unpinned on most filters),
and on a 28-core EPYC it lost 5-9% on the dataset filters, because a
pinned worker cannot move off a busy core. The workers and the
caller claim chunks from a shared
counter (`GRAIN` per worker, each at least `MIN_CHUNK` elements) and run
them through a wrapper JIT'd once per kernel that loads the arguments
from 8-byte slots, and the workers never touch Python. Claimed rather
than assigned because the dispatch waits for its slowest worker: with
one equal chunk each, a worker the scheduler put on the sibling
hyperthread of a busy core ran 1.7× slower and a compute-bound kernel
over a million cells took 29 or 50 ms from run to run; claiming, it
takes fewer chunks and the run takes 38 ms every time. Each worker
reports the time and the elements it ran, which is what the rate and
floor bookkeeping below reads. The pool is one
per process (`CPool.get`), sized by the CPUs the process may run on (its
affinity mask on Linux, the performance cores on Apple silicon), or by the
first backend's thread count if that is larger; a dispatch uses as many of
its threads as its backend has, and a backend asking for more threads than
the pool has falls back to the executor. The pool's launch state is one
set of JIT'd globals, so a run holds `CPool._txn` from the call through
the read of its report: two Python threads fanning out at once otherwise
ran each other's function and argument pack. A pool whose workers cannot
all be started is dismantled, its workers joined, before the backend falls
back. `TACK_CPU_POOL=python` selects the
other pool, a `ThreadPoolExecutor` with one `call_range` task per chunk,
which is also the fallback on Windows or when the JIT'd pool cannot be
made.

What a fan-out costs differs by an order of magnitude between the two.
Through the executor every worker wakes, takes the GIL to reach its ctypes
call and takes it again to return, and the handoffs serialize: about 60 µs
per worker on a 2 GHz Xeon (490 µs for eight, 1.2 ms after an idle gap of
2 ms or more, when the cores have entered C6); `_calibrate_fan_out`'s
docstring records 223 µs on the 10-core Apple silicon machine the policy
was first written on (August 2026). Through the JIT'd pool it is 3–5 µs
hot and about 100 µs after an idle gap (the C-state exit). Either way the
policy measures the pool it uses and a fan-out pays only when the serial
run it replaces takes meaningfully longer; with the JIT'd pool that is far
more often. Measured on the Xeon, the dataset filters at a million cells
went from a median 1.05× to 1.9× on eight threads.

### Why no fixed threshold works

The module docstring records the crossover — the range above which a fan-out
wins — for three f32 kernels on that machine:

| Kernel body | Character | Crossover |
|---|---|---:|
| `out[i] = x[i]*2 + 1` | memory bound | ~4,000,000 |
| `sqrt(..) + sin(..)` | ~10 flop/element | ~130,000 |
| 20-iteration inner loop | ~120 flop/element | ~4,000 |

The break-even moves by three orders of magnitude with arithmetic intensity,
so any element count is wrong for most kernels. The original policy threaded
every range above 1024 elements, below all three: for the memory-bound kernel
a 65,536-element dispatch took 30.7 µs serially and 262.3 µs threaded, and
threading won only at 4,194,304 elements (559 µs against 485 µs).

A threshold in elements also silently encodes the thread wake-up cost and
timer resolution of the machine it was chosen on. So the module writes down
no thresholds in elements or nanoseconds, only *ratios* — how much margin to
demand, how much of a sample to trust, how far one sample may move an
estimate — which carry across machines.

### What a wrong decision costs

| Error | What happens | Size |
|---|---|---|
| **Over-eager**: fanned out, serial would have won | The dispatch pays a whole fan-out to save less than that | Up to ~10x slower per dispatch in the measurement above, on every dispatch while the estimate stays wrong |
| **Under-eager**: ran serially, parallel would have won | Speedup not taken | Bounded by the serial run; near the crossover, small by definition |

The asymmetry shapes the design. Where the policy must err, it errs toward
serial: estimates that are too high are distrusted, estimates that are too
low are believed, and priors for the fan-out cost are pessimistic.

## The cost model

### Two paths, one crossover

With serial rate `r_s` (ns per element), fan-out cost `F`, and aggregate
parallel rate `r_p` (ns per element of the whole range when fanned out):

```
serial ≈ n · r_s      parallel ≈ F + n · r_p      crossover n* = F / (r_s − r_p)
```

`CPUBackend._min_elems` stores the crossover scaled by a margin `M`:

```
parallel_min_elems = max(num_threads, int(F · M / rate))
    rate = r_s − min(r_p, 0.9 · r_s)
```

The rate keeps `r_p` because real kernels do not divide perfectly. The
removed policy v1 (see [History](#history-policy-v1)) used `rate = r_s`: it
assumed perfect division of the work, `parallel ≈ F + T_serial / P`,
and drops `r_p`. Its margin of 2.0 (`_PARALLEL_BREAK_EVEN`) covers the
break-even factor `P / (P − 1)` for any thread count. But real kernels do not
reach speedup `P`: the effective speedup `P_eff = r_s / r_p` ranged from 1.5
to 9.2 across the kernels measured in the module's comments, and the margin
v1 really applies is `M · (1 − 1/P_eff)`. For a bandwidth-bound kernel with
`P_eff ≈ 1.5` that is 0.67 — below break-even, so the fan-outs it permits
near the crossover lose.

Until a fan-out has measured it, `r_p` is 0.0 and the rate is `r_s`, so
"not measured yet" needs no special case. The `0.9 · r_s` clamp keeps the
denominator positive when a noisy `r_p` approaches `r_s`, and caps the
threshold at ten times the serial-rate-only one for the same inputs.

The `max(num_threads, …)` floor keeps chunks at one element or more. An
unmeasured kernel, or a one-thread backend, gets `_NEVER` (`1 << 62`).

### The margin

| Case | `_margin()` |
|---|---|
| default | `clamp(1 + _MARGIN_K · cv, _MARGIN_MIN, _MARGIN_MAX)` = `clamp(1 + 1.3 · cv, 1.0, 2.5)` |
| `TACK_CPU_MARGIN` set | that value |

A margin insures against mis-estimates, so it is sized by how unreliable the
estimates currently are. `cv` is a smoothed relative deviation of new fan-out
samples from the stored estimate (`_update_fan_out_knot`; weight
`_CV_SMOOTHING` = 0.10, one sample capped at `_CV_SAMPLE_CAP` = 3.0).
`_margin()`'s docstring records why: on one machine, a quiet run wanted the
smallest margin tried while a loaded run needed at least 1.5.

Before any fan-out sample has been compared, `cv` is 0 and the margin is
1.0. (v1 used a constant 2.0, `_PARALLEL_BREAK_EVEN`.)

### One integer compare on the hot path

`parallel_min_elems` is stored on the `CompiledKernel`, so the common case in
`_run` is a single `loop_end >= compiled.parallel_min_elems`. `_set_serial_cost`
recomputes it after every timed serial run and every fan-out (through
`_record_parallel_cost`); `_run` recomputes it after the first calibration.

Being cached, the threshold is priced with the fan-out estimate *at the
moment it was recomputed*. The comment in `_dispatch` makes the trade
explicit: a fresher estimate could only have changed the next decision. The
paths that call `_fan_out_estimate()` directly — `_serial_is_affordable`,
`_sample_elems`, `_run_untimed`, `_parallel_recheck_due` — read the current
value.

## What is measured, and how

### Serial rate: `ns_per_elem`

`_run_serial` times one `call_range` with `time.perf_counter_ns()` and
subtracts the kernel's fixed per-call cost:

```
sample = max((elapsed − call_overhead_ns) / elements, _MIN_NS_PER_ELEM)
```

`call_overhead_ns` is measured once per compiled kernel by
`_measure_call_overhead`, the minimum of five calls on an empty range. An
empty range runs no iterations, so the probe is safe on any kernel, including
ones that accumulate with atomics. Without the subtraction the call's cost is
charged as per-element work; the comment in `_run_serial` records a cheap
kernel reading 2.6x too expensive and fanning out where serial was faster.
`_MIN_NS_PER_ELEM` (1e-3) keeps an all-but-empty kernel reading as measured.

The raw sample is kept in `last_sample`; the estimate moves asymmetrically,
with weight `_COST_SMOOTHING` (0.25) and outlier ratio `_MAX_SAMPLE_RATIO` (8):

| Sample vs. estimate `prev` | New estimate | Why |
|---|---|---|
| no estimate yet | the sample | nothing to smooth against |
| more than 8x higher | `prev + 0.25 · (8·prev − prev)` = 2.75x `prev` | usually a descheduled run; a real rise still arrives in a few dispatches |
| more than 8x lower | the sample, outright | easing down from a far-too-high estimate would keep losing fan-outs for thousands of dispatches |
| otherwise | `prev + 0.25 · (sample − prev)` | ordinary smoothing |

Two more rules depend on whether the range was **scattered**.
`CompiledKernel.scattered` is true initially and after any fan-out, and is
cleared when a serial run covers the whole range. A serial run over a
scattered range first pulls data back from other cores' caches, or faults in
untouched pages, so its sample is an upper bound:

- A **clean** sample sets `rate_confirmed`; the first clean sample, if lower
  than the smoothed result, is believed outright.
- A **scattered** sample may lower the estimate but not raise it, while a
  serial run of the current range is [affordable](#confirmation-and-affordability).
  Where serial is expensive, a rise is real evidence and is kept.

`_set_serial_cost` then floors the estimate at `serial_floor_ns`
([below](#parallel-rate-ns_per_elem_parallel)) and recomputes the threshold.

### How much to time: `_sample_elems`

A recheck times a slice, so measuring does not cost the dispatch its
parallelism. The slice targets a *duration* — work worth
`_SAMPLE_OVERHEAD_RATIO` (10) times the fixed call cost being subtracted —
which is two elements of an expensive kernel or thousands of a cheap one.

Sizing a sample from the estimate it is meant to check is circular, so two
guards override the target:

1. **At least 1/64 of the range.** Otherwise a corrupt-high estimate asks for
   a handful of elements, whose timing is nearly all fixed cost, which keeps
   it corrupt; the docstring records an estimate still 640x high after four
   dispatches of three-element samples.
2. **Timed whole if a fan-out would cost as much.** The test is
   `loop_end · guard_rate ≤ _fan_out_estimate()`, with
   `guard_rate = max(serial_floor_ns, min(rate, _PROBE_REFERENCE_NS_PER_ELEM))`.
   Believing the estimate only up to the 24 ns/element reference rate stops a
   corrupt-high estimate from forcing a too-short slice on a small range; a
   trusted parallel rate lifts the cap, so a genuinely expensive kernel is
   not re-run whole on every recheck. The docstring records the old
   element-count guard making every 384² frame of a 3500 ns/pixel volume
   renderer recheck as a whole-frame serial run (0.75 s against 100 ms
   threaded).

### Fan-out cost: `_fan_out_ns` and the idle curve

**Calibration.** The first time a range looks worth threading,
`_calibrate_fan_out` fans out *empty* ranges through the real path five times
and stores the minimum as `_fan_out_ns`. A pure-Python no-op probe would
understate it (the docstring: 122 µs against the real path's 223 µs). Until
then `_DEFAULT_FAN_OUT_NS` (200 µs) stands in, high on purpose so being wrong
only delays the first fan-out. (Under the removed v1, `_fan_out_ns` was the
fan-out cost for the life of the backend.)

**The idle curve.** A dispatch after a pause pays more than one after
another fan-out, because idle cores descend into deeper sleep states, and the
shape differs by machine. The comment on `_FAN_OUT_GAPS_MS` records idle
fan-outs at 2.45x the back-to-back cost on a 2-socket Xeon and 3.81x on an
M1 Max (August 2026), the Xeon's penalty complete by 10 ms and the M1 Max's
still growing at 50 ms; `_FAN_OUT_IDLE_PRIOR`'s comment adds 2.1x on a
1-socket Xeon (October 2026). So the backend keeps the cost as a curve over
*idleness*, with knots at 0, 10 and 50 ms.

`_fan_out_estimate()` takes the gap since `_last_dispatch_ns` — set at the end
of every fan-out and refresh, and not by serial runs, which leave the
workers idle — and interpolates piecewise-linearly with two rules on read:

- **Monotone.** A longer pause cannot make threads wake faster. This is
  imposed on read, not on the stored knots: enforced on storage it made the
  curve a ratchet that could only rise (comment at the end of
  `_record_fan_out`).
- **Unmeasured knots are steps.** A knot still holding its prior prices only
  gaps at or beyond it; ramping toward it priced a dispatch's own preamble as
  partly idle.

**Seeding without sleeping.** The hot knot is the median of three empty
fan-outs taken after the calibration reps (the first reps also start the
pool's threads and read high). The idle knots start at `_FAN_OUT_IDLE_PRIOR`
(4x hot, above every ratio measured), marked unmeasured, and each is replaced
by the first fan-out measured nearest it and moved to the gap where that
happened (`_update_fan_out_knot`). The previous design calibrated them by
sleeping through each gap: 180 ms of `time.sleep` inside whichever dispatch
first looked worth threading.

**Keeping it current.** A startup measurement records whatever the machine
was doing at startup, so two paths refine the curve:

- **Free samples:** `_record_fan_out`, after every fan-out, takes elapsed
  time minus the slowest worker's span — both measured — when the work is at
  most `_FAN_OUT_MAX_WORK_SHARE` (35%) of the dispatch, where the subtraction
  is well conditioned. It updates the nearest knot with weight
  `_FAN_OUT_SMOOTHING` (0.15).
- **Scheduled samples:** `_refresh_fan_out`, after every dispatch once a
  curve exists. If `_FAN_OUT_REFRESH_NS` (100 ms) has passed since the last
  measurement, or the knot nearest the current gap still holds its prior, it
  times one empty fan-out at the current gap. If the dispatch arrived cold
  (gap over 5 ms) and the hot knot is over `_FAN_OUT_HOT_REFRESH_NS` (1 s)
  old, it also warms the pool and records the minimum of three more as the
  hot knot.

Learning only from real fan-outs deadlocks: load raises the estimate, which
raises thresholds, which stops the fan-outs that would lower it. An empty
fan-out has no work to subtract, so it can always be measured. The refresh
runs after the dispatch, never before, so it neither delays the caller nor
disturbs what the dispatch measures.

### Parallel rate: `ns_per_elem_parallel`

Each worker is timed *inside* the worker, around its kernel calls
only: through the JIT'd pool every worker reports its span and the
elements it claimed (`CPool.run(..., measure=True)`, read in
`_pool_execute`), and through the executor each task times its own
`call_range`. A worker's own rate is roughly the serial rate — the
speedup comes from running chunks together, not from cheaper elements — so
`_record_parallel_cost` converts and smooths (weight 0.25):

```
r_p sample = median(worker ns/element) / workers_that_ran
```

counting only workers whose span reached `_RP_MIN_WORKER_NS` (1 µs); a
pool worker that claimed no chunk is not counted.

- **Why in-worker timing.** `r_p` used to be inferred as
  `(elapsed − fan_out_estimate) / elements`, charging every fan-out error to
  `r_p`. Near the crossover, where work and fan-out are the same size by
  definition, no ratio guard could both admit samples and keep them honest.
- **Why the median.** The dispatch waits for its slowest worker, but under
  load the max is whichever worker was descheduled: `_record_parallel_cost`'s
  docstring records the max at 10–134x the fitted rate on a loaded M1 Max
  (August 2026), the median at 0.5–6.3x. The minimum worker and per-thread
  CPU time were tried on a 1-socket Xeon (October 2026), with the
  executor, and scored worse; its workers returning from ctypes together
  queue for the GIL, a real cost the median tracks. The pool's workers
  never take the GIL.
  For an even number of workers it is the lower of the two middle values
  (`_worker_median`). The upper one of two workers is the slower worker,
  the max again: on a two-thread machine, stalling one worker for 2 ms
  raised the serial estimate of a 0.3 ns/element kernel 1900 times through
  the floor below, with no serial sample taken (October 2026). With two
  workers the lower median is the minimum, the statistic that scored worse
  on the Xeon across many workers; with two there is no third value, and
  the choice is between a statistic one preempted thread controls and one
  that misses the second worker's wait for the GIL. It has not been scored
  on a two-thread machine.

**The serial floor.** A fan-out cannot make an element cheaper than it is
serially, so `r_s ≥ r_p` — if `r_p` is trustworthy. Short spans are not: a
worker's clock stops only once it has the GIL back, and the comment on
`_RP_BOUND_SPAN_RATIO` records 256-element chunks of a 1 ns/element kernel
reading 60–340 ns/element across 16 workers. So a fan-out qualifies as a
bound only when its median measured span reaches

```
trusted_span = _RP_BOUND_SPAN_RATIO · workers · call_overhead_ns     (ratio 20)
```

and `serial_floor_ns` then becomes the *lesser* of that sample and the
smoothed `r_p`: one high draw could otherwise set the floor at the serial
rate, and the smoothed rate carries samples from short fan-outs.
`_set_serial_cost` applies the floor at once, so an estimate taken from a
cheap region is corrected on the first fan-out. The floor is still `P_eff`
below the true serial rate; it only stops a range the kernel has shown to
thread well from flipping to "too small to thread".

**Retiring the floor.** Runtime scalars and array contents can change the
work without a new compiled variant. When a whole-range fan-out
(`whole_range=True`) finishes with *every* worker span below `trusted_span` —
including spans too short to yield a rate — `_record_parallel_cost` clears
the floor and resets the recheck schedule, so the next parallel-bound
dispatch takes a fresh serial sample. A short median with one long worker,
or a fast head or tail of a recheck, cannot retire it.

### Confirmation and affordability

`_serial_is_affordable(compiled, n)` asks whether `n` elements cost at most
`_CONFIRM_BAND · margin · fan_out` (band 4) serially, priced from the serial
estimate alone; the docstring records two dearer prices that each switched
the mechanism off for the kernel it was built for.

`_needs_confirmation` uses it: while a kernel has never had a clean
sample, an affordable range above the threshold runs serially instead. A
first sample is never clean, and `_CONFIRM_BAND`'s comment records such
samples reading a bandwidth-bound kernel 3–4x dear on a 1-socket Xeon, a
first sample up to 15x. Confirmation costs at most two serial dispatches per
kernel and is checked before calibration, so a cheap kernel with a dear first
sample never pays to calibrate a fan-out it will not use.

## The decision procedure

`_dispatch` binds the arguments once, calls `_run`, and afterwards (once
a curve exists) calls `_refresh_fan_out`. `_run` decides:

```mermaid
flowchart TD
    A["_run(compiled, prefix, loop_end)"] --> B{"loop_end >= parallel_min_elems?"}
    B -- yes --> C{"_needs_confirmation?"}
    C -- yes --> WS["_run_whole_serial"]
    C -- no --> D{"_fan_out_ns measured?"}
    D -- no --> E["_calibrate_fan_out, then recompute threshold"]
    E --> E2{"still at or above threshold?"}
    E2 -- no --> WS
    E2 -- yes --> F
    D -- yes --> F{"recheck_due()?"}
    F -- no --> PW["_parallel_execute(0, loop_end, whole_range=True)"]
    F -- yes --> G{"_sample_elems covers the range?"}
    G -- yes --> WS
    G -- no --> H["head: _run_untimed(0, start)<br/>slice: _run_serial(start, start + probe)"]
    H --> I{"slice over 8x below the estimate,<br/>and room for two more slices?"}
    I -- yes --> J["_second_opinion"]
    I -- no --> K["_parallel_execute(rest of range)"]
    J --> K
    B -- no --> L{"_parallel_recheck_due?"}
    L -- yes --> PW
    L -- no --> M{"never measured, loop_end >= _probe_min_range,<br/>and more than one thread?"}
    M -- no --> WS
    M -- yes --> N["first sight: time an interior slice,<br/>then run head and tail"]
```

### Above the threshold

1. **Confirmation.** If `_needs_confirmation` holds, run the whole range
   serially, which gathers it on this thread so this sample or the next is
   clean.
2. **Calibration.** If this backend has never measured its fan-out, do it now
   (once per backend, shared by all kernels) and recompute the threshold; if
   the range now falls below it, run serially.
3. **Recheck.** `recheck_due()` counts parallel-bound dispatches and fires on
   the first, then after 2, 4, 8 … more, doubling to `_RECHECK_CAP` (1024) and
   then every 1024th. A recheck:
    - sizes a slice with `_sample_elems`, running the range whole and serially
      if the slice would cover it;
    - places it with `next_sample_start(loop_end − probe)`, which returns
      `room · frac(k · 0.618…)` for the kernel's k-th sample — a golden-ratio
      walk that spreads positions evenly however many samples there are,
      never repeats one early, and needs no random number generator;
    - runs the head `[0, start)` via `_run_untimed`, fanned out if
      `start · r_s` exceeds the current fan-out estimate, otherwise untimed on
      this thread;
    - times the slice with `_run_serial`;
    - calls `_second_opinion` if the slice read more than 8x below
      the previous estimate and two slices' worth of range remain;
    - fans out the remainder.
4. **Otherwise** fan out the whole range (`whole_range=True`).

`_second_opinion` exists because one slice cannot tell a workload that got
cheap from a cheap *region*. It runs on to the next golden-ratio position
(untimed, fanned out if that pays), times a second slice, and compares. If
both read far below the old estimate, the workload changed and the low
reading stands; otherwise the range is uneven, and the estimate moves a
quarter of the way from the old value toward the mean of the two slices.

### Below the threshold

1. **Parallel recheck.** `_parallel_recheck_due` fires when the kernel is
   confirmed, `r_p` and the fan-out have been measured, and the range *would*
   fan out on the serial rate alone (`loop_end · r_s ≥ margin · fan_out`) —
   when `r_p` is what keeps it serial. It uses the same 1, 2, 4 … 1024
   back-off, counted separately (`serial_since_rp`, `rp_recheck_after`), and
   fans out the whole range. A range that size costs at least a margin's
   worth of fan-outs serially, so a losing recheck costs about one fan-out.
2. **Measured, or too small to probe:** run the whole range serially.
   `_probe_min_range()` is `max(num_threads, F · 2.0 / 24 ns)`, with `F` the
   calibrated or default fan-out — the range at which a hypothetical
   24 ns/element kernel (`_PROBE_REFERENCE_NS_PER_ELEM`) could repay a
   fan-out. Dearer kernels pay one serial dispatch before they have an
   estimate.
3. **First sight** of an unmeasured kernel on a range large enough to
   matter. A slice of `max(probe_min // 16, loop_end // 64)` elements is
   timed at the first golden-ratio position, a little past 60% of the way
   in, not at the front. If `_needs_confirmation` now holds, head and tail
   run on this thread (the tail timed). Otherwise each of head and tail fans
   out if it alone reaches the new threshold, else runs on this thread
   (head untimed, tail timed). If nothing fanned out, the range is marked not
   scattered. A backend that has not calibrated yet judges this against the
   200 µs default and calibrates on the next dispatch above threshold.

## Failure modes and their defenses

Each mechanism above exists because a specific failure was measured:

| Problem | Mechanism | Where |
|---|---|---|
| **Prefix bias.** An image's first rows are often background — 10–70x cheaper than the frame's average for a volume renderer (`next_sample_start`'s docstring) — so a sample from the front calls a frame that threads well too small to thread. | Interior golden-ratio positions for first sight and rechecks; a second opinion before believing a far-lower slice; a trusted parallel rate as a floor. | `next_sample_start`, first-sight branch of `_run`, `_second_opinion`, `_set_serial_cost` |
| **Corrupt-high estimates.** A preempted dispatch can time a thousand times the real cost; a high estimate then asks for short samples that are mostly fixed cost and read high again. | Rises above 8x capped at 2.75x per sample; slices at least 1/64 of the range; whole-range guard believes at most the reference rate. | `_run_serial`, `_MAX_SAMPLE_RATIO`, `_sample_elems` |
| **One-way door, threading side.** Only serial runs measure `r_s`, and threading stops serial runs; a corrupt-high estimate predicts an even slower serial run, so it cannot be caught against itself. | Rechecks on a geometric back-off that time a slice; a far-lower sample believed outright. | `recheck_due`, recheck branch of `_run` |
| **One-way door, serial side.** Only fan-outs measure `r_p`. Worker rates for identical fan-outs varied 8x on a 1-socket Xeon (October 2026), and one high draw held 6–34 M-element dispatches of a cheap kernel serial, up to 6x slower (`_parallel_recheck_due`'s docstring). | Parallel rechecks on the same back-off; the floor takes the lesser of sample and smoothed rate. | `_parallel_recheck_due`, `_record_parallel_cost` |
| **Scattered samples.** A sample after a fan-out, or on untouched pages, reads dear; believed, it keeps the backend fanning out, which keeps the next sample scattered. | `scattered` tracking; scattered samples only lower an affordable estimate; confirmation before the first near-threshold fan-out; first clean sample believed. | `scattered`, `rate_confirmed`, `_run_serial`, `_needs_confirmation` |
| **Stale floors.** A floor learned on expensive inputs survives when the same compiled kernel gets cheap inputs, and keeps it threading. | A whole-range fan-out with every span short retires the floor and schedules a serial sample. | `retire_floor` in `_parallel_execute`, `_record_parallel_cost` |
| **Fixed cost dominating short samples.** For a cheap kernel the call is most of a sample; for a short chunk the GIL hand-back is most of a span. | Call overhead subtracted; slices sized to 10x it; worker rates ignored below 1 µs; spans trusted as a floor only above `trusted_span`. | `_measure_call_overhead`, `_SAMPLE_OVERHEAD_RATIO`, `_RP_MIN_WORKER_NS`, `_RP_BOUND_SPAN_RATIO` |
| **Load, descheduling, a moving fan-out cost.** Load and idle states change the fan-out cost mid-program; a descheduled worker dominates a max; a high estimate stops the fan-outs that would lower it. | Idle curve; free and scheduled refreshes, including the hot knot's recovery probe; median worker; work-share condition; derived margin. | `_fan_out_estimate`, `_record_fan_out`, `_refresh_fan_out`, `_margin` |
| **Calibration dearer than the work.** Sleeping through idle gaps cost a 100k-element add a thousand times its own runtime (`_FAN_OUT_IDLE_PRIOR`'s comment). | Pessimistic priors replaced by measurements at real pauses; confirmation checked before calibration. | `_seed_fan_out_curve`, `_prior_knot_near`, order of checks in `_run` |

## Policies, environment and hardware

### History: policy v1

The policy described here was "v2" while an earlier one, v1, could still be
selected with `TACK_CPU_POLICY=v1`. v2 became the default on 2026-08-10 and
v1 was removed on 2026-10-15. `TACK_CPU_POLICY` is still read once, when the
backend is created: unset, empty or `v2` is accepted, and anything else,
`v1` included, raises `ValueError`, so a script that relied on v1 says so.
What v1 lacked:

| Mechanism | v1 | v2 (now the policy) |
|---|---|---|
| Serial rate: overhead subtraction, outlier cap, far-lower samples believed | yes | yes |
| Serial rechecks at rotating positions | yes | yes |
| Margin | constant 2.0 | derived from fan-out variability |
| Fan-out cost | one probe, minimum of 5 | idle curve, continuously refreshed |
| `r_p` in the threshold, per-worker timing | no | yes |
| Serial floor and its retirement | no | yes |
| Scattered-sample rule, confirmation, second opinion | no | yes |
| Parallel rechecks | no | yes |

The case for v2 as the default was recorded on a 2-socket Xeon in August 2026:
across 26 benchmark runs at four background loads, v1 made 30 fan-outs that
lost to serial and v2 made 5. The deciding variable was not load but whether
the fan-out cost *moved during the run*. An otherwise idle machine is the
unsteady case — idle cores fall into deep sleep states and the cost swings by
about 2x — while sustained load holds it steady, and on the steady machine v1
had less regret. v1's failures are fan-outs that lose outright; v2's are
speedups not taken. The module docstring keeps that reasoning.

### Environment variables

| Variable | Effect |
|---|---|
| `TACK_CPU_THREADS` | Worker count, parsed as an integer; values below 1 become 1. With one thread the pool is never created and every threshold is `_NEVER`. An explicit `num_threads`, from `tack.device("cpu", num_threads=…)` or `CPUBackend(num_threads=…)`, takes precedence. |
| `TACK_CPU_POLICY` | Accepted for old scripts only: unset, empty or `v2`. Anything else, `v1` included, raises `ValueError`. |
| `TACK_CPU_MARGIN` | A float that replaces the derived margin. For experiments. |
| `TACK_CPU_POOL` | `python` (also `0`, `off`, `no`, `false`) fans out through a `ThreadPoolExecutor` instead of the JIT'd pool. Anything else, or unset, uses the pool where it can be made. |
| `TACK_CPU_POOL_SPIN` | How many pause iterations a pool worker spins before sleeping (default 20000). For experiments. |
| `TACK_CPU_MALLOC` | Linux only, off by default (read by `dispatch.env_flag`). When on, the first CPU backend tells glibc to keep freed pages (`_tune_malloc`: `mallopt` sets the mmap threshold to its 32 MiB cap, the trim threshold to 4 GiB and the top pad to 256 MiB), process-wide. A filter that frees and reallocates the same temporaries otherwise takes their page faults on every call: threshold at 5M cells on 8 threads of one bound NUMA node 126 → 104 ms. Opt-in because it changes `malloc` for NumPy and every other library, and unbound on a 28-core EPYC it gained nothing (6–8% slower, within noise); a reused block also keeps the placement of its first touch. |

The ratios, caps, intervals and priors in the module's underscore-prefixed
constants are private, with no supported override. Each carries a comment on
what it trades and, where calibrated, against what.

### Counting cores and NUMA

`_physical_core_count` picks the default thread count. A hyperthread buys
little for compute while costing a slot in every fan-out, so each platform is
asked for physical cores, falling back to `os.cpu_count()`:

| Probe | Source | Notes |
|---|---|---|
| `_linux_core_count` | `/sys/devices/system/cpu/cpu0/topology/thread_siblings_list` | Logical CPUs divided by cpu0's sibling count |
| `_macos_core_count` | `sysctlbyname("hw.perflevel0.physicalcpu")`, then `hw.physicalcpu` | Performance cores only: with equal chunks an efficiency core paced the whole dispatch (the docstring: 8 threads beat 10 on an 8+2 machine). The JIT'd pool is sized the same way (`cpu_pool._performance_cores`): a pool of `os.cpu_count()` had two workers more than any launch used |
| `_windows_core_count` | `GetLogicalProcessorInformationEx(RelationProcessorCore)` | Never run on Windows; any failure returns `None` |

A new array is not filled: `allocate_array` returns uninitialized
storage, and its pages are first touched by the kernel that writes it, in
parallel when the policy fans that kernel out. Filling at allocation was
serial, and dearer after a fan-out, while the workers' caches owned the
lines: at 5M cells on eight threads of a 2 GHz Xeon it was 130 of
threshold's 276 ms and 77 of cell-to-point's 135. A fill of 256 KiB or
more (`_PARALLEL_FILL_MIN_BYTES`; `tack.zeros`, `ones` and `full` among
them) runs `tack.algorithms.copy._fill_kernel` through the normal dispatch
(`CPUBackend.fill_array`, offered first by `Array.fill`), after NumPy has
converted the value, so it stores and refuses what `ndarray.fill` would.
Smaller fills, a one-thread backend and another backend's buffers fill in
place.

On x86-64 Linux with libnuma and two or more memory nodes, the JIT'd
pool's and the executor's threads are made under an interleave policy,
which they inherit (`_jit_pool`, `_get_pool`; the executor's threads are
all started there rather than lazily by whichever dispatch first needs
one), so the pages they first touch spread across nodes.
`_init_numa` reads the policy back with `get_mempolicy` and
stays off unless the kernel confirms it; wrapped external memory is not
affected. Only a thread whose policy is the default is changed, and the
policy found is put back: a process started under `numactl --membind` (or
bound through libnuma) keeps its binding and its arrays stay on its nodes.
The threading
policy has no NUMA term — remote-memory effects show up in the measured
rates — and workers are not pinned.

## Implementation notes

The developer guide's summary of the mechanism, kept here with its
measurements when the guide was shortened to its rules.

The CPU backend fans a loop range out to its thread pool only when the serial run would cost meaningfully more than the fan-out. Both sides are measured, not assumed:

- The fan-out itself is one ctypes call into a pool JIT'd by `runtime/cpu_pool.py` (LLVM IR built with `llvmlite.ir`, no C compiler): `N - 1` detached pthreads each wait on their own mailbox, a launch count (spinning briefly, then blocking on a futex on Linux or `os_sync_wait_on_address` on macOS 14.4+, woken only if asleep; `nanosleep` polls backing off to 1 ms elsewhere, which on the Mac made every launch wait for its slowest sleeper and kept mid-size launches serial); a launch of `k` threads posts to workers `0 .. k - 2` only and runs as worker `k - 1` itself, so the others are not woken and the `k` of every launch are the same `k` (the hot fan-out is 3 + 0.3k µs on a 2 GHz Xeon, where waking all 16 for a launch of 2 cost 11; workers are not pinned: pinning lost on a two-socket Xeon and on jura's EPYC, since a pinned worker cannot move off a busy core); the workers and the caller claim chunks from a counter (`GRAIN` per worker, at least `MIN_CHUNK` elements) and run them through a per-kernel wrapper that unpacks the arguments from 8-byte slots (`chunk_wrapper`, `pack_args`); a worker on a busy core's sibling hyperthread, or over the dear part of an uneven range, simply takes fewer (one equal chunk per worker made a compute-bound 1M-cell kernel take 29 or 50 ms as the scheduler placed the workers). The pool is one per process, sized by the CPUs the process may run on (on Apple silicon the performance cores, `hw.perflevel0.physicalcpu`, so no idle efficiency-core workers are woken and waited for), and a dispatch uses as many of its threads as its backend has. The pool program uses inline asm (arm64 `yield`), so `_target_machine` initializes the native asm parser: without it LLVM aborts the process on Apple silicon instead of raising. Its launch state (function, pack, range, counters, the spans and counts a run leaves) is one set of JIT'd globals, so a run is a transaction under `CPool._txn`, held from the ctypes call through the read of its report: two Python threads dispatching at once each wrote the other's run and read its results (TA-45). A start whose `pthread_create` fails part way sets `quit`, wakes the workers it did make and joins them before returning failure, since they were waiting in code the constructor was about to free (TA-46, a SIGSEGV). Its workers, like the executor's (`_get_pool`, which starts them all at once), are made under the interleave policy, or keep a bound task's binding, so the pages of a new array they first-touch are spread the way a buffer filled under the policy was. Through `ThreadPoolExecutor` a fan-out cost ~60 µs per worker on a 2 GHz Xeon (490 µs for 8, 1.2 ms after an idle gap), because every worker's wake-up and GIL handoff serialized; through the pool it is 3–5 µs hot and ~100 µs after an idle gap, and the dataset filters at 1M cells went from 1.05× to 1.9× on 8 threads there. `TACK_CPU_POOL=python` keeps the executor path, which is also the fallback on Windows or if the pool cannot be made (`CPUBackend.pool_kind`). Tests that simulate workers through `call_range` or `_get_pool` must set the Python path; `test_cpu_threading.py` does so for its whole module.

- Each `CompiledKernel` carries `ns_per_elem`, a smoothed estimate of its serial cost, updated on every serial dispatch. From it the backend precomputes `parallel_min_elems`, so the dispatch hot path is one integer compare.
- The backend measures its own fan-out cost by dispatching *empty* ranges through the real path — no loop iterations, so the probe has no side effects. That is a curve over idle gaps, interpolated on the actual idleness at dispatch, refined from real fan-outs, and re-measured directly at most every 100 ms. Only its hot end is measured at the first fan-out; each idle knot starts as a pessimistic prior (`_FAN_OUT_IDLE_PRIOR`, 4× hot, applied as a step) and is replaced, and moved, by the first fan-out measured at a pause the program actually took — measured at once rather than on the 100 ms clock. Nothing sleeps: calibrating by sleeping through each gap cost 180 ms inside whichever dispatch first looked worth threading (P8).
- The first time a kernel is seen at a range large enough to matter, a small slice is timed serially and the rest is decided on that sample. The slice comes from inside the range (the same golden-ratio walk as the rechecks), not its prefix.
- Once a kernel threads, occasional rechecks (dispatches 1, 2, 4, … 1024, then every 1024) re-time a serial slice. The slice walks the range in golden-ratio steps rather than always sampling the prefix, because an image kernel's first rows are background and a prefix sample read a volume render 10–70× too cheap. A range whose serial run costs no more than a fan-out is re-timed whole; sufficiently long worker spans establish a floor under the serial estimate. Worker spans and rates are summarized by `_worker_median`, the lower middle value for an even count: the upper one of two workers is the slower worker, so one descheduled thread set the floor on two-thread machines. If every worker of a complete dispatch later finishes below that duration, the old floor is retired and the next call rechecks serial cost. Cheap partial slices or a short median alone cannot retire it.
- A serial sample is *clean* only when a serial run laid the range out before it (`CompiledKernel.scattered` is false). One taken on untouched pages, or straight after a fan-out while other cores own the lines, reads a bandwidth-bound kernel 3–5× dear, and believing it was a fixed point: it kept the kernel fanning out, which kept the next sample dear (P7 in the audit). A scattered sample may lower the estimate but not raise it, wherever a serial run of the range costs at most `_CONFIRM_BAND` margin-weighted fan-outs; and until a kernel has one clean sample, a fan-out inside that budget runs serially instead, at most twice per kernel. The first clean sample is believed outright when it is lower, since everything before it was an upper bound; smoothing it in left the estimate stuck near 3× in two harness runs of three. Outside the budget a rise is real evidence (an image kernel's estimate starts low) and is believed as before. A recheck slice reading 8× under the estimate is believed outright only if a second slice from later in the range agrees (`_second_opinion`); otherwise one background-row slice could send an image kernel's whole frame serial.
- The parallel rate has the same one-way door from the other side (P9): only fan-outs measure `r_p`, so a high one holds a kernel serial and nothing corrects it. When a range would fan out on the serial rate alone but `r_p` keeps it serial, the backend fans out anyway on the recheck back-off schedule (`_parallel_recheck_due`). The serial floor takes the lesser of the qualifying sample and the smoothed `r_p`: worker rates for identical fan-outs spread 8× run to run, and a floor from one high draw raised `r_s` to `r_p`, where the formula credits threading with no gain. `scattered` is cleared by `_run_whole_serial` around `_run_serial`, not inside it, because tests replace `_run_serial` with a four-argument lambda.

A fixed element count cannot work here: the crossover moves ~1000× with arithmetic intensity (~4M elements for `out[i] = x[i]*2+1`, ~130K for a `sqrt`/`sin` expression, ~4K for a 20-iteration inner loop). The previous constant of 1024 sat below all of them, making mid-size dispatches of cheap kernels 3–10× slower than running them serially.

## The contract

### Guaranteed: the decision does not change results

Every path through `_run` executes each index of `[0, loop_end)` exactly
once. `_parallel_execute` chunks are contiguous, disjoint, and cover
`[start, end)`, whether the executor assigns them or the pool's workers
claim them from a counter. First sight and rechecks split the range into head, slice and
tail (plus the second opinion's head and slice), each starting where the last
ended. Every probe that is not real work — `_measure_call_overhead`,
`_calibrate_fan_out`, `_refresh_fan_out` — runs on an empty range, which
executes no iterations. No range is re-run to time it. Each of these calls
does run the statements outside the parallel loop again, which is why the
frontend accepts only effect-free statements there and gives every
iteration fresh copies of the outer locals it assigns (see
[Parallel Execution](parallel-execution.md#statements-outside-the-parallel-loop)).

So for any program that meets the
[execution contract](../reference/language-contract.md#execution-and-ordering)
— no conflicting cross-iteration accesses — serial execution, a fan-out, and
every split between them produce the same array contents. Atomic updates are
applied once per iteration under every decision. The contract leaves
[atomic accumulation order](../reference/language-contract.md#atomic-array-updates)
unspecified and a fan-out changes it, so floating-point atomic sums may round
differently between dispatches.

### Not guaranteed: performance

No dispatch is promised the faster path, nor the same choice across runs,
processes or machines; decisions near a crossover flip on noise. Some
dispatches take a slower path on purpose: rechecks run part of a range
serially, and parallel rechecks fan out a range the policy believes is better
serial. Refreshes add about one empty fan-out per 100 ms of
dispatching, plus the hot knot's probe at most once a second. The policy
promises to keep measuring, not to be right.

### How it is tested

`packages/tack-core/tests/test_cpu_threading.py` has three kinds of check.
Its tests drive the policy through the Python path's seams (a patched
`call_range`, an inline pool, a fake clock), so an autouse fixture sets
`TACK_CPU_POOL=python` and every backend they make fans out through the
executor. The JIT'd pool has its own tests in `test_cpu_pool.py` (whole
chunks covering the range, a launch of `k` using only the first `k`
workers, concurrent fan-outs, a failed start that falls back and exits
cleanly), the fills in `test_cpu_fill.py`, and first-touch placement in
`test_numa.py`.

**Correctness through every path.** Results against a reference for ranges of
1 to 300,000 elements and over repeated dispatches; an atomic accumulation
that must count each element once; a first-sight dispatch that must visit
each element once (`test_first_sight_samples_inside_the_range_not_its_prefix`);
floor retirement executing each range once. These hold whatever was chosen.

**Controlled-clock policy tests.** A policy that measures itself needs tests
for the measurement going bad, so most policy tests supply their timings:
`_fixed_timing` and `_controlled_serial_clock` script `perf_counter_ns`
around `_run_serial` (the kernel still runs), and `_pin_fan_out` fixes the
fan-out estimate and margin so thresholds are arithmetic. They cover the
outlier cap, a believable rise being tracked, recovery from a wrong decision
to thread, scattered samples, the first clean sample, a stale `r_p`, the
floor and its retirement, the absence of sleeps, the idle knots, and the
recheck back-off. Several have a **negative control** that disables the
defending mechanism and asserts the failure appears:
`test_negative_control_no_rechecks_means_no_correction`,
`test_negative_control_prefix_only_sampling_is_detected`, and
`test_negative_control_stale_rate_without_recovery_stays_serial`.

**The `timing` group.** Tests marked `@pytest.mark.timing` assert what the
real scheduler did: an expensive kernel really fans out, and a front-loaded
kernel's rechecks never re-run the whole range. They run in the default suite
but call `_require_idle`, which skips them when the one-minute load average
exceeds half the thread count. Each has a controlled-clock counterpart, so a
timing failure with the controlled test passing points at the host, not the
policy's logic.

**Decision quality** is measured by `benchmarks/threading_decisions.py`, which
times both paths for three kernels and scores what the backend chose:

| Aspect | What the harness does |
|---|---|
| Bracketing | Grid sizes straddle each kernel's crossover, where the errors are. Coverage requires *measured* serial wins below measured parallel wins, not a fitted crossover; `--scale` moves the grid. A perfect score usually means the grid missed the crossovers. |
| Scoring | Per row: stored choice, faster measured path, regret. Summary: wrong and over-eager counts. Regret matters more than the count. |
| Isolation | The choice and inputs are read *before* the parallel path is timed (which would update `r_p` and the curve); estimator state is then restored, except `scattered`, which describes where the data really is. |
| Snapshot provenance | Inputs read for a row need not be those the cached threshold was built from, so each snapshot is labelled consistent, different or unavailable; scores always use the stored threshold. Covered by `packages/tack-core/tests/test_threading_benchmark.py`. |
| Conditions | `--load N` adds N CPU-bound subprocesses; empty-fan-out median and spread are reported before and after the sweep, with drift flagged; `--json` records everything with host metadata. |

## Known limitations and open questions

**Which policy decides better depends on the host.** On a 1-socket Xeon VM
(20 threads, 2026-10-04), v2 had lower whole-sweep regret than v1 in 9 of 10
paired repetitions, and v1 made 2–3 losing fan-outs per sweep. On an M1 Max
workstation (8 threads, 2026-10-04, ambient load not verified idle), v2 had
lower regret in 2 of 15 pairs; neither policy made a losing fan-out, and v2's
extra regret — missed parallel wins — was attributed mainly to its fan-out
estimates and derived margins rising under added load. Grids, thread counts
and conditions differ, so no matched cross-host claim follows. The default
and the margin constants were left unchanged.

**The derived margin measures unpredictability, not slowness.** A steadily
loaded machine has low `cv` and gets a small margin, which is right only if
the estimates have tracked the load. `_margin()`'s docstring names faster
tracking, not a wider margin, as what that case needs.

**Rechecks cost the most on the most expensive kernels.** A slice is at least
1/64 of the range; for a 3500 ns/pixel volume renderer that was recorded as
10–40 ms of serial time plus an extra fan-out per recheck on a 2-socket Xeon
(2026-10-02), on dispatches 1, 2, 4, … 1024 and every 1024th after. A smaller
fraction would reopen the short-sample failure the rule guards against.
Climbing out of a large underestimate also takes several rechecks, since one
sample raises the estimate at most 2.75x.

**Which idleness a cached threshold was priced at.** When
`_record_parallel_cost` recomputes the threshold at the end of a fan-out,
`_last_dispatch_ns` has not been updated yet, so the gap includes the fan-out
just finished and a long dispatch is priced as if the workers had been idle
throughout. Before the first fan-out or refresh, the gap runs from the
clock's zero. Neither affects correctness; their effect on decisions has not
been studied.

**Core counting is coarse.** The Linux probe reads cpu0's sibling list as
comma-separated entries and counts every physical core, including efficiency
cores on hybrid x86 parts. No probe consults CPU affinity or a container's CPU
quota. `TACK_CPU_THREADS` is the override.

**Windows is unverified for this policy.** `_windows_core_count` has never
run, and every estimate depends on `perf_counter_ns` resolving short intervals
and on the scheduler's wake-up behavior; neither has been measured on Windows.

**CI checks the logic, not the tuning.** The controlled tests pin the decision
for given measurements. Whether the measurements and ratios suit a machine is
checked only by running `benchmarks/threading_decisions.py` on it.

## A thread count per launch

Tried and dropped (2026-10-14). `TACK_CPU_SUBSETS=1` let a fan-out use
the first `k` of the `N` threads: each worker's chunk had to cost ten
shares of the hot fan-out, and a kernel whose full fan-out reached a
speedup `S` of at most `N / 4` got `ceil(2 S)` threads, with full runs
on a back-off schedule to keep `r_p` at `N` current. On 16 threads of a
hyperthreaded Xeon it lost nothing to always-`N`. On jura's 28 cores it
lost: the dataset filters ran 12% slower with it, its own score (every
`k` timed directly) had rows losing over 5% to always-`N` and 2.4-3.2 ms
of regret, it kept 28 threads for bandwidth-bound launches that run
10-16% faster on 4-6, and through the rates it fed back it turned the
serial-or-parallel decision under-eager (14 of 18 rows wrong at
`threading_decisions.py --scale 1`). The per-worker mailboxes it came
with stayed: a launch still wakes only the workers it uses.
