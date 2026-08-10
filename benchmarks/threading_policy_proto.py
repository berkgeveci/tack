"""Prototype: one estimator for item 20 and P6, scored against ground truth.

The backend's threshold is

    parallel_min_elems = fan_out * BREAK_EVEN / ns_per_elem       (cpu.py)

and `threading_decisions.py` showed two things wrong with it, on two
machines. `fan_out` comes from `_calibrate_fan_out`, which takes the
minimum of back-to-back samples and so reads 0.4-0.65x the cost a
dispatch meeting idle workers actually pays (item 20). And the formula
assumes the parallel speedup of the work is the thread count, when it is
`P_eff` = 1.5-9.2 depending on the kernel (P6).

The second one has an exact form. The real crossover is where

    fixed + n*r_s  ==  fan_out + n*r_p          r_p = r_s / P_eff

so  n* = (fan_out - fixed) / (r_s - r_p) ~= fan_out / (r_s * (1 - 1/P_eff))

and therefore

    threshold / n*  ==  BREAK_EVEN * (1 - 1/P_eff)

which is the `model` column of the decision harness. Measured against it
this identity holds to 1% for `medium` and `heavy` on both machines. So
the margin the backend actually applies is not 2.0; it is 2.0 scaled by
how much parallelism the kernel can offer, and it goes to zero as P_eff
goes to 1. That is P6, and it is a derivation rather than a tuning
accident.

Three policies are scored here against measured crossovers:

  A  current      fan_out_probe * M / r_s
  B  P_eff-fixed  fan_out_probe * M / (r_s * (1 - 1/P_eff))
  C  regression   M * fan_out_hat / (r_s - r_p), with BOTH fan_out_hat
                  and r_p recovered by fitting real parallel dispatches

C is the interesting one: a + b*n fitted through observed parallel times
yields the fan-out as the intercept and the parallel rate as the slope,
so one mechanism supplies both quantities the formula needs, learned
under whatever idleness the caller's own workload has. That also
dissolves the "hot floor or idle floor" question -- the observations
carry it instead of a constant choosing it.

    uv run python benchmarks/threading_policy_proto.py
    uv run python benchmarks/threading_policy_proto.py --gap-ms 50

Measured on yavin (M1 Max, 8 threads). Back-to-back, against a 2.0x
target:

    kernel     A       B       C
    cheap    0.66x   1.36x   2.58x
    medium   1.00x   1.20x   2.29x
    heavy    0.77x   0.89x   1.70x

With a 50 ms gap before each dispatch, parallel loses at every size in
the grid, so the right call is serial everywhere: A fans out on 3 of 3,
B on 2 of 3, C on none. That is the regime `_calibrate_fan_out` cannot
see by construction, and the one where the current policy is wrong about
everything.

Two things this does NOT establish, both load-bearing before any of it
ships:

  * **C's fit is not yet trustworthy in the idle regime.** At 50 ms its
    per-kernel intercepts read 878 / 1708 / 3473 us for a quantity that
    is a property of the machine, and `heavy` came back with a *negative*
    parallel slope. C makes the right decisions there because a huge
    intercept pushes the threshold above the grid, not because the fit
    is sound -- right for the wrong reason, which is exactly the failure
    shape this benchmark exists to catch. Pooling the intercept and using
    four fit points rather than two both helped and neither is enough.

  * **Censoring is untested.** In a real backend the observations are not
    a designed grid: only ranges above the current threshold are ever
    dispatched in parallel, so the estimator sees a truncated sample of
    its own choosing, and a threshold set too high starves the fit that
    would lower it. P3's recheck mechanism is the existing precedent for
    breaking that loop. Nothing here simulates it.
"""

import argparse
import itertools
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, "packages/tack-core/src")

import tack
from tack.runtime.dispatch import get_backend

sys.path.insert(0, "benchmarks")

MARGIN = 2.0


# Imported rather than redefined. The first version of this script wrote
# its own `medium` and `heavy` from memory and got kernels an order of
# magnitude cheaper -- r_s of 0.19 and 0.51 against the harness's 1.98 and
# 5.99 -- so neither had a crossover anywhere near the scoring grid and
# two of the three rows came back empty. Scoring a new policy against a
# different workload than the old one was measured on is not a comparison.
from threading_decisions import GRIDS, KERNELS

# A decade above the scoring grid, matching the harness's MODEL_SCALE, so
# the slopes are real rather than floor-plus-noise.
#
# Four points, not two. Two is enough to define a line and nothing else:
# with a 50 ms gap before each dispatch the intercept is ~1.7 ms against
# work of a similar size, and a two-point fit turned that into a negative
# slope and P_eff of infinity. The regression's whole claim is that it
# can separate the intercept from the slope, so it has to be given enough
# points to do so.
FIT_GRID = {k: tuple(int(v[0] * 8 * f) for f in (1, 2, 4, 8))
            for k, v in GRIDS.items()}
SCORE_GRID = {k: list(v) for k, v in GRIDS.items()}


def best_of(fn, reps):
    times = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        fn()
        times.append(time.perf_counter_ns() - t0)
    return min(times)


def median_of(fn, reps, gap_ms=0.0):
    times = []
    for _ in range(reps):
        if gap_ms:
            time.sleep(gap_ms / 1000.0)
        t0 = time.perf_counter_ns()
        fn()
        times.append(time.perf_counter_ns() - t0)
    return statistics.median(times)


def variant_for(backend, name):
    for slot in list(backend._cache.values()):
        for variant in slot.values():
            if variant.ir.name.startswith(name):
                return variant.payload
    raise LookupError(name)


def fit(xs, ys):
    slope, intercept = np.polyfit(np.asarray(xs, float), np.asarray(ys, float), 1)
    return float(intercept), float(slope)


def measure(backend, name, gap_ms):
    """Ground truth for one kernel: both paths, fitted where they are real."""
    kernel = KERNELS[name]
    big = max(FIT_GRID[name])
    x = tack.field(dtype=tack.f32, shape=(big,))
    out = tack.field(dtype=tack.f32, shape=(big,))
    x.from_numpy(np.ones(big, dtype=np.float32))

    kernel(x, out, 1024)
    compiled = variant_for(backend, name)

    pts = []
    for n in FIT_GRID[name]:
        prefix = compiled.bind([x, out, n])
        s = best_of(lambda: compiled.call_range(prefix, 0, n), 4)
        p = median_of(
            lambda: backend._parallel_execute(compiled, prefix, 0, n), 5, gap_ms)
        pts.append((n, s, p))

    _, r_s = fit([p[0] for p in pts], [p[1] for p in pts])
    fan_hat, r_p = fit([p[0] for p in pts], [p[2] for p in pts])
    fixed_s, _ = fit([p[0] for p in pts], [p[1] for p in pts])

    score = []
    for n in SCORE_GRID[name]:
        prefix = compiled.bind([x, out, n])
        s = best_of(lambda: compiled.call_range(prefix, 0, n), 4)
        p = median_of(
            lambda: backend._parallel_execute(compiled, prefix, 0, n), 5, gap_ms)
        score.append({"n": n, "serial": s, "parallel": p})

    # "No crossover in the grid" is a real answer, not a failure: with a
    # 50 ms gap before every dispatch, parallel loses at every size here,
    # and the right decision for every row is serial. Scored as such.
    crossover = None
    never_wins = all(r["serial"] <= r["parallel"] for r in score)
    for lo, hi in itertools.pairwise(score):
        if lo["serial"] <= lo["parallel"] and hi["serial"] > hi["parallel"]:
            # Linear interpolation on the difference.
            d0 = lo["serial"] - lo["parallel"]
            d1 = hi["serial"] - hi["parallel"]
            crossover = lo["n"] + (hi["n"] - lo["n"]) * (-d0) / (d1 - d0)
            break

    return {
        "name": name, "r_s": r_s, "r_p": r_p, "fan_hat": fan_hat,
        "fixed_s": fixed_s, "p_eff": r_s / r_p if r_p > 0 else float("inf"),
        "score": score, "crossover": crossover,
        "never_wins": never_wins,
        "field": (x, out), "compiled": compiled,
    }


def thresholds(m, probe_ns, backend_rate, fan_pooled):
    """What each policy would set, from the inputs it is entitled to.

    C takes a *pooled* intercept rather than its own. Fan-out is a
    property of the machine, so it cannot legitimately differ by which
    kernel measured it -- and per-kernel intercepts here read 222, 315
    and 221 us, which is fit noise pretending to be signal. The decision
    harness made exactly this mistake in its first cut and fixed it the
    same way. In a real backend the pooled value is the natural one
    anyway: there is one thread pool, so one intercept, updated by every
    kernel that fans out.
    """
    r_s, r_p, p_eff = m["r_s"], m["r_p"], m["p_eff"]
    a = probe_ns * MARGIN / backend_rate
    b = (probe_ns * MARGIN / (backend_rate * (1 - 1 / p_eff))
         if p_eff > 1 else float("inf"))
    c = (MARGIN * fan_pooled / (r_s - r_p)) if r_s > r_p else float("inf")
    return {"A current": a, "B P_eff-fixed": b, "C regression": c}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gap-ms", type=float, default=0.0,
                    help="idle gap before each parallel dispatch; 0 is a "
                         "back-to-back loop, 50 a script that pauses")
    args = ap.parse_args()

    tack.init(arch=tack.cpu)
    backend = get_backend()

    print(f"threads {backend.num_threads}, gap {args.gap_ms:g} ms\n")

    # Force the probe to calibrate. The first big dispatch only *sets*
    # ns_per_elem -- parallel_min_elems starts at _NEVER, so nothing
    # considers fanning out until a later call sees a lowered threshold,
    # and the probe runs then.
    warm = tack.field(dtype=tack.f32, shape=(8 << 20,))
    warm_o = tack.field(dtype=tack.f32, shape=(8 << 20,))
    for _ in range(4):
        KERNELS["cheap"](warm, warm_o, 8 << 20)
        if backend._fan_out_ns is not None:
            break
    probe = backend._fan_out_ns
    if probe is None:
        raise SystemExit("the backend never calibrated its fan-out probe")
    print(f"the probe reads {probe/1000:.1f} us\n")

    print(f"{'kernel':7s} {'r_s':>7s} {'r_p':>7s} {'P_eff':>6s} "
          f"{'fan_hat':>9s} {'crossover':>10s}")
    models = {}
    for name in KERNELS:
        m = measure(backend, name, args.gap_ms)
        models[name] = m
        cross = f"{m['crossover']:.0f}" if m["crossover"] else "none"
        print(f"{name:7s} {m['r_s']:7.2f} {m['r_p']:7.2f} {m['p_eff']:6.2f} "
              f"{m['fan_hat']/1000:8.1f}u {cross:>10s}")

    fan_pooled = statistics.median([m["fan_hat"] for m in models.values()])
    per_kernel = ", ".join(f"{m['fan_hat'] / 1000:.0f}" for m in models.values())
    print(f"\npooled fan-out intercept: {fan_pooled / 1000:.1f} us "
          f"(per-kernel: {per_kernel} us) -- the probe reads "
          f"{probe / 1000:.1f} us")

    print(f"\n--- threshold / true crossover (1.00x is the crossover, "
          f"{MARGIN:.1f}x is the margin asked for) ---")
    print(f"{'kernel':7s} {'A current':>11s} {'B P_eff-fixed':>14s} "
          f"{'C regression':>13s}")
    for name, m in models.items():
        if not m["crossover"]:
            # Parallel never wins here, so the correct threshold is above
            # the grid. Report whether each policy would have stayed
            # serial across all of it, which is the only thing to get
            # right in this regime.
            th = thresholds(m, probe, m["r_s"], fan_pooled)
            hi = max(SCORE_GRID[name])
            verdict = "  ".join(
                f"{k.split()[0]}:{'serial' if th[k] > hi else 'FANS OUT'}"
                for k in ("A current", "B P_eff-fixed", "C regression"))
            note = "parallel never wins in grid" if m["never_wins"] else \
                   "no clean crossing"
            print(f"{name:7s} {note:28s} {verdict}")
            continue
        # The backend's own rate estimate, as the harness reads it.
        rate = m["r_s"]
        th = thresholds(m, probe, rate, fan_pooled)
        cells = "".join(
            f"{th[k]/m['crossover']:>{w}.2f}x"
            for k, w in (("A current", 10), ("B P_eff-fixed", 13),
                         ("C regression", 12)))
        print(f"{name:7s} {cells}")

    print("\nA collapses toward 0 as P_eff -> 1 (that is P6) and inherits "
          "the probe's\nunderstatement (that is item 20). B fixes the second "
          "term only. C derives\nboth from the same fit, so it should land "
          f"near {MARGIN:.1f}x for every kernel.")
    return models


if __name__ == "__main__":
    main()
