"""Is policy v2's learned `r_p` any good? Run this before trusting v2 anywhere.

v2's threshold is `fan_out * M / (r_s - r_p)`, so `r_p` sits in a
denominator: bias it toward `r_s` and the threshold inflates without
bound. `threading_decisions.py` records `r_s` but not `r_p` -- it was
written before v2 existed -- so a v2 run that overshoots its margin
cannot be decomposed there. This supplies the missing column.

`r_p` is learned as a residual:

    sample = (elapsed - fan_out_estimate) / elems     (_record_parallel_cost)

which charges every error in the fan-out estimate to `r_p`, amplified by
1/elems. Near the crossover -- the only place the threshold matters --
work and fan-out are the same size by definition, so that amplification
is order 1 and the fan-out curve's own run-to-run spread is enough to
ruin it.

    TACK_CPU_POLICY=v2 uv run python benchmarks/threading_rp_check.py

**What good looks like:** `bias` near 1.0x and `P_eff` above 1. A `P_eff`
below 1 says the backend believes fanning out makes the work *slower per
element*, which is not a thing that happens; it means `r_p` is being read
off noise.

Measured on yavin (M1 Max, 8 threads), where it fails:

    kernel   r_s be  r_s fit  r_p be  r_p fit  bias    P_eff be  P_eff fit
    cheap      0.11     0.11    0.59     0.05  11.3x       0.18       2.12
    medium     2.47     1.91    1.08     0.48   2.2x       2.28       3.95
    heavy      6.40     6.05    2.67     0.92   2.9x       2.39       6.56

reproduced at load 4.2 and 8.4, so it is not contention. The knock-on is
that v2's thresholds land at 1.90x and 2.10x the crossover against a
margin of 1.5, and yavin pays 220-241 us of regret where v1 paid 6-87.

`_RP_MIN_WORK_RATIO` does not rescue it: raised to 3 or beyond, no
dispatch in the scoring grid qualifies at all, `r_p` stays 0.0, and v2
reduces exactly to v1 -- correct, but not a fix. That is not a tuning
range, it is the absence of one.
"""

import statistics
import sys
import time

import numpy as np

sys.path.insert(0, "benchmarks")

from threading_decisions import GRIDS, KERNELS

import tack
from tack.runtime.dispatch import get_backend


def variant_for(backend, name):
    for slot in list(backend._cache.values()):
        for v in slot.values():
            if v.ir.name.startswith(name):
                return v.payload
    raise LookupError(name)


def fit(xs, ys):
    slope, intercept = np.polyfit(np.asarray(xs, float), np.asarray(ys, float), 1)
    return float(intercept), float(slope)


def main():
    tack.init(arch=tack.cpu)
    be = get_backend()
    if be.policy != "v2":
        raise SystemExit("run with TACK_CPU_POLICY=v2 -- v1 has no r_p")

    print(f"policy {be.policy}  margin {be.margin}  threads {be.num_threads}\n")
    print(f"{'kernel':7s} {'r_s be':>8s} {'r_s fit':>8s} {'r_p be':>8s} "
          f"{'r_p fit':>8s} {'bias':>7s} {'P_eff be':>9s} {'P_eff fit':>10s}")

    worst = 1.0
    impossible = []
    for name, kernel in KERNELS.items():
        grid = GRIDS[name]
        big = grid[-1] * 8
        x = tack.field(dtype=tack.f32, shape=(big,))
        out = tack.field(dtype=tack.f32, shape=(big,))
        x.from_numpy(np.ones(big, dtype=np.float32))

        # Drive the whole grid repeatedly, so the estimates settle exactly
        # as they do in the scoring run whose threshold is in question.
        for _ in range(6):
            for n in grid:
                kernel(x, out, n)

        compiled = variant_for(be, name)
        r_s_be = compiled.ns_per_elem
        r_p_be = compiled.ns_per_elem_parallel

        # Reference: both paths timed directly, well above the crossover
        # where the slopes are real rather than floor-plus-noise.
        pts = []
        for n in (grid[0] * 8, grid[-1] * 8):
            prefix = compiled.bind([x, out, n])
            serial = []
            for _ in range(4):
                t0 = time.perf_counter_ns()
                compiled.call_range(prefix, 0, n)
                serial.append(time.perf_counter_ns() - t0)
            par = []
            for _ in range(5):
                t0 = time.perf_counter_ns()
                be._parallel_execute(compiled, prefix, 0, n)
                par.append(time.perf_counter_ns() - t0)
            pts.append((n, min(serial), statistics.median(par)))

        _, r_s_fit = fit([p[0] for p in pts], [p[1] for p in pts])
        _, r_p_fit = fit([p[0] for p in pts], [p[2] for p in pts])

        bias = r_p_be / r_p_fit if r_p_fit > 0 else float("nan")
        peff_be = r_s_be / r_p_be if r_p_be > 0 else float("inf")
        peff_fit = r_s_fit / r_p_fit if r_p_fit > 0 else float("inf")
        if bias == bias:
            worst = max(worst, bias, 1 / bias if bias > 0 else 1.0)
        if peff_be < 1.0:
            impossible.append(name)
        print(f"{name:7s} {r_s_be:8.2f} {r_s_fit:8.2f} {r_p_be:8.2f} "
              f"{r_p_fit:8.2f} {bias:6.2f}x {peff_be:9.2f} {peff_fit:10.2f}")

    print(f"\nworst bias {worst:.1f}x", end="")
    if impossible:
        print(f"; P_eff below 1 for {', '.join(impossible)} -- the backend "
              f"believes\nfanning out makes the work slower per element, "
              f"so r_p is noise there.")
    else:
        print(". No kernel reports a P_eff below 1.")


if __name__ == "__main__":
    main()
