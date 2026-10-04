"""Does the CPU backend fan out when it should?

The backend decides between a serial run and a thread fan-out by comparing
a measured per-kernel cost against a measured fan-out cost. This measures
*both paths directly* and then asks what it chose, so the score is decision
quality rather than wall-clock — which matters, because wall-clock
comparisons of this drown in cross-process noise.

    uv run python benchmarks/threading_decisions.py
    TACK_CPU_THREADS=4 uv run python benchmarks/threading_decisions.py

Two things to know before reading the output.

The sizes are picked to *bracket each kernel's crossover*, because that is
the only place the decision is in doubt. A coarser sweep of the same three
kernels scored a perfect 18/18 while a fine one scored 8/18 — the errors
live within about a factor of two of the crossover and nowhere else.

Regret matters more than the count. Being wrong by 20 us on a near-tie is
not the same as fanning out a range that serial would have finished in a
third of the time, and the count cannot tell them apart. Near-ties also
flip between runs, so treat a change of one or two as noise.

What good looks like, on the machine this was written on (Apple silicon,
8 performance cores): 1-4 wrong of 18, under ~250 us of regret, and the
mistakes *under*-eager — serial chosen where parallel would have won by
less than the 2x margin the backend demands. Over-eager mistakes are the
ones worth chasing: those are fan-outs that lost to a serial run.

**A perfect score is a failure mode, not the goal.** It usually means the
grid has walked off the crossovers and every row is a decision that was
never in doubt — this happened on mustafar-linux with `--scale 8`, read
as a passing score for a day. The MODEL section below fits approximate
crossovers, while the coverage verdict requires measured serial wins below
measured parallel wins. Ties are neutral; reversed or interleaved winners
need remeasurement. Read that verdict before reading the score.

The MODEL section exists to decompose a wrong decision instead of
attributing it. The backend's threshold is

    parallel_min_elems = fan_out_ns * BREAK_EVEN / ns_per_elem

so two measured inputs feed it, and a wrong threshold can come from
either — or from the formula itself, which assumes parallel time is
`overhead + T_serial/P` with P the thread count. Fitting `a + b*n`
through both the serial and parallel columns gives all of it directly:
the true fan-out cost (a), the true per-element rates, the effective
parallelism P_eff = s/b (which is *not* the thread count on a
bandwidth-bound machine), and the real crossover. The report then splits
the threshold error into a MEASUREMENT part (the backend's inputs vs the
fitted ones) and a MODEL part (the formula fed correct inputs vs the
real crossover).

Use `--json out.json` to collect runs from several machines and compare
them; every derived quantity is in there, keyed by kernel.
"""

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
import time

import numpy as np

import tack
from tack.runtime.cpu import _PARALLEL_BREAK_EVEN, _physical_core_count
from tack.runtime.dispatch import get_backend


@tack.kernel
def cheap(x, out, n):
    for i in range(n):
        out[i] = x[i] * 2.0 + 1.0


@tack.kernel
def medium(x, out, n):
    for i in range(n):
        out[i] = tack.sqrt(x[i] * x[i] + 1.0) + tack.sin(x[i])


@tack.kernel
def heavy(x, out, n):
    for i in range(n):
        v = x[i]
        for _ in range(20):
            v = tack.sqrt(v * v + 1.0)
        out[i] = v


# Bracketing each kernel's crossover. If a machine's threads are much
# cheaper or dearer than this one's, the crossovers move and these want
# re-centring — the fan-out cost the backend reports is the clue.
GRIDS = {
    "cheap": (786432, 1048576, 1572864, 2097152, 3145728, 4194304),
    "medium": (49152, 65536, 98304, 131072, 196608, 262144),
    "heavy": (12288, 16384, 24576, 32768, 49152, 65536),
}
KERNELS = {"cheap": cheap, "medium": medium, "heavy": heavy}


def _estimator_state(obj):
    """The plain-data attributes an estimator decides from, copied."""
    return {k: (list(v) if isinstance(v, list) else v)
            for k, v in vars(obj).items()
            if isinstance(v, (bool, int, float, list, type(None)))}


def _restore(obj, state):
    for k, v in state.items():
        setattr(obj, k, list(v) if isinstance(v, list) else v)


def best_of(fn, reps):
    fn()
    fn()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        fn()
        times.append(time.perf_counter_ns() - t0)
    return min(times)


# How far above the crossover to fit the slopes. Near the crossover the
# parallel column is mostly fan-out floor, so a slope fitted there fits
# noise -- on mustafar-linux that produced P_eff = 95.8 on 16 threads,
# and a "fan-out" that varied 3x across kernels when it is a property of
# the machine. Up here the work dominates and the slope is the rate.
MODEL_SCALE = 8

# Idle gaps to sweep the fan-out cost across, in ms. The backend probes
# back-to-back (the 0 row); a real dispatch usually meets workers that
# have been idle, which is the rest of the table.
FLOOR_GAPS_MS = (0, 1, 10, 50)
FLOOR_REPS = 9


def fan_out_floor(backend, compiled, prefix):
    """What a fan-out costs after the workers have been idle a while.

    Mirrors `_calibrate_fan_out` exactly -- num_threads empty ranges
    through the real pool -- so these are comparable to the probe's own
    number. The differences are the idle gap before each rep, and
    reporting the spread instead of the minimum.

    An empty range runs no loop iterations, so this measures fan-out and
    nothing else, and is safe on any kernel. It is also why the floor
    cannot depend on which kernel is used to measure it.
    """
    pool = backend._get_pool()
    run = compiled.call_range
    out = {}
    for gap in FLOOR_GAPS_MS:
        times = []
        for _ in range(FLOOR_REPS):
            if gap:
                time.sleep(gap / 1000.0)
            t0 = time.perf_counter_ns()
            futures = [pool.submit(run, prefix, 0, 0)
                       for _ in range(backend.num_threads)]
            for f in futures:
                f.result()
            times.append(time.perf_counter_ns() - t0)
        times.sort()
        out[gap] = {"min": times[0], "median": times[len(times) // 2],
                    "max": times[-1]}
    return out


def measure_model_grid(backend, x, out_field, scale):
    """Time both paths well above the crossover, where the slopes are real."""
    result = {}
    for name, kernel in KERNELS.items():
        grid = GRIDS[name]
        pts = []
        for base in (grid[0], grid[len(grid) // 2], grid[-1]):
            n = int(base * scale * MODEL_SCALE)
            kernel(x, out_field, n)          # ensure a compiled variant
            compiled = variant_for(backend, name)
            prefix = compiled.bind([x, out_field, n])
            serial = best_of(lambda: compiled.call_range(prefix, 0, n), 4)
            parallel = best_of(
                lambda: backend._parallel_execute(compiled, prefix, 0, n), 4)
            pts.append({"n": n, "serial_ns": serial, "parallel_ns": parallel})
        result[name] = pts
    return result


def fit_line(xs, ys):
    """Least-squares `a + b*x`, returned as (intercept, slope)."""
    slope, intercept = np.polyfit(np.asarray(xs, dtype=float),
                                  np.asarray(ys, dtype=float), 1)
    return float(intercept), float(slope)


def machine_id(backend):
    """What a reader needs to tell two machines' runs apart."""
    return {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "logical_cores": os.cpu_count(),
        "physical_cores": _physical_core_count(),
        "num_threads": backend.num_threads,
    }


def _measured_bracket(points):
    """Describe an ordered split between directly measured winning paths."""
    serial = sorted(p['n'] for p in points if p['serial_ns'] < p['parallel_ns'])
    parallel = sorted(p['n'] for p in points if p['parallel_ns'] < p['serial_ns'])
    ties = sorted(p['n'] for p in points if p['parallel_ns'] == p['serial_ns'])
    lower = upper = None
    if not points:
        verdict = 'no measured points'
    elif not serial and not parallel:
        verdict = 'all measured points tie'
    elif not parallel:
        verdict = 'no measured parallel wins'
    elif not serial:
        verdict = 'no measured serial wins'
    elif max(serial) >= min(parallel):
        verdict = 'non-monotonic measured winners'
    else:
        verdict = 'brackets it'
        lower, upper = max(serial), min(parallel)
    return {
        'brackets': verdict, 'bracket_basis': 'measured timings',
        'serial_win_sizes': serial, 'parallel_win_sizes': parallel, 'tie_sizes': ties,
        'bracket_lo': lower, 'bracket_hi': upper,
    }


def model_report(rows, model_pts, floor, backend, scale):
    """Decompose the threshold error, from a measured floor and real slopes.

    Returns the per-kernel dicts so `--json` can carry the same numbers
    the table shows.
    """
    print("\n--- fan-out cost vs idle gap ---")
    print(f"{'gap before':>11s} {'min':>9s} {'median':>9s} {'max':>9s}")
    for gap, s in floor.items():
        label = "back-to-back" if gap == 0 else f"{gap} ms"
        print(f"{label:>11s} {s['min']/1000:8.1f}us {s['median']/1000:8.1f}us "
              f"{s['max']/1000:8.1f}us")
    probe = backend._fan_out_ns
    realistic = floor[10]["median"] if 10 in floor else None
    hot = floor[0]["median"]
    print(f"\nthe probe reads {probe/1000:.1f}us (min of back-to-back). A "
          f"dispatch meeting workers\nidle for 10 ms pays "
          f"{realistic/1000:.1f}us -- "
          f"{realistic/probe:.2f}x what the threshold assumes.")
    print("there is no single right answer here: a loop dispatching "
          "back-to-back really does\nmeet the hot floor. Both crossovers "
          "are reported below.")

    fits = {}
    for name in KERNELS:
        pts = [r for r in rows if r["kernel"] == name]
        if len(pts) < 3:
            continue
        ns = [p["n"] for p in pts]
        serial_c, serial_s = fit_line(ns, [p["serial_ns"] for p in pts])

        # Slopes from well above the crossover; floor measured, not fitted.
        mp = model_pts.get(name, [])
        m_ns = [p["n"] for p in mp]
        _, serial_s_hi = fit_line(m_ns, [p["serial_ns"] for p in mp])
        _, par_b = fit_line(m_ns, [p["parallel_ns"] for p in mp])
        par_a = float(realistic)

        # P_eff is the honest parallel speedup of the *work*, which the
        # backend's formula assumes is the thread count. Both rates come
        # from the same regime, so cache effects do not bias the ratio.
        p_eff = serial_s_hi / par_b if par_b > 0 else float("inf")

        # Two honest crossovers, because there are two honest floors.
        # A loop dispatching back-to-back meets the hot floor; a script
        # dispatching now and then meets the idle one. The scoring table
        # above is measured back-to-back (best_of repeats immediately),
        # so its own decisions belong to the hot regime -- scoring them
        # against the idle crossover would mix conditions.
        def _cross(floor_ns):
            return ((floor_ns - serial_c) / (serial_s - par_b)
                    if serial_s > par_b else None)

        crossover = _cross(par_a)
        crossover_hot = _cross(float(hot))

        # The backend's own inputs, read at the grid point nearest the
        # crossover -- that is where the decision is in doubt, so it is
        # the estimate that mattered.
        anchor = (min(pts, key=lambda p: abs(p["n"] - crossover))
                  if crossover else pts[-1])

        # The same formula the backend uses, fed the fitted inputs.
        ideal = par_a * _PARALLEL_BREAK_EVEN / serial_s if serial_s > 0 else None

        # Locate the fitted hot crossover, but keep it separate from the
        # directly measured bracket. A fit inside the grid does not imply
        # that any sampled size was actually faster in parallel.
        lo, hi = min(ns), max(ns)
        c = crossover_hot
        if c is None:
            fitted_location = "no fitted crossover"
        elif c < lo:
            fitted_location = f"BELOW grid ({c/lo:.2f}x under {lo})"
        elif c > hi:
            fitted_location = f"ABOVE grid ({c/hi:.2f}x over {hi})"
        else:
            fitted_location = "inside grid"

        fits[name] = {
            "serial_fixed_ns": serial_c, "serial_ns_per_elem": serial_s,
            "serial_ns_per_elem_hi": serial_s_hi,
            "parallel_fan_out_ns": par_a, "parallel_ns_per_elem": par_b,
            "p_eff": p_eff, "crossover": crossover,
            "crossover_hot": crossover_hot, "fan_out_hot_ns": float(hot),
            "backend_ns_per_elem": anchor["ns_per_elem"],
            "backend_threshold": anchor["parallel_min_elems"],
            "backend_fan_out_ns": backend._fan_out_ns,
            "ideal_threshold": ideal,
            "grid_lo": lo, "grid_hi": hi,
            "fitted_crossover_location": fitted_location,
            **_measured_bracket(pts),
        }

    if not fits:
        return fits

    print("\n--- model (floor measured; slopes fitted above the crossover) ---")
    print(f"{'kernel':7s} {'serial ns/el':>13s} {'par ns/el':>10s} "
          f"{'P_eff':>7s} {'cross (hot)':>12s} {'cross (idle)':>13s}")
    for name, f in fits.items():
        def _n(v):
            return f"{v:.0f}" if v else "none"
        print(f"{name:7s} {f['serial_ns_per_elem']:13.2f} "
              f"{f['parallel_ns_per_elem']:10.2f} {f['p_eff']:7.1f} "
              f"{_n(f['crossover_hot']):>12s} {_n(f['crossover']):>13s}")

    print(f"\nthe backend assumes P_eff is the thread count "
          f"({backend.num_threads}); the column above is what it is.")
    for name, f in fits.items():
        lo, hi = f["serial_ns_per_elem"], f["serial_ns_per_elem_hi"]
        if hi > 0 and not 0.7 <= lo / hi <= 1.4:
            print(f"  note: {name}'s serial rate differs between regimes "
                  f"({lo:.2f} near the crossover, {hi:.2f} above it) -- "
                  f"P_eff\n        is the ratio up top, so read the "
                  f"crossover as approximate.")

    print("\n--- where the threshold error comes from ---")
    print(f"{'kernel':7s} {'fan-out':>18s} {'ns/elem':>18s} "
          f"{'threshold':>11s} {'measure':>8s} {'model':>7s} "
          f"{'hot':>6s} {'idle':>6s}")
    for name, f in fits.items():
        thr, ideal = f["backend_threshold"], f["ideal_threshold"]
        meas = thr / ideal if ideal else float("nan")
        model = (ideal / f["crossover"]
                 if (ideal and f["crossover"]) else float("nan"))
        t_hot = thr / f["crossover_hot"] if f["crossover_hot"] else float("nan")
        t_idle = thr / f["crossover"] if f["crossover"] else float("nan")
        print(f"{name:7s} "
              f"{f['backend_fan_out_ns']/1000:7.0f}/{f['parallel_fan_out_ns']/1000:<6.0f}us "
              f"{f['backend_ns_per_elem']:8.2f}/{f['serial_ns_per_elem']:<8.2f} "
              f"{thr:11d} {meas:7.2f}x {model:6.2f}x {t_hot:5.2f}x {t_idle:5.2f}x")
    print("read each pair as backend/fitted. measure = the backend's inputs "
          "vs the fitted\nones through the same formula; model = that formula "
          "with correct inputs vs the\nidle crossover. hot and idle are the "
          "threshold over each real crossover.\n1.00x is right; below 1.00x "
          "fans out too early.")

    print("\n--- does the measured grid bracket the crossovers? ---")
    for name, f in fits.items():
        mark = "ok " if f["brackets"] == "brackets it" else "XX "
        counts = '/'.join(str(len(f[key])) for key in
                          ('serial_win_sizes', 'parallel_win_sizes', 'tie_sizes'))
        print(f"{mark}{name:7s} grid {f['grid_lo']}-{f['grid_hi']} "
              f"(scale {scale:g}): {f['brackets']} (serial/parallel/ties: {counts})")
        if f['bracket_lo'] is not None:
            print(f"    measured bracket: {f['bracket_lo']}-{f['bracket_hi']}")
        fitted = f"{f['crossover_hot']:.0f}" if f['crossover_hot'] is not None else 'none'
        print(f"    fitted hot crossing: {fitted} — {f['fitted_crossover_location']}")
    if any(f["brackets"] != "brackets it" for f in fits.values()):
        print("\nSome kernels have incomplete or ambiguous measured crossover "
              "coverage.\nRepeat ambiguous timings, or re-centre with --scale "
              "until serial wins lie\nbelow parallel wins. Treat those decision "
              "scores as incomplete crossover\ncoverage; a fitted crossing "
              "inside the grid does not establish a measured bracket.")
    return fits


def variant_for(backend, name):
    for slot in list(backend._cache.values()):
        for variant in slot.values():
            if variant.ir.name.startswith(name):
                return variant.payload
    raise LookupError(f"no compiled variant for {name}")


class BackgroundLoad:
    """N CPU-bound subprocesses, for scoring the policy on a busy machine.

    Tack does not run on idle machines. It runs on workstations with a
    browser open, on shared cluster nodes, and -- as this file learned the
    hard way -- on a laptop in a video call. Treating that as contamination
    to be waited out means the policy is never scored under the conditions
    it actually meets, so load belongs here as an *independent variable*.

    Subprocesses, not threads. Python threads spinning hold the GIL, so the
    measuring thread cannot run at all and every reading comes back tens of
    milliseconds long -- that measures GIL starvation, not CPU contention.
    Cost one wrong result before it was noticed.

    Killed rather than terminated, and in a finally, because a benchmark
    that leaves spinners behind poisons every run after it on that machine.
    """

    SPIN = "\nwhile True:\n    sum(i * i for i in range(10000))\n"

    def __init__(self, procs: int):
        self.procs = procs
        self._children: list = []

    def __enter__(self):
        for _ in range(self.procs):
            self._children.append(
                subprocess.Popen([sys.executable, "-c", self.SPIN],
                                 stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL))
        if self._children:
            time.sleep(1.0)          # let the scheduler settle on them
        return self

    def __exit__(self, *exc):
        for p in self._children:
            p.kill()
        for p in self._children:
            p.wait()
        if self._children:
            time.sleep(1.0)
        return False


def contention_index(backend, compiled, prefix, reps=21):
    """How much dearer a fan-out is right now than at its own best.

    Measured on an *empty* fan-out, which is deliberate twice over. It
    isolates thread scheduling from the work, and it is the quantity load
    actually attacks: waking `num_threads` workers needs that many cores
    to be free, where a serial run needs one.

    The first version of this timed a serial dispatch instead and read
    1.00x on a machine at load 6 -- correctly, because a 50 us
    single-threaded run is rarely preempted at all. It measured something
    real and useless.

    Reported instead of a load average because on macOS the load average
    counts threads this has no reason to care about: yavin read 3.7 while
    the CPU was 85% idle, and 34 while it was 70% idle.
    """
    pool = backend._get_pool()
    run = compiled.call_range
    times = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        futures = [pool.submit(run, prefix, 0, 0)
                   for _ in range(backend.num_threads)]
        for f in futures:
            f.result()
        times.append(time.perf_counter_ns() - t0)
    times.sort()
    lo = times[0]
    return {
        "min_ns": lo,
        "median_ns": times[len(times) // 2],
        "p90_ns": times[int(len(times) * 0.9)],
        "index": times[len(times) // 2] / lo if lo else float("nan"),
        "tail": times[int(len(times) * 0.9)] / lo if lo else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", type=float, default=1.0,
                        help="multiply every range size, for a machine whose "
                             "crossovers sit elsewhere")
    parser.add_argument("--json", metavar="PATH",
                        help="also write every measured and derived number "
                             "here, for comparing machines")
    parser.add_argument("--load", type=int, default=0, metavar="N",
                        help="run with N CPU-bound background processes, so "
                             "the score describes a busy machine rather than "
                             "an idle one. Try 0, then a quarter and a half "
                             "of the core count.")
    args = parser.parse_args()

    tack.init(arch=tack.cpu)
    backend = get_backend()
    machine = machine_id(backend)
    print(f"host       : {machine['host']}  ({machine['machine']}, "
          f"{machine['physical_cores']}p/{machine['logical_cores']}l cores)")
    print(f"platform   : {machine['platform']}  py{machine['python']}")

    # MODEL_SCALE, because the slope fit runs well above the scoring grid.
    biggest = int(max(max(g) for g in GRIDS.values()) * args.scale * MODEL_SCALE)
    x = tack.field(dtype=tack.f32, shape=(biggest,))
    x.from_numpy(np.ones(biggest, dtype=np.float32))
    out = tack.field(dtype=tack.f32, shape=(biggest,))

    print(f"threads    : {backend.num_threads}")
    if args.load:
        print(f"load       : {args.load} background CPU processes")

    with BackgroundLoad(args.load):
        conditions_before = _conditions(backend, x, out)
        print(f"fan-out now: {conditions_before['median_ns']/1000:.0f}us "
              f"median, {conditions_before['tail']:.1f}x p90/min spread")
        result = _score(args, backend, x, out, biggest)
        result["conditions_before"] = conditions_before
        result["conditions_after"] = _conditions(backend, x, out)

    _report(args, backend, machine, result)


def _conditions(backend, x, out):
    """Measure how contended this machine is, using the real pool."""
    n = GRIDS["cheap"][0]
    KERNELS["cheap"](x, out, n)
    compiled = variant_for(backend, "cheap")
    return contention_index(backend, compiled, compiled.bind([x, out, n]))


def _score(args, backend, x, out, biggest):
    """The scoring sweep. Everything timed happens inside the load."""
    print(f"{'kernel':7s} {'n':>9s} {'serial':>10s} {'parallel':>10s} "
          f"{'faster':>9s} {'chose':>9s}   regret")

    wrong = over_eager = 0
    regret_total = 0.0
    rows = []

    for name, kernel in KERNELS.items():
        for base in GRIDS[name]:
            n = int(base * args.scale)
            if n < 1 or n > biggest:
                continue
            for _ in range(30):            # let the estimate settle
                kernel(x, out, n)

            compiled = variant_for(backend, name)
            prefix = compiled.bind([x, out, n])
            reps = 30 if n <= 262144 else 8

            # The decision is the one the backend reached on its own
            # dispatches, so it is read here -- before timing. Timing the
            # parallel path goes through `_parallel_execute`, which updates
            # r_p, the fan-out curve and the margin's cv: read afterwards,
            # the score described a state the measurement had made, and the
            # carried-over r_p hid P9 from this harness entirely. The state
            # is restored after timing for the same reason, so no grid
            # point inherits another's measurement fan-outs.
            chose = ("parallel" if n >= compiled.parallel_min_elems
                     else "serial")
            inputs = {
                # the backend's own inputs, as they stood for this call
                "ns_per_elem": compiled.ns_per_elem,
                "parallel_min_elems": compiled.parallel_min_elems,
                # The rest of the threshold's inputs, so a wrong one can be
                # factored into fan-out, margin and rate error rather than
                # attributed to whichever is easiest to name.
                "ns_per_elem_parallel": compiled.ns_per_elem_parallel,
                "margin": backend._margin(),
                "fan_out_estimate_ns": backend._fan_out_estimate(),
            }
            saved = (_estimator_state(compiled), _estimator_state(backend))

            serial = best_of(lambda: compiled.call_range(prefix, 0, n), reps)
            parallel = best_of(
                lambda: backend._parallel_execute(compiled, prefix, 0, n), reps)
            _restore(compiled, saved[0])
            _restore(backend, saved[1])
            # Except what describes the data rather than an estimate: the
            # last timing run was a fan-out, so the range really is spread
            # across the workers' caches now. Restoring "laid out serially"
            # let the next grid point take a scattered sample as clean.
            compiled.scattered = True

            faster = "parallel" if parallel < serial else "serial"
            got = parallel if chose == "parallel" else serial
            regret = (got - min(serial, parallel)) / 1000
            regret_total += regret
            if faster != chose:
                wrong += 1
                over_eager += chose == "parallel"
            rows.append({
                "kernel": name, "n": n,
                "serial_ns": serial, "parallel_ns": parallel,
                **inputs,
                "faster": faster, "chose": chose, "regret_us": regret,
            })
            flag = "" if faster == chose else (
                "   <-- fanned out and lost" if chose == "parallel"
                else "   <-- missed a win")
            print(f"{name:7s} {n:9d} {serial/1000:9.1f}us {parallel/1000:9.1f}us "
                  f"{faster:>9s} {chose:>9s}  {regret:7.1f}us{flag}")

    total = sum(len(g) for g in GRIDS.values())
    print(f"\nwrong: {wrong}/{total}   of which fanned out and lost: "
          f"{over_eager}   total regret: {regret_total:.0f} us")
    print(f"fan-out cost measured here: "
          f"{backend._fan_out_ns and round(backend._fan_out_ns / 1000, 1)} us")
    if over_eager:
        print("\nFan-outs that lost to a serial run are the ones to chase: "
              "they mean the cost estimate reads high, or the fan-out here "
              "costs more than the threshold assumes.")

    # Both after scoring, so neither perturbs it -- and both inside the
    # load, so the floor and the slopes describe the same machine the
    # decisions were scored on.
    anchor = variant_for(backend, "cheap")
    floor = fan_out_floor(backend, anchor,
                          anchor.bind([x, out, GRIDS["cheap"][0]]))
    model_pts = measure_model_grid(backend, x, out, args.scale)
    return {
        "rows": rows, "floor": floor, "model_pts": model_pts,
        "score": {"wrong": wrong, "over_eager": over_eager,
                  "regret_us": regret_total, "total": total},
    }


def _report(args, backend, machine, result):
    """Everything that only reads what was measured, after the load stops."""
    rows, floor = result["rows"], result["floor"]
    s = result["score"]
    fits = model_report(rows, result["model_pts"], floor, backend, args.scale)

    before, after = result["conditions_before"], result["conditions_after"]
    lo, hi = before["median_ns"] / 1000, after["median_ns"] / 1000
    print("\n--- conditions this score was measured under ---")
    print(f"empty fan-out, median: {lo:.0f}us before the sweep, "
          f"{hi:.0f}us after   (p90/min spread {before['tail']:.1f}x -> "
          f"{after['tail']:.1f}x)")
    print("compare that median between runs; it is the cost the policy pays "
          "and the\nthing background load moves. The ratio is jitter, and is "
          "itself noisy --\ntreat it as a hint, not a measurement.")

    drift = abs(hi - lo) / max(lo, 1e-9)
    if drift > 0.4:
        print(f"\n  The machine changed under the run ({drift*100:.0f}% "
              f"drift in the fan-out median),\n  so this score describes a "
              f"moving target. Re-run, or pin conditions with --load.")
    if not args.load and max(before["tail"], after["tail"]) > 3.0:
        print("\n  Large jitter with --load 0. If that was not intended, the "
              "machine had\n  company: the numbers describe conditions nobody "
              "chose. Either quieten it\n  or set --load and make the load a "
              "variable rather than a surprise.")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({
                "machine": machine, "scale": args.scale,
                "load_procs": args.load,
                "conditions_before": before, "conditions_after": after,
                "break_even": _PARALLEL_BREAK_EVEN,
                "policy": getattr(backend, "policy", "v1"),
                "margin": getattr(backend, "margin", _PARALLEL_BREAK_EVEN),
                "fan_out_probe_ns": backend._fan_out_ns,
                "fan_out_floor": {str(k): v for k, v in floor.items()},
                "model_points": result["model_pts"],
                "score": s,
                "rows": rows, "fits": fits,
            }, fh, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
