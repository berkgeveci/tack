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
as a passing score for a day. The MODEL section below now fits the real
crossover and says outright whether the grid still brackets it. Read that
verdict before reading the score.

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


def best_of(fn, reps):
    fn()
    fn()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        fn()
        times.append(time.perf_counter_ns() - t0)
    return min(times)


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


def model_report(rows, backend, scale):
    """Fit both paths, then decompose the threshold error.

    Returns the per-kernel dicts so `--json` can carry the same numbers
    the table shows.
    """
    fits = {}
    for name in KERNELS:
        pts = [r for r in rows if r["kernel"] == name]
        if len(pts) < 3:
            continue
        ns = [p["n"] for p in pts]
        serial_c, serial_s = fit_line(ns, [p["serial_ns"] for p in pts])
        par_a, par_b = fit_line(ns, [p["parallel_ns"] for p in pts])

        # P_eff is the honest parallel speedup of the *work*, which the
        # backend's formula assumes is the thread count.
        p_eff = serial_s / par_b if par_b > 0 else float("inf")
        # Where the two fitted lines actually meet.
        crossover = ((par_a - serial_c) / (serial_s - par_b)
                     if serial_s > par_b else None)

        # The backend's own inputs, read at the grid point nearest the
        # crossover -- that is where the decision is in doubt, so it is
        # the estimate that mattered.
        anchor = (min(pts, key=lambda p: abs(p["n"] - crossover))
                  if crossover else pts[-1])

        # The same formula the backend uses, fed the fitted inputs.
        ideal = par_a * _PARALLEL_BREAK_EVEN / serial_s if serial_s > 0 else None

        lo, hi = min(ns), max(ns)
        if crossover is None:
            bracket = "no crossover — parallel never wins on this grid"
        elif crossover < lo:
            bracket = f"BELOW grid ({crossover/lo:.2f}x under {lo}) — grid too coarse"
        elif crossover > hi:
            bracket = f"ABOVE grid ({crossover/hi:.2f}x over {hi}) — grid too coarse"
        else:
            bracket = "brackets it"

        fits[name] = {
            "serial_fixed_ns": serial_c, "serial_ns_per_elem": serial_s,
            "parallel_fan_out_ns": par_a, "parallel_ns_per_elem": par_b,
            "p_eff": p_eff, "crossover": crossover,
            "backend_ns_per_elem": anchor["ns_per_elem"],
            "backend_threshold": anchor["parallel_min_elems"],
            "backend_fan_out_ns": backend._fan_out_ns,
            "ideal_threshold": ideal,
            "grid_lo": lo, "grid_hi": hi, "brackets": bracket,
        }

    if not fits:
        return fits

    print("\n--- model ---")
    print(f"{'kernel':7s} {'fan-out fit':>12s} {'serial ns/el':>13s} "
          f"{'par ns/el':>10s} {'P_eff':>7s} {'crossover':>11s}")
    for name, f in fits.items():
        cross = f"{f['crossover']:11.0f}" if f["crossover"] else f"{'none':>11s}"
        print(f"{name:7s} {f['parallel_fan_out_ns']/1000:10.1f}us "
              f"{f['serial_ns_per_elem']:13.2f} {f['parallel_ns_per_elem']:10.2f} "
              f"{f['p_eff']:7.1f} {cross}")

    print(f"\nthe backend assumes P_eff is the thread count "
          f"({backend.num_threads}); the fits above are what it is.")

    print("\n--- where the threshold error comes from ---")
    print(f"{'kernel':7s} {'fan-out':>18s} {'ns/elem':>18s} "
          f"{'threshold':>11s} {'measure':>8s} {'model':>7s} {'total':>7s}")
    for name, f in fits.items():
        fan_err = (f["backend_fan_out_ns"] / f["parallel_fan_out_ns"]
                   if f["parallel_fan_out_ns"] > 0 else float("nan"))
        rate_err = (f["backend_ns_per_elem"] / f["serial_ns_per_elem"]
                    if f["serial_ns_per_elem"] > 0 else float("nan"))
        thr, ideal, cross = (f["backend_threshold"], f["ideal_threshold"],
                             f["crossover"])
        meas = thr / ideal if ideal else float("nan")
        model = ideal / cross if (ideal and cross) else float("nan")
        total = thr / cross if cross else float("nan")
        print(f"{name:7s} "
              f"{f['backend_fan_out_ns']/1000:7.0f}/{f['parallel_fan_out_ns']/1000:<6.0f}us "
              f"{f['backend_ns_per_elem']:8.2f}/{f['serial_ns_per_elem']:<8.2f} "
              f"{thr:11d} {meas:7.2f}x {model:6.2f}x {total:6.2f}x")
    print("read each pair as backend/fitted. measure = the backend's inputs "
          "vs the fitted\nones through the same formula; model = that formula "
          "with correct inputs vs the\nreal crossover. 1.00x is right; below "
          "1.00x fans out too early.")

    print("\n--- does the grid still bracket the crossovers? ---")
    for name, f in fits.items():
        mark = "ok " if f["brackets"] == "brackets it" else "XX "
        print(f"{mark}{name:7s} grid {f['grid_lo']}-{f['grid_hi']} "
              f"(scale {scale:g}): {f['brackets']}")
    if any(f["brackets"] != "brackets it" for f in fits.values()):
        print("\nA grid that does not bracket the crossover cannot score "
              "the decision:\nevery row is a call that was never in doubt. "
              "Re-centre with --scale before\nreading the count above as "
              "anything.")
    return fits


def variant_for(backend, name):
    for slot in list(backend._cache.values()):
        for variant in slot.values():
            if variant.ir.name.startswith(name):
                return variant.payload
    raise LookupError(f"no compiled variant for {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", type=float, default=1.0,
                        help="multiply every range size, for a machine whose "
                             "crossovers sit elsewhere")
    parser.add_argument("--json", metavar="PATH",
                        help="also write every measured and derived number "
                             "here, for comparing machines")
    args = parser.parse_args()

    tack.init(arch=tack.cpu)
    backend = get_backend()
    machine = machine_id(backend)
    print(f"host       : {machine['host']}  ({machine['machine']}, "
          f"{machine['physical_cores']}p/{machine['logical_cores']}l cores)")
    print(f"platform   : {machine['platform']}  py{machine['python']}")

    biggest = int(max(max(g) for g in GRIDS.values()) * args.scale)
    x = tack.field(dtype=tack.f32, shape=(biggest,))
    x.from_numpy(np.ones(biggest, dtype=np.float32))
    out = tack.field(dtype=tack.f32, shape=(biggest,))

    print(f"threads    : {backend.num_threads}")
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

            serial = best_of(lambda: compiled.call_range(prefix, 0, n), reps)
            parallel = best_of(
                lambda: backend._parallel_execute(compiled, prefix, 0, n), reps)

            faster = "parallel" if parallel < serial else "serial"
            chose = ("parallel" if n >= compiled.parallel_min_elems
                     else "serial")
            got = parallel if chose == "parallel" else serial
            regret = (got - min(serial, parallel)) / 1000
            regret_total += regret
            if faster != chose:
                wrong += 1
                over_eager += chose == "parallel"
            rows.append({
                "kernel": name, "n": n,
                "serial_ns": serial, "parallel_ns": parallel,
                # the backend's own inputs, as they stood for this call
                "ns_per_elem": compiled.ns_per_elem,
                "parallel_min_elems": compiled.parallel_min_elems,
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

    fits = model_report(rows, backend, args.scale)

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({
                "machine": machine, "scale": args.scale,
                "break_even": _PARALLEL_BREAK_EVEN,
                "fan_out_probe_ns": backend._fan_out_ns,
                "score": {"wrong": wrong, "over_eager": over_eager,
                          "regret_us": regret_total, "total": total},
                "rows": rows, "fits": fits,
            }, fh, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
