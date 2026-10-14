"""Does a thread count per launch pick the right one?

`TACK_CPU_SUBSETS=1` lets the CPU backend fan a launch out to the first k
of its threads instead of all N (`CPUBackend._threads_for`: a size rule
and a saturation rule). This scores that choice the way
`threading_decisions.py` scores the serial-or-parallel one: by measuring
every option directly and asking what the backend chose.

    TACK_CPU_SUBSETS=1 uv run python benchmarks/thread_subsets.py
    TACK_CPU_THREADS=16 TACK_CPU_SUBSETS=1 uv run python benchmarks/thread_subsets.py

For three kernels -- bandwidth-bound (`out = 2x + 1`), mixed (`sqrt`,
`sin`) and compute-bound (a 24-term inner loop) -- over sizes from the
fan-out threshold up, the fan-out is timed through the pool at every k in
2..N and serially (k = 1), hot, median of repeats; the best k and its time
are the reference. Then the kernel is dispatched through the real path a
dozen times, so the policy learns its rates, and the k the next dispatch
uses is read off the pool. Its regret is the sweep's time at that k minus
the best; the always-N policy's regret is reported beside it, so the
subset's gain (or loss) is visible. Times are from the sweep, not from the
dispatch itself, so the score is decision quality.

What good looks like: regret near zero on every row, and on the
bandwidth-bound kernel's large sizes a chosen k well under N with no
loss. A chosen k whose time is worse than N's is the mistake worth
chasing.
"""
import argparse
import json
import os
import statistics as st
import sys
import time
from math import cos, sin, sqrt

import numpy as np

import tack
from tack.runtime import cpu as cpu_mod
from tack.runtime import cpu_pool
from tack.runtime.kernel_utils import resolve_variant


@tack.kernel
def scale(x, out, n):
    for i in range(n):
        out[i] = x[i] * 2.0 + 1.0


@tack.kernel
def mixed(x, out, n):
    for i in range(n):
        v = x[i]
        out[i] = sqrt(v + 1.0) * sin(v) + cos(v) * 0.5


@tack.kernel
def heavy(x, out, n):
    for i in range(n):
        v = x[i]
        acc = 0.0
        for j in range(24):
            acc = acc + sin(v + float(j)) * cos(v - float(j))
        out[i] = acc


KERNELS = [
    ("scale (bandwidth)", scale, [1 << k for k in range(14, 25)]),
    ("mixed (sqrt, sin)", mixed, [1 << k for k in range(12, 23)]),
    ("heavy (24 sin/cos)", heavy, [1 << k for k in range(9, 20)]),
]


def _compiled(backend, kernel, args):
    variant, effective = resolve_variant(backend, kernel, tuple(args), {},
                                         build=backend._build_variant,
                                         specialize_disjoint=cpu_mod._SPECIALIZE_DISJOINT)
    _, extents = cpu_mod._get_launch(variant.ir, list(effective))
    return variant.payload, variant.payload.bind(list(effective), extents)


def _timed(fn, budget_s, min_reps=5):
    """Median seconds of `fn()` over repeats within `budget_s`."""
    xs = []
    t_end = time.perf_counter() + budget_s
    while len(xs) < min_reps or time.perf_counter() < t_end:
        t0 = time.perf_counter_ns()
        fn()
        xs.append(time.perf_counter_ns() - t0)
        if len(xs) >= 200:
            break
    return st.median(xs) / 1e9


def sweep(backend, jit, compiled, prefix, n, budget_s):
    """Seconds at k = 1 (serial) and at every k in 2..N through the pool."""
    wrapper = cpu_pool.chunk_wrapper(compiled)
    pack = cpu_pool.pack_args(compiled, prefix)
    times = {1: _timed(lambda: compiled.call_range(prefix, 0, n), budget_s)}
    for k in range(2, backend.num_threads + 1):
        times[k] = _timed(lambda: jit.run(wrapper, pack, 0, n, k), budget_s)
    return times


def chosen_k(backend, kernel, args, n, warm=12):
    """The k the backend uses for this launch after `warm` dispatches: 1
    when it runs serially, else the pool's thread count."""
    for _ in range(warm):
        kernel(*args)
    used = []
    jit = backend._jit_pool()
    real = jit.run

    def spy(wrapper, pack, start, end, threads, measure=False):
        used.append(threads)
        return real(wrapper, pack, start, end, threads, measure=measure)
    jit.run = spy
    try:
        kernel(*args)
    finally:
        jit.run = real
    return used[-1] if used else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--budget", type=float, default=0.15,
                        help="seconds per (kernel, size, k) measurement")
    parser.add_argument("--json", metavar="PATH", help="write the rows here")
    parser.add_argument("--verbose", action="store_true",
                        help="print the backend's model and the whole sweep per row")
    args = parser.parse_args()
    threads = int(os.environ.get("TACK_CPU_THREADS", 0) or 0) or None
    tack.init(arch="cpu", **({"num_threads": threads} if threads else {}))
    backend = tack.runtime.dispatch.get_backend()
    n_threads = backend.num_threads
    if not backend._subsets:
        print("TACK_CPU_SUBSETS is off: the backend will always choose N", file=sys.stderr)
    jit = backend._jit_pool()
    if jit is None:
        sys.exit("needs the JIT'd pool (TACK_CPU_POOL unset) and more than one thread")
    print(f"# {n_threads} threads, pool of {jit.threads}, pinned {jit.pinned}, "
          f"subsets {'on' if backend._subsets else 'off'}, load {os.getloadavg()[0]:.2f}")
    rows = []
    print(f"{'kernel':<20}{'n':>10}{'best k':>7}{'t best':>9}{'chose':>6}{'t chosen':>10}"
          f"{'regret':>9}{'t at N':>9}{'gain vs N':>10}")
    for name, kernel, sizes in KERNELS:
        for n in sizes:
            x = tack.zeros(tack.f32, (n,))
            out = tack.zeros(tack.f32, (n,))
            x.from_numpy(np.linspace(0, 3, n, dtype=np.float32))
            compiled, prefix = _compiled(backend, kernel, [x, out, n])
            times = sweep(backend, jit, compiled, prefix, n, args.budget)
            best = min(times, key=times.get)
            k = chosen_k(backend, kernel, [x, out, n], n)
            regret = times[k] - times[best]
            gain = times[n_threads] - times[k]
            if args.verbose:
                r_s, r_p = compiled.ns_per_elem, compiled.ns_per_elem_parallel
                print(f"    model: r_s {r_s:.3f} r_p {r_p:.3f} S {r_s / r_p if r_p else 0:.1f} "
                      f"hot fan-out {(backend._fan_out_ns or 0) / 1e3:.1f} us, estimate "
                      f"{backend._fan_out_estimate() / 1e3:.1f} us, min elems {compiled.parallel_min_elems}, "
                      f"subset runs {compiled.subset_runs}/{compiled.subset_recheck_after} "
                      f"failed {compiled.subset_failed}; sweep us "
                      + " ".join(f"{j}:{t * 1e6:.0f}" for j, t in sorted(times.items())))
            rows.append({"kernel": name, "n": n, "times_us": {str(j): t * 1e6 for j, t in times.items()},
                         "best_k": best, "chosen_k": k, "regret_us": regret * 1e6,
                         "gain_vs_full_us": gain * 1e6})
            print(f"{name:<20}{n:>10}{best:>7}{times[best]*1e6:>9.1f}{k:>6}{times[k]*1e6:>10.1f}"
                  f"{regret*1e6:>9.1f}{times[n_threads]*1e6:>9.1f}{gain*1e6:>10.1f}")
    fanned = [r for r in rows if r["chosen_k"] > 1]
    print(f"\n{len(rows)} rows, {len(fanned)} fanned out: regret {sum(r['regret_us'] for r in fanned):.0f} us "
          f"({sum(r['regret_us'] > 0.05 * r['times_us'][str(r['best_k'])] for r in fanned)} rows over 5% of best), "
          f"gain over always-N {sum(r['gain_vs_full_us'] for r in fanned):.0f} us "
          f"({sum(r['gain_vs_full_us'] < -0.05 * r['times_us'][str(n_threads)] for r in fanned)} rows lose over 5% to N)")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"threads": n_threads, "subsets": backend._subsets, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
