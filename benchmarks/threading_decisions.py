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
"""

import argparse
import time

import numpy as np

import tack
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
    args = parser.parse_args()

    tack.init(arch=tack.cpu)
    backend = get_backend()

    biggest = int(max(max(g) for g in GRIDS.values()) * args.scale)
    x = tack.field(dtype=tack.f32, shape=(biggest,))
    x.from_numpy(np.ones(biggest, dtype=np.float32))
    out = tack.field(dtype=tack.f32, shape=(biggest,))

    print(f"threads    : {backend.num_threads}")
    print(f"{'kernel':7s} {'n':>9s} {'serial':>10s} {'parallel':>10s} "
          f"{'faster':>9s} {'chose':>9s}   regret")

    wrong = over_eager = 0
    regret_total = 0.0

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


if __name__ == "__main__":
    main()
