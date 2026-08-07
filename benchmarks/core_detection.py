"""What this machine reports about its cores, and what Tack makes of it.

Tack wants one compute thread per *physical* core. A hyperthread shares an
execution unit, so a second thread on one buys little for compute-bound
work while costing a full fan-out slot — and a fan-out slot is ~10 us of
Python.

    uv run python benchmarks/core_detection.py

Each platform is asked in its own way and every probe returns None off its
own platform, so exactly one should answer. If none does, the fallback is
os.cpu_count(), which counts logical processors — on an SMT machine that is
twice what Tack wants.

Known gaps, both wanting a machine to check them on:

  * The Windows probe has never run. It calls
    GetLogicalProcessorInformationEx through ctypes and returns None on any
    failure, so the worst case is the logical-processor fallback rather
    than a crash — but "returns something plausible" is unverified.

  * On Linux the answer comes from sysfs thread_siblings_list. On a
    hyperthreaded x86 box the count should be half of os.cpu_count(); if it
    equals it, either SMT is off or the probe silently failed.
"""

import os
import platform

from tack.runtime.cpu import (
    _linux_core_count,
    _macos_core_count,
    _physical_core_count,
    _windows_core_count,
)


def main():
    print(f"platform          : {platform.system()} {platform.machine()}")
    print(f"os.cpu_count()    : {os.cpu_count()}   (logical processors)")
    print()

    probes = (("linux  ", _linux_core_count),
              ("macos  ", _macos_core_count),
              ("windows", _windows_core_count))
    answered = []
    for label, probe in probes:
        try:
            result = probe()
        except Exception as exc:                        # noqa: BLE001
            result = f"raised {type(exc).__name__}: {exc}"
        print(f"  {label} probe   : {result}")
        if isinstance(result, int):
            answered.append((label.strip(), result))

    chosen = _physical_core_count()
    print(f"\nTack will use     : {chosen} threads")

    logical = os.cpu_count() or 1
    print()
    if not answered:
        print("NO PROBE ANSWERED. Falling back to the logical count, so on an "
              "SMT machine Tack is using twice the threads it should. This is "
              "the case worth reporting.")
    elif len(answered) > 1:
        print(f"MORE THAN ONE PROBE ANSWERED ({answered}) — they are meant to "
              f"be mutually exclusive.")
    elif chosen > logical:
        print(f"Reported {chosen} cores, more than the {logical} logical "
              f"processors that exist. That cannot be right.")
    elif chosen == logical and logical > 1:
        print("Physical equals logical. Correct on a machine without SMT "
              "(Apple silicon, some server parts); on an x86 box with "
              "hyperthreading enabled it means the probe did not work.")
    elif answered[0][0] == "macos":
        print(f"{chosen} of {logical}. Apple silicon has no SMT, so this is "
              f"performance cores counted and efficiency cores left out: the "
              f"fan-out splits a range into equal chunks and waits for the "
              f"slowest, and an efficiency core sets that pace. On an Intel "
              f"Mac the same gap would instead be hyperthreads.")
    else:
        print(f"{chosen} physical of {logical} logical — as expected where "
              f"SMT is enabled.")

    print("\nTACK_CPU_THREADS overrides this if the answer is wrong; 1 keeps "
          "everything on the calling thread.")


if __name__ == "__main__":
    main()
