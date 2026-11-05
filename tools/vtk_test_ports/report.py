# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""What ``drive.py``'s ``summary.json`` says, as text: each test's outcome, and per
VTK filter how Tack's output compared, what was not supported and what failed.

    python report.py <summary.json>"""

import collections
import json
import sys


def outcome(vtk, tack):
    used = [r for r in tack["records"] if r.get("used") == "tack"]
    if not vtk["passed"]:
        return "VTK fails here"
    if not tack["records"]:
        return "no stand-in ran" if tack["passed"] else "fails before a stand-in ran"
    if not used:
        return "VTK's output used throughout"
    if tack["passed"]:
        return "Tack used, test passes"
    return "Tack used, test fails"


def main():
    results = json.load(open(sys.argv[1]))
    by_test = collections.defaultdict(dict)
    for r in results:
        by_test[r["test"]][r["mode"]] = r
    outcomes = collections.defaultdict(list)
    for test, modes in sorted(by_test.items()):
        if "vtk" in modes and "tack" in modes:
            outcomes[outcome(modes["vtk"], modes["tack"])].append(test)
    print("== Tests")
    for name, tests in outcomes.items():
        print(f"{len(tests):4}  {name}")
    print()
    print("== Tack used, test fails")
    for test in outcomes["Tack used, test fails"]:
        tack = by_test[test]["tack"]
        verdicts = collections.Counter(r.get("verdict") or r.get("used") for r in tack["records"])
        print(f"  {test}  image {tack['image_error']}  {dict(verdicts)}")
        if not tack["image_error"]:
            print("    " + tack["traceback"][-400:].replace("\n", "\n    "))
    records = [rec for modes in by_test.values() if "tack" in modes for rec in modes["tack"]["records"]]
    print()
    print("== Per VTK filter: verdicts of Tack's output against VTK's")
    per = collections.defaultdict(collections.Counter)
    for rec in records:
        per[rec["filter"]][rec.get("verdict") or ("unsupported" if "why" in rec else "error")] += 1
    for name, counts in sorted(per.items(), key=lambda kv: -sum(kv[1].values())):
        print(f"  {name:28} " + ", ".join(f"{k} {v}" for k, v in counts.most_common()))
    print()
    print("== Errors (Tack raised), by message")
    errors = collections.Counter((rec["filter"], rec["error"][:160]) for rec in records if "error" in rec)
    for (name, message), n in errors.most_common():
        print(f"  {n:3}  {name}: {message}")
    print()
    print("== Unsupported settings")
    why = collections.Counter((rec["filter"], rec["why"]) for rec in records if "why" in rec)
    for (name, message), n in why.most_common():
        print(f"  {n:3}  {name}: {message}")
    print()
    print("== Outputs that differ")
    for rec in records:
        if rec.get("verdict") == "differ":
            print(f"  {rec['filter']} ({rec['test'].split('/')[-1]}): points {rec['points']} "
                  f"cells {rec['cells']}: " + "; ".join(rec["notes"])[:400])
    print()
    print("== Active attributes VTK sets and Tack does not")
    active = collections.Counter((rec["filter"], note.split(" active ")[1].split(" in ")[0])
                                 for rec in records for note in rec.get("active", []))
    for (name, what), n in active.most_common():
        print(f"  {n:3}  {name}: {what}")


if __name__ == "__main__":
    main()
