# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Run VTK's Python tests with VTK's filters and with Tack's (``run_one.py``), in
parallel, and gather what each gave into ``summary.json``:

    python drive.py <VTK source> <VTK ExternalData> <output directory> <tests file> [workers]

``tests file`` lists test scripts, one per line, relative to the VTK source."""

import concurrent.futures
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def run(source, data, out, test, mode):
    module = test.split("/Testing/Python/")[0]
    name = os.path.splitext(os.path.basename(test))[0]
    baseline = os.path.join(data, module, "Testing/Data/Baseline", name + ".png")
    where = os.path.join(out, module.replace("/", "_") + "_" + name, mode)
    command = [sys.executable, os.path.join(HERE, "run_one.py"), os.path.join(source, test),
               mode, where, os.path.join(data, "Testing")]
    if os.path.exists(baseline):
        command.append(baseline)
    start = time.time()
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=600,
                              cwd=os.path.dirname(os.path.join(source, test)))
        output, code = done.stdout + done.stderr, done.returncode
    except subprocess.TimeoutExpired as expired:
        output, code = (expired.stdout or b"").decode(errors="replace") if isinstance(
            expired.stdout, bytes) else (expired.stdout or ""), "timeout"
    error = re.findall(r'name="ImageError" type="numeric/double">([^<]+)<', output)
    passed = "RESULT PASSED" in output
    traceback = ""
    if not passed:
        tb = output.rfind("Traceback (most recent call last)")
        traceback = output[tb:tb + 3000] if tb >= 0 else output[-1500:]
    records = []
    path = os.path.join(where, "records.json")
    if mode == "tack" and os.path.exists(path):
        with open(path) as f:
            records = json.load(f)["records"]
    return {"test": test, "mode": mode, "passed": passed, "code": code,
            "image_error": [float(e) for e in error], "baseline": os.path.exists(baseline),
            "seconds": round(time.time() - start, 1), "records": records,
            "traceback": traceback}


def main():
    source, data, out, listing = sys.argv[1:5]
    workers = int(sys.argv[5]) if len(sys.argv) > 5 else 4
    tests = [line.strip() for line in open(listing) if line.strip()]
    os.makedirs(out, exist_ok=True)
    results = []
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(run, source, data, out, t, m) for t in tests for m in ("vtk", "tack")]
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            r = future.result()
            results.append(r)
            print(f"[{i + 1}/{len(futures)}] {r['mode']:4} {'ok  ' if r['passed'] else 'FAIL'} "
                  f"{r['test']} {r['image_error']} {r['seconds']}s", flush=True)
            with open(os.path.join(out, "summary.json"), "w") as f:
                json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
