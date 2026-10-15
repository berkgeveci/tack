"""Compare generic deepcopy with specialized cloning on dataset filters' IR.

    uv run --no-sync python benchmarks/ir_clone.py --output /tmp/ir-clone.json

Capture real pristine templates and typed CPU variants by running a contour
and a clip of VTK's wavelet after a warm-up run and clearing the variant
cache. Then alternate copy methods on each unchanged graph.
Setup, compilation and filtering are excluded from the clone timings. This
is a component benchmark, not an end-to-end or GPU performance claim.
"""

import argparse
import copy
import json
import platform
import statistics
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rounds', type=int, default=9)
    parser.add_argument('--copies', type=int, default=20)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.rounds < 3 or args.copies < 1:
        parser.error('--rounds must be at least 3 and --copies must be positive')
    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root / 'packages' / package / 'src')
                    for package in ('tack-core', 'tack-vis')]

    import llvmlite
    import numpy as np

    import tack
    from tack.lang import ir
    from tack.lang.ir_traversal import clone_ir, walk_ir
    from tack.runtime import cpu, kernel_utils
    from tack.runtime.dispatch import get_backend

    records = []
    original_compile = cpu._compile_kernel
    original_clone = kernel_utils.clone_ir

    def capture_clone(function):
        records.append(('template', function))
        return original_clone(function)

    def capture_compile(function):
        records.append(('typed', function))
        return original_compile(function)

    tack.init(arch=tack.cpu)
    import tack.dataset as td

    def filters():
        data = td.wavelet(extent=10)
        td.contour(data, "RTData", 157.0)
        td.clip(data, td.Plane(origin=(0, 0, 0), normal=(1, 0, 0)))

    kernel_utils.clone_ir = capture_clone
    cpu._compile_kernel = capture_compile
    try:
        filters()
        get_backend()._cache.clear()
        for per_backend in kernel_utils._fast_caches.values():
            per_backend.clear()                # the dispatch fast path's variants
        records.clear()  # Exclude the warm-up; recapture the filters' kernels.
        filters()
    finally:
        cpu._compile_kernel = original_compile
        kernel_utils.clone_ir = original_clone

    results = []
    for stage, function in records:
        reference = copy.deepcopy(function)
        cloned = clone_ir(function)
        assert ir.dump(reference) == ir.dump(cloned) == ir.dump(function)
        timings = {'deepcopy': [], 'clone_ir': []}
        methods = [('deepcopy', copy.deepcopy), ('clone_ir', clone_ir)]
        for trial in range(args.rounds):
            for name, method in methods[::1 if trial % 2 == 0 else -1]:
                start = time.perf_counter_ns()
                for _ in range(args.copies):
                    result = method(function)
                    del result  # Include disposal symmetrically.
                timings[name].append((time.perf_counter_ns() - start) / args.copies / 1e6)
        old, new = (statistics.median(timings[name]) for name in ('deepcopy', 'clone_ir'))
        nodes = list(walk_ir(function))
        results.append({'kernel': function.name, 'stage': stage,
                        'node_occurrences': len(nodes),
                        'unique_nodes': len({id(node) for node in nodes}),
                        'deepcopy_median_ms': old, 'clone_median_ms': new,
                        'speedup': old / new, 'samples_ms': timings})

    output = {'host': platform.node(), 'platform': platform.platform(),
              'python': platform.python_version(), 'numpy': np.__version__,
              'llvmlite': llvmlite.__version__, 'tack_source': tack.__file__,
              'workload': 'wavelet extent 10: contour at 157, clip by x = 0',
              'rounds': args.rounds, 'copies_per_round': args.copies,
              'kernels': results}
    text = json.dumps(output, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)


if __name__ == '__main__':
    main()
