"""Measure CPU volume startup in a fresh process, separating JIT and dispatch.

    uv run --no-sync python benchmarks/volume_cold_start.py --resolution 32
    uv run --no-sync python benchmarks/volume_cold_start.py --resolution 512

Each invocation creates the same 64-cube gyroid scene and clears the backend
cache after setup. The first render includes frontend work, LLVM compilation,
threading decisions, and actual rendering. Compare revisions at the SAME
resolution; a large canvas adds execution time even when compilation is
unchanged. Run alternating fresh processes, with no profiler for timings.

Use --source-root with a checkout or git archive to compare historical source
with the same interpreter and dependencies. --disable-disjoint isolates the
CPU specialization. --profile writes a separate cProfile artifact; profiling
inflates timings. --output-dir also saves the JSON result and raw colour planes.
"""

import argparse
import contextlib
import cProfile
import io
import json
import os
import platform
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resolution', type=int, default=32)
    parser.add_argument('--source-root', type=Path,
                        default=Path(__file__).resolve().parents[1])
    parser.add_argument('--disable-disjoint', action='store_true')
    parser.add_argument('--profile', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if args.resolution <= 0:
        parser.error('--resolution must be positive')
    root = args.source_root.resolve()
    if not (root / 'packages/tack-core/src/tack/__init__.py').is_file():
        parser.error('--source-root must contain the Tack source packages')
    sys.path[:0] = [str(root / 'packages' / package / 'src')
                    for package in ('tack-core', 'tack-rendering', 'tack-vis')]

    import llvmlite
    import numpy as np

    import tack
    from tack.rendering import (
        Canvas,
        PerspectiveCamera,
        TransferFunction,
        Volume,
        render_volume,
    )
    from tack.runtime import cpu
    from tack.runtime.dispatch import get_backend

    if not Path(tack.__file__).resolve().is_relative_to(root):
        parser.error('Tack was imported from outside --source-root')
    tack.init(arch=tack.cpu)
    if args.disable_disjoint:
        cpu._SPECIALIZE_DISJOINT = False
    size = 64
    axis = np.linspace(-np.pi, np.pi, size, dtype=np.float32)
    x, y, z = np.meshgrid(axis, axis, axis, indexing='ij')
    values = (np.sin(x) * np.cos(y) + np.sin(y) * np.cos(z)
              + np.sin(z) * np.cos(x)).astype(np.float32)
    transfer = TransferFunction(
        'cool_to_warm', range=(float(values.min()), float(values.max())),
        opacity_func=lambda t: 0.005 + 0.06 * abs(2 * t - 1) ** 1.5,
    )
    spacing = 2 * np.pi / (size - 1)
    volume = Volume(
        values.ravel(), dims=(size,) * 3, origin=(-np.pi,) * 3,
        spacing=(spacing,) * 3, transfer_function=transfer, opacity_scale=8.0,
    )
    resolution = args.resolution
    camera = PerspectiveCamera(position=(8, 6, 8), look_at=(0, 0, 0), fov=45,
                               width=resolution, height=resolution)
    canvas = Canvas(resolution, resolution)
    backend = get_backend()
    backend._cache.clear()

    def render_once():
        with contextlib.redirect_stdout(io.StringIO()):
            render_volume(canvas, volume, camera)

    compiles = []
    dispatches = []
    original_compile = cpu._compile_kernel
    original_dispatch = backend._dispatch

    def timed_compile(ir_func):
        start = time.perf_counter_ns()
        compiled = original_compile(ir_func)
        compiles.append({
            'kernel': ir_func.name,
            'ms': (time.perf_counter_ns() - start) / 1e6,
            'disjoint': getattr(ir_func, 'disjoint_fields', False),
        })
        return compiled

    def timed_dispatch(compiled, kernel_args, loop_end):
        start = time.perf_counter_ns()
        original_dispatch(compiled, kernel_args, loop_end)
        dispatches.append({
            'kernel': compiled._func_name, 'elements': loop_end,
            'ms': (time.perf_counter_ns() - start) / 1e6,
        })

    profiler = cProfile.Profile() if args.profile else None
    cpu._compile_kernel = timed_compile
    backend._dispatch = timed_dispatch
    try:
        if profiler:
            profiler.enable()
        start = time.perf_counter_ns()
        render_once()
        cold_ms = (time.perf_counter_ns() - start) / 1e6
    finally:
        if profiler:
            profiler.disable()
        cpu._compile_kernel = original_compile
        backend._dispatch = original_dispatch
    if profiler:
        args.profile.parent.mkdir(parents=True, exist_ok=True)
        profiler.dump_stats(str(args.profile))

    for _ in range(4):
        render_once()
    warm = []
    for _ in range(15):
        start = time.perf_counter_ns()
        render_once()
        warm.append((time.perf_counter_ns() - start) / 1e6)
    result = {
        'source_root': str(root), 'tack_source': tack.__file__,
        'host': platform.node(), 'platform': platform.platform(),
        'python': platform.python_version(), 'numpy': np.__version__,
        'llvmlite': llvmlite.__version__, 'threads': backend.num_threads,
        'thread_override': os.environ.get('TACK_CPU_THREADS'),
        'resolution': resolution, 'volume_size': size, 'profiled': bool(profiler),
        'disjoint_enabled': getattr(cpu, '_SPECIALIZE_DISJOINT', False),
        'cold_ms': cold_ms, 'jit_ms': sum(c['ms'] for c in compiles),
        'dispatch_ms': sum(d['ms'] for d in dispatches),
        'compiles': compiles, 'dispatches': dispatches,
        'warm_min_ms': min(warm), 'warm_median_ms': float(np.median(warm)),
    }
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        image = np.stack([canvas.color_r.to_numpy(), canvas.color_g.to_numpy(),
                          canvas.color_b.to_numpy()])
        np.save(args.output_dir / 'image.npy', image)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
