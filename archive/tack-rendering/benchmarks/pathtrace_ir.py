"""Dump example 33's CPU kernels for a matched LLVM comparison.

    uv run --no-sync python benchmarks/pathtrace_ir.py --output-dir /tmp/ir-current
    uv run --no-sync python benchmarks/pathtrace_ir.py \
        --source-root /tmp/tack-before --output-dir /tmp/ir-before \
        --triple x86_64-unknown-linux-gnu --cpu sandybridge

The scene is rendered once at 32-square, four samples and two bounces, to
capture the variants actually compiled by CPU dispatch. Each output directory
contains raw LLVM IR, target-optimized IR, assembly, instruction counts and
source/toolchain metadata. Counts inspect instruction opcodes, not occurrences
of opcode names in textual SSA names or comments.

Historical source can come from a checkout or a git archive. Always use the
same interpreter, dependencies and target settings for both revisions. An
explicit target changes only the diagnostic optimization/assembly; execution
continues using the host's native target. Cross-target output is not a hardware
performance measurement. No profiler or timing comparison runs in this probe.
"""

import argparse
import contextlib
import io
import json
import platform
import sys
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path,
                        default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--triple')
    parser.add_argument('--cpu')
    parser.add_argument('--features')
    args = parser.parse_args()
    root = args.source_root.resolve()
    if not (root / 'packages/tack-core/src/tack/__init__.py').is_file():
        parser.error('--source-root must contain the Tack source packages')
    sys.path[:0] = [str(root / 'packages' / package / 'src')
                    for package in ('tack-core', 'tack-rendering', 'tack-vis')]

    import llvmlite
    import numpy as np
    from llvmlite import binding as llvm

    import tack
    from tack.runtime import cpu

    if not Path(tack.__file__).resolve().is_relative_to(root):
        parser.error('Tack was imported from outside --source-root')
    llvm.initialize_all_targets()
    llvm.initialize_all_asmprinters()
    triple = args.triple or llvm.get_default_triple()
    native = triple == llvm.get_default_triple()
    target_cpu = args.cpu or (str(llvm.get_host_cpu_name()) if native else 'generic')
    features = args.features
    if features is None:
        features = llvm.get_host_cpu_features().flatten() if native else ''
    target = llvm.Target.from_triple(triple).create_target_machine(
        cpu=target_cpu, features=features, opt=3,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    original_optimize = cpu._optimize_module
    records = []

    def instruction_counts(module):
        counts = Counter()
        vector_loads = 0
        for function in module.functions:
            for block in function.blocks:
                for instruction in block.instructions:
                    counts[instruction.opcode] += 1
                    if instruction.opcode == 'load' and instruction.type.is_vector:
                        vector_loads += 1
        return {'opcodes': dict(sorted(counts.items())),
                'vector_loads': vector_loads}

    def dump_optimize(module, native_target):
        name = next(f.name for f in module.functions if not f.is_declaration)
        # Keep separate artifacts if one kernel has multiple variants.
        variant = sum(record['kernel'] == name for record in records)
        prefix = name if variant == 0 else f'{name}-{variant}'
        raw = str(module)
        (args.output_dir / f'{prefix}.raw.ll').write_text(raw)
        diagnostic = llvm.parse_assembly(raw)
        raw_counts = instruction_counts(diagnostic)
        diagnostic.triple = triple
        diagnostic.data_layout = str(target.target_data)
        diagnostic.verify()
        original_optimize(diagnostic, target)
        diagnostic.verify()
        optimized_counts = instruction_counts(diagnostic)
        (args.output_dir / f'{prefix}.opt.ll').write_text(str(diagnostic))
        (args.output_dir / f'{prefix}.s').write_text(target.emit_assembly(diagnostic))
        records.append({'kernel': name, 'prefix': prefix, 'raw': raw_counts,
                        'optimized': optimized_counts})
        original_optimize(module, native_target)

    example = root / 'packages/tack-rendering/examples/33_pathtrace_primitives.py'
    if not example.is_file():
        parser.error('--source-root must contain rendering example 33')
    # Execute only scene setup; the example's image-writing code is excluded.
    setup = example.read_text().split('# Warmup (JIT compile)')[0]
    scope = {'__name__': '__main__', '__file__': str(example)}
    original_argv = sys.argv
    cpu._optimize_module = dump_optimize
    try:
        sys.argv = [str(example), '--arch', 'cpu', '--resolution', '32']
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(setup, str(example), 'exec'), scope)
            scope['render'](scope['canvas'], scope['scene'], scope['camera'],
                            samples=4, max_bounces=2)
    finally:
        cpu._optimize_module = original_optimize
        sys.argv = original_argv
    result = {
        'source_root': str(root), 'tack_source': tack.__file__,
        'host': platform.node(), 'python': platform.python_version(),
        'numpy': np.__version__, 'llvmlite': llvmlite.__version__,
        'llvm': llvm.llvm_version_info, 'triple': triple, 'cpu': target_cpu,
        'features': features, 'resolution': 32, 'samples': 4, 'bounces': 2,
        'kernels': records,
    }
    (args.output_dir / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
