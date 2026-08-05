"""GPU backends must not depend on the CPU backend.

``tack.runtime.cpu`` imports llvmlite at module scope, and llvmlite is a
CPU-only extra.  When the GPU backends pulled their shared dispatch helpers
from ``tack.runtime.cpu``, a ``pip install tack-core[cuda]`` (or [metal],
[hip], [level_zero]) installed no llvmlite and then died with ImportError on
the user's first kernel launch.  The helpers live in
``tack.runtime.kernel_utils``, which has no backend-specific dependencies.

The static test below runs anywhere, including CPU-only CI.
"""

import pathlib
import subprocess
import sys
import textwrap

import pytest

import tack

RUNTIME = pathlib.Path(tack.__file__).parent / "runtime"
GPU_BACKEND_MODULES = [
    "cuda_backend.py",
    "hip_backend.py",
    "level_zero_backend.py",
    "metal.py",
]


@pytest.mark.parametrize("module", GPU_BACKEND_MODULES)
def test_gpu_backend_does_not_import_cpu_backend(module):
    """Static check — runs without any GPU present."""
    source = (RUNTIME / module).read_text()
    offenders = [
        f"  line {i}: {line.strip()}"
        for i, line in enumerate(source.splitlines(), 1)
        if "tack.runtime.cpu" in line
    ]
    assert not offenders, (
        f"{module} imports the CPU backend, which pulls in llvmlite and breaks "
        f"GPU-only installs. Import from tack.runtime.kernel_utils instead:\n"
        + "\n".join(offenders)
    )


def _first_gpu_backend():
    for arch in ["metal", "cuda", "hip", "level_zero"]:
        try:
            tack.init(arch=getattr(tack, arch))
            return arch
        except (ImportError, RuntimeError, OSError):
            continue
    return None


_GPU = _first_gpu_backend()


@pytest.mark.skipif(_GPU is None, reason="no GPU backend available")
def test_gpu_dispatch_works_without_llvmlite():
    """End-to-end: compile and run on the GPU with llvmlite unimportable."""
    script = textwrap.dedent(f"""
        import sys
        sys.modules["llvmlite"] = None  # any `import llvmlite` now raises

        import numpy as np
        import tack
        tack.init(arch=tack.{_GPU})

        n = 64
        x = tack.field(dtype=tack.f32, shape=(n,))
        out = tack.field(dtype=tack.f32, shape=(n,))
        x.from_numpy(np.arange(n, dtype=np.float32))

        @tack.kernel
        def double(x, out):
            for i in range(x.shape[0]):
                out[i] = x[i] * 2.0

        double(x, out)
        assert np.allclose(out.to_numpy(), np.arange(n) * 2.0)
        assert "tack.runtime.cpu" not in sys.modules, \\
            "GPU dispatch imported the CPU backend"
        print("OK")
    """)
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True,
    )
    assert proc.returncode == 0, (
        f"GPU dispatch failed without llvmlite:\n{proc.stdout}\n{proc.stderr}"
    )
    assert "OK" in proc.stdout
