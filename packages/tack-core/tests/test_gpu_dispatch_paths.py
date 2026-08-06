"""Drive the GPU backends' execute() on a machine with no GPU.

Everything the CUDA/HIP/Level Zero backends do between "kernel called" and
"launch" is plain Python: argument detection, the IR passes, variant
caching, scalar packing, loop-range resolution.  Only compilation and the
launch itself need a device.  So each backend is instantiated without its
__init__, those two are stubbed, and the dispatch path is exercised.

Their modules import device bindings at module scope, so the checks run in
a subprocess with those bindings stubbed — a mock left in `sys.modules`
would otherwise make the arch-detection loops in other test files believe
a GPU is present.  That is the same isolation `test_backend_isolation.py`
uses for its llvmlite check.

Without this, three of the five execute() bodies are only ever verified by
being read.
"""

import subprocess
import sys
import textwrap

import pytest

BACKENDS = {
    "cuda": dict(
        cls="from tack.runtime.cuda_backend import CUDABackend as Backend",
        stubs=["cuda", "cuda.bindings"],
        attrs="",
    ),
    "hip": dict(
        cls="from tack.runtime.hip_backend import HIPBackend as Backend",
        stubs=["hip"],
        attrs="",
    ),
    "level_zero": dict(
        cls="from tack.runtime.level_zero_backend import LevelZeroBackend as Backend",
        stubs=[],
        attrs=(
            "backend.supported_dtypes = {tack.lang.types.f32, tack.lang.types.i32,\n"
            "                            tack.lang.types.i64, tack.lang.types.f64}\n"
            "    backend._max_image_3d = 16384\n"
            "    backend._has_hw_sampler = True"
        ),
    ),
}

_PREAMBLE = '''
import sys, types
from unittest.mock import MagicMock

for name in {stubs!r}:
    mod = MagicMock()
    mod.__name__ = name
    sys.modules[name] = mod
for sub in ("driver", "nvrtc"):
    sys.modules.setdefault("cuda.bindings." + sub, MagicMock())

import numpy as np
import tack
import tack.lang.types
from tack.lang.field import Field, NumpyBuffer
from tack.runtime.kernel_utils import new_kernel_cache

tack.init(arch=tack.cpu)   # fields are allocated on the host
{cls}


class Recorder:
    def __init__(self, ir_func):
        self.ir = ir_func
        self.launches = []

    def __call__(self, kernel_args, loop_end, *extra):
        self.launches.append((list(kernel_args), loop_end))


def make_backend():
    backend = Backend.__new__(Backend)
    backend._cache = new_kernel_cache()
    backend.compiled = []

    def _compile_kernel(ir_func):
        rec = Recorder(ir_func)
        backend.compiled.append(rec)
        return rec

    backend._compile_kernel = _compile_kernel
    backend.allocate_field = (
        lambda dtype, shape, exportable=False: NumpyBuffer(dtype.numpy_dtype, shape))
    {attrs}
    return backend


@tack.kernel
def elementwise(x, out, n):
    for i in range(n):
        out[i] = x[i] * 2.0 + 1.0


@tack.kernel
def rowwise(a, out, rows, cols):
    for i in range(rows):
        for j in range(cols):
            out[i, j] = a[i, j] + 1.0


def field(shape, dtype=None):
    return tack.field(dtype=dtype or tack.f32, shape=shape)


def check(label, cond):
    assert cond, label
'''

_CHECKS = '''
# --- a launch happens, with the right range -------------------------
b = make_backend()
x, out = field((128,)), field((128,))
b.execute(elementwise, (x, out, 128), {})
check("one compile", len(b.compiled) == 1)
check("one launch", len(b.compiled[0].launches) == 1)
check("loop range", b.compiled[0].launches[0][1] == 128)
check("launch args present",
      all(a is not None for a in b.compiled[0].launches[0][0]))
check("fields reach the launch",
      any(isinstance(a, Field) for a in b.compiled[0].launches[0][0]))

# --- repeat dispatches reuse the variant ----------------------------
b = make_backend()
x, out = field((64,)), field((64,))
for _ in range(5):
    b.execute(elementwise, (x, out, 64), {})
check("compiled once for 5 dispatches", len(b.compiled) == 1)
check("five launches", len(b.compiled[0].launches) == 5)

# --- the IR passes do not re-run ------------------------------------
import tack.lang.ir_optimize as opt
calls = []
real = opt.optimize_ir
opt.optimize_ir = lambda f: (calls.append(1), real(f))[1]
b = make_backend()
x, out = field((64,)), field((64,))
for _ in range(6):
    b.execute(elementwise, (x, out, 64), {})
opt.optimize_ir = real
check("optimize_ir ran once, not per dispatch, got %d" % len(calls), len(calls) == 1)

# --- a flat length must not multiply variants -----------------------
b = make_backend()
for n in (16, 32, 64, 128, 256):
    x, out = field((n,)), field((n,))
    b.execute(elementwise, (x, out, n), {})
check("one variant across lengths, got %d" % len(b.compiled), len(b.compiled) == 1)
check("ranges", [l[1] for l in b.compiled[0].launches] == [16, 32, 64, 128, 256])

# --- but a changed row stride must ----------------------------------
b = make_backend()
a1, o1 = field((4, 8)), field((4, 8))
b.execute(rowwise, (a1, o1, 4, 8), {})
check("first 2d variant", len(b.compiled) == 1)
a2, o2 = field((8, 4)), field((8, 4))
b.execute(rowwise, (a2, o2, 8, 4), {})
check("row stride must not reuse code, got %d" % len(b.compiled), len(b.compiled) == 2)
a3, o3 = field((4, 8)), field((4, 8))
b.execute(rowwise, (a3, o3, 4, 8), {})
check("returning to the first shape reuses it, got %d" % len(b.compiled),
      len(b.compiled) == 2)

# --- dtype still separates variants ---------------------------------
b = make_backend()
x, out = field((64,)), field((64,))
b.execute(elementwise, (x, out, 64), {})
xi, oi = field((64,), tack.i32), field((64,), tack.i32)
b.execute(elementwise, (xi, oi, 64), {})
check("dtype variant", len(b.compiled) == 2)

# --- the template keeps its IRDimSize nodes -------------------------
from tack.lang import ir as _ir
from tack.runtime.kernel_utils import _walk_ir
tmpl = rowwise.get_ir().functions[0]
dims = [n for n in _walk_ir(tmpl.body) if isinstance(n, _ir.IRDimSize)]
check("template lost its IRDimSize nodes to an in-place pass", bool(dims))

# --- the declared capability contract -------------------------------
from tack.runtime.backend import Backend as BaseBackend
b = make_backend()
check("subclasses Backend", isinstance(b, BaseBackend))
check("declares an arch name", b.name != BaseBackend.name)
check("arch name is an init value", hasattr(tack, b.name))
check("declares supported dtypes", bool(b.supported_dtypes))
check("supports_f64 is derived from supported_dtypes",
      b.supports_f64 == (tack.lang.types.f64 in b.supported_dtypes))
check("label is prose, not an identifier", "_" not in b.label)
check("memory_space answers", isinstance(b.memory_space(0), str))
if b.device_memory_spaces:
    check("classifies the pointers it validates",
          type(b).memory_space is not BaseBackend.memory_space)

print("OK")
'''


@pytest.mark.parametrize("name", sorted(BACKENDS))
def test_gpu_dispatch_path(name, tmp_path):
    spec = BACKENDS[name]
    script = textwrap.dedent(_PREAMBLE).format(**spec) + textwrap.dedent(_CHECKS)

    # Run from a real file, not `python -c`. @tack.kernel reads its function's
    # source with inspect.getsource, and only Python 3.13+ registers a `-c`
    # command in linecache — on 3.11 and 3.12 the kernels in this script fail
    # to build with "could not get source code".
    script_path = tmp_path / f"gpu_dispatch_{name}.py"
    script_path.write_text(script)

    proc = subprocess.run([sys.executable, str(script_path)],
                          capture_output=True, text=True)
    assert proc.returncode == 0, (
        f"{name} dispatch path failed:\n{proc.stdout}\n{proc.stderr}")
    assert "OK" in proc.stdout
