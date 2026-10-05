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
    "cuda": {
        "cls": "from tack.runtime.cuda_backend import CUDABackend as Backend",
        "stubs": ["cuda", "cuda.bindings"],
        # The launch limit is queried in __init__ too.
        "attrs": "backend._max_launch = (2**31 - 1) * 256",
    },
    "hip": {
        "cls": "from tack.runtime.hip_backend import HIPBackend as Backend",
        "stubs": ["hip"],
        # Texture capability is queried in __init__, which __new__ skips.
        # Stand in for a device that has texture units, so the dispatch
        # path under test is the hardware one.
        "attrs": (
            "backend._has_image_support = True\n"
            "    backend._max_image_3d = 16384\n"
            "    backend._max_launch = 2**32 - 256"
        ),
    },
    "level_zero": {
        "cls": "from tack.runtime.level_zero_backend import LevelZeroBackend as Backend",
        "stubs": [],
        "attrs": (
            "backend.supported_dtypes = {tack.lang.types.f32, tack.lang.types.i32,\n"
            "                            tack.lang.types.i64, tack.lang.types.f64}\n"
            "    backend._max_image_3d = 16384\n"
            "    backend._has_hw_sampler = True\n"
            "    backend._launch_lock = threading.RLock()\n"
            "    backend._max_launch = (2**32 - 1) * 256"
        ),
    },
}

_PREAMBLE = '''
import sys, threading, types
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

# --- the launch and its scalar pack are serialized ------------------
# Pack buffers (and Level Zero's command lists) are shared launch state,
# so another thread must not reach them mid-launch.
def launch_lock_held(backend):
    lock = getattr(backend, "_launch_lock", None)
    if lock is not None:
        return lock._is_owned()
    return any(v.dispatch_lock.locked()
               for slot in backend._cache.values() for v in slot.values())

b = make_backend()
held = []
x, out = field((32,)), field((32,))
b.execute(elementwise, (x, out, 32), {})
real_call = type(b.compiled[0]).__call__
type(b.compiled[0]).__call__ = (
    lambda self, *a: (held.append(launch_lock_held(b)), real_call(self, *a))[1])
b.execute(elementwise, (x, out, 32), {})
type(b.compiled[0]).__call__ = real_call
check("launch runs under the dispatch lock", held == [True])

# --- repeat dispatches reuse the variant ----------------------------
b = make_backend()
x, out = field((64,)), field((64,))
for _ in range(5):
    b.execute(elementwise, (x, out, 64), {})
check("compiled once for 5 dispatches", len(b.compiled) == 1)
check("five launches", len(b.compiled[0].launches) == 5)

# --- a launch past the grid's limit is refused, not wrapped ---------
b = make_backend()
b._max_launch = 64
x, out = field((128,)), field((128,))
b.execute(elementwise, (x, out, 64), {})
try:
    b.execute(elementwise, (x, out, 128), {})
except ValueError as error:
    check("limit named, got %s" % error, "128 iterations exceed the 64" in str(error))
else:
    raise AssertionError("an over-limit launch reached the device")
check("only the launch within the limit", [l[1] for l in b.compiled[0].launches] == [64])

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
# Deliberately not `memory_space(0)`. Zero is a plausible address, so it
# reaches the bindings -- and under MagicMock the bindings cannot answer,
# so whatever came back was the mock's invention rather than the
# backend's. Asserting it was a string was asserting that the stub had
# produced *something*, which it always will.
#
# What is knowable under stubs is the part decided before the driver is
# involved: an input that cannot be an address is refused by `as_address`,
# so these answers are the backend's own.
check("refuses a non-integer as an address",
      b.memory_space("not a pointer") == "cpu")
check("refuses an integer too large to be an address",
      b.memory_space(1 << 200) == "cpu")
if b.device_memory_spaces:
    check("classifies the pointers it validates",
          type(b).memory_space is not BaseBackend.memory_space)

    # And a real address does reach the driver, which is the point of
    # moving the call outside the `try`: a fault there must not come back
    # as host memory, because the damage lands upstream -- field_from_ptr
    # then refuses a good device pointer with a message about the wrong
    # thing. D9 is what that costs when it goes unnoticed.
    # Import only this run's backend: each subprocess stubs the bindings
    # for its own and no other, so reaching for a sibling module fails on
    # the real import rather than the thing under test.
    import importlib
    _where = {"cuda": ("tack.runtime.cuda_backend", "driver",
                       "cuPointerGetAttribute"),
              "hip": ("tack.runtime.hip_backend", "hip",
                      "hipPointerGetAttributes")}.get(b.name)
    if _where is not None:
        _mod_name, _api_name, call = _where
        api = getattr(importlib.import_module(_mod_name), _api_name)
        original = getattr(api, call)

        def _boom(*a, **kw):
            raise RuntimeError("driver fault")

        setattr(api, call, _boom)
        try:
            escaped = False
            try:
                b.memory_space(0)
            except RuntimeError:
                escaped = True
            check("a driver fault is not reported as host memory", escaped)
        finally:
            setattr(api, call, original)

# --- verification stays off the repeated dispatch path --------------
import tack.lang.ir_verify as verifier
verified = []
real_verify = verifier.verify_ir
def record_verify(function, stage):
    verified.append(stage)
    real_verify(function, stage)
verifier.verify_ir = record_verify
b = make_backend()
for _ in range(5):
    b.execute(elementwise, (x, out, 64), {})
verifier.verify_ir = real_verify
check("GPU boundaries verified once", verified ==
      ["resolved", "inferred", "localized", "optimized", "packed", "typed"])

# --- bad packing cannot reach device compilation or the cache -------
import tack.lang.ir_pack_scalars as packing
real_pack = packing.pack_scalars
packing.pack_scalars = lambda function, args: (args, None)
b = make_backend()
try:
    b.execute(elementwise, (x, out, 64), {})
except verifier.IRVerificationError as error:
    check("failure attributed to packing", "after packed" in str(error))
else:
    raise AssertionError("unpacked scalars reached GPU compilation")
finally:
    packing.pack_scalars = real_pack
check("invalid packing not compiled", not b.compiled)
check("invalid packing not cached", not b._cache.get(elementwise))

# --- a texture binds its own snapshot, never its field --------------
@tack.kernel
def sample(out, tex, n):
    for i in range(n):
        out[i] = tex.sample(0.5, 0.5, 0.5)

for shape in ((4, 4, 4), (4, 4, 20000)):
    b = make_backend()
    # Made while CPU is active, so its storage is a private field whatever
    # this backend would build; what matters is which object reaches the
    # launch, through the scalar-packing path the count takes.
    data = field((shape[0] * shape[1] * shape[2],))
    tex = tack.texture3d(data, shape=shape)
    out = field((1,))
    b.execute(sample, (out, tex, 1), {})
    bound = b.compiled[0].launches[0][0]
    check("texture snapshot reaches the launch",
          any(a is tex._storage for a in bound))
    check("texture's field does not", not any(a is data for a in bound))
    param = next(p for p in b.compiled[0].ir.params if p.name == "tex")
    check("variant and Texture3D agree on hardware sampling for %s" % (shape,),
          param._is_texture == b.texture_in_hardware(shape))

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
