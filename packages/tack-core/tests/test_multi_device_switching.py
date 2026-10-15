"""Several CUDA and HIP devices in one process, against fake runtimes.

The CUDA driver works in the thread's current context and HIP on its current
device, so once Tack drives a device that is not the thread's own, every
operation must make its own current for its duration and put the thread's
back. Getting one wrong is a CUDA_ERROR_INVALID_CONTEXT, or memory and
launches on the wrong GPU, on a machine with two of them. The fakes here
refuse every call made in the wrong context or on the wrong device, which
checks the switching on any machine; `test_multi_device.py` runs the same
operations on real devices where there are two.

As in `test_gpu_dispatch_paths.py`, each check runs in a subprocess, so the
fake bindings never reach `sys.modules` of the test session. Kernels are not
compiled: the backend's `_compile_kernel` returns a recorder of the context
or device current at the launch.
"""

import os
import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.quick

_FAKE_CUDA = '''
import enum, itertools, sys, threading, types
import numpy as np


class CUresult(enum.IntEnum):
    CUDA_SUCCESS = 0
    CUDA_ERROR_INVALID_VALUE = 1
    CUDA_ERROR_INVALID_CONTEXT = 201


class nvrtcResult(enum.IntEnum):
    NVRTC_SUCCESS = 0


class _Names:
    def __getattr__(self, name):
        return name


class CUevent_flags:
    CU_EVENT_BLOCKING_SYNC = 1
    CU_EVENT_DISABLE_TIMING = 2


OK = CUresult.CUDA_SUCCESS
BAD = CUresult.CUDA_ERROR_INVALID_CONTEXT
DEVICES = 2
_tls = threading.local()
_ids = itertools.count(0x1000, 0x1000)
contexts = {}           # handle -> device
streams = {}            # handle -> context
events = {}
memory = {}             # base -> (bytearray, context)
wrong = []              # every call refused for its context


def current():
    stack = getattr(_tls, "stack", None)
    return stack[-1] if stack else 0


def _stack():
    if not hasattr(_tls, "stack"):
        _tls.stack = []
    return _tls.stack


def _need(ctx, call):
    if current() != ctx:
        wrong.append((call, current(), ctx))
        return BAD
    return OK


def _alloc(addr):
    for base, (data, ctx) in memory.items():
        if base <= addr < base + max(len(data), 1):
            return base, data, ctx
    raise KeyError(hex(addr))


def cuInit(flags): return OK
def cuDriverGetVersion(): return OK, 13000
def cuDeviceGetCount(): return OK, DEVICES
def cuDeviceGet(n): return (OK, n) if 0 <= n < DEVICES else (CUresult.CUDA_ERROR_INVALID_VALUE, 0)
def cuDeviceGetAttribute(attr, dev): return OK, 65535
def cuDeviceGetDefaultMemPool(dev): return OK, ("pool", dev)
def cuMemPoolSetAttribute(*a): return OK
CUdevice_attribute = CUmemPool_attribute = CUpointer_attribute = _Names()
cuuint64_t = int


def cuCtxGetCurrent(): return OK, current()
def cuCtxGetDevice():
    return (OK, contexts[current()]) if current() else (BAD, 0)
def cuCtxCreate(params, flags, dev):
    ctx = next(_ids)
    contexts[ctx] = dev
    _stack().append(ctx)
    return OK, ctx
def cuCtxPushCurrent(ctx):
    _stack().append(ctx)
    return OK
def cuCtxPopCurrent():
    return OK, _stack().pop()
def cuCtxDestroy(ctx):
    contexts.pop(ctx, None)
    return OK


def cuStreamCreate(flags):
    if not current():
        wrong.append(("cuStreamCreate", 0, None))
        return BAD, 0
    stream = next(_ids)
    streams[stream] = current()
    return OK, stream
def cuStreamSynchronize(stream): return _need(streams[stream], "cuStreamSynchronize")
def cuStreamDestroy(stream): return OK


def cuEventCreate(flags):
    event = next(_ids)
    events[event] = current()
    return OK, event
def cuEventRecord(event, stream):
    if events[event] != streams[stream]:
        wrong.append(("cuEventRecord", events[event], streams[stream]))
        return BAD
    return _need(streams[stream], "cuEventRecord")
def cuEventSynchronize(event): return OK
def cuEventDestroy(event): return OK


def cuMemAllocAsync(n, stream):
    err = _need(streams[stream], "cuMemAllocAsync")
    if err != OK:
        return err, 0
    base = next(_ids) * 0x100
    memory[base] = (bytearray(n), current())
    return OK, base
def cuMemFreeAsync(ptr, stream):
    err = _need(streams[stream], "cuMemFreeAsync") or _need(_alloc(int(ptr))[2], "cuMemFreeAsync")
    memory.pop(int(ptr), None)
    return err


def cuPointerGetAttribute(attr, addr):
    try:
        _alloc(int(addr))
    except KeyError:
        return CUresult.CUDA_ERROR_INVALID_VALUE, 0
    return OK, 2                        # CU_MEMORYTYPE_DEVICE


def _bytes(arr, n):
    return np.ascontiguousarray(arr).reshape(-1).view(np.uint8)[:n]


def _write(ptr, src, n, call, stream):
    base, data, ctx = _alloc(int(ptr))
    err = _need(streams[stream], call) or _need(ctx, call)
    off = int(ptr) - base
    data[off:off + n] = bytes(_bytes(src, n))
    return err
def cuMemcpyHtoDAsync(ptr, src, n, stream): return _write(ptr, src, n, "cuMemcpyHtoDAsync", stream)
def cuMemcpyDtoHAsync(dst, ptr, n, stream):
    base, data, ctx = _alloc(int(ptr))
    err = _need(streams[stream], "cuMemcpyDtoHAsync") or _need(ctx, "cuMemcpyDtoHAsync")
    off = int(ptr) - base
    dst.reshape(-1).view(np.uint8)[:n] = np.frombuffer(bytes(data[off:off + n]), np.uint8)
    return err
def cuMemcpyDtoDAsync(dst, src, n, stream):
    sbase, sdata, sctx = _alloc(int(src))
    chunk = bytes(sdata[int(src) - sbase:int(src) - sbase + n])
    return _write(dst, np.frombuffer(chunk, np.uint8), n, "cuMemcpyDtoDAsync", stream)
def _memset(ptr, pattern, count, stream, call):
    return _write(ptr, np.frombuffer(pattern * count, np.uint8), len(pattern) * count,
                  call, stream)
def cuMemsetD8Async(ptr, value, n, stream):
    return _memset(ptr, bytes([value]), n, stream, "cuMemsetD8Async")
def cuMemsetD16Async(ptr, value, n, stream):
    return _memset(ptr, int(value).to_bytes(2, "little"), n, stream, "cuMemsetD16Async")
def cuMemsetD32Async(ptr, value, n, stream):
    return _memset(ptr, int(value).to_bytes(4, "little"), n, stream, "cuMemsetD32Async")


driver = types.ModuleType("cuda.bindings.driver")
nvrtc = types.ModuleType("cuda.bindings.nvrtc")
driver.__dict__.update({k: v for k, v in dict(globals()).items() if not k.startswith("__")})
driver.__getattr__ = lambda name: _Names()     # enum namespaces nothing here reads
nvrtc.nvrtcResult = nvrtcResult
nvrtc.nvrtcVersion = lambda: (nvrtcResult.NVRTC_SUCCESS, 13, 0)
bindings = types.ModuleType("cuda.bindings")
bindings.driver, bindings.nvrtc = driver, nvrtc
cuda = types.ModuleType("cuda")
cuda.bindings = bindings
sys.modules.update({"cuda": cuda, "cuda.bindings": bindings,
                    "cuda.bindings.driver": driver, "cuda.bindings.nvrtc": nvrtc})
'''

_FAKE_HIP = '''
import enum, itertools, sys, threading, types
import numpy as np


class hipError_t(enum.IntEnum):
    hipSuccess = 0
    hipErrorInvalidValue = 1
    hipErrorInvalidDevice = 101
    hipErrorNotSupported = 801


class hiprtcResult(enum.IntEnum):
    HIPRTC_SUCCESS = 0


class _Names:
    def __getattr__(self, name):
        return name


class hipMemcpyKind(enum.IntEnum):
    hipMemcpyHostToDevice = 1
    hipMemcpyDeviceToHost = 2
    hipMemcpyDeviceToDevice = 3


OK = hipError_t.hipSuccess
BAD = hipError_t.hipErrorInvalidDevice
DEVICES = 2
_tls = threading.local()
_ids = itertools.count(0x1000, 0x1000)
streams = {}            # handle -> device
events = {}
memory = {}             # base -> (bytearray, device)
wrong = []
hipEventBlockingSync, hipEventDisableTiming = 1, 2
hipDeviceAttribute_t = hipMemPoolAttr = _Names()


def current():
    return getattr(_tls, "device", 0)


def _need(device, call):
    if current() != device:
        wrong.append((call, current(), device))
        return BAD
    return OK


def _alloc(addr):
    for base, (data, dev) in memory.items():
        if base <= addr < base + max(len(data), 1):
            return base, data, dev
    raise KeyError(hex(addr))


class hipPointerAttribute_t:
    type = 0


def hipDeviceptr_t(addr):
    return addr


def hipPointerGetAttributes(attrs, addr):
    try:
        _alloc(int(addr))
    except KeyError:
        return hipError_t.hipErrorInvalidValue
    attrs.type = 2                      # hipMemoryTypeDevice
    return OK


def hipInit(flags): return OK
def hipGetDevice(): return OK, current()
def hipSetDevice(n):
    _tls.device = n
    return OK
def hipGetDeviceCount(): return OK, DEVICES


_ATTRIBUTES = {"hipDeviceAttributeImageSupport": 0,
               "hipDeviceAttributeMemoryPoolsSupported": 1}


def hipDeviceGetAttribute(attr, dev): return OK, _ATTRIBUTES.get(attr, 65535)
def hipDeviceGetDefaultMemPool(dev): return OK, ("pool", dev)
def hipMemPoolSetAttribute(*a): return OK


def hipStreamCreate():
    stream = next(_ids)
    streams[stream] = current()
    return OK, stream
def hipStreamSynchronize(stream): return _need(streams[stream], "hipStreamSynchronize")


def hipEventCreateWithFlags(flags):
    event = next(_ids)
    events[event] = current()
    return OK, event
def hipEventRecord(event, stream):
    if events[event] != streams[stream]:
        wrong.append(("hipEventRecord", events[event], streams[stream]))
        return BAD
    return _need(streams[stream], "hipEventRecord")
def hipEventSynchronize(event): return OK
def hipEventDestroy(event): return OK


def hipMallocAsync(n, stream):
    err = _need(streams[stream], "hipMallocAsync")
    base = next(_ids) * 0x100
    memory[base] = (bytearray(n), current())
    return err, base
def hipFreeAsync(ptr, stream):
    err = _need(streams[stream], "hipFreeAsync") or _need(_alloc(int(ptr))[2], "hipFreeAsync")
    memory.pop(int(ptr), None)
    return err


def _write(ptr, payload, call, stream):
    base, data, dev = _alloc(int(ptr))
    err = _need(streams[stream], call) or _need(dev, call)
    off = int(ptr) - base
    data[off:off + len(payload)] = payload
    return err


def hipMemcpyAsync(dst, src, n, kind, stream):
    if kind == hipMemcpyKind.hipMemcpyHostToDevice:
        payload = bytes(np.ascontiguousarray(src).reshape(-1).view(np.uint8)[:n])
        return _write(dst, payload, "hipMemcpyAsync", stream)
    base, data, dev = _alloc(int(src))
    off = int(src) - base
    chunk = bytes(data[off:off + n])
    if kind == hipMemcpyKind.hipMemcpyDeviceToDevice:
        return _write(dst, chunk, "hipMemcpyAsync", stream)
    err = _need(streams[stream], "hipMemcpyAsync") or _need(dev, "hipMemcpyAsync")
    dst.reshape(-1).view(np.uint8)[:n] = np.frombuffer(chunk, np.uint8)
    return err
def hipMemsetAsync(ptr, value, n, stream): return _write(ptr, bytes([value]) * n, "hipMemsetAsync", stream)
def hipMemsetD8Async(ptr, value, n, stream): return hipMemsetAsync(ptr, value, n, stream)
def hipMemsetD16Async(ptr, value, n, stream):
    return _write(ptr, (int(value) & 0xFFFF).to_bytes(2, "little") * n, "hipMemsetD16Async", stream)
def hipMemsetD32Async(ptr, value, n, stream):
    return _write(ptr, (int(value) & 0xFFFFFFFF).to_bytes(4, "little") * n,
                  "hipMemsetD32Async", stream)


hip = types.ModuleType("hip.hip")
hip.__dict__.update({k: v for k, v in dict(globals()).items() if not k.startswith("__")})
hiprtc = types.ModuleType("hip.hiprtc")
hiprtc.hiprtcResult = hiprtcResult
chip = types.ModuleType("hip.chip")
chip.DLL = b"/nonexistent/libamdhip64.so"
package = types.ModuleType("hip")
package.hip, package.hiprtc, package.chip, package.LONG_VERSION = hip, hiprtc, chip, "fake"
sys.modules.update({"hip": package, "hip.hip": hip, "hip.hiprtc": hiprtc, "hip.chip": chip})
'''

_COMMON = '''
import threading
import numpy as np
import tack


class Recorder:
    """Stands in for a compiled kernel: records what was current at its launch."""

    def __init__(self, ir_func):
        self.ir = ir_func
        self.launches = []
        self.ptx = self.code_object = None

    def __call__(self, kernel_args, loop_end, *extra):
        self.launches.append(fake.current())


def recording(device):
    backend = device._backend
    backend.compiled = []

    def compile_kernel(ir_func):
        backend.compiled.append(Recorder(ir_func))
        return backend.compiled[-1]

    backend._compile_kernel = compile_kernel
    return backend


@tack.kernel
def scale(x, out, a):
    for i in range(x.shape[0]):
        out[i] = x[i] * a


def check(label, cond):
    assert cond, label
'''

_CUDA_CHECKS = '''
import cuda.bindings.driver as fake
from tack.runtime import cuda_backend

check("two devices", tack.device_count("cuda") == 2)
first = tack.device("cuda")
check("bare cuda is device 0 with nothing current", first is tack.device("cuda:0"))
own = fake.current()
check("device 0's context is left current", own != 0)
check("one device switches nothing", not cuda_backend._switching)

second = tack.device("cuda:1")
check("index and name", (second.index, str(second)) == (1, "cuda:1"))
check("the thread keeps device 0's context", fake.current() == own)
check("a second context switches", cuda_backend._switching)
check("domains differ", first._backend.memory_domain != second._backend.memory_domain)
check("bare cuda still device 0", tack.device("cuda") is first)
ctx1 = second._backend._token.handle

for dev in (first, second):
    backend = recording(dev)
    x = tack.array(np.arange(8, dtype=np.float32), device=dev)
    out = tack.zeros((8,), device=dev)
    scale(x, out, 2.0)
    scale(x, out, 3.0)
    launches = backend.compiled[0].launches
    check(f"{dev} launches in its context: {launches}",
          launches == [dev._backend._token.handle] * 2)
    check("the thread's context after a launch", fake.current() == own)
    np.testing.assert_array_equal(x.numpy(), np.arange(8))
    check("one element", x[3] == 3.0)
    out.fill(2.0)
    np.testing.assert_array_equal(out.numpy(), [2.0] * 8)
    tack.copyto(x, np.ones(8, np.float32))
    check("a copy back", x.numpy().tolist() == [1.0] * 8)
    check("on its device", x.device == dev and x.copy().device == dev)
    del x, out
check("the thread's context after it all", fake.current() == own)

try:
    scale(tack.zeros((4,), device=first), tack.zeros((4,), device=second), 1.0)
except ValueError as e:
    check("two devices named", "cuda:0" in str(e) and "cuda:1" in str(e))
else:
    raise AssertionError("arrays on two devices ran")

try:
    tack.device("cuda:2")
except ValueError as e:
    check("past the last device: " + str(e), "this machine has 2 CUDA devices" in str(e))
else:
    raise AssertionError("cuda:2 was made")

held = tack.array(np.arange(4, dtype=np.float32), device=second)
seen = []
def other_thread():
    seen.append(fake.current())
    seen.append(held.numpy().tolist())
    tack.copyto(held, np.zeros(4, np.float32))
    seen.append(fake.current())
t = threading.Thread(target=other_thread)
t.start(); t.join()
check(f"another thread works in the device's context: {seen}",
      seen == [0, [0.0, 1.0, 2.0, 3.0], 0])

# DLPack carries the device number, and an import lands on it.
check("exported as device 1", held.__dlpack_device__() == (2, 1))
tack.set_default_device("cuda:0")
imported = tack.from_dlpack(held)
check(f"imported on its device, not the default: {imported.device}", imported.device == second)
tack.set_default_device("cpu")
tack.sync()
check("every call in its own context: " + repr(fake.wrong), not fake.wrong)
'''

_CUDA_ADOPTS = '''
import cuda.bindings.driver as fake
from tack.runtime import cuda_backend

# An embedding framework made a context on device 1 current.
err, foreign = fake.cuCtxCreate(None, 0, 1)
dev = tack.device("cuda")
check("bare cuda is the framework's device", str(dev) == "cuda:1")
check("its context adopted", dev._backend._token.handle == foreign)
check("not owned", not dev._backend._token.owned)
check("still current", fake.current() == foreign)
check("one context switches nothing", not cuda_backend._switching)
zero = tack.device("cuda:0")
check("device 0 is made beside it", zero._backend._token.handle != foreign)
check("the framework's context stays current", fake.current() == foreign)
check("and the backends switch", cuda_backend._switching)
x = tack.array(np.arange(4, dtype=np.float32), device=zero)
check("device 0 works", x.numpy().tolist() == [0.0, 1.0, 2.0, 3.0])
check("framework's context after", fake.current() == foreign)
check("every call in its own context: " + repr(fake.wrong), not fake.wrong)
'''

_CUDA_SECOND_ONLY = '''
import cuda.bindings.driver as fake
from tack.runtime import cuda_backend

# Only device 1, on a thread with no context: left current, as device 0's is.
dev = tack.device("cuda:1")
check("its context is current", fake.current() == dev._backend._token.handle)
check("bare cuda is still device 0", tack.device("cuda") is not dev)
check("every call in its own context: " + repr(fake.wrong), not fake.wrong)
'''

_HIP_CHECKS = '''
import hip.hip as fake
from tack.runtime import hip_backend

check("two devices", tack.device_count("hip") == 2)
first = tack.device("hip")
check("bare hip is the current device", first is tack.device("hip:0"))
check("one device switches nothing", not hip_backend._switching)
second = tack.device("hip:1")
check("index and name", (second.index, str(second)) == (1, "hip:1"))
check("the thread stays on device 0", fake.current() == 0)
check("a second device switches", hip_backend._switching)

for dev in (first, second):
    backend = recording(dev)
    x = tack.array(np.arange(8, dtype=np.float32), device=dev)
    out = tack.zeros((8,), device=dev)
    scale(x, out, 2.0)
    scale(x, out, 3.0)
    launches = backend.compiled[0].launches
    check(f"{dev} launches on its device: {launches}", launches == [dev.index] * 2)
    check("the thread's device after a launch", fake.current() == 0)
    np.testing.assert_array_equal(x.numpy(), np.arange(8))
    check("one element", x[3] == 3.0)
    out.fill(2.0)
    np.testing.assert_array_equal(out.numpy(), [2.0] * 8)
    tack.copyto(x, np.ones(8, np.float32))
    check("a copy back", x.numpy().tolist() == [1.0] * 8)
    del x, out

fake.hipSetDevice(1)
check("bare hip follows the thread's device", tack.device("hip") is second)
fake.hipSetDevice(0)

held = tack.array(np.arange(4, dtype=np.float32), device=second)
seen = []
def other_thread():
    seen.append(held.numpy().tolist())
    seen.append(fake.current())
t = threading.Thread(target=other_thread)
t.start(); t.join()
check(f"another thread reads device 1: {seen}", seen == [[0.0, 1.0, 2.0, 3.0], 0])

check("exported as device 1", held.__dlpack_device__() == (10, 1))
tack.set_default_device("hip:0")
imported = tack.from_dlpack(held)
check(f"imported on its device, not the default: {imported.device}", imported.device == second)
tack.set_default_device("cpu")
tack.sync()
check("every call on its own device: " + repr(fake.wrong), not fake.wrong)
'''

_HIP_SECOND_ONLY = '''
import hip.hip as fake
from tack.runtime import hip_backend

# Only device 1, on a thread whose device is 0: it must switch from the start.
dev = tack.device("hip:1")
check("switching", hip_backend._switching)
x = tack.array(np.arange(4, dtype=np.float32), device=dev)
check("reads back", x.numpy().tolist() == [0.0, 1.0, 2.0, 3.0])
check("the thread stays on device 0", fake.current() == 0)
check("every call on its own device: " + repr(fake.wrong), not fake.wrong)
'''


def _run(fake, checks):
    script = "\n".join(textwrap.dedent(part) for part in (fake, _COMMON, checks))
    env = dict(os.environ, TACK_CACHE="off")
    env.pop("TACK_NO_REINIT", None)
    result = subprocess.run([sys.executable, "-c", script], capture_output=True,
                            text=True, timeout=300, env=env)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("checks", [_CUDA_CHECKS, _CUDA_ADOPTS, _CUDA_SECOND_ONLY],
                         ids=["two-devices", "adopts-a-framework-context", "second-only"])
def test_cuda_works_in_each_devices_context(checks):
    _run(_FAKE_CUDA, checks)


@pytest.mark.parametrize("checks", [_HIP_CHECKS, _HIP_SECOND_ONLY],
                         ids=["two-devices", "second-only"])
def test_hip_works_on_each_device(checks):
    _run(_FAKE_HIP, checks)
