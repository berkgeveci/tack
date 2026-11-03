# Choose a device

A Tack program runs the same kernels on the CPU and on GPUs from Apple,
NVIDIA, AMD and Intel. Once a first kernel works, the next question is where
it runs. You pick a device, make it the default, and move arrays between
devices. One program can also use several GPUs, ask a device what it
supports, and know when a kernel call has finished.

A *device* is one CPU or one GPU, as a Python value. Every array lives on one
device, and a kernel runs on the device its arrays live on:

<!-- tack-check: devices-first -->
```python
import numpy as np
import tack

tack.set_default_device("cpu")


@tack.kernel
def scale(x, out, factor):
    for i in range(x.shape[0]):
        out[i] = x[i] * factor


x = tack.arange(8, dtype=tack.f32)     # on the default device
out = tack.empty_like(x)               # on x's device
scale(x, out, 2.0)                     # runs where x and out are
assert str(x.device) == "cpu" and x.device == tack.get_default_device()
np.testing.assert_allclose(out.numpy(), np.arange(8) * 2.0)

# On a machine with a GPU, name it instead of the CPU:
#     tack.set_default_device("cuda")     # or "metal", "hip", "level_zero"
# and the same lines allocate on the GPU and run the kernel there.
```

The kernel's source does not change between devices. Tack compiles it for
whichever device runs it, the first time it runs there.

## Name a device

`tack.device(spec)` returns the device that `spec` names. The spec is a
string that names the kind of device and, for GPUs that come in numbers,
which one:

| spec | device | install extra |
|---|---|---|
| `"cpu"` | the CPU's cores, through LLVM | `cpu` |
| `"metal"` | the GPU of an Apple silicon Mac | `metal` |
| `"cuda"`, `"cuda:1"` | an NVIDIA GPU | `cuda` |
| `"hip"`, `"hip:1"` | an AMD GPU, through ROCm | `hip` |
| `"level_zero"`, `"level_zero:1"` | an Intel GPU | none: system libraries |
| `"warp"` | NVIDIA Warp's device (a prototype) | `warp` |

Each kind of device has its own *backend*: the code generator and runtime
that compile kernels for it. [Install Tack](../install.md) says what each
extra needs.

Asking for the same device twice gives the same device. Its arrays and the
kernels it has compiled are shared by everything that asks for it. Two
devices compare equal when they are the same device, and `str(device)` gives
its spec, such as `"cuda:0"`.

Options make a separate device. The CPU takes `num_threads`, so
`tack.device("cpu", num_threads=1)` is a second CPU device that runs every
kernel on the calling thread. It shares memory with the default CPU device,
so it can run kernels on CPU arrays made before it. An option a backend does
not accept raises `ValueError` naming the ones it does.

## Set the default device

The default device is where arrays are allocated when you do not say, and
where kernels with no array arguments run. Three calls control it:

- `tack.set_default_device("cuda")` makes a device the default for the whole
  process, and returns it. With nothing set, the default is the CPU.
- `tack.get_default_device()` returns the default device.
- `with tack.device("cuda"):` makes a device the default inside the block,
  on the current thread only. Blocks nest, and the previous default returns
  when the block ends.

Every array constructor also takes `device=`, as a device or a spec. Every
array records its device in `a.device`. The `*_like` constructors, such as
`tack.empty_like`, allocate on their prototype's device.

Changing the default invalidates nothing. Arrays made before the change keep
their device and keep working, and compiled kernels stay compiled.

<!-- tack-check: devices-default -->
```python
import numpy as np
import tack

tack.set_default_device("cpu")


@tack.kernel
def scale(x, out, factor):
    for i in range(x.shape[0]):
        out[i] = x[i] * factor


x = tack.arange(8, dtype=tack.f32)
out = tack.empty_like(x)

serial = tack.device("cpu", num_threads=1)
with serial:                        # the default inside the block
    y = tack.ones(8)                # allocated on serial
    scale(x, out, 3.0)              # serial can address CPU arrays: runs there
assert y.device == serial
assert tack.get_default_device() == tack.device("cpu")
assert x.to_device(serial) is x     # the same memory, so no copy
np.testing.assert_allclose(out.numpy(), np.arange(8) * 3.0)
```

## Know where a kernel runs

A kernel call runs on one device, chosen from its arguments:

1. If the default device can address every array argument, the kernel runs
   there. This is why a one-thread CPU device made the default still runs
   kernels on arrays from the ordinary CPU device.
2. Otherwise it runs on the arrays' own device. A CUDA array passed while
   the CPU is the default runs on its GPU.
3. Arrays on two devices that cannot address each other's memory raise
   `ValueError`. Move one first.

`a.to_device(d)` moves an array. It returns the array itself when `d` can
already address its memory, and otherwise a copy on `d`. Copies between two
GPUs go through the host.

The array methods follow the same rule. Copies, reductions such as `a.sum()`,
and the functions in `tack.algorithms` run on their input's device, whatever
the default is.

## Use several GPUs

On a machine with several GPUs of one kind, `tack.device_count("cuda")` says
how many there are, and `tack.device("cuda:1")` is the second one. The same
holds for `"hip"` and `"level_zero"`. The number is the one the vendor's
runtime uses, so it agrees with PyTorch's numbering and with DLPack's.
`device_count` makes no device, and it returns 0 where the backend cannot
start.

A bare `"cuda"` or `"hip"` means the GPU the calling thread already uses.
That is the one a framework embedding Tack has chosen, or else the first.

Each device compiles its own kernels, and the arrays of one kernel call must
all be on one device. To split one problem over several GPUs, give each GPU
its own arrays and launch on each. Tack's communication layer does this for
you, over GPUs and over MPI ranks. See
[Blocks and worlds](../communication/blocks-and-worlds.md).

## Check what a device supports

Every backend runs the same kernel language, within what its hardware can do.
A kernel that needs something a backend lacks is rejected with an error that
names the kernel and the missing capability. It never runs with a different
meaning.

| | CPU | Metal | CUDA | HIP | Level Zero | Warp (prototype) |
|---|---|---|---|---|---|---|
| `f64` arrays | yes | no | yes | yes | if the device has it | yes |
| Workgroups: shared memory, barriers, `thread_id`, block reductions | no | yes | yes | yes | yes | on a CUDA device |
| Atomic array types | all ten | `i32`, `u32`, `f32` | 32- and 64-bit integers, `f32`, `f64` | as CUDA | `i32`, `u32`, `f32` | as CUDA |
| `sum`, `min`, `max` on the device | no: NumPy | `f32` | `f32` | `f32` | `f32` | no: NumPy |
| 3D textures | in software | in hardware | in hardware | in hardware if the device has images | in hardware if the device has samplers | no |
| Iterations per launch | no limit | 2^32 | max grid × 256 | max grid × 256, at most 2^32 − 256 | max group count × group size | 2^31 − 1 |

A launch longer than its backend's limit raises `ValueError` naming the
kernel, the count and the limit. Split such work over several calls.

In code, ask the backend rather than checking its name. `backend_of` takes a
device, a spec, or `None` for the default device:

<!-- tack-check: devices-capabilities -->
```python
import tack
from tack.runtime.dispatch import backend_of

tack.set_default_device("cpu")
backend = backend_of(tack.get_default_device())
print(backend.name, backend.supports_f64, backend.supports_workgroups)
# cpu True False

assert backend.supports_f64 and not backend.supports_workgroups
assert tack.f64 in backend.supported_dtypes
assert tack.i64 in backend.supported_atomic_dtypes
```

Use this to choose between a GPU path and a portable one. The
[N-body tutorial](../users-guide/tutorials/nbody.md) runs a tiled kernel
where `supports_workgroups` is true and a plain one elsewhere. The
[backend capability contract](../contracts/backend-capabilities.md) has the
full matrix and what each capability rejects.

## Know when a call returns

On the CPU a kernel call returns when the kernel has finished. On Metal, CUDA,
HIP and Level Zero it returns as soon as the launch is queued, and the GPU
runs queued launches in the order they were made. Fills and writes from the
host take their place in that order too.

The host waits only when it needs a result. Reading an array waits for the
launches before it: `numpy()`, `np.asarray`, indexing one element, a
reduction such as `a.sum()`, and exporting through DLPack all wait. So does
`tack.sync()`, which waits for every device, and `device.synchronize()`,
which waits for one.

Most programs never call `tack.sync()`. You need it in two cases:

- **Timing.** Without it, a GPU time measures only the queuing. See
  [Debug and time a kernel](debug-and-time.md).
- **Memory another library reads without Tack.** Usually Tack handles this
  itself. See [Share arrays](share-arrays.md).

Because launches are queued, an error inside a launch is raised at the next
wait rather than at the call. In an `asyncio` program, the waits can run off
the event loop. See [Wait without blocking](asyncio.md).

## Control the CPU's threads

The CPU backend compiles each kernel to native code through LLVM and runs it
on a pool of worker threads, one per core. Spreading a launch over the
workers costs a few microseconds, so the backend measures each kernel's cost
per element and fans out only when the work is long enough to gain from it.
Short launches run on the calling thread. The
[CPU threading design note](../design/cpu-threading.md) explains how it
decides.

Two settings change the thread count:

- `TACK_CPU_THREADS=4` in the environment sets it for the process.
- `tack.device("cpu", num_threads=4)` makes a device with that many threads.

A count of 1 runs everything on the calling thread. That helps when
profiling, and when Tack is embedded in a program that manages its own
threads.

## Details

### Backend notes

**CPU.** The CPU has no workgroups, and Tack does not emulate them. A kernel
that uses shared memory, a barrier, `thread_id` or a block reduction is
rejected on the CPU before it compiles, even if that code would never run.
For scratch storage private to each iteration, use `tack.local_array`, which
works everywhere. See
[Use workgroups, atomics, textures and random numbers](workgroups-atomics-textures.md).

**Metal.** Apple silicon shares memory between the CPU and the GPU, but
`tack.array`, `tack.copyto` and `numpy()` still copy values. To share
storage, see [Share arrays](share-arrays.md). Metal has no `f64`, so an `f64`
array cannot be made on it, and a kernel that computes in `f64` is refused
with the reason.

**CUDA.** Tack works in the GPU's primary context, the one the CUDA runtime,
PyTorch, CuPy and Warp use. These libraries and Tack can therefore take turns
on one thread in any order, and pass arrays to each other through DLPack. If
an application already made another context current before Tack first used
the GPU, Tack adopts that one instead.

Tack makes the context current on the thread that first asks for the device.
Another thread that calls kernels needs the context current too. A thread on
which PyTorch, CuPy or the CUDA runtime already used the GPU has it.
Otherwise, in that thread, retain the primary context with
`cuDevicePrimaryCtxRetain` and make it current with `cuCtxSetCurrent`, both
from `cuda.bindings.driver`. Once a program uses two CUDA GPUs, Tack makes
the right context current around every operation itself.

**HIP.** ROCm 7.0.2 and 7.1.1 can miscompile some integer code. Their
compiler can evaluate a 64-bit signed comparison wrongly inside a long
integer expression, and the kernel then returns a wrong value without an
error. Use a ROCm release newer than 7.1 if you can. On 7.0 and 7.1, check
integer-heavy kernels against the CPU. Go by the ROCm release, not by
`hiprtcVersion()`, which reports 9.0 on 7.0.2.

AMD's CDNA GPUs, such as the MI300, have no texture hardware, and the MI200
cannot filter 3D textures linearly. On those, Tack samples 3D textures in
software.

**Level Zero.** Some Intel GPUs have `f64` and some do not. The backend asks
the device when it starts. Atomics stay limited to `i32`, `u32` and `f32`
even on a device with `f64`. Intel's Xe-HPC GPUs (Ponte Vecchio) have no
texture samplers, so Tack samples 3D textures in software there.

A Level Zero pointer means something only in the context that allocated it.
To share device arrays with VTK, start Tack with
`tack.interop.vtk.init_level_zero()` instead of
`tack.set_default_device("level_zero")`. It makes a device in VTK's context
the default and returns it.

**Warp (prototype).** `tack.device("warp")` runs Tack kernels through NVIDIA
Warp, on Warp's default device: its first CUDA GPU, else its CPU. Launches go
on Warp's current stream, so a `wp.ScopedStream` around Tack calls applies.
It lacks `tack.inspect`, 3D textures, device reductions, Tack's disk cache
of compiled code, and launches of more than 2^31 − 1 iterations. Its
interface may change.

### When a backend cannot start

If a backend's libraries are missing, `tack.device()` raises `RuntimeError`
saying what is missing and how to install it:

```text
RuntimeError: Cannot start the 'hip' device: missing dependency.
  No module named 'hip'
  Requires an AMD GPU with ROCm, plus hip-python; in a clone:
    pip install -e 'packages/tack-core[hip]'   (or: uv sync --extra hip)
```

If the device cannot be made, the previous default stays.

### Keeping an embedding program's default

Set `TACK_NO_REINIT=1` to make `set_default_device` keep the default once one
is set. This is for programs that embed Tack beside other code that would set
it again, such as an ANARI device sharing the process.

## Where to go next

- [Share arrays with NumPy, PyTorch, CuPy and VTK](share-arrays.md), to
  bring data in and out without copying.
- [Debug and time a kernel](debug-and-time.md), to measure what a device
  gains you.
- The [runtime API](../contracts/runtime-api.md#devices), for every argument
  and error of the device functions.
