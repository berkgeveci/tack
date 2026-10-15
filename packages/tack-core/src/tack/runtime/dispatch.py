"""Tack runtime dispatch — devices, the default device, and where a launch runs.

A device is a value: ``tack.device("cuda")`` names one backend, made on first
use and the same object afterwards, so its compiled kernels and its arrays
are shared by everything that asks for it. Arrays remember their device, and
a kernel runs where its arrays are.
"""

import functools
import os
import platform
import threading
import weakref

_current_backend = None
# Debug mode (bounds checks, see lang/ir_bounds.py): `set_debug`, or
# TACK_DEBUG when Tack is imported.
_debug = os.environ.get("TACK_DEBUG", "").strip().lower() not in ("", "0", "false", "no", "off")

# What to actually do when a backend will not start. Two of these want more
# than the extra's name, for opposite reasons: [hip] installs hip-python but
# cannot know which ROCm the machine has, and [level_zero] installs nothing
# at all because its dependencies are system libraries.
#
# This text has been wrong once already. It used to route people to Test
# PyPI, which was the only place hip-python was published; the package moved
# to PyPI proper, the extra started declaring it, and the message stayed
# behind — pointing at the wrong index, for a version the extra deliberately
# does not pin. It prints only where there is no ROCm and somebody asks for
# HIP, so nothing catches it going stale except reading it.
_BACKEND_HELP = {
    "cpu": "Requires llvmlite:  pip install 'tack-core[cpu]'",
    "metal": ("Requires macOS on Apple Silicon:  "
              "pip install 'tack-core[metal]'"),
    "cuda": ("Requires an NVIDIA GPU and the CUDA toolkit:  "
             "pip install 'tack-core[cuda]'"),
    "hip": (
        "Requires an AMD GPU with ROCm, plus hip-python:\n"
        "    pip install 'tack-core[hip]'\n"
        "  hip-python ships manylinux x86_64 wheels only, and its version\n"
        "  tracks the ROCm release it binds to — 7.1.x against ROCm 7.1,\n"
        "  7.2.x against 7.2. The extra sets a lower bound rather than a\n"
        "  pin, so a system on a different ROCm wants it spelled out:\n"
        "    pip install 'hip-python~=7.2.0'"
    ),
    "level_zero": (
        "Requires an Intel GPU with the Level Zero runtime.\n"
        "  These are system libraries, not Python packages, so the\n"
        "  [level_zero] extra has nothing to install: you need\n"
        "  libze_loader.so (level-zero runtime) and libocloc.so\n"
        "  (intel-opencl-icd)."
    ),
}


def env_flag(name: str) -> bool:
    """Whether a boolean environment variable is switched on.

    Unset, empty, ``0``, ``false``, ``no`` and ``off`` all mean off, so
    ``TACK_NO_REINIT=0`` disables the option rather than enabling it.
    """
    import os
    value = os.environ.get(name, "").strip().lower()
    return value not in ("", "0", "false", "no", "off")


_CONSTRUCTORS = {
    "cpu": ("tack.runtime.cpu", "CPUBackend"),
    "metal": ("tack.runtime.metal", "MetalBackend"),
    "cuda": ("tack.runtime.cuda_backend", "CUDABackend"),
    "hip": ("tack.runtime.hip_backend", "HIPBackend"),
    "level_zero": ("tack.runtime.level_zero_backend", "LevelZeroBackend"),
}
# Device types that number their devices in a spec ("cuda:0"); the others
# have one device and are written bare ("cpu", "metal").
_NUMBERED = frozenset({"cuda", "hip", "level_zero"})

# One backend per (type, index) made without options, the same object on
# every request. A backend made with options (a CPU thread count, a Level
# Zero context owned by another library) is its own.
_devices = {}
_live = weakref.WeakSet()       # every backend made, for sync()
_lock = threading.RLock()
_scope = threading.local()      # `with tack.device(...)` stack, per thread
_open_scopes = 0                # blocks open on any thread: none, no lookup


class Device:
    """A device: where arrays live and kernels run. ``tack.device(...)`` makes one.

    ``type`` is ``"cpu"``, ``"metal"``, ``"cuda"``, ``"hip"`` or
    ``"level_zero"``; ``index`` numbers the devices of a type. Two devices are
    equal when they are the same backend. ``with device:`` makes it the
    default device inside the block, on this thread.
    """

    __slots__ = ("__weakref__", "_backend")

    def __init__(self, backend):
        self._backend = backend

    @property
    def type(self) -> str:
        return self._backend.name

    @property
    def index(self) -> int:
        return self._backend.ordinal

    def synchronize(self):
        """Wait for every kernel launched on this device so far."""
        self._backend.synchronize()

    def __enter__(self):
        global _open_scopes
        _scope.__dict__.setdefault("stack", []).append(self._backend)
        with _lock:
            _open_scopes += 1
        return self

    def __exit__(self, *exc):
        global _open_scopes
        _scope.stack.pop()
        with _lock:
            _open_scopes -= 1
        return False

    def __eq__(self, other):
        return isinstance(other, Device) and other._backend is self._backend

    def __hash__(self):
        return id(self._backend)

    def __str__(self):
        return f"{self.type}:{self.index}" if self.type in _NUMBERED else self.type

    def __repr__(self):
        return f"Device('{self}')"


def as_device(backend) -> Device:
    """The `Device` for ``backend``, one per backend."""
    found = backend.__dict__.get("_public_device")
    if found is None:
        found = backend.__dict__["_public_device"] = Device(backend)
    return found


def _parse(spec):
    """``(type, index)`` from a device spec: ``"cuda"``, ``"cuda:1"``, ``"cpu"``."""
    if not isinstance(spec, str):
        raise TypeError(f"a device is a Device or a string such as 'cpu' or 'cuda:0', "
                        f"not {spec!r}")
    kind, _, number = spec.partition(":")
    if kind not in _CONSTRUCTORS:
        available = ", ".join(sorted(_CONSTRUCTORS))
        raise ValueError(f"Unknown device type: '{kind}'. Available: {available}")
    if not number:
        return kind, 0
    if kind not in _NUMBERED or not number.isdigit():
        raise ValueError(f"'{spec}' is not a device: write '{kind}'"
                         + (f" or '{kind}:N'" if kind in _NUMBERED else ""))
    return kind, int(number)


def device(spec="cpu", **options) -> Device:
    """The device ``spec`` names: ``"cpu"``, ``"metal"``, ``"cuda"``, ``"cuda:0"``,
    ``"hip"``, ``"level_zero"``; a `Device` is returned as it is.

    Without options it is made on first use and the same device afterwards,
    so its compiled kernels and arrays are shared by everything that asks.
    Options go to the backend and must be ones it declares in
    `Backend.device_options`; they make a new device each call. The CPU
    takes ``num_threads``; Level Zero takes ``external_context``, the
    driver, device and context handles of a context owned by another
    library, so both can address the same memory. Only device 0 of each
    type is reachable for now.

    ``with tack.device("cuda"):`` makes it the default device in the block.
    """
    if isinstance(spec, Device):
        if options:
            raise TypeError("options make a new device: pass a device spec, not a Device")
        return spec
    kind, index = _parse(spec)
    if index != 0:
        raise ValueError(f"device {index} of '{kind}': Tack reaches device 0 only, for now")
    if options:
        return as_device(_construct(kind, options))
    with _lock:
        found = _devices.get((kind, index))
        if found is None:
            found = _devices[(kind, index)] = _construct(kind, {})
        return as_device(found)


def _construct(kind, options):
    module_name, class_name = _CONSTRUCTORS[kind]
    help_msg = _BACKEND_HELP[kind]
    try:
        import importlib
        mod = importlib.import_module(module_name)
        cls = getattr(mod, class_name)
        unknown = sorted(set(options) - cls.device_options)
        if unknown:
            accepted = ", ".join(sorted(cls.device_options)) or "none"
            raise ValueError(
                f"The '{kind}' device does not accept the option(s) "
                f"{', '.join(unknown)}. Accepted: {accepted}.")
        backend = cls(**options)
    except ImportError as e:
        raise RuntimeError(
            f"Cannot start the '{kind}' device: missing dependency.\n"
            f"  {e}\n"
            f"  {help_msg}"
        ) from e
    except RuntimeError as e:
        raise RuntimeError(
            f"Cannot start the '{kind}' device on {platform.system()}.\n"
            f"  {e}\n"
            f"  {help_msg}"
        ) from e
    _live.add(backend)
    return backend


def backend_of(device_or_spec):
    """The backend behind a `Device`, a spec, or None (the default device)."""
    if device_or_spec is None:
        return get_backend()
    return device(device_or_spec)._backend


def set_default_device(spec="cpu", **options) -> Device:
    """Make ``spec``'s device the default: where arrays are allocated, and where
    kernels without arrays run. Returns it.

    The device is ``tack.device(spec, **options)``: without options the same
    one as before, so arrays and compiled kernels survive another call.

    Set ``TACK_NO_REINIT=1`` to leave the default alone once one is set
    (useful when embedded in an ANARI device that shares the process).
    """
    global _current_backend
    if _current_backend is not None and env_flag("TACK_NO_REINIT"):
        return as_device(_current_backend)
    _current_backend = device(spec, **options)._backend
    return as_device(_current_backend)


def get_default_device() -> Device:
    """The default device: the innermost ``with tack.device(...)`` block's, else
    the one `set_default_device` chose, else the CPU."""
    return as_device(get_backend())


def set_debug(enabled: bool = True):
    """Check every array index in every kernel launched from now on.

    A launch that went out of bounds raises ``IndexError`` naming the access;
    the access itself is skipped. Kernels compile separately in debug mode
    and run slower; each launch waits to finish. ``TACK_DEBUG=1`` turns it on
    when Tack is imported.
    """
    global _debug
    _debug = bool(enabled)


def debug_enabled() -> bool:
    """Whether debug mode's bounds checks are on (`set_debug`)."""
    return _debug


def on_inputs_device(func):
    """Run ``func`` with its first array argument's device as the default.

    For library functions that allocate temporaries and launch kernels on
    their inputs (scans, sorts): both then happen where the inputs are.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        dev = next((d for d in (getattr(a, "_device", None) for a in args) if d is not None),
                   None)
        if dev is None:
            return func(*args, **kwargs)
        with as_device(dev):
            return func(*args, **kwargs)
    return wrapper


_SCALARS = frozenset((int, float, bool))


def launch_backend(name, args):
    """The backend a launch of kernel ``name`` with ``args`` runs on.

    Its arrays must all be in one device's memory. The default device runs
    it when it can address them (or there are none), so a default CPU
    device made with ``num_threads=1`` still governs CPU arrays made before
    it; else the arrays' own device does. A device class's arrays count
    when no plain array decides; dispatch checks them all (`check_devices`).
    """
    default = get_backend()
    found = None
    templates = False
    for arg in args:
        if type(arg) in _SCALARS:
            continue                              # cheaper than a failed getattr
        dev = getattr(arg, "_device", None)       # an Array's or a Texture3D's
        if dev is None:
            templates = templates or getattr(arg, "_device_class", False)
            continue
        if found is None:
            found = dev
        elif dev is not found and dev.memory_domain != found.memory_domain:
            raise ValueError(_two_devices(name, found, dev))
    if found is None and templates:
        found = _template_device(args)
    if found is None or found is default or found.memory_domain == default.memory_domain:
        return default
    return found


def check_devices(name, backend, args):
    """Raise if an array among ``args`` (template arrays expanded) is in another
    device's memory than ``backend``'s."""
    for arg in args:
        dev = getattr(arg, "_device", None)
        if dev is not None and dev is not backend and dev.memory_domain != backend.memory_domain:
            raise ValueError(_two_devices(name, backend, dev))


def _template_device(args):
    from tack.lang.template_rewrite import classify_template_attrs
    for arg in args:
        if getattr(arg, "_device_class", False):
            for array in classify_template_attrs(arg)[1].values():
                dev = getattr(array, "_device", None)
                if dev is not None:
                    return dev
    return None


def _two_devices(name, first, second):
    return (f"Kernel '{name}' was given arrays on two devices ({as_device(first)} and "
            f"{as_device(second)}); a launch runs on one. Copy them to one device first "
            f"(a.to_device(...)).")


def sync():
    """Wait until every kernel launched so far, on every device, has finished.

    Reading or writing an array from the host (``numpy``, ``tack.copyto``,
    indexing, reductions, DLPack export) already waits for the kernels
    before it, so most code never needs this. It is for timing kernels, and
    for memory shared with another library that reads it without Tack.
    """
    for backend in list(_live):
        backend.synchronize()


async def sync_async():
    """`sync` for ``asyncio``: ``await tack.sync_async()`` waits for every
    kernel launched so far on the default device without blocking the event
    loop.

    On CUDA, HIP, Metal and Level Zero the wait runs in a worker thread and
    the loop goes on; launches made after the call are not waited for. A
    launch's error is raised here. On the CPU, whose launches complete
    before returning, it returns at once. See `tack.runtime.aio`.
    """
    from tack.runtime.aio import sync_async as _sync_async
    await _sync_async(get_backend())


def get_backend():
    """The default device's backend: the innermost ``with tack.device(...)``
    block's, else the one `set_default_device` chose, starting the CPU if
    none was."""
    global _current_backend
    if _open_scopes:     # a failed thread-local lookup costs ~0.2 us a launch
        stack = _scope.__dict__.get("stack")
        if stack:
            return stack[-1]
    if _current_backend is None:
        _current_backend = device("cpu")._backend
    return _current_backend
