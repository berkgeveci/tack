"""The contract every Tack backend implements.

Backends were duck-typed: the contract had to be reconstructed by reading
five files, and callers asked what a backend could do by probing for
methods —

    if hasattr(backend, 'reduce_array'):     # does it reduce on device?
    if hasattr(backend, 'memory_space'):     # can it classify a pointer?
    if "CUDA" in type(backend).__name__:     # which backend is this?

That answers "is this method defined" rather than "is this supported",
which is not the same question and drifted apart. `supports_f64` existed
on Level Zero alone, so `getattr(backend, 'supports_f64', True)` reported
True for Metal, which has no f64 at all — and three test files worked
around it by comparing the arch name to the string "metal". Meanwhile
`memory_space()` was missing on Metal and Level Zero, so `tack.memory_space()`
silently answered `"cpu"` for a Level Zero device pointer.

Capabilities are now declared. A backend states what it supports; callers
read the attribute. Anything derivable is derived, so there is one source
of truth: `supports_f64` comes from `supported_dtypes`, and nothing else
needs to be kept in step with it.

This class deliberately does not force `execute()` into a template method.
Backends already share that path through `kernel_utils.resolve_variant()`;
what was missing was the declared contract around it.
"""

from tack.lang.types import ScalarType, f64


class Backend:
    """Base class for Tack compute backends.

    Subclasses declare their capabilities as class attributes (or set them
    in `__init__` when they are device-dependent, as Level Zero does for
    f64) and implement the methods below.
    """

    # ── Identity ─────────────────────────────────────────────────────

    #: Device type, as a device spec writes it (`tack.device("cuda")`).
    name: str = "unknown"

    #: Which of the type's devices this backend drives: N in ``"cuda:N"``,
    #: the device's number on its runtime (CUDA's device ordinal, HIP's
    #: device id, the index among the Level Zero driver's devices).
    ordinal: int = 0

    #: Whether the type has more than one device to reach: the constructor
    #: then takes ``ordinal=`` (None for `current_ordinal`'s), and
    #: `device_count` says how many there are.
    multi_device: bool = False

    @classmethod
    def device_count(cls) -> int:
        """How many devices of this type the machine has."""
        return 1

    @classmethod
    def current_ordinal(cls) -> int:
        """The device a bare spec names (``"cuda"``): the one the calling
        thread already uses, as another library may have chosen it."""
        return 0

    #: How the backend is written in messages meant for people. Defaults to
    #: `name`; backends whose arch id is not how you would spell it in prose
    #: (metal → Metal, level_zero → Level Zero) override it.
    display_name: str = ""

    # ── Capabilities ─────────────────────────────────────────────────

    #: Scalar types this backend accepts as array dtypes. Dispatch checks
    #: every array argument against this set.
    supported_dtypes: frozenset[ScalarType] = frozenset()

    #: True when `reduce_array()` runs reductions on the device. When
    #: False, `Array.sum()`/`min()`/`max()` fall back to numpy on the host.
    supports_device_reductions: bool = False

    #: Native workgroup execution for shared memory, barriers, local thread
    #: IDs and block reductions. This does not promise uniform participation,
    #: supported atomic types/scopes, or safe partial-workgroup execution.
    supports_workgroups: bool = False

    #: Keyword options `tack.device()` forwards to this backend's constructor.
    #: Anything else is rejected there, so a misspelt or misdirected option
    #: cannot be silently dropped.
    device_options: frozenset[str] = frozenset()

    #: Memory-space names (as returned by `memory_space()`) that a pointer
    #: must be in for `array_from_ptr()` to wrap it. Empty means this
    #: backend does not distinguish, so no check is made.
    device_memory_spaces: frozenset[str] = frozenset()

    #: Appended to the error when `from_dlpack` refuses a tensor this
    #: backend cannot wrap, for backends whose reason the DLPack device
    #: type does not convey. Metal sets it: its own memory is
    #: host-addressable, so "cannot address it" is the wrong summary and
    #: would send a reader looking for the wrong problem.
    dlpack_refusal_note: str = ""

    @property
    def supported_atomic_dtypes(self) -> frozenset[ScalarType]:
        """Global array types supporting add/min/max on this target."""
        from tack.lang.atomic_support import ATOMIC_DTYPES
        return ATOMIC_DTYPES.get(self.name, frozenset()) & self.supported_dtypes

    @property
    def supports_f64(self) -> bool:
        """Whether kernels can be dispatched with f64 arrays.

        Derived, so it cannot disagree with `supported_dtypes`.
        """
        return f64 in self.supported_dtypes

    @property
    def memory_domain(self) -> tuple:
        """What an array's memory is valid in: two backends with one domain can use
        each other's arrays (two CPU backends with different thread counts,
        say). CUDA and Level Zero add their context, since a pointer belongs
        to one."""
        return (self.name, self.ordinal)

    @property
    def label(self) -> str:
        """The backend's name as it should appear in a message."""
        return self.display_name or self.name

    # ── Required of every backend ────────────────────────────────────

    def allocate_array(self, dtype: ScalarType, shape: tuple[int, ...],
                       exportable: bool = False):
        """Allocate backend storage for an array, returning a DeviceBuffer."""
        raise NotImplementedError

    def wrap_ptr(self, ptr, dtype: ScalarType, shape: tuple[int, ...]):
        """Wrap existing memory as a DeviceBuffer without copying."""
        raise NotImplementedError

    def fill_array(self, array, value) -> bool:
        """Fill an array with one scalar the backend's own way; false to let
        `Array.fill` fill the buffer in place.

        The CPU backend runs a large fill as a kernel through its dispatch,
        so the threads first-touch the pages. GPU buffers fill on the
        device through `DeviceBuffer.fill` and need nothing here.
        """
        return False

    def execute(self, kernel, args, kwargs):
        """Compile (or reuse) and run a kernel with these arguments."""
        raise NotImplementedError

    # ── Optional, with honest defaults ───────────────────────────────

    def synchronize(self):
        """Wait until every kernel launched so far has finished.

        Backends whose launches complete before ``execute`` returns need not
        override it. Metal queues launches and waits only when the host
        touches array memory, so ``tack.sync()`` is how a caller timing
        kernels, or handing memory to a library Tack cannot see, waits.
        """

    def async_marker(self):
        """A mark after every launch queued so far that another thread can
        wait on -- an object with a blocking ``wait()``, which raises a
        launch's error as `synchronize` would -- or None.

        `tack.sync_async` waits on it in a worker thread so an ``asyncio``
        loop keeps running. None, the default, means there is no such mark,
        and the coroutine calls `synchronize` inline: right for a backend
        whose launches complete before returning, and still correct, if
        blocking, for one that queues launches and has not added a mark.
        """
        return

    def memory_space(self, ptr) -> str:
        """Classify where an integer pointer lives.

        The default says host memory, which is correct for backends whose
        allocations are CPU-addressable. Backends with distinct device
        memory override this and list the results in
        `device_memory_spaces`.
        """
        return "cpu"

    def _store_texture_shapes(self, ir_func, effective_args):
        """Record Texture3D extents on the params, for codegen and the key.

        HIP and Level Zero override this to fall back to software sampling
        where the device has no texture hardware. `tack.inspect` calls it
        too, so the source it shows is the source dispatch would compile.
        """
        from tack.runtime.kernel_utils import _store_texture_shapes
        _store_texture_shapes(ir_func, effective_args)

    def cache_target(self):
        """What compiled code for this backend depends on besides the kernel, as a
        string (the device, the compiler's version), or None if this backend
        does not store compiled variants on disk. See ``disk_cache``."""

    def variant_artifact(self, payload):
        """What of a compiled variant's payload to store on disk: picklable data
        from which ``load_variant`` rebuilds it in another process."""
        raise NotImplementedError

    def load_variant(self, ir_func, artifact, effective_args):
        """The payload of a variant, from what ``variant_artifact`` stored.

        ``effective_args`` are the call's arguments, for what the payload
        holds per process rather than on disk: the GPU backends' scalar
        pack buffers. CPU and Metal ignore them."""
        raise NotImplementedError

    def reduce_array(self, array, op: str) -> float:
        """Reduce an array on the device. Only called when
        `supports_device_reductions` is True."""
        raise NotImplementedError(
            f"{self.label} backend does not implement device reductions")

    def texture_in_hardware(self, shape_3d) -> bool:
        """Whether a `Texture3D` of this extent is sampled by texture units.

        False means the texture keeps a private array and the generated
        code interpolates it in software, which is what CPU does. `Texture3D`
        asks when choosing its storage, and backends whose answer varies by
        device or extent ask again when preparing a variant, so a texture
        always holds the storage its compiled kernel binds.
        """
        return False

    def create_texture_image(self, shape_3d):
        """Allocate a single-channel f32 3D texture image.

        Only called when `texture_in_hardware()` is True. The result is
        owned by one `Texture3D`: it exposes ``upload(array)``, which copies
        the array into it, and frees the device objects when collected.
        """
        raise NotImplementedError(
            f"{self.label} backend has no hardware textures")
