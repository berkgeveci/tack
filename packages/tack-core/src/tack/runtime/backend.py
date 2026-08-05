"""The contract every Tack backend implements.

Backends were duck-typed: the contract had to be reconstructed by reading
five files, and callers asked what a backend could do by probing for
methods —

    if hasattr(backend, 'reduce_field'):     # does it reduce on device?
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

    #: Arch identifier, matching `tack.init(arch=...)`.
    name: str = "unknown"

    #: How the backend is written in messages meant for people. Defaults to
    #: `name`; backends whose arch id is not how you would spell it in prose
    #: (metal → Metal, level_zero → Level Zero) override it.
    display_name: str = ""

    # ── Capabilities ─────────────────────────────────────────────────

    #: Scalar types this backend accepts as field dtypes. Dispatch checks
    #: every field argument against this set.
    supported_dtypes: frozenset[ScalarType] = frozenset()

    #: True when `reduce_field()` runs reductions on the device. When
    #: False, `Field.sum()`/`min()`/`max()` fall back to numpy on the host.
    supports_device_reductions: bool = False

    #: Memory-space names (as returned by `memory_space()`) that a pointer
    #: must be in for `field_from_ptr()` to wrap it. Empty means this
    #: backend does not distinguish, so no check is made.
    device_memory_spaces: frozenset[str] = frozenset()

    @property
    def supports_f64(self) -> bool:
        """Whether kernels can be dispatched with f64 fields.

        Derived, so it cannot disagree with `supported_dtypes`.
        """
        return f64 in self.supported_dtypes

    @property
    def label(self) -> str:
        """The backend's name as it should appear in a message."""
        return self.display_name or self.name

    # ── Required of every backend ────────────────────────────────────

    def allocate_field(self, dtype: ScalarType, shape: tuple[int, ...],
                       exportable: bool = False):
        """Allocate backend storage for a field, returning a DeviceBuffer."""
        raise NotImplementedError

    def wrap_ptr(self, ptr, dtype: ScalarType, shape: tuple[int, ...]):
        """Wrap existing memory as a DeviceBuffer without copying."""
        raise NotImplementedError

    def execute(self, kernel, args, kwargs):
        """Compile (or reuse) and run a kernel with these arguments."""
        raise NotImplementedError

    # ── Optional, with honest defaults ───────────────────────────────

    def memory_space(self, ptr) -> str:
        """Classify where an integer pointer lives.

        The default says host memory, which is correct for backends whose
        allocations are CPU-addressable. Backends with distinct device
        memory override this and list the results in
        `device_memory_spaces`.
        """
        return "cpu"

    def reduce_field(self, field, op: str) -> float:
        """Reduce a field on the device. Only called when
        `supports_device_reductions` is True."""
        raise NotImplementedError(
            f"{self.label} backend does not implement device reductions")
