"""Every backend declares the same capabilities, honestly.

The point of the Backend base class is that callers ask what a backend
supports instead of probing for methods. That only works if the answers
are true, and if they cannot drift out of step with each other — so these
tests check the declarations against the behaviour they describe.
"""

import pytest

import tack
from tack.lang.types import f32, f64
from tack.runtime.backend import Backend

ALL_BACKEND_CLASSES = []
for _mod, _cls in [("cpu", "CPUBackend"), ("metal", "MetalBackend"),
                   ("cuda_backend", "CUDABackend"), ("hip_backend", "HIPBackend"),
                   ("level_zero_backend", "LevelZeroBackend")]:
    try:
        _m = __import__(f"tack.runtime.{_mod}", fromlist=[_cls])
        ALL_BACKEND_CLASSES.append(getattr(_m, _cls))
    except ImportError:
        pass   # device bindings not installed here


# ── Declarations on the classes (no device needed) ───────────────────

@pytest.mark.parametrize("cls", ALL_BACKEND_CLASSES,
                         ids=lambda c: c.__name__)
def test_backend_subclasses_the_base(cls):
    assert issubclass(cls, Backend)


@pytest.mark.parametrize("cls", ALL_BACKEND_CLASSES,
                         ids=lambda c: c.__name__)
def test_arch_name_is_declared(cls):
    assert cls.name != Backend.name, f"{cls.__name__} did not declare a name"
    assert cls.name == cls.name.lower()
    assert hasattr(tack, cls.name), \
        f"'{cls.name}' is not a tack.init(arch=...) value"


@pytest.mark.parametrize("cls", ALL_BACKEND_CLASSES,
                         ids=lambda c: c.__name__)
def test_supported_dtypes_are_declared(cls):
    assert cls.supported_dtypes, f"{cls.__name__} declared no dtypes"
    assert f32 in cls.supported_dtypes, "every backend should handle f32"


def test_arch_names_are_unique():
    names = [c.name for c in ALL_BACKEND_CLASSES]
    assert len(names) == len(set(names))


# ── Declarations against reality (needs the device) ──────────────────

def test_supports_f64_matches_supported_dtypes(backend):
    """One source of truth — the property is derived, not duplicated."""
    from tack.runtime.dispatch import get_backend
    be = get_backend()
    assert be.supports_f64 == (f64 in be.supported_dtypes)


def test_f64_dispatch_agrees_with_the_declaration(backend):
    """Declaring f64 support means an f64 kernel actually runs.

    This is the check that would have caught Metal reporting True: the
    old `getattr(backend, 'supports_f64', True)` said yes and dispatch
    said no.
    """
    import numpy as np

    from tack.runtime.dispatch import get_backend
    be = get_backend()

    @tack.kernel
    def double_it(x, out, n):
        for i in range(n):
            out[i] = x[i] * 2.0

    x = tack.field(dtype=tack.f64, shape=(4,)) if be.supports_f64 else None
    if be.supports_f64:
        out = tack.field(dtype=tack.f64, shape=(4,))
        x.from_numpy(np.arange(4, dtype=np.float64))
        double_it(x, out, 4)
        np.testing.assert_allclose(out.to_numpy(),
                                   np.arange(4) * 2.0, rtol=1e-12)
    else:
        with pytest.raises(TypeError, match="f64"):
            x = tack.field(dtype=tack.f64, shape=(4,))
            out = tack.field(dtype=tack.f64, shape=(4,))
            double_it(x, out, 4)


def test_label_is_human_readable(backend):
    from tack.runtime.dispatch import get_backend
    be = get_backend()
    assert be.label
    assert "_" not in be.label, "label is for prose, not an identifier"


def test_reductions_declaration_matches_behaviour(backend):
    """Reductions give the right answer either way; the flag says how."""
    import numpy as np

    from tack.runtime.dispatch import get_backend
    be = get_backend()

    f = tack.field(dtype=tack.f32, shape=(8,))
    f.from_numpy(np.arange(8, dtype=np.float32))
    assert f.sum() == pytest.approx(28.0)
    assert f.min() == pytest.approx(0.0)
    assert f.max() == pytest.approx(7.0)

    if not be.supports_device_reductions:
        # The base class raises rather than silently doing something else.
        with pytest.raises(NotImplementedError, match="device reductions"):
            Backend.reduce_field(be, f, 'sum')


def test_memory_space_is_answered_not_guessed(backend):
    """Every backend answers; it is no longer a missing-method default."""
    from tack.runtime.dispatch import get_backend
    be = get_backend()
    space = be.memory_space(0)
    assert isinstance(space, str) and space


def test_device_memory_spaces_are_self_consistent(backend):
    """A backend that validates pointers must be able to classify them."""
    from tack.runtime.dispatch import get_backend
    be = get_backend()
    if be.device_memory_spaces:
        assert type(be).memory_space is not Backend.memory_space, \
            f"{be.name} lists device memory spaces but inherits the default"


def test_field_from_ptr_checks_every_form_of_address(backend):
    """The memory-space check used to run for a Python int alone.

    A host address handed over as a numpy integer, or as the driver's own
    pointer type, skipped it and was wrapped as device memory.
    """
    import numpy as np

    from tack.runtime.dispatch import get_backend
    be = get_backend()
    if not be.device_memory_spaces:
        pytest.skip(f"{be.label} does not distinguish device memory")

    host = np.zeros(8, dtype=np.float32)
    for ptr in (host.ctypes.data, np.uint64(host.ctypes.data)):
        with pytest.raises(ValueError, match="requires a device"):
            tack.field_from_ptr(ptr, tack.f32, (8,))
    with pytest.raises(TypeError, match="device address"):
        tack.field_from_ptr("not a pointer", tack.f32, (8,))

    device = tack.field(dtype=tack.f32, shape=(8,))
    device.from_numpy(np.arange(8, dtype=np.float32))
    alias = tack.field_from_ptr(np.uint64(device._buffer.address), tack.f32, (8,))
    np.testing.assert_array_equal(alias.to_numpy(), np.arange(8))


def test_numpy_integer_alias_copies_reach_the_device_memory(backend):
    """Copies through a pointer given as a numpy integer reach that memory.

    hip-python reads a numpy scalar through the buffer protocol, as the
    address of the scalar's own storage. HIP stored the scalar as given,
    so every copy through the alias failed with hipErrorInvalidValue.
    Write through one handle and read through the other: equal values
    alone cannot tell an alias from a copy.

    A signed integer names only the lower half of the address space, and
    `as_address` refuses negatives, so np.int64 is tried only where the
    address fits. Level Zero's device addresses on a Max 1100 start at
    0xff00..., past 2^63, where np.int64 itself raises OverflowError.
    """
    import numpy as np

    from tack.runtime.dispatch import get_backend
    if not get_backend().device_memory_spaces:
        pytest.skip(f"{get_backend().label} does not distinguish device memory")

    device = tack.field(dtype=tack.f32, shape=(8,))
    address = device._buffer.address
    ptr_types = [np.uint64] + ([np.int64] if address < 1 << 63 else [])
    for ptr_type in ptr_types:
        alias = tack.field_from_ptr(ptr_type(address), tack.f32, (8,), writable=True)
        alias.from_numpy(np.arange(8, dtype=np.float32) + 1)
        np.testing.assert_array_equal(device.to_numpy(), np.arange(8) + 1)
        alias.fill(3.0)
        np.testing.assert_array_equal(device.to_numpy(), np.full(8, 3.0))
        device.from_numpy(np.arange(8, dtype=np.float32) * 2)
        np.testing.assert_array_equal(alias.to_numpy(), np.arange(8) * 2)


# ── What counts as an address ────────────────────────────────────────
#
# D9: HIP's memory_space() answered "cpu" for every pointer it was ever
# given, because two binding mistakes landed in a handler wide enough to
# absorb them. The handler was wide because it had to cover "not an
# address" as well, and those two questions want separating -- a handler
# that cannot tell "not a pointer" from "I called this wrong" should not
# answer either confidently.
#
# `as_address` is that separation, so it is worth pinning directly: it is
# the reason both GPU backends can now make their API call outside a
# `try`, where a mistaken one is a traceback.

@pytest.mark.parametrize("value,expected", [
    (0, 0),
    (4096, 4096),
    ((1 << 64) - 1, (1 << 64) - 1),      # the last address there is
])
def test_addresses_pass_through(value, expected):
    from tack.runtime.kernel_utils import as_address
    assert as_address(value) == expected


@pytest.mark.parametrize("value", [
    "not a pointer",
    None,
    object(),                            # Metal hands MTLBuffer objects around
])
def test_things_that_are_not_integers_are_refused(value):
    from tack.runtime.kernel_utils import as_address
    assert as_address(value) is None


@pytest.mark.parametrize("value", [
    1 << 64,                             # one past the last address
    1 << 200,                            # the case that motivated the range check
    -1,
])
def test_integers_too_big_or_negative_to_be_addresses_are_refused(value):
    """The half `int()` alone cannot see.

    `int(1 << 200)` succeeds -- Python integers are unbounded -- and the
    bindings then refuse it from inside their own marshalling, raising
    after the API call rather than before it. Caught, that became "cpu";
    uncaught, it is an OverflowError from somewhere the caller has no
    context for. Asking here makes it neither.
    """
    from tack.runtime.kernel_utils import as_address
    assert as_address(value) is None


# ── Initialization options and environment ───────────────────────────

def test_cpu_declares_num_threads():
    """CPUBackend takes num_threads; tack.init used to refuse to pass it."""
    from tack.runtime.dispatch import get_backend
    try:
        tack.init(arch=tack.cpu, num_threads=3)
        assert get_backend().num_threads == 3
    finally:
        tack.init(arch=tack.cpu)


@pytest.mark.parametrize("policy", ["v1", "v2"])
def test_cpu_policy_accepts_its_values(monkeypatch, policy):
    from tack.runtime.cpu import CPUBackend
    monkeypatch.setenv("TACK_CPU_POLICY", policy)
    assert CPUBackend(num_threads=2).policy == policy


@pytest.mark.parametrize("policy", ["V2", "v3", "2"])
def test_cpu_policy_refuses_anything_else(monkeypatch, policy):
    """Any value but "v2" used to select v1 without a word."""
    from tack.runtime.cpu import CPUBackend
    monkeypatch.setenv("TACK_CPU_POLICY", policy)
    with pytest.raises(ValueError, match="Accepted: v1, v2"):
        CPUBackend(num_threads=2)


@pytest.mark.parametrize("value,keeps", [
    ("1", True), ("yes", True), ("0", False), ("false", False),
    ("off", False), ("", False),
])
def test_no_reinit_reads_as_a_flag(monkeypatch, value, keeps):
    """`TACK_NO_REINIT=0` used to count as set."""
    from tack.runtime.dispatch import get_backend
    tack.init(arch=tack.cpu)
    before = get_backend()
    monkeypatch.setenv("TACK_NO_REINIT", value)
    tack.init(arch=tack.cpu)
    assert (get_backend() is before) == keeps
