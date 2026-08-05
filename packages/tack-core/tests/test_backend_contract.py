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
