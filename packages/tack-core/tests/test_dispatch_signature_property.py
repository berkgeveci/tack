"""The dispatch fast path's invariant, as a property: equal call signatures
mean the same variant and the same arguments.

A warm call skips variant resolution: `_fast_resolve` looks the call's
signature (`_call_signature`, `_template_signature`) up and reuses the
variant, and the expansion plan, that the first call with that signature
resolved. That is right only if everything the variant key and the
expansion read is in the signature. A kind of argument or attribute the
signature missed would hand one call another call's compiled code, or
leave an argument out, without an error.

`test_dispatch_fast_path.py` checks chosen cases. Here calls are drawn at
random from small pools of values -- arrays, Python and NumPy scalars,
device classes whose attributes are any of the kinds the key reads, nested
device classes, device functions, class constants changed between calls
-- so that many different calls share a signature. Each call is resolved
the full way (`_resolve_variant`), and the fast path's answer for it, from
whichever earlier call recorded its signature, must be the same variant
with the same expanded arguments. The seed is fixed, so a failure repeats.
"""

import random

import numpy as np
import pytest

import tack
from tack.lang.template_rewrite import template_scope
from tack.runtime import cpu as cpu_mod
from tack.runtime.dispatch import get_backend
from tack.runtime.kernel_utils import _call_signature, _fast_resolve, _resolve_variant

pytestmark = pytest.mark.quick


@tack.func
def _one(x):
    return x


@tack.func
def _two(x):
    return x + 1.0


@tack.device_class
class _Leaf:
    SCALE = 2.0


@tack.device_class
class _Node:
    LIMIT = 4


@tack.device_class
class _Other:
    pass


@tack.device_class
class _Grid:
    SHAPE = (2, 3)            # a class tuple of numbers: one constant per entry


@tack.kernel
def _touch(t, x, s):
    for i in range(x.shape[0]):
        x[i] = x[i]


_NAMES = ("a", "b", "c", "SCALE", "LIMIT", "SHAPE", "_hidden")


def _arrays():
    return [lambda: tack.zeros((4,)), lambda: tack.zeros((5,)),
            lambda: tack.zeros((4,), tack.i32), lambda: tack.zeros((4,), tack.f64),
            lambda: tack.zeros((2, 2)), lambda: tack.zeros((2, 3)),
            lambda: tack.zeros((4,), tack.vec3()), lambda: tack.zeros((4,), tack.vec2()),
            lambda: tack.zeros((4,), tack.mat2()), lambda: tack.zeros((4,), tack.vec4())]


_SCALARS = (1, 7, 2**40, -1, True, False, 1.0, 2.5, np.int32(1), np.int64(2**40),
            np.float32(1.0), np.float64(1.0), np.uint8(3))
_OTHERS = (None, "text", [1, 2], {}, ("x",))
# Tuples of numbers: runtime scalars per entry, or, replacing a class tuple
# such as _Grid.SHAPE, compiled-in constants; same widths, other values too.
_TUPLES = ((1, 2), (2, 3), (4, 4), (1.0, 2), (1, 2**40), (2, 3, 4))


def _value(rng, depth):
    kind = rng.choice(("array", "scalar", "scalar", "func", "child", "other", "tuple"))
    if kind == "array":
        return rng.choice(_arrays())()
    if kind == "scalar":
        return rng.choice(_SCALARS)
    if kind == "func":
        return rng.choice((_one, _two))
    if kind == "tuple":
        return rng.choice(_TUPLES)
    if kind == "child" and depth < 2:
        return _template(rng, depth + 1)
    return rng.choice(_OTHERS)


def _template(rng, depth=0):
    obj = rng.choice((_Leaf, _Node, _Other, _Grid))()
    for name in rng.sample(_NAMES, rng.randint(0, 3)):
        setattr(obj, name, _value(rng, depth))
    return obj


def _call(rng):
    s = rng.choice((*_SCALARS, 0.5))
    return (_template(rng), rng.choice(_arrays())(), s)


def _same(a, b):
    """Two expanded argument lists bind the same: arrays the same objects,
    scalars equal with the same type."""
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if isinstance(x, tack.Array) or isinstance(y, tack.Array):
            if x is not y:
                return False
        elif type(x) is not type(y) or x != y:
            return False
    return True


def _slow(backend, args):
    with template_scope():
        return _resolve_variant(backend, _touch, args, {}, backend._build_variant, None,
                                cpu_mod._SPECIALIZE_DISJOINT)


def test_equal_signatures_resolve_to_the_same_variant_and_arguments():
    from tack.runtime import dispatch
    if dispatch._debug:
        pytest.skip("debug mode never takes the fast path")
    tack.set_default_device("cpu")
    backend = get_backend()
    rng = random.Random(20261015)
    checked = 0
    seen = set()
    for step in range(1600):
        if step % 50 == 25:                   # a class constant changes between calls
            _Leaf.SCALE = rng.choice((2.0, 3.0))
            _Node.LIMIT = rng.choice((4, 5))
            _Grid.SHAPE = rng.choice(((2, 3), (3, 3)))
        args = _call(rng)
        sig = _call_signature(args)
        # As dispatch does: the fast path first, from what earlier calls
        # recorded -- the full path below records this call, and would
        # answer for its signature itself.
        hit = None
        if sig in seen:
            hit = _fast_resolve(backend, _touch, args, cpu_mod._SPECIALIZE_DISJOINT)
        try:
            variant, effective = _slow(backend, args)
        except (TypeError, ValueError, NotImplementedError):
            continue                          # a call dispatch refuses either way
        if sig is None:
            continue
        if hit is None:
            seen.add(sig)
            continue
        fast_variant, fast_effective = hit
        assert fast_variant is variant, (
            f"call {step}: the fast path gave another call's variant\n  args: {args!r}")
        assert _same(fast_effective, effective), (
            f"call {step}: the fast path expanded the arguments differently\n"
            f"  fast: {fast_effective!r}\n  full: {effective!r}")
        checked += 1
    # The pools are small so that calls share signatures; make sure enough
    # did for the check to mean something.
    assert checked > 100, checked


@tack.device_class
class _Scaled:
    SCALE = 2.0

    @tack.func
    def scale(self):
        return self.SCALE


@tack.kernel
def _fill_scale(t, x):
    for i in range(x.shape[0]):
        x[i] = t.scale()


def test_an_instance_attribute_that_is_not_a_number_drops_the_class_constant():
    """A class constant is a number. An instance attribute of the same name
    holding a number overrides it, still compiled in; one holding anything
    else drops it, as a subclass attribute does. Kept as a constant, a list
    made dispatch fail with "unhashable type: 'list'"."""
    tack.set_default_device("cpu")
    x = tack.zeros((2,))
    unread = _Leaf()
    unread.SCALE = [1, 2]
    _touch(unread, x, 1.0)                    # no method reads it: nothing to refuse
    listed = _Scaled()
    listed.SCALE = [1, 2]
    with pytest.raises(ValueError, match="self.SCALE"):
        _fill_scale(listed, x)                # read: no constant of that name
    _fill_scale(_Scaled(), x)
    np.testing.assert_array_equal(x.numpy(), [2.0, 2.0])
    overridden = _Scaled()
    overridden.SCALE = 5.0
    _fill_scale(overridden, x)
    np.testing.assert_array_equal(x.numpy(), [5.0, 5.0])
