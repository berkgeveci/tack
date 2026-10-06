"""Random numbers in kernels, from a counter-based generator with explicit state.

Nothing here is hidden: a kernel seeds a ``u32`` state from the element it
works on and a stream number, and every draw returns the value and the next
state. The sequence is a pure function of those integers, so the same
program draws the same numbers on every backend and for any thread order,
and a NumPy reference that calls the ``np_`` mirrors below draws exactly
the same ones::

    from tack import random

    @tack.kernel
    def scatter(pos, n, frame):
        for i in range(n):
            state = random.seed(i, frame)        # element i of this frame
            u, state = random.uniform(state)     # f32 in [0, 1)
            z, state = random.normal(state)      # standard normal
            d, state = random.direction3(state)  # a unit 3-vector
            pos[i] = [u, z, d.x]

The generator is PCG's output permutation (RXS-M-XS on 32 bits) over a
linear congruential step, in wrapping ``u32`` arithmetic. ``uniform`` is
the state's top 24 bits scaled, so it is exact and identical everywhere;
``normal`` and the directions go through ``log``, ``sqrt``, ``cos`` and
``sin``, so they agree across backends to those functions' rounding.
Taichi's ``ti.random()`` keeps hidden per-thread state instead; its results
depend on the thread schedule and cannot be reproduced by a reference.
"""

import math

import numpy as np

import tack

MULTIPLIER = tack.constant(747796405, tack.u32)
INCREMENT = tack.constant(2891336453, tack.u32)
MIX = tack.constant(277803737, tack.u32)
S28 = tack.constant(28, tack.u32)
S22 = tack.constant(22, tack.u32)
S8 = tack.constant(8, tack.u32)
S4 = tack.constant(4, tack.u32)
TWO_PI = tack.constant(2 * math.pi)
SCALE = tack.constant(1.0 / 16777216.0)


@tack.func
def advance(state):
    """The state after ``state``; also the 32 random bits it stands for."""
    x = tack.u32(state) * MULTIPLIER + INCREMENT
    word = ((x >> ((x >> S28) + S4)) ^ x) * MIX
    return (word >> S22) ^ word


@tack.func
def seed(index, stream):
    """A state for element ``index`` of ``stream`` (a frame, a pass, a purpose)."""
    return advance(tack.u32(index) ^ advance(tack.u32(stream)))


@tack.func
def uniform(state):
    """An f32 in [0, 1) and the next state."""
    return tack.f32(state >> S8) * SCALE, advance(state)


@tack.func
def normal(state):
    """A standard normal draw (Box-Muller) and the next state; uses two states."""
    u1 = 1.0 - tack.f32(state >> S8) * SCALE              # in (0, 1], so log is finite
    state = advance(state)
    u2 = tack.f32(state >> S8) * SCALE
    return sqrt(-2.0 * log(u1)) * cos(TWO_PI * u2), advance(state)


@tack.func
def direction2(state):
    """A unit 2-vector with uniformly distributed direction, and the next state."""
    a = TWO_PI * tack.f32(state >> S8) * SCALE
    return tack.Vector([cos(a), sin(a)]), advance(state)


@tack.func
def direction3(state):
    """A unit 3-vector uniformly distributed on the sphere, and the next state; two states."""
    a = TWO_PI * tack.f32(state >> S8) * SCALE
    state = advance(state)
    z = tack.f32(state >> S8) * SCALE * 2.0 - 1.0
    r = sqrt(1.0 - z * z)
    return tack.Vector([r * cos(a), r * sin(a), z]), advance(state)


# NumPy mirrors: the same functions over arrays of states, for references.

def np_advance(state):
    x = np.asarray(state).astype(np.uint32)
    with np.errstate(over="ignore"):
        x = x * np.uint32(MULTIPLIER) + np.uint32(INCREMENT)
        word = ((x >> ((x >> np.uint32(S28)) + np.uint32(S4))) ^ x) * np.uint32(MIX)
    return (word >> np.uint32(S22)) ^ word


def np_seed(index, stream):
    return np_advance(np.asarray(index).astype(np.uint32) ^ np_advance(stream))


def _bits_to_unit(state):
    return (np.asarray(state, np.uint32) >> np.uint32(S8)).astype(np.float32) * np.float32(SCALE)


def np_uniform(state):
    return _bits_to_unit(state), np_advance(state)


def np_normal(state):
    f = np.float32
    u1 = f(1.0) - _bits_to_unit(state)
    state = np_advance(state)
    u2 = _bits_to_unit(state)
    return np.sqrt(f(-2.0) * np.log(u1)) * np.cos(f(TWO_PI) * u2), np_advance(state)


def np_direction2(state):
    a = np.float32(TWO_PI) * _bits_to_unit(state)
    return np.stack([np.cos(a), np.sin(a)], axis=-1).astype(np.float32), np_advance(state)


def np_direction3(state):
    f = np.float32
    a = f(TWO_PI) * _bits_to_unit(state)
    state = np_advance(state)
    z = _bits_to_unit(state) * f(2.0) - f(1.0)
    r = np.sqrt(f(1.0) - z * z)
    return np.stack([r * np.cos(a), r * np.sin(a), z], axis=-1).astype(f), np_advance(state)
