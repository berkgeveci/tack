# Control flow and math

Inside a kernel's parallel loop you write ordinary Python: conditions, loops
and math. This page shows what each iteration can do: `if` and conditional
expressions, `for` and `while` loops with `break` and `continue`, the math
functions, and how integer and floating-point arithmetic behave. It is for
readers who have written a first kernel ([Kernels](kernels.md)). After it you
can write iterative per-element algorithms, such as Newton's method or a
bounded search, and know which arithmetic gives the same answer as NumPy.

## A first example

This kernel clamps each value to a range and counts the halvings that bring
it below one. It uses an `if`, a `while` loop and a math function:

<!-- tack-check: clamp-and-halve -->
```python
import numpy as np
import tack

tack.set_default_device("cpu")

@tack.kernel
def clamp_and_halve(data, lo, hi, clamped, halvings):
    for i in range(len(data)):
        value = data[i]
        if value < lo:
            value = lo
        elif value > hi:
            value = hi
        clamped[i] = value

        count = 0
        v = abs(data[i])
        while v > 1.0 and count < 100:      # a cap keeps the loop bounded
            v = v * 0.5
            count += 1
        halvings[i] = count

data = tack.array(np.array([-3.0, 0.5, 9.0], np.float32))
clamped = tack.empty((3,))
halvings = tack.empty((3,), tack.i32)
clamp_and_halve(data, 0.0, 4.0, clamped, halvings)
assert clamped.numpy().tolist() == [0.0, 0.5, 4.0]
assert halvings.numpy().tolist() == [2, 0, 4]
```

Each iteration makes its own decisions. One iteration can take the `if` branch
and loop four times while its neighbour does neither. On a GPU, iterations that
run side by side but take different branches take longer than iterations that
agree, so the cost of a branch depends on how often neighbours disagree. The
result never does.

## Conditions

`if`, `elif` and `else` work as in Python, and so do comparisons, chained
comparisons such as `0 <= i < n`, and `and`, `or` and `not`. A comparison has
the value 1 when true and 0 when false.

A conditional expression, `a if condition else b`, evaluates only the arm it
selects. `and` and `or` stop as soon as the answer is known, as in Python.
Both are useful at the edges of an array, where reading past the end would be
an error:

<!-- tack-check: guarded-loads -->
```python
import numpy as np
import tack

tack.set_default_device("cpu")

@tack.kernel
def differences(data, out):
    for i in range(len(data)):
        previous = data[i - 1] if i > 0 else data[i]     # no read of data[-1]
        out[i] = data[i] - previous

data = tack.array([1.0, 4.0, 9.0, 16.0])
out = tack.empty((4,))
differences(data, out)
assert out.numpy().tolist() == [0.0, 3.0, 5.0, 7.0]
```

Comparisons of vectors give one result per component. They need `any` or
`all` to become a condition, as [Vectors and matrices](vectors-and-matrices.md) explains.

## Loops inside an iteration

Inside the parallel loop, `for` over a `range` and `while` run in order, within
the iteration. `range` takes a start, an end and a step, and its bounds can be
any value the iteration has, including values read from arrays. `break` leaves
the nearest of these loops, and `continue` goes on to its next round.

A loop that stops when a calculation converges should also stop after a
fixed number of rounds. A kernel cannot raise an exception from one iteration,
so a value that never converges would otherwise keep its thread running:

<!-- tack-check: newton -->
```python
import numpy as np
import tack

tack.set_default_device("cpu")

@tack.kernel
def newton_sqrt(x, out, converged):
    for i in range(len(x)):
        value = x[i]
        guess = max(value, 1.0)
        done = 0
        if value == 0.0:
            guess = 0.0
            done = 1
        for step in range(30):                 # at most 30 rounds
            if done == 1:
                break
            better = 0.5 * (guess + value / guess)
            if abs(better - guess) <= 1e-6 * guess:
                done = 1
            guess = better
        out[i] = guess
        converged[i] = done                    # tell the host which converged

x = tack.array(np.array([0.0, 2.0, 9.0, 1e6], np.float32))
out = tack.empty((4,))
converged = tack.empty((4,), tack.i32)
newton_sqrt(x, out, converged)
np.testing.assert_allclose(out.numpy(), np.sqrt(x.numpy()), rtol=1e-6)
assert converged.numpy().all()
```

The `converged` array is how an iteration reports a failure: write a status
and let host code decide what to do with it.

## Math functions

These functions work in kernels and device functions without importing
anything. Writing `from math import sqrt` does no harm, and `math.sqrt(x)`
works too.

| Function | Computes |
|---|---|
| `sqrt(x)` | Square root |
| `sin`, `cos`, `tan` | Trigonometric functions |
| `asin`, `acos`, `atan`, `atan2(y, x)` | Inverse trigonometric functions |
| `sinh`, `cosh`, `tanh` | Hyperbolic functions |
| `exp`, `exp2` | `e` and 2 to a power |
| `log`, `log2`, `log10` | Logarithms |
| `floor`, `ceil` | Rounding down and up |
| `abs` | Absolute value |
| `min(a, b, ...)`, `max(a, b, ...)` | Smallest and largest of two or more values |
| `pow(base, exponent)` | Power, as `base ** exponent` |

Each one compiles to the device's own math function. Given a vector, each
applies to every component. A function computes in the precision of its
argument: `sqrt` of an `f64` is an `f64`. An integer argument is converted to
`f32` first.

Math functions have domains, and values outside them follow IEEE floating
point: `sqrt(-1.0)` and `log(-1.0)` are NaN, and `log(0.0)` is minus
infinity. Decide how your algorithm treats such values before you run it on
large inputs.

### Helpers from shader languages

`tack.math` adds the small functions graphics and simulation code reaches for,
with the definitions GLSL gives them:

| Function | Computes |
|---|---|
| `fract(x)` | `x - floor(x)` |
| `mix(x, y, a)` | `x * (1 - a) + y * a` |
| `clamp(x, lo, hi)` | `min(max(x, lo), hi)` |
| `saturate(x)` | `clamp(x, 0, 1)` |
| `step(edge, x)` | 0 below `edge`, 1 from it |
| `smoothstep(edge0, edge1, x)` | A smooth step from 0 to 1 between the edges |
| `sign(x)` | -1, 0 or 1 |
| `length(v)`, `distance(a, b)`, `normalize(v)` | A vector's length, a distance, a unit vector |

They are [device functions](device-functions.md), so they work on scalars and
vectors alike, and only kernels and other device functions can call them:

<!-- tack-check: colour-map -->
```python
import numpy as np
import tack
from tack import math as tm

tack.set_default_device("cpu")

@tack.kernel
def colour_map(data, colour):
    for i in range(len(data)):
        t = tm.clamp(data[i], 0.0, 1.0)
        colour[i] = tm.mix([0.1, 0.2, 0.8], [1.0, 0.8, 0.1], t)   # blue to yellow

data = tack.array([-1.0, 0.5, 2.0])
colour = tack.empty((3,), tack.vec3())
colour_map(data, colour)
np.testing.assert_allclose(colour.numpy()[1], [0.55, 0.5, 0.45], rtol=1e-6)
```

## How arithmetic behaves

Kernels compute in fixed-width types, as NumPy does, not with Python's
unlimited integers. Most arithmetic gives what NumPy gives for the same types.
These are the rules that surprise people:

| Expression | Result | Why |
|---|---|---|
| `7 / 2` | `3.5`, an `f32` | `/` always divides as floating point |
| `-7 // 2` | `-4` | `//` rounds down, as in Python |
| `-7 % 2` | `1` | `%` takes the divisor's sign, as in Python |
| `int(-0.3)` | `0` | `int` truncates toward zero |
| `int(floor(-0.3))` | `-1` | `floor` rounds down first |
| `tack.i32(2147483647) + 1` | `-2147483648` | Integers wrap at their width |
| `tack.u8(-1)` | `255` | A cast keeps the low bits |
| `tack.i8(3) ** 5` | `-13` | An integer power keeps the base's type, and wraps |

Four of these need care in practice.

**Division.** Integer `/` gives an `f32`, even when you store it into an
`f64` array. For a double-precision quotient, cast first:
`tack.f64(a) / tack.f64(b)`. `//` and `%` on floats also round down and take
the divisor's sign, like Python's.

**Grid cells.** To find the cell of a coordinate that can be negative, use
`int(floor(x))`. `int(x)` maps both `-0.3` and `0.3` to cell 0.

**Wrapping.** Integer `+`, `-` and `*` wrap at their type's width instead of
growing. When two integer types meet, the result is a type that holds both
ranges: `i8 + u16` is `i32`, and `i32 + u32` is `i64`. Mixing a signed type
with `u64` has no such type, so it needs an explicit cast. Use a typed
constant, such as `tack.constant(747796405, tack.u32)`, for arithmetic that
must wrap at 32 bits.

**Float literals.** A literal such as `0.1` takes the precision of what it
meets, as Python numbers do in NumPy. With an `f64` array `x`, `x[i] * 0.1`
computes in `f64` with the `f64` nearest 0.1. Alone, or beside integers, a
literal is `f32`. A local assigned a literal once, such as `a = 0.1`, behaves
like the literal wherever it is read.

Floating-point results can differ between devices in their last bits. Each
device has its own math library, and a compiler may fuse a multiply and an add
within one expression into a single operation. No device reorders a sum or
uses fast approximate math. Compare results across devices with
a tolerance. The
[floating-point policy](../reference/language-contract.md#floating-point-execution-policy)
says exactly what is and is not guaranteed.

## Details

The [kernel language contract](../reference/language-contract.md) has the
complete rules. These details come up less often:

- Shift counts must be at least zero and less than the type's width. A signed
  right shift copies the sign bit, and an unsigned one fills with zeros.
- An integer divisor must not be zero. The smallest signed value divided by
  -1 is outside the defined domain too.
- An integer power needs a non-negative exponent, and `0 ** 0` is 1. Cast the
  base to a float for negative powers: `tack.f32(a) ** e`.
- Converting a float to an integer needs a finite value whose truncation fits
  the integer type.
- Integer `abs`, `min` and `max` are exact. `abs` keeps its type, so the
  smallest signed value stays negative.
- Floating `min` and `max` return the number when the other argument is NaN.

## Where to go next

- [Device functions](device-functions.md) shows how to move a loop like
  Newton's method into a helper that several kernels share.
- [Vectors and matrices](vectors-and-matrices.md) covers vector arithmetic,
  component masks and `tack.select`.
