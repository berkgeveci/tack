# Control Flow and Math

## Conditionals

```python
@tack.kernel
def clamp(data, lo, hi, n):
    for i in range(n):
        val = data[i]
        if val < lo:
            val = lo
        if val > hi:
            val = hi
        data[i] = val
```

`if`/`elif`/`else` works as expected, and so do conditional expressions
(`a if cond else b`), which evaluate only the selected arm.

## While Loops

```python
@tack.kernel
def newton_sqrt(x, out, n):
    for i in range(n):
        val = x[i]
        guess = 0.0
        if val > 0.0:
            guess = max(val, 1.0)
            k = 0
            while k < 20:
                guess = 0.5 * (guess + val / guess)
                k += 1
        out[i] = guess
```

The example requires nonnegative inputs; the zero case is handled before the
iteration, avoiding division by zero. A fixed iteration cap makes runtime
bounded. For a convergence loop, combine a tolerance with such a cap and write
a status field if the host needs to know which elements converged. A kernel
cannot raise a Python exception from an individual iteration.

## Break and Continue

```python
@tack.kernel
def find_threshold(data, out, threshold, n):
    for i in range(n):
        count = 0
        for j in range(100):
            if data[i * 100 + j] > threshold:
                break
            count = count + 1
        out[i] = count
```

`break` leaves the nearest sequential loop; `continue` advances it to its next
iteration. They do not exit a whole launch. Do not use lane-dependent exits
around a GPU barrier: all workgroup lanes must still participate. A `return`
inside a sequential loop of a device function remains unsupported; record a
result, `break`, then return after the loop.

## Math Builtins

The following math builtins are recognized inside kernels and device functions:

| Function | Description |
|----------|-------------|
| `sqrt(x)` | Square root |
| `sin(x)`, `cos(x)`, `tan(x)` | Trigonometric |
| `asin(x)`, `acos(x)`, `atan(x)` | Inverse trig |
| `atan2(y, x)` | Two-argument arctangent |
| `sinh(x)`, `cosh(x)`, `tanh(x)` | Hyperbolic |
| `exp(x)`, `exp2(x)` | Exponential |
| `log(x)`, `log2(x)`, `log10(x)` | Logarithmic |
| `floor(x)`, `ceil(x)` | Rounding |
| `abs(x)` | Absolute value |
| `min(a, b, ...)`, `max(a, b, ...)` | Min/max of two or more values |
| `pow(base, exp)` | Power |

These are imported automatically — no `import math` needed. They compile to
native math operations (e.g., `sinf` on CUDA, `metal::sin` on MSL).
With two integer operands, `pow` and `**` compute exact fixed-width power
and preserve the base type; the exponent must be nonnegative. With a
floating operand they use floating-point power at the promoted precision.
See [Fields and Types](02-fields-and-types.md) for wrapping and division rules.

```python
from math import sqrt, sin, cos, exp, log, floor, ceil, pow  # optional

@tack.kernel
def wave(out, t, n):
    for i in range(n):
        x = float(i) / float(n)
        out[i] = sin(x * 6.2832) * exp(-t)
```

## Conditions can guard loads

Scalar `and`/`or` and conditional expressions short-circuit. This is useful at
boundaries: the following load is evaluated only for an in-range index.

```python
@tack.kernel
def read_previous(data, out, n):
    for i in range(n):
        out[i] = data[i - 1] if i > 0 else 0.0
```

A vector comparison instead produces one mask value per component. Use
`any(mask)` or `all(mask)` for an `if`, and `tack.select(mask, a, b)` to pick
component values. Vector masks evaluate component expressions rather than
providing a scalar short-circuit guard for invalid loads. See
[Vectors and matrices](14-vectors-and-matrices.md#vectors).

## Cast at the point where the meaning changes

`float(i)` converts to `f32`, and `int(x)` truncates a finite representable value
toward zero. For a grid cell containing a coordinate, use `int(floor(x))` when
negative coordinates are possible. Those are different operations:

| Input | `int(x)` | `int(floor(x))` |
|---|---:|---:|
| `1.7` | 1 | 1 |
| `-0.3` | 0 | −1 |

`tack.f64(x)` explicitly requests double precision where the backend provides
it. An `f64` output field does not retroactively change all intermediate
operations: integer true division produces `f32`, so cast its operands for a
double-precision quotient. Plain integer literals also participate in promotion;
use typed constants for intentional unsigned wrapping arithmetic.

Math operations have input domains. A sampler's indices must be valid; an
inverse matrix needs a nonzero determinant; `log` needs positive input for a
finite result. Choose how your algorithm handles values outside those domains
before running large inputs. The [numerical contract](../reference/language-contract.md#floating-point-execution-policy)
defines which floating special cases are portable.

## Shader-style helpers

`from tack import math as tm` gives reusable functions such as `clamp`, `mix`,
`fract`, `smoothstep`, `step`, `sign`, `length`, `distance` and `normalize`.
These are device functions, so call them inside kernels or other device functions:

```python
from tack import math as tm

@tack.kernel
def map_values(data, colour, n):
    for i in range(n):
        t = tm.clamp(data[i], 0.0, 1.0)
        colour[i] = tm.mix([0.1, 0.2, 0.8], [1.0, 0.8, 0.1], t)
```

Here `colour` is a three-component vector field. `mix` interpolates its arms;
`clamp` limits a value component by component. `fract(x)` is `x - floor(x)`,
which is useful for repeating patterns. The [dye tutorial](tutorials/dye.md)
uses interpolation over actual fields, and [marching squares](tutorials/contours.md)
uses interpolated edge crossings to build geometry.
