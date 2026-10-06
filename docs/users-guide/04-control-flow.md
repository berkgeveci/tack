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
        guess = val * 0.5
        k = 0
        while k < 20:
            guess = 0.5 * (guess + val / guess)
            k = k + 1
        out[i] = guess
```

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

## Math Builtins

All standard math functions are available inside kernels:

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
