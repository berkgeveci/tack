# Control Flow and Math

## Conditionals

```python
@pgc.kernel
def clamp(data, lo, hi, n):
    for i in range(n):
        val = data[i]
        if val < lo:
            val = lo
        if val > hi:
            val = hi
        data[i] = val
```

`if`/`elif`/`else` works as expected. Ternary expressions are not supported;
use `if`/`else` blocks instead.

## While Loops

```python
@pgc.kernel
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
@pgc.kernel
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
| `exp(x)`, `exp2(x)` | Exponential |
| `log(x)`, `log2(x)`, `log10(x)` | Logarithmic |
| `floor(x)`, `ceil(x)` | Rounding |
| `abs(x)` | Absolute value |
| `min(a, b)`, `max(a, b)` | Min/max |
| `pow(base, exp)` | Power |

These are imported automatically — no `import math` needed. They compile to
the native GPU math intrinsics (e.g., `sinf` on CUDA, `metal::sin` on MSL).

```python
from math import sqrt, sin, cos, exp, log, floor, ceil, abs, min, max, pow

@pgc.kernel
def wave(out, t, n):
    for i in range(n):
        x = float(i) / float(n)
        out[i] = sin(x * 6.2832) * exp(-t)
```
