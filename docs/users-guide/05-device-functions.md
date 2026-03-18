# Device Functions

## @pgc.func

Functions decorated with `@pgc.func` are inlined into kernels at the AST
level. They let you write reusable device-side code:

```python
@pgc.func
def lerp(a, b, t):
    return a + t * (b - a)

@pgc.kernel
def interpolate(src, dst, alpha, n):
    for i in range(n):
        dst[i] = lerp(src[i], 1.0, alpha)
```

When the kernel is compiled, `lerp` is inlined — no function call overhead.

## Multiple Return Values

`@pgc.func` supports returning tuples:

```python
@pgc.func
def polar_to_cart(r, theta):
    x = r * cos(theta)
    y = r * sin(theta)
    return x, y

@pgc.kernel
def convert(r_field, theta_field, x_out, y_out, n):
    for i in range(n):
        x, y = polar_to_cart(r_field[i], theta_field[i])
        x_out[i] = x
        y_out[i] = y
```

## Nested Calls

`@pgc.func` functions can call other `@pgc.func` functions:

```python
@pgc.func
def square(x):
    return x * x

@pgc.func
def length(x, y, z):
    return sqrt(square(x) + square(y) + square(z))
```

## Passing Fields

Fields can be passed to `@pgc.func` — they work just like in kernels:

```python
@pgc.func
def safe_load(data, i, default_val):
    result = default_val
    if i >= 0:
        result = data[i]
    return result
```

## Texture Sampling in Functions

`tex.sample()` works inside `@pgc.func`. PGC automatically propagates
texture metadata through the inlining:

```python
@pgc.func
def sample_bilinear(tex, u, v, w):
    return tex.sample(u, v, w)

@pgc.kernel
def render(tex, output, n):
    for i in range(n):
        output[i] = sample_bilinear(tex, 0.5, 0.5, float(i) / float(n))
```

## Local Arrays in Functions

Local arrays (`pgc.local_array`) can be passed to `@pgc.func`. The
function accesses the caller's array directly — no copy:

```python
@pgc.func
def fill_weights(w, ct, pc0, pc1, pc2):
    for v in range(ct.num_points):
        w[v] = ct.weight(v, pc0, pc1, pc2)

@pgc.kernel
def interp(ct: pgc.template(), data, out, n):
    for i in range(n):
        w = pgc.local_array(pgc.f32, ct.num_points)
        fill_weights(w, ct, 0.5, 0.5, 0.5)  # fills w in place
        # ... use w[v] ...
```

This also works with `@pgc.data_oriented` methods — see
[Advanced Features](07-advanced.md) for the cell set pattern.
