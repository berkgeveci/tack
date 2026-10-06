# Device Functions

## @tack.func

Functions decorated with `@tack.func` are inlined into kernels at the AST
level. They let you write reusable device-side code:

```python
@tack.func
def lerp(a, b, t):
    return a + t * (b - a)

@tack.kernel
def interpolate(src, dst, alpha, n):
    for i in range(n):
        dst[i] = lerp(src[i], 1.0, alpha)
```

When the kernel is compiled, `lerp` is inlined — no function call overhead.

Calls use the defining kernel or device function's Python bindings. You can
import a device function under an alias or call it through an imported
module. Two modules may define the same function name without interfering.
Nested calls use the callee's namespace, and closure-bound device functions
are supported. Template methods keep their defining bindings too.

Bindings are resolved when the IR specialization is first built. Cached
kernels keep the inlined body; recreate a kernel to capture a changed
callable binding. Runtime function parameters, ordinary Python callables,
and recursive device calls are unsupported. This does not enable capturing
numerical Python variables: pass those values as kernel arguments.

## Multiple Return Values

`@tack.func` supports returning tuples:

```python
@tack.func
def polar_to_cart(r, theta):
    x = r * cos(theta)
    y = r * sin(theta)
    return x, y

@tack.kernel
def convert(r_field, theta_field, x_out, y_out, n):
    for i in range(n):
        x, y = polar_to_cart(r_field[i], theta_field[i])
        x_out[i] = x
        y_out[i] = y
```

The values may be vectors and matrices as well as scalars, in any mix,
and the targets may be field elements:

```python
@tack.func
def closest_hit(origin, direction):
    ...
    return distance, normal, color              # a scalar and two vectors

@tack.func
def polar(F):
    ...
    return R, S                                 # two matrices

@tack.kernel
def shade(origins, directions, depth, normals, n):
    for i in range(n):
        depth[i], normals[i], color = closest_hit(origins[i], directions[i])
```

A result made of several values has to be unpacked into one target each;
binding it to a single name is rejected.

## Nested Calls

`@tack.func` functions can call other `@tack.func` functions:

```python
@tack.func
def square(x):
    return x * x

@tack.func
def length(x, y, z):
    return sqrt(square(x) + square(y) + square(z))
```

## Passing Fields

Fields can be passed to `@tack.func` — they work just like in kernels:

```python
@tack.func
def safe_load(data, i, default_val):
    result = default_val
    if i >= 0:
        result = data[i]
    return result
```

## Texture Sampling in Functions

`tex.sample()` works inside `@tack.func`. Tack automatically propagates
texture metadata through the inlining:

```python
@tack.func
def sample_bilinear(tex, u, v, w):
    return tex.sample(u, v, w)

@tack.kernel
def render(tex, output, n):
    for i in range(n):
        output[i] = sample_bilinear(tex, 0.5, 0.5, float(i) / float(n))
```

## Local Arrays in Functions

Local arrays (`tack.local_array`) can be passed to `@tack.func`. The
function accesses the caller's array directly — no copy:

```python
@tack.func
def fill_weights(w, ct, pc0, pc1, pc2):
    for v in range(ct.num_points):
        w[v] = ct.weight(v, pc0, pc1, pc2)

@tack.kernel
def interp(ct: tack.template(), data, out, n):
    for i in range(n):
        w = tack.local_array(tack.f32, ct.num_points)
        fill_weights(w, ct, 0.5, 0.5, 0.5)  # fills w in place
        # ... use w[v] ...
```

This also works with `@tack.data_oriented` methods — see
[Advanced Features](07-advanced.md) for the cell set pattern.
