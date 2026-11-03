# Device functions

When the same few lines appear in several kernels, move them into a device
function. A device function is a helper that kernels call, a Python function
decorated with `@tack.func`. Tack inlines it into every kernel that calls it,
so splitting a kernel into named pieces costs nothing at run time. A device
function can return several values, take arrays and use small scratch
arrays. Helpers can live in modules of their own and be shared between
kernels.

## Write a device function

<!-- tack-check: lerp -->
```python
import numpy as np
import tack

tack.set_default_device("cpu")

@tack.func
def lerp(a, b, t):
    return a + t * (b - a)

@tack.kernel
def blend(src, dst, out, t):
    for i in range(len(out)):
        out[i] = lerp(src[i], dst[i], t)

src = tack.zeros((4,))
dst = tack.array([4.0, 8.0, 12.0, 16.0])
out = tack.empty((4,))
blend(src, dst, out, 0.25)
assert out.numpy().tolist() == [1.0, 2.0, 3.0, 4.0]
```

Tack *inlines* a device function: when it compiles `blend`, it copies the
body of `lerp` into the kernel in place of the call. There is no call at run
time, so splitting a kernel into helpers costs nothing.

Because it is inlined, a device function takes the types of the values each
call passes. The same `lerp` works on `f32` and `f64` numbers, on integers,
and on vectors, compiled separately for each. Calling a device function from
Python, outside a kernel, raises `RuntimeError`. Write a NumPy version for
host code when you need one, which also makes a good reference to test
against.

A device function contains what a kernel's loop body can contain: arithmetic,
conditions, loops, math functions, and calls to other device functions. It may
`return` early, outside a loop:

```python
@tack.func
def side(x):
    if x < 0.0:
        return -1
    return 1
```

Every `return` of a function must give values of the same kind: all scalars
of compatible types, or all vectors of one width.

## Return several values

A device function can return a tuple, which the caller unpacks:

<!-- tack-check: polar -->
```python
import numpy as np
import tack

tack.set_default_device("cpu")

@tack.func
def to_cartesian(r, theta):
    return r * cos(theta), r * sin(theta)

@tack.kernel
def convert(r, theta, x, y):
    for i in range(len(r)):
        x[i], y[i] = to_cartesian(r[i], theta[i])

r = tack.array([1.0, 2.0])
theta = tack.array(np.array([0.0, np.pi / 2], np.float32))
x, y = tack.empty((2,)), tack.empty((2,))
convert(r, theta, x, y)
np.testing.assert_allclose(x.numpy(), [1.0, 0.0], atol=1e-6)
np.testing.assert_allclose(y.numpy(), [0.0, 2.0], atol=1e-6)
```

The values can be scalars, vectors and matrices in any mix, and the targets
can be local variables or array elements. A ray tracer's hit function might
return a distance, a normal and a colour:

```python
@tack.func
def closest_hit(origin, direction):
    ...
    return distance, normal, colour          # a scalar and two 3-vectors

@tack.kernel
def shade(origins, directions, depth, normals):
    for i in range(len(depth)):
        depth[i], normals[i], colour = closest_hit(origins[i], directions[i])
```

A device function may return another call's result as it is, with
`return closest_hit(o, d)`. A local name may hold such a result until it is
unpacked, as in `hit = closest_hit(o, d)` and then `depth, normal, colour = hit`.
Using that name as a single value raises `UnsupportedSyntaxError`.

## Call device functions from device functions

Device functions call each other, to any depth, and all of it is inlined into
the kernel:

```python
@tack.func
def square(x):
    return x * x

@tack.func
def length3(x, y, z):
    return sqrt(square(x) + square(y) + square(z))
```

Tack finds the function a call names the way Python would, at the place the
call is written. So you can import a device function from another module, call
it as `module.function`, or import it under another name, and two modules can
each define a `square` without interfering. A device function defined inside
another Python function, and returned from it, works too.

Tack resolves these names the first time it compiles the kernel, and the
compiled kernel keeps the inlined body. Binding a name to a different function
afterwards does not change a kernel already compiled. Make the kernel again
to pick up the change.

## Pass arrays

A device function can take arrays and index them as a kernel does. The array
is the caller's array, not a copy, so a store through it changes the caller's
data:

```python
@tack.func
def load_or(data, i, default):
    value = default
    if 0 <= i < len(data):
        value = data[i]
    return value
```

Vectors and matrices, in contrast, are passed as values. Changing a vector
argument inside the function changes the function's copy only.

A device function can also take a texture and sample it, and a
[device class](device-classes.md) and call its methods. See
[Workgroups, atomics, textures and random numbers](workgroups-atomics-textures.md)
for textures.

## Use small scratch arrays

Sometimes one iteration needs a few values at a time: the weights of a cell's
points, or a short list to sort. `tack.local_array(dtype, size)` gives each
iteration a small array of its own. On a GPU it lives in the thread's
registers or private memory, and on the CPU on the thread's stack. Its values
are not set, so write each element before reading it.

Pass a local array to a device function, and the function fills it in place.
That lets a helper produce several values at once without returning them:

<!-- tack-check: local-array -->
```python
import numpy as np
import tack

tack.set_default_device("cpu")

ORDER = tack.constant(4)

@tack.func
def powers(w, x):                       # fills w with 1, x, x**2, x**3
    p = 1.0
    for k in range(ORDER):
        w[k] = p
        p *= x

@tack.kernel
def polynomial(x, coefficients, out):
    for i in range(len(x)):
        w = tack.local_array(tack.f32, ORDER)
        powers(w, x[i])
        total = 0.0
        for k in range(ORDER):
            total += coefficients[k] * w[k]
        out[i] = total

x = tack.array([0.0, 1.0, 2.0])
coefficients = tack.array([1.0, 0.0, 0.0, 1.0])     # 1 + x**3
out = tack.empty((3,))
polynomial(x, coefficients, out)
assert out.numpy().tolist() == [1.0, 2.0, 9.0]
```

The size must be known when the kernel compiles. It can be a literal, a
`tack.constant`, a device class's class constant, or an expression of these
and the array dimensions the kernel reads, such as `2 * data.shape[1]`. A
scalar argument cannot be a size, because its value is known only at the
launch. Keep local arrays small, a few dozen elements at most: a GPU has
little private memory per thread.

`tack.local_array_like(array, size)` makes a local array of `array`'s element
type, so one kernel serves `f32` and `f64` inputs:

```python
--8<-- "docs/examples/array_sizes.py:scratch"
```

Here the size, `PAIR * data.shape[1]`, depends on the input's width, so an
input of another width compiles another version of the kernel.

## Details

What a device function cannot do, each refused with `UnsupportedSyntaxError`
when the kernel compiles:

- `return` inside a loop. Record the result in a local variable, `break`, and
  return after the loop.
- Call itself, directly or through another function. Rewrite the recursion
  as a loop, with a local array as the stack if it needs one.
- Call an ordinary Python function, or a function passed in as an argument.
  Every call must name a `@tack.func` that Tack can find when it compiles.
- Read a Python variable from around it. As for kernels, use
  `tack.constant` for fixed values, and arguments for everything else.

[Device classes](device-classes.md) take this further: an object whose
methods are device functions, whose arrays and parameters kernels read, and
whose class constants are compiled in.

## Where to go next

- [Vectors and matrices](vectors-and-matrices.md) shows the vector and matrix
  values device functions often take and return.
- [Device classes](device-classes.md) groups device functions with the data
  they work on.
