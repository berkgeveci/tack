# Use workgroups, atomics, textures and random numbers

Most kernels are a parallel loop in which each iteration reads what it likes
and writes its own output. This page is for kernel writers who need more than
that. It covers four tools. Atomics let many iterations update one element.
Random numbers come out the same on every device. 3D textures sample a volume
with trilinear interpolation. Workgroups let GPU threads cooperate through
shared memory. After reading it you can use each one, and you know which
devices support it.

A histogram of random samples uses the first two:

<!-- tack-check: advanced-first -->
```python
import numpy as np
import tack
from tack import random

tack.set_default_device("cpu")


@tack.kernel
def histogram(bins, n, frame):
    for i in range(n):
        state = random.seed(i, frame)             # sample i of this frame
        u, state = random.uniform(state)          # f32 in [0, 1)
        b = min(int(u * float(bins.shape[0])), bins.shape[0] - 1)
        tack.atomic_add(bins, b, 1)               # many samples, one bin


bins = tack.zeros(10, dtype=tack.i32)
histogram(bins, 100_000, 0)
counts = bins.numpy()
assert counts.sum() == 100_000
assert abs(counts - 10_000).max() < 500       # close to uniform
```

## Update one element from many iterations

An *atomic* makes one update of one array element indivisible, so that two
iterations adding to the same bin both count. Tack has three:

- `tack.atomic_add(array, index, value)` adds `value` to the element.
- `tack.atomic_min(array, index, value)` keeps the smaller of the two.
- `tack.atomic_max(array, index, value)` keeps the larger.

The array must hold its starting values before the launch: zeros for a sum,
a large value for a minimum. Initialize it with `tack.zeros`, `tack.full` or
`fill`, since a new array's contents are unspecified.

### Claim a slot with the old value

Each atomic is also an expression. Its value is the element's value just
before this update. A counter therefore hands each iteration a slot of its
own, which is how iterations append to a shared list:

<!-- tack-check: advanced-compact -->
```python
import numpy as np
import tack

tack.set_default_device("cpu")


@tack.kernel
def keep_positive(values, out, count):
    for i in range(values.shape[0]):
        if values[i] > 0.0:
            slot = tack.atomic_add(count, 0, 1)   # the old count: this iteration's slot
            out[slot] = values[i]


values = tack.array(np.array([1.0, -1.0, 2.0, -3.0, 4.0], dtype=np.float32))
out = tack.zeros(5)
count = tack.zeros(1, dtype=tack.i32)             # reset before every call
keep_positive(values, out, count)
n = int(count[0])
assert n == 3 and sorted(out.numpy()[:n]) == [1.0, 2.0, 4.0]
```

The slots come out in whatever order the iterations ran, so the kept values
may be in any order. Allocate enough room for the most that can be kept. When
the order matters, or you want the output exactly as large as it needs to be,
count first, scan the counts, then write. The
[marching squares tutorial](../users-guide/tutorials/contours.md) shows that
pattern, and [Reductions, scans and sorts](reductions-scans-sorts.md) the
scan.

### Update a grid cell or a vector

For a multi-dimensional array, the index is a tuple with one index per
dimension, or a vector of integers. On a vector array, a vector value updates
every component, one atomic per component:

<!-- tack-check: advanced-scatter -->
```python
import numpy as np
import tack

tack.set_default_device("cpu")


@tack.kernel
def scatter(pos, vel, mass, momentum, inv_dx):
    for p in range(pos.shape[0]):
        cell = tack.vec2(int(pos[p].x * inv_dx), int(pos[p].y * inv_dx))
        tack.atomic_add(mass, cell, 1.0)                     # mass[cell[0], cell[1]]
        tack.atomic_add(momentum, (cell[0], cell[1]), vel[p])  # both components


pos = tack.array(np.array([[0.1, 0.1], [0.2, 0.3], [0.7, 0.2]], np.float32), tack.vec2())
vel = tack.array(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], np.float32), tack.vec2())
mass = tack.zeros((2, 2))
momentum = tack.zeros((2, 2), tack.vec2())
scatter(pos, vel, mass, momentum, 2.0)
assert mass.numpy()[0, 0] == 2.0 and mass.numpy()[1, 0] == 1.0
np.testing.assert_allclose(momentum.numpy()[0, 0], [1.0, 1.0])
```

A plain integer index is the flat, row-major index. On a vector array with a
scalar value, it is the flat index of one component. The
[MPM tutorial](../users-guide/tutorials/mpm.md) scatters particles to a grid
this way.

### What atomics do not do

An atomic makes one element's update indivisible. It does not order other
reads and writes, so it cannot publish data from one iteration to another.
A vector update is one atomic per component, not one for the whole vector.
Floating-point sums by atomics add in whatever order iterations arrive, so the
low bits can change from run to run. When a sum must be reproducible, use
the reductions in [Reductions, scans and sorts](reductions-scans-sorts.md),
which add in a fixed order.

Each device supports atomics on some element types. The CPU supports all ten.
CUDA and HIP support 32- and 64-bit integers, `f32` and `f64`. Metal and Level
Zero support `i32`, `u32` and `f32`. A kernel with an atomic on another type
is rejected before it compiles, with the reason.

## Draw random numbers

`tack.random` draws random numbers in kernels with no hidden state. A kernel
seeds a state from the element it works on and a stream number, and every
draw returns a value and the next state:

- `random.seed(index, stream)` makes a state.
- `random.uniform(state)` draws an `f32` in [0, 1).
- `random.normal(state)` draws from the standard normal distribution.
- `random.direction2(state)` and `random.direction3(state)` draw a unit
  vector.

The numbers are a pure function of the seed's two integers. A kernel
therefore draws the same numbers on every device and in any thread order, and
a different stream, such as a frame number, gives a different sequence. The
module's NumPy mirrors (`np_seed`, `np_uniform`, `np_normal`,
`np_direction2`, `np_direction3`) draw the same numbers on the host, so a
reference can check a kernel:

<!-- tack-check: advanced-random -->
```python
import numpy as np
import tack
from tack import random

tack.set_default_device("cpu")


@tack.kernel
def draw(u, z, frame):
    for i in range(u.shape[0]):
        state = random.seed(i, frame)
        u[i], state = random.uniform(state)
        z[i], state = random.normal(state)


u = tack.empty(1000)
z = tack.empty(1000)
draw(u, z, 7)

state = random.np_seed(np.arange(1000), 7)
u_ref, state = random.np_uniform(state)
z_ref, state = random.np_normal(state)
assert np.array_equal(u.numpy(), u_ref)                  # uniform: to the bit
np.testing.assert_allclose(z.numpy(), z_ref, rtol=1e-5, atol=1e-6)
```

`uniform` and the states match the mirrors to the bit on every device.
`normal` and the directions go through `log`, `sqrt`, `cos` and `sin`, so
they match to those functions' rounding. The
[trail-following agents tutorial](../users-guide/tutorials/physarum.md) seeds
each agent's stream this way.

## Sample a 3D texture

A *texture* is a volume of values that kernels sample at any point, with
trilinear interpolation between the stored values. `tack.texture3d(array,
shape)` makes one from an `f32` array of `W * H * D` values with x varying
fastest. In a kernel, `tex.sample(u, v, w)` interpolates at coordinates in
[0, 1], where 0 is the first value along an axis and 1 is the last:

<!-- tack-check: advanced-texture -->
```python
import numpy as np
import tack

tack.set_default_device("cpu")

W, H, D = 4, 3, 2
volume = np.arange(W * H * D, dtype=np.float32)   # value = x + 4 y + 12 z
data = tack.array(volume)
tex = tack.texture3d(data, shape=(W, H, D))


@tack.kernel
def along_x(tex, out):
    for i in range(out.shape[0]):
        u = float(i) / float(out.shape[0] - 1)
        out[i] = tex.sample(u, 0.5, 0.5)          # y = 1, z = 0.5


out = tack.empty(7)
along_x(tex, out)
np.testing.assert_allclose(out.numpy(), np.linspace(0.0, 3.0, 7) + 4.0 + 6.0, atol=1e-5)

tack.copyto(data, 2.0 * volume)    # the texture still holds the old values
along_x(tex, out)
assert abs(out.numpy()[0] - 10.0) < 1e-5
tex.update()                       # copy the array again
along_x(tex, out)
assert abs(out.numpy()[0] - 20.0) < 1e-5
```

The texture holds a copy of the array, taken when it is made. Later writes to
the array do not reach it until you call `tex.update()`. This holds on every
device. Coordinates outside [0, 1] clamp to the edge. `tex.sample` also works
inside a device function given the texture.

Where the hardware has texture units, Tack samples with them:

| device | sampling |
|---|---|
| CPU | software trilinear interpolation, compiled into the kernel |
| Metal | hardware |
| CUDA | hardware |
| HIP | hardware on GPUs with image support; software on CDNA GPUs such as the MI300 |
| Level Zero | hardware on GPUs with samplers; software on Xe-HPC (Ponte Vecchio) |

## Cooperate in a workgroup

On a GPU, the iterations of a launch run in *workgroups* of 256 threads, also
called lanes. The threads of one workgroup can share fast on-chip memory and
wait for each other. This lets them load data once and reuse it, as a tiled
matrix product or an N-body force calculation does. Four primitives give
access to it:

- `tack.shared(dtype, size)` allocates memory shared by the workgroup's
  threads. `tack.shared_like(array, size)` takes the element type from an
  array argument.
- `tack.barrier()` waits until every thread of the workgroup reaches it. It
  orders the workgroup's reads and writes of shared and global memory.
- `tack.thread_id()` is the thread's index in its workgroup, from 0 to 255.
- `tack.block_sum`, `tack.block_min` and `tack.block_max` reduce an `f32`
  value over the workgroup. [Reductions, scans and sorts](reductions-scans-sorts.md)
  covers them.

The [N-body tutorial](../users-guide/tutorials/nbody.md) loads positions in
tiles of 256. Each thread loads one position into shared memory, and then
every thread reads all 256:

```python
--8<-- "docs/examples/nbody.py:tiled"
```

The first barrier makes sure the tile is full before any thread reads it. The
second makes sure every thread has finished reading before the next tile
overwrites it.

### Follow the workgroup rules

A barrier works only if every thread of the workgroup reaches it. Tack checks
the conditions under which that holds, and rejects a kernel it cannot prove
safe:

- In a kernel with a barrier or a block reduction, every workgroup must be
  full. The launch's iteration count must be a multiple of 256.
- A barrier or block reduction may sit inside a branch or loop only when its
  condition is the same for every thread: a constant, a scalar argument, an
  array's shape. A condition that depends on `thread_id()` or on a value
  loaded from an array is rejected.
- Shared memory starts with unspecified values. Write it before you read it,
  and keep its indices in bounds.

A rejected kernel raises `ValueError` naming the kernel, the primitive and the
place. The [language contract](../reference/language-contract.md#workgroups-and-synchronization)
lists the cases exactly.

### Keep a path for the CPU

The CPU has no workgroups, and Tack does not emulate them. A kernel that uses
any of these primitives is rejected on the CPU before it compiles, even if the
primitive sits in a branch that never runs. Keep a plain kernel beside the
tiled one, and choose by what the device supports:

```python
--8<-- "docs/examples/nbody.py:choose"
```

<!-- tack-check: advanced-workgroups-cpu -->
```python
import tack

tack.set_default_device("cpu")


@tack.kernel
def partial_sums(data, sums):
    for i in range(data.shape[0]):
        total = tack.block_sum(data[i])           # one value per workgroup
        if tack.thread_id() == 0:
            tack.atomic_add(sums, 0, total)


data = tack.ones(512)
sums = tack.zeros(1)
try:
    partial_sums(data, sums)                      # runs on "cuda", "metal", ...
except RuntimeError as error:
    assert "does not support workgroup execution" in str(error)
else:
    raise AssertionError("the CPU ran a workgroup kernel")
assert data.sum() == 512.0                        # the portable reduction
```

For scratch memory private to one iteration, rather than shared by a
workgroup, use `tack.local_array`, which works on every device. See
[Device functions](device-functions.md).

## Details

- **Atomic alignment.** An atomic's array must be aligned for its element
  type. Tack checks this for wrapped pointers too.
- **Atomic scope.** Atomics are relaxed. They are indivisible across the CPU's
  worker threads, or across the workgroups of one GPU, and are not fences
  between the host and the device.
- **Floating-point minimum and maximum.** `atomic_min` and `atomic_max` on
  floats are defined for finite values and zeros. Between `-0.0` and `0.0`
  either may win. NaNs and infinities are outside their domain.
- **Shared memory sizes.** A shared array's size must be known when the
  kernel compiles: a literal, a `tack.constant`, a class constant, or an
  expression of those and array dimensions.
- **Texture limits.** The array must be `f32` and hold exactly `W * H * D`
  values. Linear interpolation is the only mode. Hardware filtering computes
  its weights at reduced precision, so GPU samples can differ from the CPU's
  in the low bits.
- **Workgroups on Metal and Level Zero.** A device whose limits do not admit
  256 threads a group rejects workgroup kernels rather than run smaller
  groups.

The [language contract](../reference/language-contract.md#atomic-array-updates)
gives the precise rules for atomics, and the
[backend implementations note](../design/backend-implementations.md) explains
how each device stores textures.

## Where to go next

- [Memory and parallel correctness](memory-and-correctness.md), for when an
  atomic is needed and when a gather avoids it.
- [Reductions, scans and sorts](reductions-scans-sorts.md), for sums and
  counts that are reproducible.
- The [N-body tutorial](../users-guide/tutorials/nbody.md), for workgroups in
  a complete program.
