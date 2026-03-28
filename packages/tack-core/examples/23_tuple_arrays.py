"""23 -- Multi-component tuple arrays: AOS vs SOA layouts.

Inspired by VTK's vtkDataArray hierarchy.  Both AOS and SOA array types
expose the same get_value(i, comp) / set_value(i, comp, val) interface
via @tack.data_oriented templates.  Kernels written against this interface
work with either layout -- the template system inlines the correct indexing.

AOS (Array of Structures):
    Storage: one flat field -- [x0, y0, z0, x1, y1, z1, ...]
    get_value(i, c) = data[i * num_components + c]

SOA (Structure of Arrays):
    Storage: one field per component -- x=[x0, x1, ...], y=[y0, y1, ...], ...
    get_value(i, c) dispatches to the correct component field

Storage is provided externally, just like example 21.

Usage:
  uv run python examples/23_tuple_arrays.py
  uv run python examples/23_tuple_arrays.py --arch metal
"""

import numpy as np
import linecache
import tack

import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument('--arch', default='cpu', choices=['cpu', 'metal', 'cuda', 'hip', 'level_zero'])
_arch = getattr(tack, _parser.parse_args().arch)
tack.init(arch=_arch)


# ================================================================
# ARRAY TYPES -- same get_value/set_value interface, different layouts
# ================================================================

@tack.data_oriented
class AOSTupleArray:
    """Multi-component array in Array-of-Structures layout.

    Takes a single externally-provided field of size num_tuples * num_components.
    Memory: [x0, y0, z0, x1, y1, z1, ...]
    """

    def __init__(self, data, num_tuples, num_components):
        self.data = data                # tack.field -- provided externally
        self.num_tuples = num_tuples
        self.num_components = num_components

    @tack.func
    def get_value(self, i, c):
        return self.data[i * self.num_components + c]

    @tack.func
    def set_value(self, i, c, val):
        self.data[i * self.num_components + c] = val


# ================================================================
# SOA FACTORY -- generates SOATupleArray classes for any N components
# ================================================================
# Tack templates need named field attributes (self.c0, self.c1, ...),
# and @tack.func needs inspectable source.  We generate the class
# source dynamically and register it with linecache so that
# inspect.getsource() works.

_soa_cache = {}  # nc -> class


def make_soa_type(nc):
    """Generate a @tack.data_oriented SOA class with nc component fields."""
    if nc in _soa_cache:
        return _soa_cache[nc]

    params = ", ".join(f"c{i}" for i in range(nc))
    init_body = "\n".join(f"        self.c{i} = c{i}" for i in range(nc))

    # get_value: conditional chain dispatching to the right component
    get_lines = ["        result = self.c0[i]"]
    for i in range(1, nc):
        get_lines.append(f"        if c == {i}:")
        get_lines.append(f"            result = self.c{i}[i]")
    get_lines.append("        return result")
    get_body = "\n".join(get_lines)

    # set_value: conditional chain writing to the right component
    set_lines = []
    for i in range(nc):
        set_lines.append(f"        if c == {i}:")
        set_lines.append(f"            self.c{i}[i] = val")
    set_body = "\n".join(set_lines)

    source = f"""@tack.data_oriented
class SOATupleArray{nc}:
    def __init__(self, {params}):
{init_body}

    @tack.func
    def get_value(self, i, c):
{get_body}

    @tack.func
    def set_value(self, i, c, val):
{set_body}
"""
    # Register source with linecache so inspect.getsource() works
    filename = f"<soa_tuple_array_{nc}>"
    lines = source.splitlines(True)
    linecache.cache[filename] = (len(source), None, lines, filename)

    code = compile(source, filename, "exec")
    ns = {"tack": tack}
    exec(code, ns)

    cls = ns[f"SOATupleArray{nc}"]
    _soa_cache[nc] = cls
    return cls


# ================================================================
# GENERIC KERNELS -- work with either AOS or SOA
# ================================================================

@tack.kernel
def compute_magnitude(arr: tack.template(), output, n):
    """Compute 3D vector magnitude: sqrt(x^2 + y^2 + z^2)."""
    for i in range(n):
        x = arr.get_value(i, 0)
        y = arr.get_value(i, 1)
        z = arr.get_value(i, 2)
        output[i] = sqrt(x * x + y * y + z * z)


@tack.kernel
def add_tuple_arrays(a: tack.template(), b: tack.template(),
                     out: tack.template(), n, nc):
    """Component-wise addition: out[i,c] = a[i,c] + b[i,c]."""
    for i in range(n):
        for c in range(nc):
            out.set_value(i, c, a.get_value(i, c) + b.get_value(i, c))


@tack.kernel
def scale_tuple_array(arr: tack.template(), factor, out: tack.template(), n, nc):
    """Scale every component: out[i,c] = arr[i,c] * factor."""
    for i in range(n):
        for c in range(nc):
            out.set_value(i, c, arr.get_value(i, c) * factor)


@tack.kernel
def dot_product(a: tack.template(), b: tack.template(), output, n, nc):
    """Per-tuple dot product: output[i] = sum_c(a[i,c] * b[i,c])."""
    for i in range(n):
        s = 0.0
        for c in range(nc):
            s = s + a.get_value(i, c) * b.get_value(i, c)
        output[i] = s


# ================================================================
# HELPERS
# ================================================================

def make_aos(data_np):
    """Create an AOSTupleArray from a (num_tuples, num_components) numpy array."""
    nt, nc = data_np.shape
    field = tack.field(dtype=tack.f32, shape=(nt * nc,))
    field.from_numpy(data_np.ravel().astype(np.float32))
    return AOSTupleArray(field, nt, nc)


def aos_to_numpy(arr):
    """Read back an AOSTupleArray to (num_tuples, num_components) numpy."""
    return arr.data.to_numpy().reshape(arr.num_tuples, arr.num_components)


def make_soa(data_np):
    """Create a SOATupleArray from a (num_tuples, nc) numpy array."""
    nt, nc = data_np.shape
    SOAType = make_soa_type(nc)
    fields = []
    for c in range(nc):
        f = tack.field(dtype=tack.f32, shape=(nt,))
        f.from_numpy(data_np[:, c].astype(np.float32).copy())
        fields.append(f)
    return SOAType(*fields)


def soa_to_numpy(arr, nc):
    """Read back a SOATupleArray to (num_tuples, nc) numpy."""
    columns = [getattr(arr, f"c{c}").to_numpy() for c in range(nc)]
    return np.column_stack(columns)


# ================================================================
# TEST
# ================================================================

n = 1000
nc = 3

# Random 3-component data
np.random.seed(42)
data_a = np.random.randn(n, nc).astype(np.float32)
data_b = np.random.randn(n, nc).astype(np.float32)

output = tack.field(dtype=tack.f32, shape=(n,))


# --- AOS tests ---
print("--- AOS layout ---")

a = make_aos(data_a)
b = make_aos(data_b)
out_field = tack.field(dtype=tack.f32, shape=(n * nc,))
out = AOSTupleArray(out_field, n, nc)

np.testing.assert_allclose(aos_to_numpy(a), data_a, atol=1e-6)

compute_magnitude(a, output, n)
expected = np.sqrt(np.sum(data_a ** 2, axis=1))
assert np.allclose(output.to_numpy(), expected, atol=1e-4)
print("  AOS magnitude: OK")

add_tuple_arrays(a, b, out, n, nc)
assert np.allclose(aos_to_numpy(out), data_a + data_b, atol=1e-4)
print("  AOS add: OK")

scale_tuple_array(a, 2.5, out, n, nc)
assert np.allclose(aos_to_numpy(out), data_a * 2.5, atol=1e-4)
print("  AOS scale: OK")

dot_product(a, b, output, n, nc)
expected = np.sum(data_a * data_b, axis=1)
assert np.allclose(output.to_numpy(), expected, atol=1e-3)
print("  AOS dot: OK")


# --- SOA tests ---
print("\n--- SOA layout ---")

SOAType = make_soa_type(nc)

a = make_soa(data_a)
b = make_soa(data_b)
out = SOAType(
    tack.field(dtype=tack.f32, shape=(n,)),
    tack.field(dtype=tack.f32, shape=(n,)),
    tack.field(dtype=tack.f32, shape=(n,)),
)

np.testing.assert_allclose(soa_to_numpy(a, nc), data_a, atol=1e-6)

compute_magnitude(a, output, n)
expected = np.sqrt(np.sum(data_a ** 2, axis=1))
assert np.allclose(output.to_numpy(), expected, atol=1e-4)
print("  SOA magnitude: OK")

add_tuple_arrays(a, b, out, n, nc)
assert np.allclose(soa_to_numpy(out, nc), data_a + data_b, atol=1e-4)
print("  SOA add: OK")

scale_tuple_array(a, 2.5, out, n, nc)
assert np.allclose(soa_to_numpy(out, nc), data_a * 2.5, atol=1e-4)
print("  SOA scale: OK")

dot_product(a, b, output, n, nc)
expected = np.sum(data_a * data_b, axis=1)
assert np.allclose(output.to_numpy(), expected, atol=1e-3)
print("  SOA dot: OK")


# --- Mixed layout tests ---
print("\n--- Mixed layouts ---")

def make_soa_output(nc, n):
    return SOAType(*[tack.field(dtype=tack.f32, shape=(n,)) for _ in range(nc)])

aos_a = make_aos(data_a)
soa_b = make_soa(data_b)
aos_out_field = tack.field(dtype=tack.f32, shape=(n * nc,))
aos_out = AOSTupleArray(aos_out_field, n, nc)
soa_out = make_soa_output(nc, n)
expected = data_a + data_b

# AOS + SOA -> AOS
add_tuple_arrays(aos_a, soa_b, aos_out, n, nc)
assert np.allclose(aos_to_numpy(aos_out), expected, atol=1e-4)
print("  AOS + SOA -> AOS: OK")

# AOS + SOA -> SOA
add_tuple_arrays(aos_a, soa_b, soa_out, n, nc)
assert np.allclose(soa_to_numpy(soa_out, nc), expected, atol=1e-4)
print("  AOS + SOA -> SOA: OK")

# SOA + AOS -> SOA
soa_a = make_soa(data_a)
aos_b = make_aos(data_b)
add_tuple_arrays(soa_a, aos_b, soa_out, n, nc)
assert np.allclose(soa_to_numpy(soa_out, nc), expected, atol=1e-4)
print("  SOA + AOS -> SOA: OK")

# --- Arbitrary component count ---
print("\n--- 5-component SOA (arbitrary N) ---")

nc5 = 5
data_5a = np.random.randn(n, nc5).astype(np.float32)
data_5b = np.random.randn(n, nc5).astype(np.float32)

soa5_a = make_soa(data_5a)
soa5_b = make_soa(data_5b)
SOA5 = make_soa_type(nc5)
soa5_out = SOA5(*[tack.field(dtype=tack.f32, shape=(n,)) for _ in range(nc5)])

add_tuple_arrays(soa5_a, soa5_b, soa5_out, n, nc5)
assert np.allclose(soa_to_numpy(soa5_out, nc5), data_5a + data_5b, atol=1e-4)
print("  SOA(5) add: OK")

dot_product(soa5_a, soa5_b, output, n, nc5)
expected = np.sum(data_5a * data_5b, axis=1)
assert np.allclose(output.to_numpy(), expected, atol=1e-3)
print("  SOA(5) dot: OK")

# AOS(5) + SOA(5)
aos5_a = make_aos(data_5a)
add_tuple_arrays(aos5_a, soa5_b, soa5_out, n, nc5)
assert np.allclose(soa_to_numpy(soa5_out, nc5), data_5a + data_5b, atol=1e-4)
print("  AOS(5) + SOA(5) -> SOA(5): OK")


print(f"\nAll tests passed!")
print("""
Key insight: the same kernels (compute_magnitude, add_tuple_arrays,
scale_tuple_array, dot_product) work with ANY combination of AOS and
SOA arrays.  Storage is provided externally:

  AOS: one flat field, data[i * nc + c]
  SOA: N separate fields (c0, c1, ...), generated by make_soa_type(N)

make_soa_type(nc) generates @tack.data_oriented classes for any number
of components.  You can mix layouts freely in a single kernel call.
""")
