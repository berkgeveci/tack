"""21 -- VTK-style array abstraction with compile-time dispatch.

Demonstrates using @pgc.data_oriented templates to build a generic
array interface where different array types (AOS, constant, affine)
all expose the same get_value/set_value API.  The template system
inlines the correct implementation at compile time -- a ConstantArray
compiles to a literal constant, an AffineArray compiles to a multiply-add,
and an AOSArray compiles to a buffer load.

This is analogous to VTK's vtkDataArray hierarchy where GetTypedValue,
SetTypedValue, etc. provide a uniform interface over AOS arrays,
SOA arrays, constant arrays, affine arrays, and other implicit arrays.

Usage:
  uv run python examples/21_array_abstraction.py
"""

import numpy as np
import pgc

import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument('--arch', default='cpu', choices=['cpu', 'metal', 'cuda', 'hip', 'level_zero', 'wgpu'])
_arch = getattr(pgc, _parser.parse_args().arch)
pgc.init(arch=_arch)


# ---------------------------------------------------------------------------
# Array types -- all share the same get_value / set_value interface
# ---------------------------------------------------------------------------

@pgc.data_oriented
class AOSArray:
    """Backed by a pgc.field -- standard memory-resident array."""

    def __init__(self, data):
        self.data = data  # Field -> kernel buffer parameter

    @pgc.func
    def get_value(self, i):
        return self.data[i]

    @pgc.func
    def set_value(self, i, val):
        self.data[i] = val


@pgc.data_oriented
class ConstantArray:
    """O(1) memory -- every element has the same value."""

    def __init__(self, value):
        self.value = value  # scalar -> compile-time constant

    @pgc.func
    def get_value(self, i):
        return self.value


@pgc.data_oriented
class AffineArray:
    """O(1) memory -- value[i] = slope * i + intercept."""

    def __init__(self, slope, intercept):
        self.slope = slope        # scalar -> compile-time constant
        self.intercept = intercept  # scalar -> compile-time constant

    @pgc.func
    def get_value(self, i):
        return self.slope * float(i) + self.intercept


# ---------------------------------------------------------------------------
# Generic kernels -- work with ANY array type via the template interface
# ---------------------------------------------------------------------------

@pgc.kernel
def add_arrays(input1: pgc.template(), input2: pgc.template(),
               output: pgc.template(), n):
    """output[i] = input1[i] + input2[i] for any array types."""
    for i in range(n):
        output.set_value(i, input1.get_value(i) + input2.get_value(i))


@pgc.kernel
def scale_array(input1: pgc.template(), scalar: pgc.template(),
                output: pgc.template(), n):
    """output[i] = input1[i] * scalar[i] -- scalar can be a ConstantArray."""
    for i in range(n):
        output.set_value(i, input1.get_value(i) * scalar.get_value(i))


# ---------------------------------------------------------------------------
# Test 1: AOS + AOS -> AOS (standard vector add)
# ---------------------------------------------------------------------------
n = 1000

a_data = pgc.field(dtype=pgc.f32, shape=(n,))
b_data = pgc.field(dtype=pgc.f32, shape=(n,))
out_data = pgc.field(dtype=pgc.f32, shape=(n,))

a_np = np.arange(n, dtype=np.float32)
b_np = np.arange(n, dtype=np.float32) * 2.0
a_data.from_numpy(a_np)
b_data.from_numpy(b_np)

a = AOSArray(a_data)
b = AOSArray(b_data)
out = AOSArray(out_data)

add_arrays(a, b, out, n)
result = out_data.to_numpy()
expected = a_np + b_np
assert np.allclose(result, expected), f"AOS+AOS failed: max err {np.max(np.abs(result - expected))}"
print("Test 1 -- AOS + AOS -> AOS: OK")


# ---------------------------------------------------------------------------
# Test 2: AOS + Constant -> AOS (add scalar offset to every element)
# ---------------------------------------------------------------------------
offset = ConstantArray(42.0)

add_arrays(a, offset, out, n)
result = out_data.to_numpy()
expected = a_np + 42.0
assert np.allclose(result, expected), f"AOS+Constant failed"
print("Test 2 -- AOS + Constant -> AOS: OK")


# ---------------------------------------------------------------------------
# Test 3: AOS + Affine -> AOS (add linearly varying values)
# ---------------------------------------------------------------------------
ramp = AffineArray(0.5, 10.0)  # value[i] = 0.5 * i + 10.0

add_arrays(a, ramp, out, n)
result = out_data.to_numpy()
expected = a_np + (0.5 * np.arange(n) + 10.0).astype(np.float32)
assert np.allclose(result, expected, atol=1e-3), f"AOS+Affine failed"
print("Test 3 -- AOS + Affine -> AOS: OK")


# ---------------------------------------------------------------------------
# Test 4: Constant + Affine -> AOS (no input buffers at all!)
# ---------------------------------------------------------------------------
c = ConstantArray(100.0)

add_arrays(c, ramp, out, n)
result = out_data.to_numpy()
expected = (100.0 + 0.5 * np.arange(n) + 10.0).astype(np.float32)
assert np.allclose(result, expected, atol=1e-3), f"Constant+Affine failed"
print("Test 4 -- Constant + Affine -> AOS: OK (no input buffers needed!)")


# ---------------------------------------------------------------------------
# Test 5: Scale AOS by Constant (multiply every element by 3.0)
# ---------------------------------------------------------------------------
factor = ConstantArray(3.0)

scale_array(a, factor, out, n)
result = out_data.to_numpy()
expected = a_np * 3.0
assert np.allclose(result, expected), f"Scale by constant failed"
print("Test 5 -- AOS * Constant -> AOS: OK")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\nAll tests passed on {_parser.parse_args().arch} backend!")
print("""
Key insight: the same add_arrays / scale_array kernels work with any
combination of AOSArray, ConstantArray, and AffineArray.  The template
system inlines the correct get_value/set_value at compile time:

  - ConstantArray.get_value(i) -> literal constant (zero memory access)
  - AffineArray.get_value(i)   -> slope * i + intercept (no buffer)
  - AOSArray.get_value(i)      -> buffer[i] (normal load)

This is the same pattern as VTK's vtkDataArray hierarchy where
GetTypedValue/SetTypedValue provide a uniform interface over AOS arrays,
constant arrays, affine arrays, and other implicit array types.
""")
