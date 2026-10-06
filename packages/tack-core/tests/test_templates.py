"""Tests for @tack.data_oriented template parameters."""

import numpy as np
import pytest

import tack

# Build list of available backends


# --- Structured cell set (scalar attrs only, no field attrs) ---

@tack.data_oriented
class CellSetStructured:
    def __init__(self, nx, ny):
        self.nx = nx
        self.ny = ny

    @tack.func
    def get_cell_point0(self, cell_id):
        cx = cell_id % (self.nx - 1)
        cy = cell_id // (self.nx - 1)
        return cy * self.nx + cx


# --- Explicit cell set (field attrs) ---

@tack.data_oriented
class CellSetExplicit:
    def __init__(self, conn_np):
        self.connectivity = tack.field(dtype=tack.i32, shape=(len(conn_np),))
        self.connectivity.from_numpy(conn_np)

    @tack.func
    def get_cell_point0(self, cell_id):
        return self.connectivity[cell_id * 4]


# --- Kernels that use template objects ---

@tack.kernel
def extract_point0(cell_set, output):
    for i in range(output.shape[0]):
        output[i] = cell_set.get_cell_point0(i)


def test_structured_template(backend):
    """Template with scalar attrs only (nx, ny become constants)."""
    nx, ny = 4, 3
    num_cells = (nx - 1) * (ny - 1)
    output = tack.field(dtype=tack.i32, shape=(num_cells,))

    cs = CellSetStructured(nx, ny)
    extract_point0(cs, output)

    result = output.to_numpy()
    expected = np.array([0, 1, 2, 4, 5, 6], dtype=np.int32)
    np.testing.assert_array_equal(result, expected)


def test_explicit_template(backend):
    """Template with field attrs (connectivity becomes extra parameter)."""
    # Build connectivity for a 4x3 structured grid
    nx, ny = 4, 3
    num_cells = (nx - 1) * (ny - 1)
    conn = np.empty(num_cells * 4, dtype=np.int32)
    for cj in range(ny - 1):
        for ci in range(nx - 1):
            cid = cj * (nx - 1) + ci
            p0 = cj * nx + ci
            conn[cid * 4: cid * 4 + 4] = [p0, p0 + 1, p0 + nx + 1, p0 + nx]

    output = tack.field(dtype=tack.i32, shape=(num_cells,))
    cs = CellSetExplicit(conn)
    extract_point0(cs, output)

    result = output.to_numpy()
    expected = np.array([0, 1, 2, 4, 5, 6], dtype=np.int32)
    np.testing.assert_array_equal(result, expected)


def test_same_kernel_different_templates(backend):
    """Same kernel works with different template types."""
    nx, ny = 4, 3
    num_cells = (nx - 1) * (ny - 1)

    # Build explicit connectivity
    conn = np.empty(num_cells * 4, dtype=np.int32)
    for cj in range(ny - 1):
        for ci in range(nx - 1):
            cid = cj * (nx - 1) + ci
            p0 = cj * nx + ci
            conn[cid * 4: cid * 4 + 4] = [p0, p0 + 1, p0 + nx + 1, p0 + nx]

    output_s = tack.field(dtype=tack.i32, shape=(num_cells,))
    output_e = tack.field(dtype=tack.i32, shape=(num_cells,))

    cs_s = CellSetStructured(nx, ny)
    cs_e = CellSetExplicit(conn)

    extract_point0(cs_s, output_s)
    extract_point0(cs_e, output_e)

    np.testing.assert_array_equal(output_s.to_numpy(), output_e.to_numpy())


def test_different_scalar_values(backend):
    """Same template type with different scalar values produces different code."""
    cs_a = CellSetStructured(4, 3)
    cs_b = CellSetStructured(5, 4)

    output_a = tack.field(dtype=tack.i32, shape=(6,))
    output_b = tack.field(dtype=tack.i32, shape=(12,))

    extract_point0(cs_a, output_a)
    extract_point0(cs_b, output_b)

    # 4x3 grid: point0 of cells
    expected_a = np.array([0, 1, 2, 4, 5, 6], dtype=np.int32)
    np.testing.assert_array_equal(output_a.to_numpy(), expected_a)

    # 5x4 grid: point0 of cells
    expected_b = np.array([0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13], dtype=np.int32)
    np.testing.assert_array_equal(output_b.to_numpy(), expected_b)


# --- Vector field attributes ---------------------------------------------------

@tack.data_oriented
class _Flow:
    """A template whose state includes a vector field its methods read."""

    def __init__(self, n):
        self.rho = tack.field(dtype=tack.f32, shape=(n, n))
        self.vel = tack.Vector.field(2, dtype=tack.f32, shape=(n, n))

    @tack.func
    def momentum_x(self, i, j, k):
        u = self.vel[i, j]
        return self.rho[i, j] * u[0] * float(k)


@tack.kernel
def _momentum_direct(flow: tack.template(), out, n):
    for i, j in tack.ndrange(n, n):
        u = flow.vel[i, j]
        out[i, j] = flow.rho[i, j] * u[1]


@tack.kernel
def _momentum_through_method(flow: tack.template(), out, n):
    for i, j in tack.ndrange(n, n):
        acc = 0.0
        for k in range(3):
            acc = acc + flow.momentum_x(i, j, k)
        out[i, j] = acc


def test_template_vector_field_attribute_in_kernel_and_method(backend):
    """A template's vector field attribute loads whole vectors, whether the
    kernel reads it directly or a method does. The runtime detected vector
    widths only for direct kernel arguments, so the synthetic parameter the
    attribute became was a scalar field and the method's load failed IR
    verification."""
    n = 4
    flow = _Flow(n)
    flow.rho.from_numpy(np.full((n, n), 2.0, np.float32))
    vel = np.zeros((n, n, 2), np.float32)
    vel[..., 0] = 3.0
    vel[..., 1] = 5.0
    flow.vel.from_numpy(vel.reshape(-1))
    out = tack.field(dtype=tack.f32, shape=(n, n))

    _momentum_direct(flow, out, n)
    np.testing.assert_array_equal(out.to_numpy(), np.full((n, n), 10.0, np.float32))

    _momentum_through_method(flow, out, n)
    np.testing.assert_array_equal(out.to_numpy(), np.full((n, n), 18.0, np.float32))


# --- Inheritance, and device functions held as attributes ---

@tack.func
def _cubic_falloff(r, h):
    return max(0.0, 1.0 - r / h) ** 3


@tack.func
def _linear_falloff(r, h):
    return max(0.0, 1.0 - r / h)


@tack.data_oriented
class _BaseModel:
    scale = 2.0

    def __init__(self, n, h):
        self.h = h
        self.x = tack.field(dtype=tack.f32, shape=(n,))
        self.x.from_numpy(np.arange(n, dtype=np.float32) * 0.5)
        self.out = tack.field(dtype=tack.f32, shape=(n,))

    @tack.func
    def weight(self, r):
        return _cubic_falloff(r, self.h) * self.scale

    @tack.func
    def shaped(self, r):
        return self.weight(r)


@tack.data_oriented
class _DerivedModel(_BaseModel):
    extra = 10.0

    def __init__(self, n, h, gamma):
        super().__init__(n, h)
        self.gamma = gamma

    @tack.func
    def shaped(self, r):
        return self.weight(r) ** self.gamma + self.extra


class _UndecoratedModel(_BaseModel):
    """A subclass without the decorator still gets its own methods."""
    scale = 3.0

    @tack.func
    def shaped(self, r):
        return self.weight(r) + 1.0


@tack.data_oriented
class _ModelWithKernel:
    def __init__(self, n, falloff):
        self.falloff = falloff          # a device function held as an attribute
        self.h = 2.0
        self.x = tack.field(dtype=tack.f32, shape=(n,))
        self.x.from_numpy(np.arange(n, dtype=np.float32) * 0.5)
        self.out = tack.field(dtype=tack.f32, shape=(n,))

    @tack.func
    def twice(self, r):
        return 2.0 * self.falloff(r, self.h)


@tack.kernel
def _apply_shaped(s: tack.template(), n):
    for i in range(n):
        s.out[i] = s.shaped(s.x[i])


@tack.kernel
def _apply_falloff(s: tack.template(), n):
    for i in range(n):
        s.out[i] = s.falloff(s.x[i], s.h) + s.twice(s.x[i])


def test_data_oriented_subclass_inherits_methods_and_constants(backend):
    """A subclass sees the base's @tack.func methods and class constants,
    and may override either. Inherited methods used to be unknown:
    "self.weight ... is neither a class constant ... nor a @tack.func method"."""
    n = 6
    x = np.arange(n, dtype=np.float32) * 0.5
    cubic = np.maximum(0, 1 - x / 2.0) ** 3

    base = _BaseModel(n, 2.0)
    _apply_shaped(base, n)
    np.testing.assert_allclose(base.out.to_numpy(), cubic * 2.0, rtol=1e-6)

    derived = _DerivedModel(n, 2.0, 2.0)
    _apply_shaped(derived, n)
    np.testing.assert_allclose(derived.out.to_numpy(), (cubic * 2.0) ** 2 + 10.0, rtol=1e-5)

    undecorated = _UndecoratedModel(n, 2.0)
    _apply_shaped(undecorated, n)
    np.testing.assert_allclose(undecorated.out.to_numpy(), cubic * 3.0 + 1.0, rtol=1e-6)


def test_device_function_held_as_an_attribute(backend):
    """`self.falloff = cubic` lets methods and kernels call `self.falloff(...)`.
    Which function it holds is part of the specialization: the same kernel
    compiles again for an object holding another one."""
    n = 6
    x = np.arange(n, dtype=np.float32) * 0.5
    for falloff, want in ((_cubic_falloff, np.maximum(0, 1 - x / 2.0) ** 3),
                          (_linear_falloff, np.maximum(0, 1 - x / 2.0)),
                          (_cubic_falloff, np.maximum(0, 1 - x / 2.0) ** 3)):
        model = _ModelWithKernel(n, falloff)
        _apply_falloff(model, n)
        np.testing.assert_allclose(model.out.to_numpy(), 3.0 * want, rtol=1e-6)


def test_unknown_template_method_is_reported():
    @tack.kernel
    def bad(s: tack.template(), n):
        for i in range(n):
            s.out[i] = s.missing(s.x[i])
    with pytest.raises(Exception, match="no @tack.func method 'missing'"):
        bad(_BaseModel(2, 1.0), 2)
