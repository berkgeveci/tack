"""Test tack.inspect() — kernel code inspection without execution."""

import pytest

import tack


@tack.kernel
def vector_add(x, y, out):
    for i in range(len(x)):
        out[i] = x[i] + y[i]


def _make_fields():
    n = 64
    x = tack.field(dtype=tack.f32, shape=(n,))
    y = tack.field(dtype=tack.f32, shape=(n,))
    out = tack.field(dtype=tack.f32, shape=(n,))
    return x, y, out


def test_inspect_ir(backend):
    x, y, out = _make_fields()
    result = tack.inspect(vector_add, x, y, out, mode="ir")
    assert isinstance(result, str)
    assert "Function" in result
    assert "ParallelFor" in result


# What each backend's generated source has to contain: the spelling of its
# kernel entry point. Every backend is listed, and an unknown one is a
# failure rather than a pass -- this test used to check "cpu" and "metal"
# and fall through everywhere else, so on a CUDA machine it asserted only
# that the string was longer than fifty characters.
_ENTRY_POINT = {
    "cpu": "define",              # LLVM IR
    "metal": "kernel void",       # MSL
    "cuda": "__global__",         # CUDA C
    "hip": "__global__",          # HIP C, which extends the CUDA codegen
    "level_zero": "__kernel",     # OpenCL C
}


def test_inspect_source(backend):
    x, y, out = _make_fields()
    result = tack.inspect(vector_add, x, y, out, mode="source")
    assert isinstance(result, str)
    assert len(result) > 50

    assert backend in _ENTRY_POINT, (
        f"no entry point known for {backend!r}; add it rather than let this "
        f"test pass without checking anything")
    assert _ENTRY_POINT[backend] in result


def test_hip_source_carries_its_runtime_header(backend):
    """HIP shares the CUDA codegen and differs by exactly this include."""
    if backend != "hip":
        pytest.skip("HIP only")
    x, y, out = _make_fields()
    result = tack.inspect(vector_add, x, y, out, mode="source")
    assert "#include <hip/hip_runtime.h>" in result


def test_inspect_default_mode(backend):
    x, y, out = _make_fields()
    result = tack.inspect(vector_add, x, y, out)
    assert isinstance(result, str)
    assert len(result) > 50


def test_inspect_invalid_mode(backend):
    x, y, out = _make_fields()
    with pytest.raises(ValueError, match="Unknown inspect mode"):
        tack.inspect(vector_add, x, y, out, mode="bad")


def test_inspect_not_a_kernel(backend):
    with pytest.raises(TypeError, match="Expected a @tack.kernel"):
        tack.inspect(lambda: None, mode="ir")


def test_inspect_scalar_args(backend):
    @tack.kernel
    def scale(x, out, factor):
        for i in range(len(x)):
            out[i] = x[i] * factor

    n = 64
    x = tack.field(dtype=tack.f32, shape=(n,))
    out = tack.field(dtype=tack.f32, shape=(n,))
    result = tack.inspect(scale, x, out, 2.0, mode="source")
    assert isinstance(result, str)
    assert len(result) > 50


def test_optimized_mode_is_cpu_only(backend):
    """There is no Tack-visible optimized form of GPU source to return.

    It used to hand back the unoptimized source under the optimized name.
    """
    x, y, out = _make_fields()
    if backend == "cpu":
        assert "define" in tack.inspect(vector_add, x, y, out, mode="optimized")
        return
    with pytest.raises(ValueError, match="CPU only"):
        tack.inspect(vector_add, x, y, out, mode="optimized")


@pytest.mark.parametrize("mode", ["ir", "source"])
def test_inspect_rejects_dtypes_dispatch_would(monkeypatch, mode):
    """Inspection must not show code for a call dispatch would refuse."""
    import numpy as np

    from tack.runtime.dispatch import get_backend

    tack.init(arch=tack.cpu)
    backend = get_backend()
    monkeypatch.setattr(backend, "supported_dtypes",
                        backend.supported_dtypes - {tack.f64})
    x = tack.field(dtype=tack.f64, shape=(4,))
    out = tack.field(dtype=tack.f64, shape=(4,))
    x.from_numpy(np.zeros(4))

    @tack.kernel
    def copy(x, out):
        for i in range(x.shape[0]):
            out[i] = x[i]

    with pytest.raises(TypeError, match="not supported on"):
        tack.inspect(copy, x, out, mode=mode)


def test_inspect_takes_the_backends_texture_decision(monkeypatch):
    """HIP and Level Zero sample in software on devices without texture
    hardware, which changes the generated code. Inspection has to ask the
    backend, as dispatch does, rather than assume hardware sampling."""
    import numpy as np

    from tack.runtime.dispatch import get_backend

    tack.init(arch=tack.cpu)
    backend = get_backend()
    seen = []

    def software_only(ir_func, effective_args):
        for param, arg in zip(ir_func.params, effective_args):
            if isinstance(arg, tack.Texture3D):
                seen.append(param.name)
                param._is_texture = False

    monkeypatch.setattr(backend, "_store_texture_shapes", software_only)

    @tack.kernel
    def sample(tex, out):
        for i in range(out.shape[0]):
            out[i] = tex.sample(0.5, 0.5, 0.5)

    data = tack.field(tack.f32, (8,))
    data.from_numpy(np.ones(8, dtype=np.float32))
    tex = tack.texture3d(data, shape=(2, 2, 2))
    tack.inspect(sample, tex, tack.field(tack.f32, (3,)), mode="source")
    assert seen == ["tex"]
