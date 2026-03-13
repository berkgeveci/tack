# Portable GPU Compute Framework — Project Plan

## Vision

A Python-first portable GPU compute framework inspired by VTK-m/Viskores worklet patterns with a Taichi-style Python interface. Kernels written as decorated Python functions, compiled via LLVM IR to SPIR-V (universal GPU format) or native JIT (CPU).

## Target Platforms

| Platform | Runtime | Priority |
|----------|---------|----------|
| Apple Metal | Metal Compute | Phase 1 (dev machine) |
| CPU (threads) | llvmlite native JIT | Phase 1 |
| AMD ROCm/HIP | HIP runtime | Phase 2 (Frontier) |
| Intel GPUs | Level Zero | Phase 3 (Aurora) |
| NVIDIA CUDA | Vulkan or CUDA driver API | Phase 4 |
| Vulkan | Vulkan Compute | Opportunistic (workstations) |

## Compilation Pipeline

```
Python @kernel function
    ↓
AST transformation (lift from Taichi)
    ↓
LLVM IR generation (llvmlite)
    ↓
  ┌──────────┬──────────────┐
  ↓          ↓              ↓
CPU JIT   SPIR-V         SPIR-V
(native)  (via translator) (via translator)
  ↓          ↓              ↓
threads   Metal/Vulkan   HIP/Level Zero
          (MoltenVK or    (native runtime)
           Metal direct)
```

## API Design (Taichi-style)

```python
import pgc  # portable gpu compute — name TBD

pgc.init(arch=pgc.metal)  # or pgc.cpu, pgc.hip, pgc.vulkan

# Fields (n-d arrays living on device)
x = pgc.field(dtype=pgc.f32, shape=(1024,))
y = pgc.field(dtype=pgc.f32, shape=(1024,))
out = pgc.field(dtype=pgc.f32, shape=(1024,))

@pgc.kernel
def vector_add(x: pgc.template(), y: pgc.template(), out: pgc.template()):
    for i in range(x.shape[0]):  # auto-parallelized
        out[i] = x[i] + y[i]

# Fill from numpy
x.from_numpy(np_x)
y.from_numpy(np_y)

vector_add(x, y, out)

result = out.to_numpy()
```

### VTK-m Worklet Patterns (later phases)

```python
@pgc.worklet(pgc.MapField)
def magnitude(vec: pgc.Vec3f) -> pgc.f32:
    return pgc.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)

@pgc.worklet(pgc.Reduce)
def sum_reduce(a: pgc.f32, b: pgc.f32) -> pgc.f32:
    return a + b

# Topology worklets (later)
@pgc.worklet(pgc.PointToCell)
def cell_average(cell_points: pgc.PointGroup, field: pgc.FieldIn) -> pgc.f32:
    total = 0.0
    for i in range(cell_points.count()):
        total += field[cell_points[i]]
    return total / cell_points.count()
```

## Phase 1: CPU Backend + Metal (This Machine)

### Step 1: Project skeleton
- Package structure with `pyproject.toml` (uv-managed)
- Core modules: `lang/`, `runtime/`, `backends/`

### Step 2: AST transformation
- `@kernel` decorator captures Python function
- Python AST → simplified IR (study/lift from Taichi's `taichi/lang/ast/`)
- Support: `for` loops (parallel range), scalar math, field indexing
- Type inference for kernel arguments

### Step 3: LLVM IR codegen via llvmlite
- Transform simplified IR → LLVM IR using llvmlite's IR builder
- Parallel `for` → function that takes (start, end) range
- Field access → pointer arithmetic with proper types
- Scalar math → LLVM intrinsics where applicable

### Step 4: CPU backend
- llvmlite JIT compilation of generated LLVM IR
- Thread pool dispatch (split range across threads)
- Field backed by numpy arrays (zero-copy via ctypes pointers)

### Step 5: LLVM IR → SPIR-V
- Use `spirv-tools` or implement minimal LLVM IR → SPIR-V translator
- Alternative: use `llvm-spirv` translator (LLVM project)
- Target SPIR-V 1.0 compute shaders initially

### Step 6: Metal backend
- SPIR-V → MSL via SPIRV-Cross (already available on macOS)
- Or: SPIR-V → Metal IR directly
- Metal compute pipeline: create device, compile shader, dispatch
- Metal runtime adapter in Python (via `pyobjc` or ctypes to Metal C API)

### Testing with Taichi Examples

Port these Taichi examples as validation:
1. **Vector add** — simplest kernel, validates basic pipeline
2. **SAXPY** — scalar * vector + vector
3. **Reduction** — sum of array elements
4. **Mandelbrot** — nested loops, complex math, 2D field output
5. **N-body** — multiple fields, distance calculations
6. **Jacobi iteration** — stencil pattern, read/write fields
7. **Matrix multiply** — 2D indexing, accumulation

## Dependencies

```
# Core
llvmlite          # LLVM IR generation + CPU JIT
numpy             # Field data backing

# Metal backend
pyobjc-framework-Metal        # Metal API bindings
pyobjc-framework-MetalKit     # Metal utilities

# SPIR-V toolchain
spirv-tools       # SPIR-V validation/optimization (pip or brew)
# SPIRV-Cross installed via brew for SPIR-V → MSL

# Development
pytest
```

## Taichi Source Reference

Study and selectively lift from these files (Apache 2.0 licensed):

### AST Transformation (pure Python — most relevant)
```
~/Work/taichi/taichi/lang/ast/
├── ast_transformer.py      # Main AST visitor, transforms Python AST
├── ast_transformer_utils.py # Helper utilities
├── transform.py            # Entry point for AST pipeline
├── checkers.py             # Static checks on kernels
└── builder.py              # IR builder from AST
```

### Kernel Infrastructure
```
~/Work/taichi/taichi/lang/
├── kernel_impl.py          # @ti.kernel decorator, compilation dispatch
├── field.py                # ti.field() — device array abstraction
├── matrix.py               # ti.Vector, ti.Matrix types
├── impl.py                 # Runtime: init(), scope management
├── ops.py                  # Math operations
├── types.py                # Type system (f32, i32, etc.)
└── expr.py                 # Expression IR nodes
```

### Key Patterns to Understand
```
~/Work/taichi/taichi/lang/
├── kernel_arguments.py     # How kernel args are typed/passed
├── snode.py                # Structural nodes (Taichi's data layout)
└── runtime_ops.py          # Runtime operation dispatch
```

### LLVM Backend Reference
```
~/Work/taichi/taichi/runtime/llvm/
├── llvm_context.cpp        # LLVM context management
├── llvm_offline_cache.cpp  # Kernel caching
└── CMakeLists.txt
```

## Architecture Notes

- **Why SPIR-V as universal format**: Single compilation target that can be consumed by Vulkan, translated to MSL (Metal), translated to PTX (NVIDIA), or used with HIP via SPIR-V ingestion. Avoids maintaining N backend-specific code generators.

- **Why not Vulkan-only**: Frontier and Aurora compute nodes don't have Vulkan drivers. Need native HIP and Level Zero runtime support for production HPC use.

- **Why not Numba/CUDA**: Numba is NVIDIA-only for GPU. Its abstraction level is wrong — too much magic, not enough control over data layout and dispatch patterns. No VTK-m worklet concepts.

- **Why lift from Taichi**: Its Python AST transformation is well-engineered, pure Python, and Apache 2.0 licensed. The hard problems (parallel for detection, type inference, field access patterns) are already solved there. No need to reinvent.

## Directory Structure

```
pgc/                        # package name TBD
├── pyproject.toml
├── src/
│   └── pgc/
│       ├── __init__.py     # pgc.init(), pgc.field(), pgc.kernel
│       ├── lang/
│       │   ├── ast_transform.py   # Python AST → IR
│       │   ├── ir.py              # Internal IR nodes
│       │   ├── types.py           # f32, i32, Vec3f, etc.
│       │   ├── kernel.py          # @kernel decorator
│       │   └── field.py           # Field abstraction
│       ├── codegen/
│       │   ├── llvm_gen.py        # IR → LLVM IR (llvmlite)
│       │   └── spirv_gen.py       # LLVM IR → SPIR-V
│       ├── runtime/
│       │   ├── cpu.py             # CPU JIT backend
│       │   ├── metal.py           # Metal compute backend
│       │   ├── vulkan.py          # Vulkan compute backend
│       │   └── hip.py             # ROCm/HIP backend
│       └── data/
│           ├── array_handle.py    # VTK-m ArrayHandle concept
│           └── cell_set.py        # VTK-m CellSet concept
├── tests/
│   ├── test_vector_add.py
│   ├── test_saxpy.py
│   ├── test_mandelbrot.py
│   └── ...
└── examples/
    ├── vector_add.py
    ├── mandelbrot.py
    └── nbody.py
```

## Getting Started

```bash
mkdir pgc && cd pgc
uv init
uv add llvmlite numpy pyobjc-framework-Metal pytest
# Copy this plan into the project
# Start with Step 1: project skeleton
```
