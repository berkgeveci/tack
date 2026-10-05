# Conformance and validation

A contract is only as good as the evidence that the implementation meets
it. Tack has two kinds of evidence:

- **the test suite**, which runs on every machine and exercises every
  backend that machine has
- **hardware validation**, a recorded run of the full suite and the
  example sweeps on a named device at an exact commit

This page maps the contracts to their tests at release candidate
`23d6e1d`, describes the oracles the tests trust, and gives the procedure
for a hardware validation and the most recent status of each backend,
recorded at `745e01f`.

!!! note "What counts as evidence"

    A CPU run does not validate GPU execution. A host-side syntax check of
    generated CUDA, HIP, MSL or OpenCL source shows that the source is
    well-formed, not that it runs correctly on a device. A skipped or
    deselected test is not a pass. Expected results come from independent
    oracles, never from recording what Tack currently outputs.

## How the suite is organized

`pyproject.toml` sets `testpaths` to the three packages' `tests/`
directories. The root `conftest.py` probes every backend once at import
and provides capability fixtures, so which backends a test runs on depends
on the machine, not on the test file:

| Fixture | Runs the test once per |
|---|---|
| `backend` | Available backend |
| `f64_backend` | Available backend with `supports_f64` |
| `reduction_backend` | Available backend with `supports_device_reductions` |
| `workgroup_backend` | Available backend with `supports_workgroups` |

Capability skips read the declared attributes described in
[Backend capabilities](backend-capabilities.md). They don't compare arch
names. A backend that fails to initialize with `ImportError`,
`RuntimeError`, `OSError` or `AttributeError` is left out of every fixture.

Two pytest markers are registered:

- **`slow`** marks the example sweep in `test_examples.py`. It is
  deselected by default (`addopts = "-m 'not slow'"`). Run it with
  `pytest -m slow`.
- **`timing`** marks tests in `test_cpu_threading.py` that assert what the
  CPU scheduler actually did. They skip themselves, and print the load,
  when the one-minute load average exceeds half the CPU backend's thread
  count. Let the load settle before a validation run so that they execute.

## Contract areas and their tests

All modules are in `packages/tack-core/tests/` unless another package is
named.

| Contract area | Test modules |
|---|---|
| Regression baseline LC1–LC8 | `test_compiler_contract.py` |
| Source validation and diagnostics | `test_source_validation.py`, `test_outside_parallel_loop.py`, `test_error_messages.py`, `test_error_reporting.py` |
| Execution order, short-circuiting and control flow | `test_differential.py`, `test_variable_length_loop.py`, `test_field_shape.py` |
| Fixed-width integers, division and power | `test_integer_semantics.py`, `test_integer_division.py`, `test_integer_expression_differential.py`, `test_division_and_power.py`, `test_small_int_types.py`, `test_promotion_and_safety.py`, `test_explicit_casts.py` |
| Floating-point policy | `test_float_semantics.py`, `test_float_literals.py`, `test_float_division.py`, `test_scalar_matching.py`, `test_local_var_types.py` |
| Field, block and statistical reductions | `test_reduction_semantics.py`, `test_block_reduce.py`, `test_stats.py` |
| Workgroups and participation | `test_workgroup_contract.py`, `test_workgroup_participation.py`, `test_shared_like.py` |
| Atomics | `test_atomic_contract.py` |
| Memory and aliasing, CPU disjoint specialization | `test_compiler_contract.py` (LC2), `test_disjoint_fields.py` |
| Specialization identity, caching and concurrent dispatch | `test_variant_cache.py`, `test_kernel_cache.py`, `test_concurrent_dispatch.py`, `test_templates.py` |
| GPU launch indexing and size limits | `test_launch_limits.py`, `test_cuda_gen.py`, `test_hip_gen.py` |
| Device-function binding and inlining | `test_func_bindings.py`, `test_new_features.py`, `test_texture_inline.py` |
| Identifier preservation in generated code | `test_gpu_identifiers.py` |
| IR structure and passes | `test_ir_verify.py`, `test_ir_clone.py`, `test_ir_traversal.py`, `test_ir_optimize.py`, `test_ast_transform.py`, `test_type_annotate.py`, `test_type_annotate_expr.py`, `test_type_inference.py` |
| Backend capability declarations | `test_backend_contract.py`, `test_dispatch_types.py`, `test_backend_isolation.py` |
| Runtime API: fields, inspection, local arrays | `test_field_utils.py`, `test_inspect.py`, `test_local_array.py`, `test_vector_add.py`, `test_cpu_jit.py` |
| Texture snapshots and storage | `test_texture_snapshot.py`, `test_texture_inline.py`, `test_gpu_dispatch_paths.py`, `tack-rendering/tests/test_volume.py` |
| Interop | `test_dlpack.py`, `test_level_zero_context.py`, `tack-vis/tests/test_vtk_interop.py` |
| Code generators (host-side, no device) | `test_llvm_gen.py`, `test_msl_gen.py`, `test_cuda_gen.py`, `test_hip_gen.py`, `test_opencl_gen.py`, `test_opencl_compilation.py`, `test_scalar_packing.py` |
| GPU dispatch paths without a GPU | `test_gpu_dispatch_paths.py` |
| Per-backend runtime behavior | `test_metal.py`, `test_cuda.py`, `test_hip.py` |
| CPU threading and NUMA | `test_cpu_threading.py`, `test_threading_benchmark.py`, `test_numa.py` |
| Rendering correctness | `tack-rendering/tests/test_raster_differential.py`, `test_raster_winners.py`, `test_rasterize.py`, `test_mixed_render.py`, `test_bvh.py`, and the remaining rendering modules |
| Visualization algorithms | `tack-vis/tests/test_flying_edges.py`, `test_compute_normals.py`, `test_cell_to_point.py`, `test_amr_blanking.py`, `test_fe.py`, `test_algorithms.py` |
| Package namespace | `tack-vis/tests/test_namespace.py` |
| Validation tooling | `test_examples.py`, `test_validation_harness.py`, `test_compiler_tools.py` |

The language contract describes what its central modules cover in its own
sections. For example,
[Atomic field updates](../reference/language-contract.md#atomic-field-updates)
lists the cases in `test_atomic_contract.py`, and the
[regression baseline](../reference/language-contract.md#regression-baseline)
lists LC1–LC8 with the CPU evidence that first exposed each defect.

## Oracles

A test is only independent if its expected value doesn't come from the
code under test. The suite uses these kinds of oracle:

| Oracle | Used for | Example modules |
|---|---|---|
| Plain arithmetic and NumPy | Ordering, aliasing and control flow, where the answer is obvious by hand | `test_compiler_contract.py` |
| Serial Python execution of the same source | Differential testing: the kernel body runs as ordinary Python over independent NumPy arrays, and every field is compared | `test_differential.py` |
| Python big integers | Fixed-width wrapping, division, remainder, power and casts, computed exactly and then reduced modulo the width | `test_integer_semantics.py`, `test_integer_division.py`, `test_integer_expression_differential.py`, `test_division_and_power.py` |
| Exact rationals and correctly rounded sums | Contraction and rounding (`fractions.Fraction`), and reduction error budgets (`math.fsum`) | `test_float_semantics.py`, `test_reduction_semantics.py` |
| Structural invariants | Properties that must hold whatever the output, such as BVH containment, surface watertightness and analytic surfaces | `tack-rendering/tests/test_bvh.py`, `tack-vis/tests/test_flying_edges.py` |
| Reference renders | Raster output checked against an oracle composed from renders of each candidate alone, with an index-coded palette | `tack-rendering/tests/test_raster_differential.py`, `test_raster_winners.py` |
| Negative controls | Show that an oracle can fail: planted wrong pixels, a mutation reference with Tack's IR optimizations bypassed, a final-only wrapping oracle that a late-wrapping compiler would satisfy, and generator shapes that a device previously rejected | `test_raster_differential.py`, `test_compiler_contract.py`, `test_integer_expression_differential.py`, `test_opencl_compilation.py` |

Generated cases use fixed grammars and reproducible seeds. A failure
report names the seed, backend and input, so it can be reproduced without
rerunning the whole generator.

## Host compiler checks

Several modules compile generated GPU source on the host with Clang, as
OpenCL C or C++ syntax checks, and some run host helpers under
UndefinedBehaviorSanitizer. The shared helper `require_clang()` in
`tests/compiler_tools.py` picks the compiler:

- `TACK_CLANG` selects the C/OpenCL driver. `TACK_CLANGXX` selects the C++
  driver. Without `TACK_CLANGXX`, the C++ driver is the `clang++` sibling
  of `TACK_CLANG`. If neither is set, the helper searches `PATH` for
  `clang` and then `clang-23` through `clang-14`.
- Each mode (`opencl`, `cxx`, `ubsan`) is preflighted with a real probe
  compile, and for `ubsan` a probe run. The probe result is cached per
  compiler and environment.
- On an unconfigured machine, missing or unusable tooling **skips** the
  check. An explicit `TACK_CLANG` or `TACK_CLANGXX`, or
  `TACK_REQUIRE_CLANG` set to anything other than empty or `0`, turns the
  same condition into a **failure**. An explicit choice that fails never
  falls back to another compiler on `PATH`.

Hardware validation runs with `TACK_REQUIRE_CLANG=1`, so these checks
can't silently disappear.

## Examples and the validation harness

**Example sweep.** `test_examples.py` runs every numbered example in the
three packages and in the top-level `examples/` directory as a subprocess
with `--arch $TACK_EXAMPLES_ARCH`, which defaults to `cpu`. An example
skips, with the reason, when it needs an optional third-party package or
doesn't offer the selected backend. A missing `tack` module is a failure.
The test is marked `slow`:

```bash
TACK_EXAMPLES_ARCH=cuda uv run --no-sync pytest -m slow -q -ra packages/tack-core/tests/test_examples.py
```

**Harness.** `packages/tack-core/examples/validate_all.py` runs seven
workloads (vector add, SAXPY, reduction, Mandelbrot, N-body, Jacobi and
matrix multiply). Each runs four times on fresh fields, every call is
verified, and the first and fastest warm times are printed. With `--arch`,
the harness initializes exactly that backend, confirms that
`get_backend().name` matches, and prints `Backends: <arch>`. Without
`--arch`, it validates every backend it can initialize.
`test_validation_harness.py` checks that the harness never discards a
wrong result and never reports a backend it didn't run.

## Running a hardware validation

A hardware validation establishes that one exact commit meets the
contracts on one named device. Run every step at the same hash, one job at
a time.

1. **Pin the source.** `git rev-parse HEAD` must print the full hash under
   test, and `git status --short` must be empty before and after.
2. **Prepare the environment.** Unset `TACK_NO_REINIT`, because it
   suppresses backend switching and invalidates a multi-backend run. Set
   `TACK_REQUIRE_CLANG=1` and `TACK_CLANG` to a working Clang. Use
   `uv run --no-sync` so that the installed backend extras are kept.
3. **Run the full suite with the backend required.** Initialize the backend
   explicitly, assert that it is the one requested, and print it before
   pytest starts. A backend that silently fails to initialize would
   otherwise turn every device case into a skip:

    ```bash
    uv run --no-sync python - cuda <<'PYTEST'
    import sys, pytest, tack
    from tack.runtime.dispatch import get_backend
    arch = sys.argv[1]
    tack.init(arch=getattr(tack, arch))
    assert get_backend().name == arch
    print('Required hardware backend:', get_backend().name, flush=True)
    raise SystemExit(pytest.main(['-q', '-ra']))
    PYTEST
    ```

4. **Run both example sweeps,** with `TACK_EXAMPLES_ARCH=<backend>` and
   with `TACK_EXAMPLES_ARCH=cpu`.
5. **Run the harness on both selections:** `validate_all.py --arch <backend>`
   and `validate_all.py --arch cpu`. Each must print only the requested
   backend and report 7 of 7 workloads OK.
6. **Explain every skip.** `-ra` lists them. Each must be a capability, an
   absent optional dependency, or an absent device, and the record says
   which. Confirm that the `timing` tests ran and didn't skip because of
   load.

For a focused run, prefer whole files to `-k <backend>`. `-k` is a
substring match. For example, `-k hip` also selects test IDs that contain
`relationships`, and a `[hip-...]` label doesn't guarantee that the case
executed on the device. Pre-build rejections and source checks never
launch.

**Record** the hash and clean status, the printed backend line, the
device, driver, runtime and binding versions, the Python, NumPy and
llvmlite versions, the load before the suite, the pass, skip and deselect
counts, every skip reason, whether the `timing` tests ran, and the first
failure if there was one. Don't waive a numerical failure with an
expected-failure marker. Preserve its log and report it.

There is one exception: a defect in a vendor toolchain, shown with a
reproducer that doesn't use Tack, after the generated source has been
shown to be correct. Such a marker is `strict`, applies only to the
toolchain release that was shown to fail (ROCm 7.0, by the HIP runtime's
major and minor version, for the case below), names the defect, and is listed
in the backend's documentation. The only one so far is ROCm 7.0.2's
miscompile of generated integer seed 31 (see
[Backend Implementations](../design/backend-implementations.md#hip)).

## Validation status at `745e01f`

From the project's validation records. A result recorded at another hash
is reported at that hash, not relabeled.

| Backend | Status at `745e01f` | Record |
|---|---|---|
| CPU + CUDA | **Validated clean**, 2026-10-05 | Intel Xeon E5-2650 host, NVIDIA RTX 4060 Ti, driver 615.71.09, NVRTC 13.4, Clang 14 with `TACK_REQUIRE_CLANG=1`. Full suite run at `0501c8f` with CUDA required: 3503 passed, 89 skipped, 49 deselected, no failures, and the timing tests ran. CUDA examples: 45 passed, 4 skipped. CPU examples: 44 passed, 5 skipped. `validate_all.py` 7/7 on both. `745e01f` only reorders one `__all__` list after a lint failure. At `745e01f`, lint passed and `test_vtk_interop.py`, `test_dlpack.py` and `test_level_zero_context.py` were rerun: 69 passed, 32 skipped |
| CPU + Metal | **Validated clean**, 2026-10-05 | Apple M1 Max, macOS 26.7, pyobjc 12.1, Apple Clang 21.0.0 with `TACK_REQUIRE_CLANG=1`, at `745e01f` exactly. Full suite with Metal required: 3439 passed, 88 skipped, 49 deselected, 0 failed, and the timing tests ran. Metal examples: 45 passed, 4 skipped. CPU examples: 44 passed, 5 skipped. `validate_all.py` 7/7 on both |
| CPU + HIP | **One toolchain failure**, 2026-10-05 | AMD Instinct MI300X VF (gfx942), ROCm 7.0.2, hip-python 7.2.2, at `745e01f` exactly. Full suite with HIP required: 3501 passed, 90 skipped, 1 failed, and the timing tests ran. HIP examples: 45 passed, 4 skipped. CPU examples: 44 passed, 5 skipped. `validate_all.py` 7/7 on both. The failure is ROCm 7.0.2's device compiler miscompiling generated integer seed 31; the generated source is correct. `7de38d2` adds only a strict expected-failure marker for that case under hipRTC 7.0. A rerun of that module at `7de38d2` is pending |
| CPU + Level Zero | **Pending** | Not yet run at `745e01f`, because the host is unavailable. Last clean full validation: `3decd4c`, 2026-10-04, Intel Data Center GPU Max 1100 (`supports_f64: True`): 3301 passed, 108 skipped, 0 failed, with examples and harness clean. A later full run of the Level Zero interop branch (`321a719`, 2026-10-05) had two failures, one since fixed and one NUMA-related. It predates the release candidate and doesn't count for it |

All recorded skips were capabilities or absent optional dependencies. For
example, the CPU + Metal skips are the HIP, CUDA and Level Zero suites
(no device), VTK without DLPack support, NUMA tests that need a multi-node
machine, Metal atomics outside `supported_atomic_dtypes`, and Metal's
missing `f64`.

The fixes committed after `7de38d2`, up to release candidate `23d6e1d`,
are awaiting hardware validation at the next release candidate.
