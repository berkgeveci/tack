# Tack Developer's Guide

This guide explains Tack's internals for contributors and anyone who wants to
understand how Python kernel source becomes GPU machine code.

This guide is task-oriented: how the pieces are laid out and how to change
them. Two other sections go deeper:

- [Design and Implementation](../design/index.md) explains *why* each part is
  built the way it is and how it works end to end: the compilation pipeline,
  specialization and caching, numerical semantics, parallel execution, memory
  and aliasing, the CPU threading policy, each backend, and interoperability.
- [Contracts](../contracts/index.md) states *what* is guaranteed: the
  [kernel language contract (draft)](../reference/language-contract.md), the
  backend capability and runtime API contracts, and how conformance is tested.

## Table of Contents

1. [Architecture Overview](01-architecture.md) — Compilation pipeline, module layout
2. [AST Transform](02-ast-transform.md) — Python AST to Tack IR
3. [IR Design](03-ir.md) — Node types, structure, invariants
4. [IR Passes](04-ir-passes.md) — Resolve, optimize, type annotate, scalar packing
5. [Codegen](05-codegen.md) — LLVM, MSL, CUDA, HIP, OpenCL
6. [Runtime and Dispatch](06-runtime.md) — Backend lifecycle, field allocation, kernel dispatch
7. [Template System](07-templates.md) — @tack.data_oriented, template rewrite, @tack.func inlining
8. [Adding a New Feature](08-adding-features.md) — Walkthrough of adding a new IR node

## Host compiler checks

Several codegen tests compile OpenCL C or run generated C++ on the host,
including checks with undefined-behavior sanitization. They complement the
vendor compiler and hardware tests; they do not require a GPU.

All these checks share compiler discovery and preflight. Set `TACK_CLANG`
to a Clang executable outside `PATH` if needed. C++ checks use its adjacent
`clang++` (including a version suffix); set `TACK_CLANGXX` explicitly for a
wrapper or a different C++ driver. An invalid explicit choice fails rather
than falling back to another compiler. Without overrides, discovery tries
unversioned drivers on `PATH`, followed by versioned drivers.

Developer runs may skip missing or unusable default tooling with a reason.
Set `TACK_REQUIRE_CLANG=1` for validation: absent compilers, missing headers,
and missing UBSan link/runtime support then fail. The Linux and Metal CI
suite jobs require these checks and run their preflight before the suite.

```bash
export TACK_CLANG=/absolute/path/to/clang
export TACK_REQUIRE_CLANG=1
uv run pytest -q -s packages/tack-core/tests/test_compiler_tools.py \
  -k test_host_compiler_preflight
```

Preflight compiles OpenCL C 2.0 and C++17, then links and executes a small
UBSan-instrumented program. Select an installation with its headers and
sanitizer runtimes, and supply any required library/SDK paths in the same
environment as the suite. Compiler preflight is cached per driver and
execution environment; each generated kernel is still compiled by its test.
