# PGC Developer's Guide

This guide explains PGC's internals for contributors and anyone who wants to
understand how Python kernel source becomes GPU machine code.

## Table of Contents

1. [Architecture Overview](01-architecture.md) — Compilation pipeline, module layout
2. [AST Transform](02-ast-transform.md) — Python AST to PGC IR
3. [IR Design](03-ir.md) — Node types, structure, invariants
4. [IR Passes](04-ir-passes.md) — Resolve, optimize, type annotate, scalar packing
5. [Codegen](05-codegen.md) — LLVM, MSL, CUDA, HIP, OpenCL, SPIR-V
6. [Runtime and Dispatch](06-runtime.md) — Backend lifecycle, field allocation, kernel dispatch
7. [Template System](07-templates.md) — @pgc.data_oriented, template rewrite, @pgc.func inlining
8. [Adding a New Feature](08-adding-features.md) — Walkthrough of adding a new IR node
