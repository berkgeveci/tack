# Contributing to Tack

## Getting set up

```bash
git clone https://github.com/berkgeveci/tack.git
cd tack
uv sync --extra cpu --extra dev
uv run pytest
```

Add the extra for your hardware if you have it — `--extra metal`, `--extra cuda`,
`--extra level_zero`. Note that `uv sync` makes the environment match exactly the
extras you list, so syncing without `--extra metal` will *remove* Metal from an
environment that had it.

`hip-python` lives on Test PyPI and cannot be declared as a dependency; see the
README for the install line.

## The thing to know about the tests

**A green run does not mean every backend was exercised.** Each backend's tests
detect their backend at import and skip themselves when it is unavailable, so on a
CPU-only machine most of the GPU suite silently sits out. Check the skip count, and
check which backends were found:

```bash
uv run python -c "
import tack
for a in ('cpu','metal','cuda','hip','level_zero'):
    try: tack.init(arch=getattr(tack, a)); print(a, 'available')
    except Exception: pass"
```

CI prints the same thing in its log.

The 49 shipped examples are deselected from the default run — the sweep takes
about a minute against an 18-second suite. Run them with:

```bash
uv run pytest -m slow -rs
```

`-rs` prints why anything skipped; examples needing an optional package
(matplotlib, vtk, oidn, imgui_bundle) skip themselves rather than failing. CI
runs this as its own job.

Two suites cover ground your machine probably cannot:

- `tests/test_gpu_dispatch_paths.py` drives the CUDA/HIP/Level Zero dispatch path in a
  subprocess with the device bindings stubbed, so those code paths are checked
  anywhere.
- `tests/test_backend_isolation.py` proves a GPU install works without `llvmlite`.
  CI runs it against a deliberately minimal install; it is the only check that catches
  a CPU-only dependency leaking into a GPU backend.

If you change dispatch, run both.

## Writing tests

Use the shared fixtures rather than rolling your own backend discovery:

```python
def test_something(backend):        # runs once per available backend
def test_precision(f64_backend):    # only where f64 is supported
def test_reduce(reduction_backend): # only where reductions run on device
```

They live in `packages/tack-core/tests/conftest.py` and
`packages/tack-vis/tests/conftest.py`.

Ask the backend what it supports; never compare its name to a string:

```python
if backend.supports_f64: ...          # yes
if arch == "metal": ...               # no — see runtime/backend.py
```

## Things that are easy to get wrong

**Kernels are compiled, not interpreted.** A `@tack.kernel` body never runs as Python,
so line coverage understates kernel testing badly. Test behaviour and values, not
coverage.

**Dimensions get baked in.** `resolve_ir` substitutes field dimensions as literals, so
a kernel that uses `x.shape[0]` in its body compiles a separate variant per shape.
That is intentional, and `shape_signature()` keys the cache on exactly those
dimensions. If you add a pass that bakes in something new, it has to reach the key.

**The IR from `kernel.get_ir()` is a template.** The passes mutate IR in place and run
on a deep copy. Mutating the template consumes nodes later dispatches need.

**Prefer analytic checks.** Where an algorithm has a closed-form answer — a plane cut
yields exactly `(n+1)²` merged points, a sphere encloses `4/3·π·r³` — assert that rather
than a recorded output. It catches a class of bug that golden files do not.

## Style

Match the surrounding code. Comments explain *why*, not *what*.

## Submitting

Open a PR against `main`. CI runs the suite on Python 3.11/3.12/3.13, the packaging
gate, and a best-effort Metal job. All three should be green — the Metal job is
`continue-on-error`, so read its log rather than trusting its status.
