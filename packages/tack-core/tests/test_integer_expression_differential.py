"""Generated fixed-width integer expressions against an independent oracle.

`test_integer_semantics.py` pins each integer operation on boundary and
seeded inputs in fixed expressions; `test_differential.py` generates control
flow over exact i32 arithmetic. This module fills the gap between them:
*compositions* of already-defined operations in which an intermediate wrap,
cast or promotion decides what a later operation sees -- a sum that wraps
before it is compared, a narrowing that happens before a widening, a shift
whose count is a masked wrapped value, a conditional whose arms promote.

A small typed grammar with fixed seeds generates the kernels (32 seeds, one
kernel each, plus two control kernels). Each generated kernel is written to
an inspectable source file and loaded from it. The expected results come
from an oracle implemented here, from the rules in
``docs/reference/language-contract.md`` -- promotion, modular conversion,
wrapping at each node, shift and comparison semantics -- using Python
integers and nothing from the production type-inference, IR or code
generation. Two control expressions show that the oracle wraps *inside*
the tree: a deliberately naive oracle that wraps only the final result
disagrees with it, and the device agrees with the strict one.

Kernels are reused on changed inputs and on lengths 1, 255, 256, 257 and
4097, so cold and cached variants and partial workgroups all run, and every
output allocation carries a sentinel tail beyond the launched range that
must stay untouched.
"""
from __future__ import annotations

import importlib.util
import random
import textwrap
from dataclasses import dataclass

import numpy as np
import pytest

import tack

# ── the type system, as the contract states it ───────────────────────────

INT_TYPES = ("i8", "u8", "i16", "u16", "i32", "u32", "i64", "u64")
BITS = {t: int(t[1:]) for t in INT_TYPES}
SIGNED = {t: t.startswith("i") for t in INT_TYPES}
NP = {t: getattr(np, ("int" if SIGNED[t] else "uint") + str(BITS[t])) for t in INT_TYPES}


def wrap(value: int, t: str) -> int:
    """Reduce modulo 2^N and reinterpret with the destination signedness."""
    value %= 1 << BITS[t]
    if SIGNED[t] and value >= 1 << (BITS[t] - 1):
        value -= 1 << BITS[t]
    return value


def promote(a: str, b: str) -> str:
    """Narrowest integer type preserving both full ranges (contract rule)."""
    if a == b:
        return a
    if SIGNED[a] == SIGNED[b]:
        return a if BITS[a] >= BITS[b] else b
    signed, unsigned = (a, b) if SIGNED[a] else (b, a)
    if unsigned == "u64":
        raise ValueError("signed with u64 needs an explicit cast")
    need = max(BITS[signed], 2 * BITS[unsigned])       # a signed type wider than the unsigned one
    for t in ("i8", "i16", "i32", "i64"):
        if BITS[t] >= need:
            return t
    raise ValueError("no signed type preserves both ranges")


def lo_hi(t: str) -> tuple[int, int]:
    return (-(1 << (BITS[t] - 1)), (1 << (BITS[t] - 1)) - 1) if SIGNED[t] else (0, (1 << BITS[t]) - 1)


# ── expression trees ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class Var:
    name: str
    t: str


@dataclass(frozen=True)
class Const:
    value: int                 # always an i32 literal in the source
    t: str = "i32"


@dataclass(frozen=True)
class Cast:
    t: str
    e: object


@dataclass(frozen=True)
class Un:
    op: str                    # "-" or "~"
    e: object


@dataclass(frozen=True)
class Bin:
    op: str                    # + - * & | ^
    l: object
    r: object


@dataclass(frozen=True)
class Shift:
    op: str                    # << >>
    l: object
    count: object              # an expression whose evaluated value lies in [0, N)


@dataclass(frozen=True)
class Cmp:
    op: str                    # < <= > >= == !=
    l: object
    r: object


@dataclass(frozen=True)
class Cond:
    c: object                  # a Cmp
    a: object
    b: object


def type_of(e) -> str:
    if isinstance(e, (Var, Const, Cast)):
        return e.t
    if isinstance(e, Un):
        return type_of(e.e)
    if isinstance(e, Bin):
        return promote(type_of(e.l), type_of(e.r))
    if isinstance(e, Shift):
        return type_of(e.l)                        # shifts keep the left operand's type
    if isinstance(e, Cmp):
        return "i32"                               # normalized 0 / 1
    if isinstance(e, Cond):
        return promote(type_of(e.a), type_of(e.b))
    raise TypeError(e)


_CMP = {"<": lambda a, b: a < b, "<=": lambda a, b: a <= b, ">": lambda a, b: a > b,
        ">=": lambda a, b: a >= b, "==": lambda a, b: a == b, "!=": lambda a, b: a != b}
_BIN = {"+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b,
        "&": lambda a, b: a & b, "|": lambda a, b: a | b, "^": lambda a, b: a ^ b}


def evaluate(e, env: dict, strict: bool = True) -> int:
    """The oracle. `strict=False` is the deliberately wrong control: it skips
    the wrap at every inner node and casts, leaving only the final store to
    wrap -- the thing the contract says must *not* happen."""
    t = type_of(e)
    if isinstance(e, Var):
        return env[e.name]
    if isinstance(e, Const):
        return e.value
    if isinstance(e, Cast):
        v = evaluate(e.e, env, strict)
        return wrap(v, e.t) if strict else v
    if isinstance(e, Un):
        v = evaluate(e.e, env, strict)
        v = -v if e.op == "-" else ~v
        return wrap(v, t) if strict else v
    if isinstance(e, Bin):
        a, b = evaluate(e.l, env, strict), evaluate(e.r, env, strict)
        v = _BIN[e.op](a, b)                       # promotion preserves both values exactly
        return wrap(v, t) if strict else v
    if isinstance(e, Shift):
        a, k = evaluate(e.l, env, strict), evaluate(e.count, env, strict)
        assert 0 <= k < BITS[t], (k, t)            # caller constraint, kept by construction
        if e.op == "<<":
            v = a << k
            return wrap(v, t) if strict else v
        return a >> k                              # arithmetic for signed, logical for unsigned values
    if isinstance(e, Cmp):
        return int(_CMP[e.op](evaluate(e.l, env, strict), evaluate(e.r, env, strict)))
    if isinstance(e, Cond):
        chosen = e.a if evaluate(e.c, env, strict) else e.b
        v = evaluate(chosen, env, strict)
        return wrap(v, t) if strict else v         # arms convert to the promoted type (lossless)
    raise TypeError(e)


def source(e) -> str:
    if isinstance(e, Var):
        return e.name
    if isinstance(e, Const):
        return f"({e.value})" if e.value < 0 else str(e.value)
    if isinstance(e, Cast):
        return f"tack.{e.t}({source(e.e)})"
    if isinstance(e, Un):
        return f"({e.op}{source(e.e)})"
    if isinstance(e, Bin):
        return f"({source(e.l)} {e.op} {source(e.r)})"
    if isinstance(e, Shift):
        return f"({source(e.l)} {e.op} {source(e.count)})"
    if isinstance(e, Cmp):
        return f"({source(e.l)} {e.op} {source(e.r)})"
    if isinstance(e, Cond):
        return f"({source(e.a)} if {source(e.c)} else {source(e.b)})"
    raise TypeError(e)


# ── the generator ────────────────────────────────────────────────────────

# Input type families: same width, same signedness across widths, and the
# documented mixed-signedness promotions. u64 never meets a signed type or an
# untyped literal, which the contract rejects, except through an explicit cast.
FAMILIES = [
    ("i8",), ("u8",), ("i16",), ("u16",), ("i32",), ("u32",), ("i64",), ("u64",),
    ("i8", "i16", "i32"), ("u8", "u16", "u32"), ("i32", "i64"), ("u32", "u64"),
    ("i8", "u16"), ("i16", "u32"), ("u8", "i8"), ("u8", "i32"), ("i32", "u32"),
]


class Gen:
    def __init__(self, seed: int):
        self.rng = random.Random(1000 + seed)
        self.family = FAMILIES[seed % len(FAMILIES)]
        self.inputs = [Var(n, self.rng.choice(self.family)) for n in ("x", "y", "z")]
        self.locals: list[Var] = []
        self.has_u64 = any(v.t == "u64" for v in self.inputs)

    def leaf(self):
        pool = self.inputs + self.locals
        if self.rng.random() < 0.2:
            c = Const(self.rng.choice([0, 1, 2, 3, 5, 7, 15, 31, 100, 255, 1000, -1, -2, -128]))
            # A bare literal next to u64 is rejected by the contract; cast it.
            return Cast("u64", c) if self.has_u64 else c
        return self.rng.choice(pool)

    def compatible(self, a, b) -> bool:
        try:
            promote(type_of(a), type_of(b))
            return True
        except ValueError:
            return False

    def expr(self, depth: int):
        if depth == 0 or self.rng.random() < 0.15:
            return self.leaf()
        kind = self.rng.choice(["bin", "bin", "bin", "shift", "cast", "un", "cond", "bitwise"])
        if kind in ("bin", "bitwise"):
            ops = ["+", "-", "*"] if kind == "bin" else ["&", "|", "^"]
            for _ in range(8):
                l, r = self.expr(depth - 1), self.expr(depth - 1)
                if self.compatible(l, r):
                    return Bin(self.rng.choice(ops), l, r)
            return self.leaf()
        if kind == "shift":
            l = self.expr(depth - 1)
            n = BITS[type_of(l)]
            if self.rng.random() < 0.5:
                count = Const(self.rng.randrange(0, n))
            else:
                # A masked wrapped value is always a valid count: (e & (N-1)).
                inner = self.expr(depth - 1)
                mask = Const(n - 1)
                if type_of(inner) == "u64":
                    mask = Cast("u64", mask)
                if not self.compatible(inner, mask):
                    inner = Cast("i64", inner)
                count = Bin("&", inner, mask)
            if type_of(l) == "u64" and isinstance(count, Const):
                count = Cast("u64", count)
            return Shift(self.rng.choice(["<<", ">>"]), l, count)
        if kind == "cast":
            return Cast(self.rng.choice(INT_TYPES), self.expr(depth - 1))
        if kind == "un":
            return Un(self.rng.choice(["-", "~"]), self.expr(depth - 1))
        # cond: a comparison that depends on a (possibly wrapped) intermediate
        for _ in range(8):
            l, r = self.expr(depth - 1), self.expr(depth - 1)
            a, b = self.expr(depth - 1), self.expr(depth - 1)
            if self.compatible(l, r) and self.compatible(a, b):
                return Cond(Cmp(self.rng.choice(list(_CMP)), l, r), a, b)
        return self.leaf()

    def kernel(self, n_exprs: int = 3, depth: int = 4):
        """Returns (source text, [(local name, expression, store type)])."""
        stores = []
        for k in range(n_exprs):
            e = self.expr(depth)
            if isinstance(e, (Var, Const)):            # keep every slot interesting
                e = Bin("+", e, self.leaf()) if self.compatible(e, self.inputs[0]) else Cast("i32", e)
            name = f"t{k}"
            stores.append((name, e, type_of(e)))
            self.locals.append(Var(name, type_of(e)))
        # One output per expression, in its own type; the last store narrows or
        # widens into a different type so store conversion is exercised too.
        store_types = [t for _, _, t in stores]
        if len(stores) > 1:
            store_types[-1] = self.rng.choice([t for t in INT_TYPES if t != store_types[-1]])
        lines = ["import tack", "", "",
                 "def run(x_in, y_in, z_in, n, " + ", ".join(f"out{k}" for k in range(len(stores))) + "):",
                 "    for i in range(n):",
                 "        x = x_in[i]", "        y = y_in[i]", "        z = z_in[i]"]
        for (name, e, _), st in zip(stores, store_types):
            lines.append(f"        {name} = {source(e)}")
        for k, ((name, _, _), st) in enumerate(zip(stores, store_types)):
            lines.append(f"        out{k}[i] = {name}")
        return "\n".join(lines) + "\n", stores, store_types


# ── running ──────────────────────────────────────────────────────────────

LENGTHS = (1, 255, 256, 257, 4097)
TAIL = 8
SENTINEL = {t: (lo_hi(t)[1] if SIGNED[t] else lo_hi(t)[1] - 1) for t in INT_TYPES}


def _load(tmp_path, name: str, src: str):
    path = tmp_path / f"{name}.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return tack.kernel(module.run), path


def _inputs(rng: random.Random, inputs, n: int, boundary: bool):
    """Boundary values first (min, max, 0, 1, -1 or high bit), then seeded full range."""
    arrays = {}
    for v in inputs:
        lo, hi = lo_hi(v.t)
        edges = [lo, hi, 0, 1, hi - 1, lo + 1, (hi // 2) + 1] + ([-1, -2] if SIGNED[v.t] else [1 << (BITS[v.t] - 1)])
        if boundary:
            vals = [edges[i % len(edges)] for i in range(n)]
        else:
            vals = [rng.randint(lo, hi) for _ in range(n)]
        arrays[v.name] = np.array(vals, dtype=NP[v.t])
    return arrays


def _field(values, t: str):
    data = np.asarray(values, dtype=NP[t])
    f = tack.field(getattr(tack, t), data.shape)
    f.from_numpy(data)
    return f


def _expected(stores, store_types, arrays, n):
    """Evaluate the stores in order, each local joining the environment at
    its own type, so later expressions see earlier wrapped intermediates."""
    outs = [np.empty(n, dtype=NP[st]) for st in store_types]
    for i in range(n):
        env = {k: int(a[i]) for k, a in arrays.items()}
        for col, (name, e, et), st in zip(outs, stores, store_types):
            v = evaluate(e, env)                     # already wrapped at the local's type
            env[name] = v
            col[i] = wrap(v, st)                     # the store converts to the field type
    return outs


def _run_and_check(kernel, path, stores, store_types, inputs, arrays, n, label):
    fields = [_field(arrays[v.name], v.t) for v in inputs]
    outs = [_field(np.full(n + TAIL, SENTINEL[st], dtype=NP[st]), st) for st in store_types]
    kernel(*fields, n, *outs)
    expected = _expected(stores, store_types, arrays, n)
    for k, (out, exp, st) in enumerate(zip(outs, expected, store_types)):
        got = out.to_numpy()
        assert got.dtype == NP[st], (got.dtype, st)
        bad = np.flatnonzero(got[:n] != exp)
        if len(bad):
            i = int(bad[0])
            env = {kk: int(a[i]) for kk, a in arrays.items()}
            raise AssertionError(
                f"{label}: out{k} ({st}) mismatch at index {i} of {n}: got {got[i]}, "
                f"expected {exp[i]}; inputs {env}; expression {source(stores[k][1])}; "
                f"source file {path}")
        assert np.all(got[n:] == SENTINEL[st]), f"{label}: out{k} tail beyond {n} was written"


# ── the generated cases ──────────────────────────────────────────────────

SEEDS = list(range(32))

# ROCm 7.0.2's device compiler (AMD clang 20, inside hipRTC's comgr)
# miscompiles seed 31 at -O1 and above: within the full expression a 64-bit
# signed `-253 < -1` evaluates false, so out2 is 255 instead of 127. The
# generated source is correct as host C++ under UBSan (clang and gcc,
# -O0 to -O3), on CUDA, and under ROCm's clang 23. ROCm 7.1.1 (still AMD
# clang 20) does the same on an MI210: wrong at -O1, right at -O0.
# Strict, so a toolchain that compiles it correctly reports XPASS and the
# entry can go.
#
# Keyed on the HIP runtime's (major, minor), which is the ROCm release whose
# hipRTC and comgr Tack loads. hiprtcVersion() cannot identify it: on ROCm
# 7.0.2 it reports 9.0, so a mark keyed on it never applied.
ROCM_MISCOMPILED_SEEDS = {31: {(7, 0), (7, 1)}}


def _hip_runtime_version():
    """(major, minor) of the HIP runtime, i.e. the ROCm release, or None."""
    try:
        from hip import hip
        err, version = hip.hipRuntimeGetVersion()
    except Exception:
        return None
    if err != hip.hipError_t.hipSuccess:
        return None
    # HIP_VERSION = major * 10^7 + minor * 10^5 + patch
    return version // 10_000_000, version // 100_000 % 100


@pytest.mark.parametrize("seed", SEEDS)
def test_generated_integer_expression_matches_oracle(backend, tmp_path, seed, request):
    if backend == "hip" and seed in ROCM_MISCOMPILED_SEEDS:
        version = _hip_runtime_version()
        if version in ROCM_MISCOMPILED_SEEDS[seed]:
            request.applymarker(pytest.mark.xfail(
                strict=True, reason=f"ROCm {version[0]}.{version[1]}'s device compiler "
                                    f"(hipRTC/comgr, AMD clang 20) miscompiles a "
                                    f"64-bit compare in seed {seed}"))
    gen = Gen(seed)
    src, stores, store_types = gen.kernel()
    kernel, path = _load(tmp_path, f"int_expr_seed{seed}", src)
    rng = random.Random(seed)
    # Cold, then cached: the same kernel on every length, boundary then seeded
    # inputs, so changed values and changed lengths go through the cache.
    for n in LENGTHS:
        for boundary in (True, False):
            arrays = _inputs(rng, gen.inputs, n, boundary)
            _run_and_check(kernel, path, stores, store_types, gen.inputs, arrays, n,
                           f"seed {seed} ({'/'.join(v.t for v in gen.inputs)}), n={n}, "
                           f"{'boundary' if boundary else 'seeded'} inputs, {backend}")


def test_seeds_cover_every_width_and_the_documented_mixed_promotions():
    types_seen, mixed = set(), set()
    for seed in SEEDS:
        gen = Gen(seed)
        for v in gen.inputs:
            types_seen.add(v.t)
        kinds = {SIGNED[v.t] for v in gen.inputs}
        if len(kinds) == 2:
            mixed.add(tuple(sorted({v.t for v in gen.inputs})))
    assert types_seen == set(INT_TYPES), types_seen
    assert mixed, "no mixed-signedness family was drawn"


def test_generated_sources_are_distinct_and_nested():
    sources = set()
    for seed in SEEDS:
        src, stores, _ = Gen(seed).kernel()
        sources.add(src)
        assert any(depth(e) >= 2 for _, e, _ in stores), f"seed {seed} is flat: {src}"
    assert len(sources) == len(SEEDS)


def depth(e) -> int:
    if isinstance(e, (Var, Const)):
        return 0
    if isinstance(e, (Cast, Un)):
        return 1 + depth(e.e)
    if isinstance(e, (Bin, Cmp)):
        return 1 + max(depth(e.l), depth(e.r))
    if isinstance(e, Shift):
        return 1 + max(depth(e.l), depth(e.count))
    return 1 + max(depth(e.c), depth(e.a), depth(e.b))


# ── oracle controls ──────────────────────────────────────────────────────

CONTROLS = {
    # i8 127 + 1 wraps to -128 *before* it is compared with x, so the sum is
    # not greater; an oracle that wraps only at the end says it is.
    "overflow_before_comparison": (
        ("i8", "i8"),
        Cond(Cmp(">", Bin("+", Var("x", "i8"), Var("y", "i8")), Var("x", "i8")), Const(1), Const(0)),
        {"x": 127, "y": 1}, 0, 1),
    # u16 200 narrowed to i8 is -56, and widening that to i64 keeps -56; an
    # oracle that defers wrapping widens 200.
    "narrowing_before_widening": (
        ("u16", "u16"),
        Cast("i64", Cast("i8", Var("x", "u16"))),
        {"x": 200, "y": 0}, -56, 200),
}


@pytest.mark.parametrize("name", sorted(CONTROLS))
def test_oracle_control_disagrees_with_final_only_wrapping(name):
    _, e, env, strict_value, naive_value = CONTROLS[name]
    assert evaluate(e, env, strict=True) == strict_value
    assert wrap(evaluate(e, env, strict=False), type_of(e)) == naive_value
    assert strict_value != naive_value


@pytest.mark.parametrize("name", sorted(CONTROLS))
def test_device_agrees_with_the_strict_oracle_on_controls(backend, tmp_path, name):
    types, e, env, strict_value, _ = CONTROLS[name]
    t = type_of(e)
    src = textwrap.dedent(f"""
        import tack


        def run(x_in, y_in, n, out0):
            for i in range(n):
                x = x_in[i]
                y = y_in[i]
                out0[i] = {source(e)}
    """)
    kernel, path = _load(tmp_path, f"int_expr_control_{name}", src)
    n = 257
    xs = np.full(n, env["x"], dtype=NP[types[0]])
    ys = np.full(n, env["y"], dtype=NP[types[1]])
    out = _field(np.full(n + TAIL, SENTINEL[t], dtype=NP[t]), t)
    kernel(_field(xs, types[0]), _field(ys, types[1]), n, out)
    got = out.to_numpy()
    assert np.all(got[:n] == strict_value), (
        f"{name} on {backend}: got {got[0]}, strict oracle {strict_value}; source {path}")
    assert np.all(got[n:] == SENTINEL[t])


def test_promotion_rules_match_the_contract_examples():
    assert promote("i8", "u16") == "i32"
    assert promote("i16", "u32") == "i64"
    assert promote("i32", "u32") == "i64"
    assert promote("u8", "i8") == "i16"
    assert promote("i8", "i64") == "i64"
    assert promote("u8", "u32") == "u32"
    with pytest.raises(ValueError):
        promote("i32", "u64")
    assert wrap(127 + 1, "i8") == -128
    assert wrap(65535 * 65535, "u16") == 1
    assert wrap(-1, "u64") == (1 << 64) - 1
    assert wrap(255, "i64") == 255


def _distinguishes(seed: int) -> bool:
    """Whether per-node and final-only wrapping disagree on any element of
    the exact input sequence the device test feeds this seed."""
    gen = Gen(seed)
    _, stores, store_types = gen.kernel()
    rng = random.Random(seed)
    for n in LENGTHS:
        for boundary in (True, False):
            arrays = _inputs(rng, gen.inputs, n, boundary)
            for i in range(n):
                env_s = {k: int(a[i]) for k, a in arrays.items()}
                env_n = dict(env_s)
                for (name, e, _), st in zip(stores, store_types):
                    vs = evaluate(e, env_s, strict=True)
                    vn = wrap(evaluate(e, env_n, strict=False), type_of(e))
                    env_s[name], env_n[name] = vs, vn
                    if wrap(vs, st) != wrap(vn, st):
                        return True
    return False


def test_generated_kernels_depend_on_intermediate_wrapping():
    """The generated cases are only worth running if they could catch a
    compiler that wraps late. Wrapping commutes with + - * & | ^ and <<
    (the low N bits of a sum or product depend only on the operands' low N
    bits), so late wrapping is visible only where a comparison, a right
    shift, a narrowing cast or a widening promotion consumes an intermediate.
    Measured on the device test's own inputs, 25 of the 32 seeds reach such
    a node with values where early and late wrapping diverge; the other
    seven exercise composition without hitting a divergent input. The
    assertion is a majority, so a grammar change that stops reaching these
    semantics fails it while the per-seed device checks stay exact."""
    distinguishing = [seed for seed in SEEDS if _distinguishes(seed)]
    assert len(distinguishing) > len(SEEDS) // 2, (
        f"only {len(distinguishing)} of {len(SEEDS)} generated kernels distinguish "
        f"per-node wrapping from final-only wrapping: {distinguishing}")
