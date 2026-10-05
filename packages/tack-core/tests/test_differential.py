"""Compare compiled kernels with the same source executed serially by Python.

Inputs stay in bounds and within exact i32 arithmetic. Every parallel
iteration owns its elements. Device functions get Python reference shims;
Workgroup operations, atomics, and open numerical policies are not covered.
"""

import importlib.util
import textwrap
import types

import numpy as np
import pytest

import tack
from tack.lang.func import Func


def _load_kernel(tmp_path, source):
    path = tmp_path / "differential_kernel.py"
    path.write_text(textwrap.dedent(source))
    spec = importlib.util.spec_from_file_location("differential_kernel", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name, value in list(vars(module).items()):
        if isinstance(value, types.FunctionType) and name != "run":
            setattr(module, name, tack.func(value))
    return tack.kernel(module.run)


def _python_reference(kernel):
    namespace = dict(kernel.func.__globals__)
    for name, value in list(namespace.items()):
        if isinstance(value, Func):
            fn = value.func
            namespace[name] = types.FunctionType(fn.__code__, namespace, fn.__name__)
    fn = kernel.func
    return types.FunctionType(fn.__code__, namespace, fn.__name__)


def _check(kernel, values, *scalars):
    values = np.asarray(values, dtype=np.int32)
    expected = [values.copy(), np.zeros_like(values), np.full_like(values, -99)]
    _python_reference(kernel)(*expected, *scalars)
    fields = []
    for initial in (values, np.zeros_like(values), np.full_like(values, -99)):
        field = tack.field(dtype=tack.i32, shape=initial.shape)
        field.from_numpy(initial)
        fields.append(field)
    kernel(*fields, *scalars)
    for field, reference in zip(fields, expected):
        np.testing.assert_array_equal(field.to_numpy(), reference)


@pytest.mark.parametrize("expression", [
    "x[i] < 0",
    "not (x[i] < 0)",
    "(x[i] < 0) + (x[i] > 0)",
    "(x[i] > -3) and (x[i] < 3)",
    "(x[i] < -3) or (x[i] > 3)",
    "-2 < x[i] < 4",
    "x[i] + 2 if x[i] < 0 else x[i] * 3",
    "((x[i] + 8) // 3) % 4",
    "True if x[i] < 0 else False",
    "int(x[i] >= 0) + int(x[i] <= 0)",
    "int((x[i] + 0.5 if x[i] < 0 else x[i] * 1.25) * 4)",
], ids=["compare", "not", "boolean-arithmetic", "and", "or", "chain", "conditional", "positive-division", "boolean-literals", "boolean-sum", "conditional-promotion"])
def test_expression_matches_python(backend, tmp_path, expression):
    kernel = _load_kernel(tmp_path, f"""
        def run(x, state, out):
            for i in range(out.shape[0]):
                out[i] = {expression}
    """)
    _check(kernel, [-5, -3, -1, 0, 1, 3, 5])


@pytest.mark.parametrize("expression", [
    "(x[i] < 0) and (touch(state, i) > 0)",
    "(x[i] >= 0) or (touch(state, i) > 0)",
    "touch(state, i) if x[i] < 0 else 17",
    "17 if x[i] < 0 else touch(state, i)",
    "(x[i] < 0) and ((x[i] > -3) or (touch(state, i) > 0))",
    "(touch(state, i) == 1) and (x[i] < 0) and (touch(state, i) == 2)",
], ids=["and", "or", "then", "else", "nested", "three-operands"])
def test_guarded_inline_call_matches_python(backend, tmp_path, expression):
    kernel = _load_kernel(tmp_path, f"""
        def touch(state, i):
            state[i] = state[i] + 1
            return state[i]

        def run(x, state, out):
            for i in range(out.shape[0]):
                out[i] = {expression}
    """)
    _check(kernel, [-5, -3, -1, 0, 1, 3, 5])


@pytest.mark.parametrize("expression", [
    "state[i] + touch(state, i)",
    "min(state[i], touch(state, i))",
    "state[i] < touch(state, i)",
    "combine(state[i], touch(state, i))",
    "touch(state, i) < touch(state, i) < touch(state, i)",
], ids=["arithmetic", "builtin-arguments", "compare", "func-arguments", "chain-once"])
def test_operand_evaluation_order_matches_python(backend, tmp_path, expression):
    kernel = _load_kernel(tmp_path, f"""
        def touch(state, i):
            state[i] = state[i] + 1
            return state[i]

        def combine(a, b):
            return a * 10 + b

        def run(x, state, out):
            for i in range(out.shape[0]):
                out[i] = {expression}
    """)
    _check(kernel, [-1, 0, 1])


@pytest.mark.parametrize("condition", ["if", "while"])
def test_inline_call_in_condition_matches_python(backend, tmp_path, condition):
    kernel = _load_kernel(tmp_path, f"""
        def touch(state, i):
            state[i] = state[i] + 1
            return state[i]

        def run(x, state, out):
            for i in range(out.shape[0]):
                out[i] = 0
                {condition} touch(state, i) < 3:
                    out[i] = out[i] + 1
    """)
    _check(kernel, [-1, 0, 1])


@pytest.mark.parametrize("statement", [
    "state[i] += touch(state, i)",
    "out[index(state, i)] = state[i]",
    "out[index(state, i)] += touch(state, i)",
    "a, b = state[i], touch(state, i)\nout[i] = a * 10 + b",
], ids=["augmented-load", "store-index", "augmented-index-once", "tuple-elements"])
def test_assignment_evaluation_order_matches_python(backend, tmp_path, statement):
    statement = statement.replace("\n", "\n                ")
    kernel = _load_kernel(tmp_path, f"""
        def touch(state, i):
            state[i] = state[i] + 1
            return state[i]

        def index(state, i):
            state[i] = state[i] + 1
            return i

        def run(x, state, out):
            for i in range(out.shape[0]):
                {statement}
    """)
    _check(kernel, [-1, 0, 1])


def test_sequential_range_bounds_evaluated_once(backend, tmp_path):
    kernel = _load_kernel(tmp_path, """
        def run(x, state, out):
            for i in range(out.shape[0]):
                state[i] = 5
                total = 0
                step = 1
                for j in range(0, state[i], step):
                    total = total + j
                    state[i] = state[i] - 1
                    step = step + 1
                out[i] = total
    """)
    _check(kernel, [-1, 0, 1])


def test_sequential_range_argument_order_matches_python(backend, tmp_path):
    kernel = _load_kernel(tmp_path, """
        def touch(state, i):
            state[i] = state[i] + 1
            return state[i]

        def run(x, state, out):
            for i in range(out.shape[0]):
                total = 0
                for j in range(state[i], touch(state, i) + 2):
                    total = total + j
                out[i] = total
    """)
    _check(kernel, [-1, 0, 1])


@pytest.mark.parametrize("seed", range(24))
def test_generated_control_flow_matches_python(backend, tmp_path, seed):
    # Fixed grammar/seed IDs make every failure reproducible without a
    # random runtime dependency. Vary bounds, steps, branches, and exits.
    step = 1 + seed % 3
    cutoff = 2 + seed % 5
    expression = ["x[i] + j", "x[i] * 2 - j", "x[i] if j % 2 == 0 else -x[i]"][seed % 3]
    kernel = _load_kernel(tmp_path, f"""
        def run(x, state, out, count):
            for i in range(out.shape[0]):
                total = 0
                for j in range(1, count, {step}):
                    if j == {cutoff}:
                        continue
                    if j > {cutoff + 2}:
                        break
                    total = total + ({expression})
                cursor = 0
                while cursor < {seed % 4}:
                    cursor = cursor + 1
                    if cursor == 2:
                        continue
                    total = total + cursor
                out[i] = total
    """)
    for count in (0, 1, 8):
        _check(kernel, [-3, -1, 0, 1, 3], count)
