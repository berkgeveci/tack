"""Statements outside the parallel loop may only bind locals and arrays.

They may run any number of times per launch: once per GPU thread, and on
CPU once per chunk, threading probe or timing sample. Effects there are
rejected with their source position; pure local assignments are accepted.
"""

import ast
import textwrap

import numpy as np
import pytest

import tack
from tack.lang.ast_transform import transform_kernel
from tack.lang.source_validation import UnsupportedSyntaxError


@pytest.mark.parametrize("prologue, epilogue, effect, line, column", [
    ('out[0] = 1.0', '', 'store', 2, 5),
    ('out[0] += 1.0', '', 'store', 2, 5),
    ('tack.atomic_add(out, 0, 1.0)', '', 'atomic_add()', 2, 5),
    ('tack.atomic_max(out, 0, 1.0)', '', 'atomic_max()', 2, 5),
    ('print("start")', '', 'print()', 2, 5),
    ('tack.barrier()', '', 'barrier()', 2, 5),
    ('total = tack.block_sum(1.0)', '', 'block_sum()', 2, 5),
    ('tmp = tack.local_array(tack.f32, 2)\ntmp[0] = 1.0', '', 'store', 3, 5),
    ('', 'out[0] = 1.0', 'store', 5, 5),
    ('if n > 0:\n    out[0] = 1.0', '', 'store', 3, 9),
    ('while n > 0:\n    out[0] = 1.0\n    n = n - 1', '', 'store', 3, 9),
])
def test_effects_outside_the_parallel_loop_are_rejected(prologue, epilogue, effect, line, column):
    source = ('def bad(out, n):\n' + textwrap.indent(prologue, '    ')
              + '\n    for i in range(out.shape[0]):\n        out[i] = 2.0\n'
              + textwrap.indent(epilogue, '    '))
    with pytest.raises(UnsupportedSyntaxError) as error:
        transform_kernel(ast.parse(source), bindings={'tack': tack})
    assert str(error.value).startswith(
        f"Kernel 'bad': {effect} at line {line}, column {column} "
        "is outside the parallel loop")


def test_local_assignments_and_array_declarations_are_accepted():
    module = transform_kernel(ast.parse('''
def good(x, out, k):
    scale = 2.0 * k
    first = x[0]
    smem = tack.shared(tack.f32, 256)
    tmp = tack.local_array(tack.f32, 4)
    tid = tack.thread_id()
    if k > 1:
        scale = scale + 1.0
    for i in range(out.shape[0]):
        out[i] = x[i] * scale + first
    unused = scale * 2.0
'''), bindings={'tack': tack})
    assert module.functions[0].body


@pytest.mark.parametrize("source, construct, line, column", [
    ('def bad(out):\n    scale = 2.0\n', 'definition', 1, 1),
    ('def bad(out):\n    for i in range(4):\n        out[i] = 1\n    for j in range(4):\n        out[j] = 2\n',
     'for loop', 4, 5),
    ('def bad(out, n):\n    if n > 0:\n        for i in range(4):\n            out[i] = 1\n',
     'for loop', 3, 9),
    ('def bad(out, n):\n    while n > 0:\n        for i in range(4):\n            out[i] = 1\n',
     'for loop', 3, 9),
])
def test_a_kernel_has_one_parallel_loop_in_its_body(source, construct, line, column):
    with pytest.raises(UnsupportedSyntaxError) as error:
        transform_kernel(ast.parse(source))
    message = str(error.value)
    assert message.startswith(f"Kernel 'bad': {construct} at line {line}, column {column}")
    assert "parallel loop" in message


@tack.func
def _store_first(out, v):
    out[0] = v
    return v


@tack.kernel
def _inlined_store_before_loop(x, out):
    scale = _store_first(out, 2.0)
    for i in range(out.shape[0]):
        out[i] = x[i] * scale


@tack.data_oriented
class _Counter:
    def __init__(self, n):
        self.counts = tack.field(dtype=tack.i32, shape=(n,))

    @tack.func
    def bump(self):
        tack.atomic_add(self.counts, 0, 1)


@tack.kernel
def _template_effect_before_loop(counter, out):
    counter.bump()
    for i in range(out.shape[0]):
        out[i] = 1.0


@tack.kernel
def _atomic_before_loop(counter, out):
    tack.atomic_add(counter, 0, 1)
    for i in range(out.shape[0]):
        out[i] = 1.0


def _fields(n):
    x = tack.field(dtype=tack.f32, shape=(n,))
    x.from_numpy(np.arange(n, dtype=np.float32))
    return x, tack.field(dtype=tack.f32, shape=(n,))


def test_inlined_effect_names_the_device_function(backend):
    with pytest.raises(UnsupportedSyntaxError) as error:
        _inlined_store_before_loop(*_fields(4))
    assert str(error.value).startswith(
        "Device function '_store_first' (inlined into kernel "
        "'_inlined_store_before_loop'): store at line 3, column 5 "
        "is outside the parallel loop")


def test_template_method_effect_names_the_method(backend):
    with pytest.raises(UnsupportedSyntaxError) as error:
        _template_effect_before_loop(_Counter(1), _fields(4)[1])
    assert str(error.value).startswith(
        "Device function 'bump' (inlined into kernel "
        "'_template_effect_before_loop'): atomic_add() at line 3, column 5 "
        "is outside the parallel loop")


def test_atomic_before_the_loop_is_rejected_at_dispatch(backend):
    counter = tack.field(dtype=tack.i32, shape=(1,))
    counter.from_numpy(np.zeros(1, dtype=np.int32))
    with pytest.raises(UnsupportedSyntaxError, match="atomic_add.*outside the parallel loop"):
        _atomic_before_loop(counter, _fields(1 << 16)[1])
    assert counter.to_numpy().tolist() == [0]


@tack.kernel
def _locals_before_loop(x, out, k):
    scale = 2.0 * k
    offset = x[0]
    if k > 1:
        offset = offset + 1.0
    for i in range(out.shape[0]):
        out[i] = x[i] * scale + offset


@tack.kernel
def _rebinds_outer_local(out):
    count = 0.0
    for i in range(out.shape[0]):
        count = count + 1.0
        out[i] = count


def test_locals_before_the_loop_are_used_inside_it(backend):
    x, out = _fields(1000)
    _locals_before_loop(x, out, 3)
    np.testing.assert_array_equal(out.to_numpy(), np.arange(1000, dtype=np.float32) * 6 + 1)


def test_each_iteration_starts_from_the_outer_local(backend):
    # On CPU the statements before the loop ran once per chunk, so the
    # count carried over from one iteration to the next of that chunk.
    out = tack.field(dtype=tack.f32, shape=(1 << 16,))
    _rebinds_outer_local(out)
    assert set(out.to_numpy().tolist()) == {1.0}
