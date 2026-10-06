"""Copy propagation must preserve ordering with bounded assignment analysis."""

import numpy as np
import pytest

import tack
from tack.lang import ir
from tack.lang import ir_optimize as optimize
from tack.lang.ir_traversal import walk_ir
from tack.lang.types import i32


@pytest.mark.parametrize('depth', [8, 32])
def test_assignment_analysis_does_not_revisit_nested_subtrees(monkeypatch, depth):
    body = [ir.IRAssign('result', ir.IRName('source'))]
    for level in range(depth):
        body = [ir.IRIf(ir.IRConstant(1), body, [
            ir.IRAssign(f'other_{level}', ir.IRConstant(level)),
        ])]
    node_count = sum(1 for _ in walk_ir(body))
    visits = 0

    def counted_walk(root):
        nonlocal visits
        for node in walk_ir(root):
            visits += 1
            yield node

    monkeypatch.setattr(optimize, 'walk_ir', counted_walk)
    optimize._copy_prop_body(body)
    # Count work, not wall time: deeper nesting must not multiply visits.
    assert visits <= node_count


@pytest.mark.parametrize('binding', [
    ir.IRSequentialFor('source', ir.IRConstant(0), ir.IRConstant(2), []),
    ir.IRLocalAlloc('source', i32, ir.IRConstant(2)),
    ir.IRSharedAlloc('source', i32, ir.IRConstant(2)),
])
def test_nested_bindings_prevent_following_a_modified_copy_source(binding):
    body = [
        ir.IRAssign('saved', ir.IRName('source')),
        ir.IRIf(ir.IRConstant(1), [binding], []),
        ir.IRAssign('result', ir.IRName('saved')),
    ]
    changed = optimize._copy_prop_body(body)
    assert changed[-1].value.name == 'saved'


def test_parameter_copies_still_propagate_into_nested_blocks():
    body = [
        ir.IRAssign('alias', ir.IRName('parameter')),
        ir.IRIf(ir.IRConstant(1), [
            ir.IRAssign('nested', ir.IRName('alias')),
            ir.IRAssign('result', ir.IRBinOp('+', ir.IRName('nested'), ir.IRConstant(1))),
        ], []),
    ]
    changed = optimize._copy_prop_body(body)
    assert changed[1].then_body[-1].value.left.name == 'parameter'


def test_assignment_summary_is_fresh_for_each_invocation():
    def body():
        return [ir.IRAssign('alias', ir.IRName('parameter')),
                ir.IRAssign('result', ir.IRName('alias'))]

    first = optimize._copy_prop_body(body())
    assert first[-1].value.name == 'parameter'

    modified = body()
    modified.insert(1, ir.IRAssign('parameter', ir.IRConstant(2)))
    second = optimize._copy_prop_body(modified)
    assert second[-1].value.name == 'alias'


def test_a_chain_of_copies_resolves_to_its_root_in_one_pass():
    """Nested device functions pass a field down as p -> a -> b -> c. Every
    later use must name the parameter: a link left behind is a local that
    holds a field."""
    body = [
        ir.IRAssign('a', ir.IRName('p')),
        ir.IRAssign('b', ir.IRName('a')),
        ir.IRIf(ir.IRConstant(1), [
            ir.IRAssign('c', ir.IRName('b')),
            ir.IRAssign('out', ir.IRFieldLoad(ir.IRName('c'), ir.IRConstant(0))),
        ], []),
    ]
    result = optimize._copy_prop_body(body)
    assert [stmt.value.name for stmt in result[:2]] == ['p', 'p']
    inner = result[2].then_body
    assert inner[0].value.name == 'p'
    assert inner[1].value.field.name == 'p'


def test_a_chain_stops_at_a_reassigned_link():
    body = [
        ir.IRAssign('a', ir.IRName('p')),
        ir.IRAssign('b', ir.IRName('a')),
        ir.IRAssign('b', ir.IRConstant(3)),
        ir.IRAssign('c', ir.IRName('b')),
        ir.IRAssign('out', ir.IRName('c')),
    ]
    result = optimize._copy_prop_body(body)
    assert result[1].value.name == 'p'
    assert result[3].value.name == 'b'
    assert result[4].value.name == 'c'


# --- A field handed down through nested device functions ---

@tack.func
def _load_1(f, i):
    return f[i]


@tack.func
def _load_2(f, i):
    return _load_1(f, i) + 1.0


@tack.func
def _load_3(f, i):
    return _load_2(f, i) + 1.0


@tack.func
def _load_4(f, i):
    return _load_3(f, i) + 1.0


@tack.kernel
def field_through_four_funcs(x, out, n):
    for i in range(n):
        out[i] = _load_4(x, i)


def test_field_passed_through_nested_device_functions(backend):
    """Three or more levels left a local holding the field, which failed to
    compile on every backend ("Cannot coerce float* to i32" on CPU)."""
    n = 6
    x = tack.field(dtype=tack.f32, shape=(n,))
    x.from_numpy(np.arange(n, dtype=np.float32))
    out = tack.field(dtype=tack.f32, shape=(n,))
    field_through_four_funcs(x, out, n)
    np.testing.assert_array_equal(out.to_numpy(), np.arange(n) + 3.0)
