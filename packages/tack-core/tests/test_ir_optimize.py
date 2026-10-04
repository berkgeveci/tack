"""Copy propagation must preserve ordering with bounded assignment analysis."""

import pytest

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
