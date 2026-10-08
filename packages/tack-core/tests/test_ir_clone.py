"""Fast IR copying preserves ownership, graph identity and deepcopy metadata."""

import copy
import sys
import weakref

import numpy as np
import pytest

import tack
from tack.lang import ir
from tack.lang.ir_traversal import CHILD_FIELDS, clone_ir
from tack.lang.ir_verify import IRVerificationError, verify_ir
from tack.lang.types import f32, f64, i32


def _nodes():
    leaf = ir.IRConstant(1, f32)
    return [
        ir.IRModule(), ir.IRFunction('kernel', [], []), ir.IRParam('a', f32),
        ir.IRParallelFor('i', leaf, leaf, []),
        ir.IRSequentialFor('j', leaf, leaf, [], leaf), ir.IRWhile(leaf, []),
        ir.IRBreak(), ir.IRContinue(), ir.IRIf(leaf, [], []),
        ir.IRIfExp(leaf, leaf, leaf), ir.IRBinOp('+', leaf, leaf),
        ir.IRUnaryOp('-', leaf), ir.IRCompare('<', leaf, leaf),
        ir.IRBoolOp('and', [leaf, leaf]), ir.IRFieldLoad(ir.IRName('a'), leaf),
        ir.IRFieldStore(ir.IRName('a'), leaf, leaf),
        ir.IRAtomicOp('add', ir.IRName('a'), leaf, leaf), leaf, ir.IRName('a'),
        ir.IRAttribute(ir.IRName('a'), 'shape'), ir.IRCall('sqrt', [leaf]),
        ir.IRAssign('x', leaf), ir.IRReturn(leaf), ir.IRCast(leaf, f32),
        ir.IRSharedAlloc('shared', f32, leaf), ir.IRLocalAlloc('local', f32, leaf),
        ir.IRBlockReduce('sum', leaf), ir.IRBarrier(), ir.IRThreadId(),
        ir.IRPrint([leaf], [('expr', 0)]), ir.IRDimSize('a', 0),
        ir.IRTextureSample('texture', [leaf, leaf, leaf], (2, 3, 4)),
        ir.IRTableLoad((3, 1, 4), i32, leaf),
    ]


@pytest.mark.parametrize('node', _nodes(), ids=lambda n: type(n).__name__)
def test_registered_nodes_match_deepcopy_and_isolate_annotations(node):
    shared = ir.IRName('metadata')
    node._metadata = {'state': [7], 'dtype': f64, 'shared': [shared, shared]}
    result = clone_ir(node)
    reference = copy.deepcopy(node)
    assert type(result) is type(node)
    assert result is not node
    assert ir.dump(result) == ir.dump(reference)
    assert set(vars(result)) == set(vars(reference))
    assert result._metadata['dtype'] is f64
    assert result._metadata['shared'][0] is result._metadata['shared'][1]
    assert result._metadata['shared'][0] is not shared
    result._metadata['state'].append(8)
    assert node._metadata['state'] == [7]


def test_clone_coverage_includes_every_registered_node():
    assert {type(n) for n in _nodes()} == set(CHILD_FIELDS)


@pytest.mark.skipif(sys.implementation.name != 'cpython',
                    reason='immediate reference-count release is CPython-specific')
def test_completed_clone_does_not_retain_either_graph():
    node = ir.IRBinOp('+', ir.IRConstant(1), ir.IRConstant(2))
    result = clone_ir(node)
    original_ref, clone_ref = weakref.ref(node), weakref.ref(result)
    del node, result
    assert original_ref() is None
    assert clone_ref() is None


def test_structural_and_metadata_aliases_share_one_memo():
    shared = ir.IRConstant(-0.0, f32)
    result = ir.IRBinOp('+', shared, shared)
    state = {'nodes': [shared], 'dtype': f32}
    result.first = state
    result.second = state
    result.in_tuple = (shared, state)
    clone = clone_ir(result)
    assert clone.left is clone.right is clone.first['nodes'][0] is clone.in_tuple[0]
    assert clone.first is clone.second is clone.in_tuple[1]
    assert clone.left.dtype is f32
    assert np.signbit(clone.left.value)
    clone.left.value = 4.0
    clone.first['nodes'].append(ir.IRConstant(2))
    assert shared.value == 0 and np.signbit(shared.value)
    assert state['nodes'] == [shared]


def test_metadata_cycles_and_node_attribute_dictionary_are_preserved():
    node = ir.IRConstant(1)
    node.self = node
    node.attributes = vars(node)
    node.list = [node]
    node.list.append(node.list)
    clone = clone_ir(node)
    assert clone.self is clone
    assert clone.attributes is vars(clone)
    assert clone.list[0] is clone and clone.list[1] is clone.list
    assert clone.list is not node.list


def test_structural_cycles_remain_rejected_by_verification():
    value = ir.IRUnaryOp('-', ir.IRConstant(1))
    value.operand = value
    out = ir.IRParam('out', f32)
    function = ir.IRFunction('cycle', [out], [ir.IRParallelFor(
        'i', ir.IRConstant(0), ir.IRConstant(1),
        [ir.IRFieldStore(ir.IRName('out'), ir.IRName('i'), value)],
    )])
    copied = clone_ir(function)
    assert copied.body[0].body[0].value.operand is copied.body[0].body[0].value
    with pytest.raises(IRVerificationError, match='cycle in IR tree'):
        verify_ir(copied, 'lowered')


def test_external_metadata_keeps_its_deepcopy_protocol():
    class Metadata:
        def __deepcopy__(self, memo):
            result = Metadata()
            memo[id(self)] = result
            result.owner = copy.deepcopy(self.owner, memo)
            result.data = copy.deepcopy(self.data, memo)
            return result

    node = ir.IRConstant(3, f32)
    node.extra = Metadata()
    node.extra.owner = node
    node.extra.data = np.array([1, 2, 3])
    clone = clone_ir(node)
    assert clone.extra.owner is clone
    assert clone.extra is not node.extra
    assert clone.extra.data is not node.extra.data
    clone.extra.data[0] = 99
    np.testing.assert_array_equal(node.extra.data, [1, 2, 3])


def test_cloning_runs_only_for_new_variants(backend, monkeypatch):
    import tack.lang.ir_traversal as traversal
    import tack.runtime.kernel_utils as utils

    @tack.kernel
    def fill(out):
        for i in range(out.shape[0]):
            out[i] = 3

    calls = []
    original = clone_ir

    def record(root):
        calls.append(root)
        return original(root)

    monkeypatch.setattr(traversal, 'clone_ir', record)
    monkeypatch.setattr(utils, 'clone_ir', record)
    out = tack.field(tack.f32, (8,))
    fill(out)
    initial_count = len(calls)
    assert initial_count > 0
    for _ in range(4):
        fill(out)
    assert len(calls) == initial_count
    wider = tack.field(tack.i64, (8,))
    fill(wider)
    assert len(calls) > initial_count
    np.testing.assert_array_equal(out.to_numpy(), np.full(8, 3))
    np.testing.assert_array_equal(wider.to_numpy(), np.full(8, 3))
