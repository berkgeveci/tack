"""Structural passes must reach every child, including less common nodes."""

from types import SimpleNamespace

import pytest

from tack.lang import ir
from tack.lang.ir_optimize import _copy_prop_body, _replace_names
from tack.lang.ir_pack_scalars import _rewrite
from tack.lang.ir_resolve import _resolve
from tack.lang.ir_traversal import CHILD_FIELDS, transform_ir, walk_ir
from tack.lang.types import f32


@pytest.mark.parametrize('wrap', [
    lambda x: ir.IRBinOp('+', x, ir.IRConstant(1)),
    lambda x: ir.IRUnaryOp('-', x),
    lambda x: ir.IRCompare('<', ir.IRConstant(0), x),
    lambda x: ir.IRBoolOp('and', [ir.IRConstant(1), x]),
    lambda x: ir.IRIfExp(ir.IRConstant(1), x, ir.IRConstant(0)),
    lambda x: ir.IRFieldLoad(ir.IRName('field'), x),
    lambda x: ir.IRFieldStore(ir.IRName('field'), ir.IRConstant(0), x),
    lambda x: ir.IRAtomicOp('add', ir.IRName('field'), x, ir.IRConstant(1)),
    lambda x: ir.IRAttribute(x, 'attribute'),
    lambda x: ir.IRCall('sqrt', [x]),
    lambda x: ir.IRAssign('value', x),
    lambda x: ir.IRReturn(x),
    lambda x: ir.IRCast(x, f32),
    lambda x: ir.IRSharedAlloc('shared', f32, x),
    lambda x: ir.IRLocalAlloc('local', f32, x),
    lambda x: ir.IRBlockReduce('sum', x),
    lambda x: ir.IRPrint([x], [('expr', 0)]),
    lambda x: ir.IRTextureSample('texture', [ir.IRConstant(0), x, ir.IRConstant(1)]),
    lambda x: ir.IRSequentialFor('j', ir.IRConstant(0), ir.IRConstant(4), [], step=x),
    lambda x: ir.IRParallelFor('i', ir.IRConstant(0), x, []),
    lambda x: ir.IRWhile(x, []),
    lambda x: ir.IRIf(x, [], []),
])
def test_expression_children_are_walked_and_rewritten_without_visiting_metadata(wrap):
    leaf = ir.IRName('old')
    root = wrap(leaf)
    metadata = ir.IRName('old')
    root._debug_metadata = metadata
    assert any(node is leaf for node in walk_ir(root))
    assert not any(node is metadata for node in walk_ir(root))
    changed = transform_ir(root, lambda node: ir.IRName('new') if node is leaf else node)
    assert any(isinstance(node, ir.IRName) and node.name == 'new' for node in walk_ir(changed))
    assert not any(node is leaf for node in walk_ir(changed))
    assert changed._debug_metadata is metadata
    assert metadata.name == 'old'


def test_every_declared_ir_node_has_an_explicit_child_schema():
    declared = {value for value in vars(ir).values()
                if isinstance(value, type) and issubclass(value, ir.IRNode)
                and value is not ir.IRNode}
    assert set(CHILD_FIELDS) == declared


def test_modules_parameters_and_statement_lists_are_structural():
    module = ir.IRModule()
    param = ir.IRParam('out')
    statement = ir.IRPrint([ir.IRConstant(1)])
    function = ir.IRFunction('test', [param], [
        ir.IRIf(ir.IRConstant(1), [statement], [ir.IRBarrier()]),
    ])
    module.functions = [function]
    assert list(walk_ir(module))[:3] == [module, function, param]
    assert any(node is statement for node in walk_ir(module))
    new = ir.IRPrint([ir.IRConstant(2)])
    transform_ir(module, lambda node: new if node is statement else node)
    assert function.body[0].then_body == [new]


@pytest.mark.parametrize('operation', [lambda node: list(walk_ir(node)),
                                       lambda node: transform_ir(node, lambda n: n)])
def test_unregistered_nodes_and_cycles_fail_loudly(operation):
    class NewNode(ir.IRNode):
        pass

    with pytest.raises(TypeError, match='Unknown IR node: NewNode'):
        operation(NewNode())
    node = ir.IRUnaryOp('-', ir.IRConstant(1))
    node.operand = node
    with pytest.raises(ValueError, match='Cycle in IR tree'):
        operation(node)
    cyclic_list = []
    cyclic_list.append(cyclic_list)
    with pytest.raises(ValueError, match='Cycle in IR tree'):
        operation(cyclic_list)


@pytest.mark.parametrize('wrap', [
    lambda x: ir.IRPrint([x]),
    lambda x: ir.IRBlockReduce('sum', x),
    lambda x: ir.IRTextureSample('texture', [x, ir.IRConstant(0), ir.IRConstant(1)]),
    lambda x: ir.IRSharedAlloc('shared', f32, x),
    lambda x: ir.IRLocalAlloc('local', f32, x),
])
def test_all_structural_passes_reach_uncommon_expression_slots(wrap):
    resolved = _resolve(wrap(ir.IRDimSize('input', 1)),
                        {'input': SimpleNamespace(shape=(3, 7))})
    assert not any(isinstance(node, ir.IRDimSize) for node in walk_ir(resolved))
    assert any(isinstance(node, ir.IRConstant) and node.value == 7 for node in walk_ir(resolved))

    copied = _replace_names(wrap(ir.IRName('alias')), {'alias': 'original'})
    assert not any(isinstance(node, ir.IRName) and node.name == 'alias' for node in walk_ir(copied))
    assert any(isinstance(node, ir.IRName) and node.name == 'original' for node in walk_ir(copied))

    packed = _rewrite(wrap(ir.IRName('scalar')), {'scalar': ('__pack_f32__', 2)})
    loads = [node for node in walk_ir(packed) if isinstance(node, ir.IRFieldLoad)]
    assert len(loads) == 1
    assert loads[0].field.name == '__pack_f32__'
    assert loads[0].index.value == 2


def test_control_flow_metadata_survives_copy_propagation():
    continuation = ir.IRContinue()
    continuation.outermost = True
    condition = ir.IRName('alias')
    body = ir.IRIf(condition, [continuation], [])
    changed = _replace_names(body, {'alias': 'condition'})
    assert changed.condition.name == 'condition'
    assert changed.then_body[0].outermost is True


def test_copy_substitution_does_not_change_an_earlier_shared_expression():
    shared = ir.IRBinOp('+', ir.IRName('alias'), ir.IRConstant(1))
    body = [ir.IRAssign('before', shared),
            ir.IRAssign('alias', ir.IRName('parameter')),
            ir.IRAssign('after', shared)]
    changed = _copy_prop_body(body)
    assert changed[0].value.left.name == 'alias'
    assert changed[2].value.left.name == 'parameter'
    assert changed[0].value is not changed[2].value
    assert shared.left.name == 'alias'
