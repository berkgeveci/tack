"""Malformed IR must stop at the pass that produced it, before compilation."""

import copy

import numpy as np
import pytest

import tack
from tack.lang import ir
from tack.lang.ir_type_annotate import annotate_types
from tack.lang.ir_verify import IRVerificationError, verify_ir
from tack.lang.types import f32, i32, i64


def function(value=None):
    out = ir.IRParam('out', f32)
    out._is_field = True
    count = ir.IRParam('count', i32)
    count._is_field = False
    store = ir.IRFieldStore(ir.IRName('out'), ir.IRName('i'),
                           value if value is not None else ir.IRConstant(1))
    loop = ir.IRParallelFor('i', ir.IRConstant(0), ir.IRName('count'), [store])
    return ir.IRFunction('example', [out, count], [loop])


class UnknownNode(ir.IRNode):
    pass


@pytest.mark.parametrize('value, message', [
    (UnknownNode(), 'expected expr node'),
    (None, 'expected expr node'),
    (ir.IRBarrier(), 'expected expr node'),
    (ir.IRAtomicOp('add', ir.IRName('out'), ir.IRConstant(0), ir.IRConstant(1)),
     'expected expr node'),
    (ir.IRBinOp('unsupported', ir.IRConstant(1), ir.IRConstant(2)), 'unsupported operator'),
    (ir.IRBoolOp('and', [ir.IRConstant(1)]), 'needs two values'),
    (ir.IRName('missing'), 'unbound name'),
    (ir.IRConstant('text'), 'constant must be numeric'),
    (ir.IRCast(ir.IRConstant(1), 'int'), 'cast target must be a ScalarType'),
])
def test_invalid_expression_reports_kernel_stage_and_path(value, message):
    func = function()
    func.body[0].body[0].value = value
    with pytest.raises(IRVerificationError, match=message) as error:
        verify_ir(func, 'lowered')
    assert "Kernel 'example': invalid IR after lowered" in str(error.value)
    assert 'function.body[0].body[0].value' in str(error.value)
    assert type(value).__name__ in str(error.value)


def test_cycle_is_reported_but_shared_subexpressions_are_valid():
    value = ir.IRBinOp('+', ir.IRConstant(1), ir.IRConstant(2))
    func = function(ir.IRBinOp('*', value, value))
    verify_ir(func, 'lowered')
    value.right = value
    with pytest.raises(IRVerificationError, match='cycle in IR tree'):
        verify_ir(func, 'lowered')


@pytest.mark.parametrize('corrupt, message', [
    (lambda f: f.params.append(f.params[0]), 'duplicate parameter names'),
    (lambda f: setattr(f.body[0], 'body', ()), 'body must be a list'),
    (lambda f: f.body[0].body.append(None), 'expected stmt node'),
    (lambda f: delattr(f.body[0].body[0], 'index'), 'missing attribute index'),
    (lambda f: setattr(f.body[0], 'start', ir.IRConstant(3)), 'normalized to start at zero'),
    (lambda f: f.body.append(copy.deepcopy(f.body[0])), 'exactly one'),
    (lambda f: f.body.clear(), 'exactly one'),
    (lambda f: f.body[0].body.append(ir.IRBreak()), 'break must target a sequential loop'),
    (lambda f: f.body[0].body.append(ir.IRContinue()), 'outermost flag'),
    (lambda f: f.body.insert(0, ir.IRContinue()), 'continue must target a loop'),
    (lambda f: f.body[0].body.append(ir.IRReturn(None)), 'kernel return'),
    (lambda f: f.body[0].body.append(copy.deepcopy(f.body[0])), 'function top level'),
    (lambda f: f.body.insert(0, ir.IRSequentialFor('j', ir.IRConstant(0), ir.IRConstant(1), [])),
     'sequential loop must be inside'),
    (lambda f: setattr(f.params[0], 'type_annotation', 'float'), 'parameter type'),
    (lambda f: delattr(f.params[0], '_is_field'), 'field/scalar category'),
])
def test_structure_and_loop_invariants(corrupt, message):
    func = function()
    corrupt(func)
    with pytest.raises(IRVerificationError, match=message):
        verify_ir(func, 'inferred')


def test_nested_loop_targets_and_outer_continue():
    func = function()
    outer = ir.IRContinue()
    outer.outermost = True
    inner = ir.IRContinue()
    func.body[0].body += [ir.IRSequentialFor('j', ir.IRConstant(0), ir.IRConstant(3),
                                          [inner, ir.IRBreak()]), outer]
    verify_ir(func, 'inferred')
    inner.outermost = True
    with pytest.raises(IRVerificationError, match='outermost flag'):
        verify_ir(func, 'inferred')


@pytest.mark.parametrize('container', ['store', 'print', 'reduce', 'allocation', 'condition'])
def test_unresolved_dimension_is_found_in_nested_expression_slots(container):
    func = function()
    dim = ir.IRDimSize('out', 0)
    if container == 'store':
        func.body[0].body[0].value = dim
    elif container == 'print':
        func.body[0].body.append(ir.IRPrint([dim]))
    elif container == 'reduce':
        func.body[0].body[0].value = ir.IRBlockReduce('sum', dim)
    elif container == 'allocation':
        func.body[0].body.insert(0, ir.IRLocalAlloc('tmp', f32, dim))
    else:
        func.body[0].body.append(ir.IRIf(dim, [], []))
    verify_ir(func, 'lowered')
    with pytest.raises(IRVerificationError, match='unresolved dimension in generated code'):
        verify_ir(func, 'resolved')


def test_host_bound_dimension_remains_dynamic_after_annotation():
    func = function()
    func.body[0].end = ir.IRBinOp('-', ir.IRDimSize('out', 0), ir.IRConstant(1))
    verify_ir(func, 'resolved')
    annotate_types(func)
    verify_ir(func, 'typed')
    assert isinstance(func.body[0].end.left, ir.IRDimSize)


@pytest.mark.parametrize('shape, coords, message', [
    (None, [ir.IRConstant(0)] * 3, 'texture extent'),
    ((2, 3, 0), [ir.IRConstant(0)] * 3, 'texture extent'),
    ((2, 3, 4), [ir.IRConstant(0)] * 2, 'three coordinates'),
])
def test_texture_resolution(shape, coords, message):
    func = function(ir.IRTextureSample('out', coords, shape))
    with pytest.raises(IRVerificationError, match=message):
        verify_ir(func, 'resolved')


def test_array_like_allocation_must_resolve_before_inference():
    func = function()
    alloc = ir.IRSharedAlloc('tmp', None, ir.IRConstant(8), field_name='out')
    func.body[0].body.insert(0, alloc)
    verify_ir(func, 'lowered')
    with pytest.raises(IRVerificationError, match='allocation dtype is unresolved'):
        verify_ir(func, 'resolved')
    alloc.dtype = f32
    verify_ir(func, 'resolved')


def test_scalar_parameter_cannot_be_used_as_a_buffer():
    func = function()
    func.body[0].body[0].field = ir.IRName('count')
    with pytest.raises(IRVerificationError, match='field access must name a field'):
        verify_ir(func, 'inferred')


def test_scalar_parameter_cannot_survive_packing():
    with pytest.raises(IRVerificationError, match='scalar parameter survived GPU packing'):
        verify_ir(function(), 'packed')


def test_typed_expressions_and_assignment_storage_are_required():
    func = function()
    func.body[0].body.insert(0, ir.IRAssign('value', ir.IRConstant(2)))
    func.body[0].body[1].value = ir.IRName('value')
    annotate_types(func)
    verify_ir(func, 'typed')
    # A field pointer intentionally carries no scalar dtype.
    assert func.body[0].body[1].field.dtype is None
    del func.body[0].body[0]._resolved_type
    with pytest.raises(IRVerificationError, match='assignment storage type is missing'):
        verify_ir(func, 'typed')
    annotate_types(func)
    func.body[0].body[1].value.dtype = None
    with pytest.raises(IRVerificationError, match='scalar expression dtype is missing'):
        verify_ir(func, 'typed')


def test_allocation_size_is_annotated_and_checked():
    func = function()
    alloc = ir.IRLocalAlloc('tmp', f32, ir.IRConstant(8))
    func.body[0].body.insert(0, alloc)
    annotate_types(func)
    verify_ir(func, 'typed')
    alloc.size.dtype = None
    with pytest.raises(IRVerificationError, match='scalar expression dtype is missing'):
        verify_ir(func, 'typed')


def test_logical_result_dtype_is_checked():
    func = function(ir.IRCompare('<', ir.IRConstant(1), ir.IRConstant(2)))
    annotate_types(func)
    verify_ir(func, 'typed')
    func.body[0].body[0].value.dtype = i64
    with pytest.raises(IRVerificationError, match='logical result must have i32 dtype'):
        verify_ir(func, 'typed')


def test_verification_does_not_mutate_ir_or_metadata():
    func = function()
    annotate_types(func)
    before = copy.deepcopy(func)
    verify_ir(func, 'typed')
    assert ir.dump(func) == ir.dump(before)
    assert vars(func) == {'name': before.name, 'params': func.params, 'body': func.body}
    assert '_shape_deps' not in vars(func)


@pytest.mark.parametrize('stage, module, pass_name', [
    ('resolved', 'tack.lang.ir_resolve', 'resolve_ir'),
    ('localized', 'tack.runtime.kernel_utils', '_localize_assigned_scalar_params'),
    ('optimized', 'tack.lang.ir_optimize', 'optimize_ir'),
    ('typed', 'tack.lang.ir_type_annotate', 'annotate_types'),
])
def test_corrupt_pass_stops_before_codegen_and_caching(stage, module, pass_name, monkeypatch):
    import importlib

    from tack.runtime.dispatch import get_backend

    tack.init(arch=tack.cpu)
    backend = get_backend()

    @tack.kernel
    def fill(out):
        for i in range(out.shape[0]):
            out[i] = 1

    out = tack.field(dtype=tack.f32, shape=(4,))
    target = importlib.import_module(module)
    original = getattr(target, pass_name)

    def corrupt(func, *args):
        original(func, *args)
        if stage == 'resolved':
            func.body[0].body[0].value = ir.IRDimSize('out', 0)
        elif stage == 'localized':
            func.body[0].body.append(ir.IRContinue())
        elif stage == 'optimized':
            func.body[0].body.append(None)
        else:
            func.body[0].body[0].value.dtype = None

    monkeypatch.setattr(target, pass_name, corrupt)
    compile_calls = []
    monkeypatch.setattr('tack.runtime.cpu._compile_kernel', lambda f: compile_calls.append(f))
    with pytest.raises(RuntimeError, match=f'invalid IR after {stage}') as error:
        fill(out)
    assert isinstance(error.value.__cause__, IRVerificationError)
    assert not compile_calls
    assert not backend._cache.get(fill)
    np.testing.assert_array_equal(out.to_numpy(), np.zeros(4))


def test_verifier_runs_only_when_templates_and_variants_are_new(monkeypatch):
    import tack.lang.ir_verify as verifier

    tack.init(arch=tack.cpu)

    @tack.kernel
    def fill(out):
        for i in range(out.shape[0]):
            out[i] = 3

    calls = []
    original = verifier.verify_ir

    def record(func, stage):
        calls.append(stage)
        return original(func, stage)

    monkeypatch.setattr(verifier, 'verify_ir', record)
    monkeypatch.setattr('tack.lang.kernel.verify_ir', record)
    out = tack.field(dtype=tack.f32, shape=(8,))
    for _ in range(5):
        fill(out)
    assert calls == ['lowered', 'resolved', 'inferred', 'localized', 'optimized', 'typed']
    wider = tack.field(dtype=tack.i64, shape=(8,))
    fill(wider)
    assert calls[6:] == ['resolved', 'inferred', 'localized', 'optimized', 'typed']
    np.testing.assert_array_equal(wider.to_numpy(), np.full(8, 3))


def test_inspection_localizes_assigned_scalar_parameters():
    from tack.lang.inspect_kernel import _prepare_ir
    from tack.lang.ir_traversal import walk_ir
    from tack.runtime.dispatch import get_backend

    tack.init(arch=tack.cpu)

    @tack.kernel
    def increment(out, value):
        for i in range(out.shape[0]):
            value = value + 1
            out[i] = value

    out = tack.field(dtype=tack.i32, shape=(4,))
    inspected, _ = _prepare_ir(increment, (out, 3))
    increment(out, 3)
    executed = next(iter(get_backend()._cache[increment].values())).ir
    assert ir.dump(inspected) == ir.dump(executed)
    assignments = [node.target for node in walk_ir(inspected) if isinstance(node, ir.IRAssign)]
    assert 'value' not in assignments
    assert '__value_local__' in assignments
    np.testing.assert_array_equal(out.to_numpy(), np.full(4, 4))
