"""Device calls preserve Python binding identity across namespaces and caches."""

import ast
import gc
import importlib.util
import sys
import weakref

import numpy as np
import pytest

import tack
from tack.lang.source_validation import UnsupportedSyntaxError

_MODULE_SOURCE = """
import tack
@tack.func
def adjust(x):
    return x + OFFSET
alias = adjust
@tack.func
def nested(x):
    return adjust(x) * 2
@tack.kernel
def apply(out):
    for i in range(out.shape[0]):
        out[i, 0] = adjust(i)
        out[i, 1] = alias(i)
        out[i, 2] = nested(adjust(i))
@tack.func
def cross(x):
    return peer.nested(adjust(x)) + adjust(x)
@tack.kernel
def cross_apply(out):
    for i in range(out.shape[0]):
        out[i] = cross(i)
@tack.data_oriented
class Model:
    def __init__(self, bias):
        self.bias = bias
    @tack.func
    def step(self, x):
        return adjust(x) + self.bias
    @tack.func
    def both(self, x):
        return self.step(x) + nested(x)
@tack.func
def __tmpl_obj_both__(x):
    return x + 100
@tack.kernel
def template_apply(obj, out):
    for i in range(out.shape[0]):
        out[i, 0] = obj.both(i)
        out[i, 1] = __tmpl_obj_both__(i)
@tack.kernel
def template_pair(left, right, out):
    for i in range(out.shape[0]):
        out[i] = left.step(i) + right.both(i)
"""


@pytest.fixture(params=[False, True], ids=['a-first', 'b-first'])
def modules(request, tmp_path, monkeypatch):
    result = {}
    for name in (['b', 'a'] if request.param else ['a', 'b']):
        qualified = f'_tack_binding_{name}'
        path = tmp_path / f'{qualified}.py'
        path.write_text(_MODULE_SOURCE.replace('OFFSET', '10' if name == 'a' else '30'))
        spec = importlib.util.spec_from_file_location(qualified, path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, qualified, module)
        spec.loader.exec_module(module)
        result[name] = module
    result['a'].peer = result['b']
    result['b'].peer = result['a']
    return result['a'], result['b']


def test_module_collisions_aliases_and_cached_kernels(backend, modules):
    a, b = modules
    out = tack.field(tack.i32, (6, 3))
    for module, offset in [(a, 10), (b, 30), (a, 10), (b, 30)]:
        module.apply(out)
        x = np.arange(6, dtype=np.int32)
        np.testing.assert_array_equal(out.to_numpy(),
                                      np.column_stack([x + offset, x + offset,
                                                       2 * (x + 2 * offset)]))
    # A new kernel object must bind correctly after all colliding definitions.
    tack.kernel(a.apply.func)(out)
    np.testing.assert_array_equal(out.to_numpy()[:, 0], np.arange(6) + 10)


def test_nested_callees_use_own_namespace_and_arguments_use_caller(backend, modules):
    a, b = modules
    out = tack.field(tack.i32, (6,))
    for module, own, peer in [(a, 10, 30), (b, 30, 10), (a, 10, 30)]:
        module.cross_apply(out)
        x = np.arange(6, dtype=np.int32)
        np.testing.assert_array_equal(out.to_numpy(), 2 * (x + own + peer) + x + own)


def test_templates_keep_defining_bindings_and_synthetic_calls_distinct(backend, modules):
    a, b = modules
    out = tack.field(tack.i32, (6, 2))
    for module, offset in [(a, 10), (b, 30), (a, 10)]:
        obj = module.Model(7)
        module.template_apply(obj, out)
        x = np.arange(6, dtype=np.int32)
        np.testing.assert_array_equal(out.to_numpy(),
                                      np.column_stack([3 * (x + offset) + 7, x + 100]))


def test_multiple_template_arguments_and_repeated_instance(backend, modules):
    a, b = modules
    out = tack.field(tack.i32, (6,))
    x = np.arange(6, dtype=np.int32)
    for left, right, expected in [
        (a.Model(2), b.Model(5), 4 * x + 10 + 2 + 90 + 5),
        (b.Model(2), a.Model(5), 4 * x + 30 + 2 + 30 + 5),
    ]:
        a.template_pair(left, right, out)
        np.testing.assert_array_equal(out.to_numpy(), expected)
    shared = a.Model(4)
    a.template_pair(shared, shared, out)
    np.testing.assert_array_equal(out.to_numpy(), 4 * (x + 10) + 8)
    shared.bias = 9
    a.template_pair(shared, shared, out)
    np.testing.assert_array_equal(out.to_numpy(), 4 * (x + 10) + 18)


@tack.func
def _add_ten(x):
    return x + 10


@tack.func
def _add_thirty(x):
    return x + 30


def _closure_kernel(device):
    @tack.func
    def inner(x):
        return device(x)

    @tack.kernel
    def apply(out):
        for i in range(out.shape[0]):
            out[i] = inner(i)
    return apply


def test_closure_bindings_keep_same_named_device_functions_distinct(backend):
    first, second = _closure_kernel(_add_ten), _closure_kernel(_add_thirty)
    out = tack.field(tack.i32, (6,))
    for kernel, offset in [(first, 10), (second, 30), (first, 10)]:
        kernel(out)
        np.testing.assert_array_equal(out.to_numpy(), np.arange(6) + offset)


@tack.func
def _recursive(x):
    return _recursive(x)


@tack.kernel
def _calls_recursive(out):
    for i in range(out.shape[0]):
        out[i] = _recursive(i)


def test_recursion_receives_a_diagnostic():
    with pytest.raises(UnsupportedSyntaxError, match='Recursive @tack.func'):
        _calls_recursive.get_ir()


def test_runtime_callable_binding_cannot_fall_back_to_a_global():
    from tack.lang.ast_transform import transform_kernel
    source = ast.parse('def bad(_add_ten, out):\n    out[0] = _add_ten(1)')
    with pytest.raises(UnsupportedSyntaxError, match='statically bound @tack.func'):
        transform_kernel(source, bindings={'_add_ten': _add_ten})


def test_intrinsic_validation_ignores_unrelated_device_function_names():
    from tack.lang import ir
    from tack.lang.ast_transform import transform_kernel

    @tack.func
    def barrier(x):
        return x + 7

    source = ast.parse('def supported(out):\n    for i in range(4):\n        barrier()')
    module = transform_kernel(source, bindings={'unrelated': barrier})
    assert isinstance(module.functions[0].body[0].body[0], ir.IRBarrier)
    with pytest.raises(UnsupportedSyntaxError, match='only supported as a statement'):
        transform_kernel(ast.parse('def bad(out):\n    out[0] = barrier()'),
                         bindings={'unrelated': barrier})


def test_binding_resolution_does_not_execute_object_properties():
    from tack.lang.ast_transform import transform_kernel

    class Holder:
        @property
        def adjust(self):
            pytest.fail('resolution executed a property')

    with pytest.raises(UnsupportedSyntaxError, match='statically bound @tack.func'):
        transform_kernel(ast.parse('def bad(out):\n    out[0] = holder.adjust(1)'),
                         bindings={'holder': Holder()})


def test_ordinary_python_callable_cannot_impersonate_an_intrinsic():
    from tack.lang.ast_transform import transform_kernel

    def sqrt(x):
        return x + 100

    with pytest.raises(UnsupportedSyntaxError, match='statically bound @tack.func'):
        transform_kernel(ast.parse('def bad(out):\n    out[0] = sqrt(4)'),
                         bindings={'sqrt': sqrt})


def test_device_functions_are_not_retained_by_a_global_registry():
    from tack.lang.func import Func

    def disposable(x):
        return x
    function = Func(disposable)
    reference = weakref.ref(function)
    del function
    gc.collect()
    assert reference() is None
