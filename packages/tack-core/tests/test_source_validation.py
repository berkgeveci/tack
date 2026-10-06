"""Unsupported constructs must fail in the frontend with source context."""

import ast
import re
import textwrap

import pytest

import tack
from tack.lang.ast_transform import transform_kernel
from tack.lang.source_validation import UnsupportedSyntaxError


@pytest.mark.parametrize("statement, construct", [
    ('assert False', 'Assert'),
    ('value: int = 1', 'AnnAssign'),
    ('del out[i]', 'Delete'),
    ('raise ValueError()', 'Raise'),
    ('import math', 'Import'),
    ('from math import sin', 'ImportFrom'),
    ('global value', 'Global'),
    ('nonlocal value', 'Nonlocal'),
    ('try:\n    out[i] = 1\nexcept Exception:\n    out[i] = 2', 'Try'),
    ('with manager:\n    out[i] = 1', 'With'),
    ('def nested():\n    pass', 'FunctionDef'),
    ('class Nested:\n    pass', 'ClassDef'),
    ('value = lambda x: x', 'Lambda'),
    ('value = [j for j in range(3)]', 'ListComp'),
    ('value = (j for j in range(3))', 'GeneratorExp'),
    ('value = (other := 1)', 'NamedExpr'),
    ('value = {1, 2}', 'Set'),
    ('value = {1: 2}', 'Dict'),
    ('value = [1, 2]', 'List'),
    ('value = out[1:3]', 'Slice'),
    ('yield out[i]', 'Yield'),
    ('value = f"value={i}"', 'JoinedStr'),
    ('value = b"bytes"', 'Constant'),
    ('value = out @ out', "'@'"),
    ('value = out is None', 'Is'),
    ('match i:\n    case 0:\n        pass', 'Match'),
    ('print(i, end="")', 'Call'),
    ('out[i] = sqrt(value=1)', 'Call'),
    ('out[i] = sqrt(*values)', 'Starred'),
    ('out[i] = tack.atomic_add(out, i, 1)', 'Call'),
    ('out[i] = tack.barrier()', 'Call'),
    ('for j in range(3, ignored=1):\n    out[i] = j', 'Call'),
    ('break', 'Break'),
    ('return', 'Return'),
    ('for j in range(3):\n    pass\nelse:\n    out[i] = 1', 'For'),
    ('while False:\n    pass\nelse:\n    out[i] = 1', 'While'),
    ('for j in range(0, 3, 0):\n    pass', 'Constant'),
    ('for j in range(3, 0, -1):\n    pass', 'UnaryOp'),
])
def test_unsupported_source_has_function_and_location(statement, construct):
    source = 'def bad(out):\n    for i in range(4):\n' + textwrap.indent(statement, '        ')
    with pytest.raises(UnsupportedSyntaxError) as error:
        transform_kernel(ast.parse(source))
    message = str(error.value)
    assert "Kernel 'bad'" in message
    assert construct in message
    assert re.search(r'line [1-9][0-9]*, column [1-9][0-9]*', message)


@pytest.mark.parametrize("call", ['sqrt(1, 2)', 'sqrt()', 'min(1)', 'pow(1, 2, 3)', 'tack.barrier(1)',
                                  'tack.thread_id(1)'])
def test_extra_or_missing_builtin_arguments_are_rejected(call):
    source = f'def bad(out):\n    for i in range(4):\n        {call}'
    with pytest.raises(UnsupportedSyntaxError, match="Kernel 'bad'.*takes.*arguments.*line|Kernel 'bad'.*line.*takes.*arguments"):
        transform_kernel(ast.parse(source))


@pytest.mark.parametrize("parameters", ['x=1', 'x, /', '*args', '**kwargs', 'x, *, y'])
def test_unsupported_signature_is_rejected(parameters):
    source = f'def bad({parameters}):\n    pass'
    with pytest.raises(UnsupportedSyntaxError, match='positional parameters|default parameter'):
        transform_kernel(ast.parse(source))


def test_docstrings_and_pass_are_explicit_no_ops():
    module = transform_kernel(ast.parse('''
def supported(out):
    """Kernel documentation."""
    for i in range(4):
        pass
        out[i] = 7
'''))
    assert len(module.functions[0].body[0].body) == 1


def test_diagnostic_reports_exact_captured_source_location():
    source = 'def bad(out):\n    for i in range(4):\n        assert False'
    with pytest.raises(UnsupportedSyntaxError, match="Kernel 'bad': unsupported Assert at line 3, column 9"):
        transform_kernel(ast.parse(source))


def test_bound_function_can_shadow_intrinsic():
    from tack.lang.func import Func

    def barrier(value):
        return value

    bound = Func(barrier)
    module = transform_kernel(ast.parse('''
def supported(out):
    for i in range(4):
        out[i] = barrier(i)
'''), bindings={'barrier': bound})
    assert module.functions[0].body[0].body


def test_unsupported_device_source_is_rejected_before_return_rewriting():
    @tack.func
    def unsupported_after_return(value):
        return value
        assert False  # noqa: B011 — validate even unreachable unsupported source

    @tack.kernel
    def caller(out):
        for i in range(out.shape[0]):
            out[i] = unsupported_after_return(i)

    with pytest.raises(UnsupportedSyntaxError, match="Device function 'unsupported_after_return'.*Assert.*line"):
        caller.get_ir()


def test_nested_sequential_break_remains_supported():
    module = transform_kernel(ast.parse('''
def supported(out):
    for i in range(4):
        while out[i] < 3:
            break
'''))
    assert module.functions[0].body[0].body


# ── Names the kernel never binds, and aliases of arrays ─────────────

_MODULE_SCALE = 3.0


@tack.func
def _reads_module_value(v):
    return v * _MODULE_SCALE


@tack.kernel
def _kernel_reads_module_value(x, out):
    for i in range(out.shape[0]):
        out[i] = x[i] * _MODULE_SCALE


@tack.kernel
def _func_reads_module_value(x, out):
    for i in range(out.shape[0]):
        out[i] = _reads_module_value(x[i])


@tack.kernel
def _aliases_local_array(x, out):
    for i in range(out.shape[0]):
        tmp = tack.local_array(tack.f32, 2)
        view = tmp
        view[0] = x[i]
        out[i] = tmp[0]


@tack.func
def _fill_first(arr, v):
    arr[0] = v


@tack.kernel
def _passes_local_array(x, out):
    for i in range(out.shape[0]):
        tmp = tack.local_array(tack.f32, 2)
        _fill_first(tmp, x[i] + 1.0)
        out[i] = tmp[0]


def _pair(n=4):
    x = tack.field(dtype=tack.f32, shape=(n,))
    out = tack.field(dtype=tack.f32, shape=(n,))
    x.fill(2.0)
    return x, out


@tack.func
def _sums_then_reads_j(v):
    total = 0.0
    for j in range(3):
        total = total + v
    return total + j


@tack.kernel
def _func_reads_loop_variable(x, out):
    for i in range(x.shape[0]):
        out[i] = _sums_then_reads_j(x[i])


def test_module_value_in_kernel_is_named_with_its_position(backend):
    with pytest.raises(NameError) as error:
        _kernel_reads_module_value(*_pair())
    message = str(error.value)
    assert "Kernel '_kernel_reads_module_value'" in message
    assert "'_MODULE_SCALE' at line 4, column 25" in message
    assert "pass the value as an argument" in message


def test_module_value_in_device_function_names_the_function(backend):
    with pytest.raises(NameError) as error:
        _func_reads_module_value(*_pair())
    message = str(error.value)
    assert "Device function '_reads_module_value'" in message
    assert "inlined into kernel '_func_reads_module_value'" in message
    assert "'_MODULE_SCALE' at line 3, column 16" in message


def test_undefined_name_in_lowering_reports_a_name_error():
    with pytest.raises(NameError, match="Kernel 'bad': name 'missing' at line 4, column 25"):
        transform_kernel(ast.parse('''
def bad(x, out):
    for i in range(out.shape[0]):
        out[i] = x[i] + missing
'''))


def test_unbound_field_in_dimension_query():
    with pytest.raises(NameError, match="name 'ghost'"):
        transform_kernel(ast.parse('''
def bad(out):
    for i in range(ghost.shape[0]):
        out[i] = 1
'''))


def test_branch_and_loop_bindings_are_not_unbound():
    transform_kernel(ast.parse('''
def good(x, out, n):
    for i in range(n):
        if x[i] > 0:
            v = x[i]
        for j in range(2):
            v = v + j
        out[i] = v
'''))


# --- A loop variable's binding ends with its loop ---------------------------

def _lower(source):
    return transform_kernel(ast.parse(source))


def test_reading_a_loop_variable_after_its_loop_is_rejected():
    """CPU raised NameError at codegen and the GPU backends failed to
    compile; the frontend now refuses it with the read's position."""
    with pytest.raises(NameError) as error:
        _lower('''
def bad(x, out, n):
    for i in range(n):
        for d in range(4):
            x[i] = x[i] + d
        out[i] = d
''')
    message = str(error.value)
    assert message.startswith("Kernel 'bad': name 'd' at line 6, column 18 is read after the `for` loop")
    assert "copy it to another name inside the loop" in message


def test_a_loop_variable_does_not_update_an_outer_binding():
    """Python would leave `d == 3`; every Tack backend left the outer 100.
    Neither answer is given silently."""
    with pytest.raises(NameError, match="name 'd' at line 7, column 18 is read after"):
        _lower('''
def bad(x, out, n):
    for i in range(n):
        d = 100
        for d in range(4):
            x[i] = x[i] + d
        out[i] = d
''')


def test_a_loop_variable_read_in_a_later_range_bound_is_rejected():
    with pytest.raises(NameError, match="name 'd' at line 6, column 24 is read after"):
        _lower('''
def bad(x, out, n):
    for i in range(n):
        for d in range(4):
            x[i] = x[i] + d
        for j in range(d):
            out[i] = out[i] + j
''')


def test_a_loop_variable_stale_on_one_branch_is_rejected():
    """Merged conservatively: assigned on one branch only, it is still the
    loop's binding on the other."""
    with pytest.raises(NameError, match="name 'd' at line 8, column 18 is read after"):
        _lower('''
def bad(x, out, n):
    for i in range(n):
        for d in range(4):
            x[i] = x[i] + d
        if x[i] > 0:
            d = 1
        out[i] = d
''')


def test_a_loop_variable_rebound_before_the_read_is_allowed():
    _lower('''
def good(x, out, n):
    for i in range(n):
        acc = 0
        for d in range(4):
            acc = acc + d
        d = x[i]
        if acc > d:
            d = d + 1
        else:
            d = d - 1
        out[i] = acc + d
        for d in range(2):
            out[i] = out[i] + d
''')


def test_a_loop_variable_copied_inside_its_loop_is_allowed():
    _lower('''
def good(x, out, n):
    for i in range(n):
        last = -1
        for d in range(x[i]):
            last = d
        out[i] = last
''')


def test_sibling_loops_may_share_a_variable():
    _lower('''
def good(x, out, n):
    for i in range(n):
        for d in range(4):
            out[i] = out[i] + d
        for d in range(3):
            out[i] = out[i] + d
        while out[i] > 100:
            for d in range(2):
                out[i] = out[i] - d
''')


def test_loop_variable_read_after_an_inlined_loop_names_the_device_function(backend):
    with pytest.raises(NameError) as error:
        _func_reads_loop_variable(*_pair())
    message = str(error.value)
    assert message.startswith("Device function '_sums_then_reads_j' (inlined into kernel "
                              "'_func_reads_loop_variable'): name 'j' at line 6, column 20")


def test_aliasing_a_local_array_is_rejected_with_its_position(backend):
    with pytest.raises(UnsupportedSyntaxError) as error:
        _aliases_local_array(*_pair())
    message = str(error.value)
    assert message.startswith("Kernel '_aliases_local_array': cannot bind 'view'")
    assert "array 'tmp'" in message
    assert "line 5, column 9" in message
    # Not re-wrapped as a backend failure.
    assert "failed on" not in message


def test_passing_a_local_array_to_a_device_function_still_works(backend):
    x, out = _pair()
    _passes_local_array(x, out)
    assert out.to_numpy().tolist() == [3.0] * 4


def test_unsupported_syntax_reaches_the_caller_unwrapped(backend):
    @tack.kernel
    def asserts(x, out):
        for i in range(out.shape[0]):
            assert x[i] > 0
            out[i] = x[i]

    with pytest.raises(UnsupportedSyntaxError) as error:
        asserts(*_pair())
    assert str(error.value).startswith("Kernel 'asserts': unsupported Assert")
