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
    ('value = out @ out', 'MatMult'),
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


@pytest.mark.parametrize("call", ['sqrt(1, 2)', 'sqrt()', 'min(1, 2, 3)', 'tack.barrier(1)', 'tack.thread_id(1)'])
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


def test_registered_function_can_shadow_intrinsic(monkeypatch):
    from tack.lang.func import Func, _func_registry

    def barrier(value):
        return value

    # Func registers itself; restore the original registry after this test.
    monkeypatch.setattr('tack.lang.func._func_registry', dict(_func_registry))
    Func(barrier)
    module = transform_kernel(ast.parse('''
def supported(out):
    for i in range(4):
        out[i] = barrier(i)
'''))
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
