"""Vector and matrix values, as lowering emits them.

Tack scalarizes vectors and matrices: a value is a `VectorValue`, one
scalar IR expression per component (a matrix row-major), and a variable
holding one is one IR local per component. Arithmetic, math, comparisons,
construction (`tack.vec3`, `tack.mat2`, `tack.identity`), `@`, the vector
and matrix methods and component access all lower here, component by
component.
"""

import ast

from tack.lang import ir
from tack.lang.ir_traversal import walk_ir
from tack.lang.lowering.values import (
    _VECTOR_CONSTRUCTORS,
    COMPONENT_NAMES,
    MAX_MATRIX_EXTENT,
    TupleValue,
    VectorValue,
    _like,
    _size,
)
from tack.lang.types import i32


class VectorValues:
    """Lowers vector and matrix values, variables and their components.

    Owns `_vector_vars` (variable -> component count), `_matrix_vars`
    (variable -> (rows, columns)) and `_component_names` (the generated
    name of each variable's components). Reads the array metadata of
    `ArrayAccess` for vector array elements.
    """

    def _init_vectors(self):
        # Vector tracking: name -> number of components
        self._vector_vars: dict[str, int] = {}
        # Matrix variables: name -> (rows, columns), beside _vector_vars
        self._matrix_vars: dict[str, tuple] = {}
        # (variable, component) -> the generated name of that component
        self._component_names: dict[tuple[str, int], str] = {}

    def _component_name(self, name, component):
        key = (name, component)
        if key not in self._component_names:
            self._component_names[key] = self._fresh_name(f"{name}__{component}")
        return self._component_names[key]

    def _variable_value(self, name: str) -> VectorValue:
        """The components of the vector or matrix variable ``name``."""
        return VectorValue((ir.IRName(self._component_name(name, c))
                            for c in range(self._vector_vars[name])),
                           self._matrix_vars.get(name))

    def _bind_vector(self, name: str, value: list):
        """Record that the variable ``name`` holds the vector or matrix ``value``."""
        self._vector_vars[name] = len(value)
        shape = getattr(value, 'shape', None)
        if shape is not None and len(shape) == 2:
            self._matrix_vars[name] = shape
        else:
            self._matrix_vars.pop(name, None)

    def _named_component(self, node: ast.Attribute, n: int) -> int:
        """The index that ``x``, ``y``, ``z`` or ``w`` names in an n-vector."""
        c = COMPONENT_NAMES.find(node.attr) if len(node.attr) == 1 else -1
        if not 0 <= c < n:
            names = ", ".join(COMPONENT_NAMES[:n])
            raise self._source_error(
                node, "attribute", f"'{node.attr}' is not a component of a {n}-vector; "
                f"use {names} or an index")
        return c

    def _component_target(self, target):
        """An assignment target ``v.x`` or ``vf[i].x`` as the subscript it names."""
        if not isinstance(target, ast.Attribute):
            return target
        vector = target.value
        if isinstance(vector, ast.Name) and vector.id in self._vector_vars:
            n = self._vector_vars[vector.id]
            is_matrix = vector.id in self._matrix_vars
        elif (isinstance(vector, ast.Subscript) and isinstance(vector.value, ast.Name)
                and vector.value.id in self._vector_arrays):
            n = self._vector_arrays[vector.value.id]
            is_matrix = vector.value.id in self._matrix_arrays
        else:
            return target
        if is_matrix:
            raise self._source_error(
                target, "attribute", f"'{target.attr}' names a component of a vector; "
                f"a matrix entry is m[i, j]")
        index = ast.copy_location(ast.Constant(self._named_component(target, n)), target)
        return ast.copy_location(ast.Subscript(value=vector, slice=index, ctx=target.ctx), target)

    def _vector_conditional(self, node, condition, then_value, then_stmts,
                            else_value, else_stmts):
        """``a if c else b`` with a vector arm: one selection per component.

        The condition is evaluated once. An arm that needs statements of
        its own (an inlined call) runs them only when it is selected.
        """
        shape = self._common_shape(node, "conditional expression", [then_value, else_value])
        then_value, else_value = zip(*self._componentwise(
            node, "conditional expression", [then_value, else_value]), strict=True)
        if then_stmts or else_stmts:
            names = []
            for _ in then_value:
                names.append(self._fresh_name(f"__conditional_{self._inline_counter}__"))
                self._inline_counter += 1
            self._pre_stmts.append(ir.IRIf(
                condition,
                [*then_stmts, *(ir.IRAssign(n, v) for n, v in zip(names, then_value))],
                [*else_stmts, *(ir.IRAssign(n, v) for n, v in zip(names, else_value))],
            ))
            return VectorValue([ir.IRName(n) for n in names], shape)
        condition = self._capture_value(condition, self._pre_stmts)
        return VectorValue(
            [ir.IRIfExp(condition, t, e) for t, e in zip(then_value, else_value)], shape)

    def _vector_compare(self, node, ops: list, values: list) -> VectorValue:
        """``v < w``, ``lo <= v < hi``: a vector of 0/1, one comparison per component.

        A scalar operand is compared with every component. Each link of a
        chain is a comparison vector of its own and the links are combined
        with ``and`` per component.
        """
        links = [self._elementwise(node, "comparison", [left, right],
                                   lambda row, op=op: ir.IRCompare(op, row[0], row[1]))
                 for op, left, right in zip(ops, values[:-1], values[1:], strict=True)]
        if len(links) == 1:
            return links[0]
        return self._elementwise(node, "comparison", links,
                                 lambda row: ir.IRBoolOp(op="and", values=list(row)))

    def _try_parse_vector_construct(self, value_node: ast.expr) -> list | None:
        """The component nodes of ``tack.vecN(a, b, ...)`` with N scalar arguments
        written out, or None."""
        if not isinstance(value_node, ast.Call):
            return None
        try:
            name = self._resolve_call_name(value_node)
        except (NotImplementedError, AttributeError):
            return None
        width = _VECTOR_CONSTRUCTORS.get(name)
        if width is None or len(value_node.args) != width or width == 1:
            return None
        if any(isinstance(arg, (ast.List, ast.Starred)) for arg in value_node.args):
            return None
        return value_node.args

    def _visit_vector_construct(self, node: ast.Call, width: int):
        """``tack.vecN(...)``: N scalars, one scalar for every component, or
        one vector of N components."""
        what = f"tack.vec{width}()"
        values = self._visit_ordered(node.args)
        if len(values) == width and not any(isinstance(v, list) for v in values):
            return VectorValue(values)
        if len(values) == 1:
            value = values[0]
            if isinstance(value, VectorValue):
                if value.shape != (width,):
                    raise self._source_error(
                        node, what, f"takes a vector of {width} components, not one of "
                        f"shape {value.shape}")
                return value
            if not isinstance(value, list):
                value = self._capture_value(value, self._pre_stmts)
                return VectorValue([value] * width)
        raise self._source_error(
            node, what, f"takes {width} scalars, one scalar or one {width}-vector")

    def _visit_matrix_construct(self, node: ast.Call, extent: int | None) -> VectorValue:
        """``tack.mat(row, ...)`` and ``tack.matN(...)``: rows that are lists of
        scalars or vectors, or (``matN``) N*N scalars in row-major order."""
        what = f"tack.mat{extent}()" if extent else "tack.mat()"
        if not node.args:
            raise self._source_error(node, what, "needs its rows")
        if extent and len(node.args) == extent * extent and not any(
                isinstance(arg, ast.List) for arg in node.args):
            values = self._visit_ordered(node.args)
            if not any(isinstance(v, list) for v in values):
                return VectorValue(values, (extent, extent))
        rows = []
        for row in node.args:
            if isinstance(row, ast.List):
                rows.append(VectorValue(self._visit_ordered(row.elts)))
            else:
                value, statements = self._visit_expression(row)
                if statements:
                    rows = [self._capture_value(r, self._pre_stmts) for r in rows]
                    self._pre_stmts.extend(statements)
                rows.append(value)
        for row in rows:
            if (not isinstance(row, VectorValue) or len(row.shape) != 1
                    or any(isinstance(entry, list) for entry in row)):
                raise self._source_error(
                    node, what, "needs each row to be a list of scalars or a vector")
        widths = {len(row) for row in rows}
        if len(widths) != 1:
            raise self._source_error(node, what, "has rows of different lengths")
        shape = (len(rows), widths.pop())
        if max(shape) > MAX_MATRIX_EXTENT:
            raise self._source_error(
                node, what, f"is {shape[0]}x{shape[1]}; matrices are at most "
                f"{MAX_MATRIX_EXTENT}x{MAX_MATRIX_EXTENT}")
        if extent and shape != (extent, extent):
            raise self._source_error(
                node, what, f"needs {extent} rows of {extent}, or {extent * extent} scalars; "
                f"got {shape[0]} rows of {shape[1]}")
        return VectorValue([entry for row in rows for entry in row], shape)

    def _visit_matrix_identity(self, node: ast.Call) -> VectorValue:
        """``tack.identity(n)`` for a literal ``n``, as ``np.identity``."""
        if (len(node.args) != 1 or not isinstance(node.args[0], ast.Constant)
                or not isinstance(node.args[0].value, int)
                or not 1 <= node.args[0].value <= MAX_MATRIX_EXTENT):
            raise NotImplementedError(
                f"tack.identity() takes a literal size from 1 to {MAX_MATRIX_EXTENT}")
        n = node.args[0].value
        return VectorValue(
            [ir.IRConstant(1.0 if i == j else 0.0) for i in range(n) for j in range(n)], (n, n))

    def _matmul(self, node, left, right):
        """``left @ right``: matrix products, with a vector as a row or a column.

        Matrix @ matrix is a matrix, matrix @ vector and vector @ matrix
        are vectors, and vector @ vector is their dot product.
        """
        for operand in (left, right):
            if not isinstance(operand, VectorValue):
                raise self._source_error(
                    node, "'@'", "multiplies vectors and matrices; use '*' to scale by a scalar")
        # Every entry is used once per row or column of the other operand.
        left = self._capture_value(left, self._pre_stmts)
        right = self._capture_value(right, self._pre_stmts)
        lrows, lcols = (1, len(left)) if len(left.shape) == 1 else left.shape
        rrows, rcols = (len(right), 1) if len(right.shape) == 1 else right.shape
        if lcols != rrows:
            raise self._source_error(
                node, "'@'", f"multiplies {left.describe()} by {right.describe()}")
        entries = []
        for i in range(lrows):
            for j in range(rcols):
                total = ir.IRBinOp(op="*", left=left[i * lcols], right=right[j])
                for k in range(1, lcols):
                    total = ir.IRBinOp(op="+", left=total, right=ir.IRBinOp(
                        op="*", left=left[i * lcols + k], right=right[k * rcols + j]))
                entries.append(total)
        if len(left.shape) == 1 and len(right.shape) == 1:
            return entries[0]
        if len(left.shape) == 1 or len(right.shape) == 1:
            return VectorValue(entries)
        return VectorValue(entries, (lrows, rcols))

    def _emit_matrix_method(self, method_name: str, m: VectorValue, arg_nodes: list):
        """transpose, trace, determinant and inverse of a matrix value."""
        rows, cols = m.shape
        if arg_nodes:
            raise NotImplementedError(f"{method_name}() takes no arguments")
        if method_name == "transpose":
            return VectorValue([m[i * cols + j] for j in range(cols) for i in range(rows)],
                               (cols, rows))
        if rows != cols:
            raise NotImplementedError(
                f"{method_name}() needs a square matrix, not {m.describe()}")
        if method_name == "trace":
            total = m[0]
            for i in range(1, rows):
                total = ir.IRBinOp(op="+", left=total, right=m[i * cols + i])
            return total
        if rows not in (2, 3):
            raise NotImplementedError(
                f"{method_name}() is implemented for 2x2 and 3x3 matrices, not {m.describe()}")

        def mul(a, b):
            return ir.IRBinOp(op="*", left=a, right=b)

        def sub(a, b):
            return ir.IRBinOp(op="-", left=a, right=b)

        # Each entry appears in several products.
        m = self._capture_value(m, self._pre_stmts)
        if rows == 2:
            a, b, c, d = m
            det = sub(mul(a, d), mul(b, c))
            adjugate = [d, ir.IRUnaryOp(op="-", operand=b), ir.IRUnaryOp(op="-", operand=c), a]
        else:
            a, b, c, d, e, f, g, h, i = m
            # Cofactors along the first row give the determinant; the
            # adjugate is the transpose of the cofactor matrix.
            c00, c01, c02 = sub(mul(e, i), mul(f, h)), sub(mul(f, g), mul(d, i)), \
                sub(mul(d, h), mul(e, g))
            det = ir.IRBinOp(op="+", left=ir.IRBinOp(op="+", left=mul(a, c00), right=mul(b, c01)),
                             right=mul(c, c02))
            adjugate = [c00, sub(mul(c, h), mul(b, i)), sub(mul(b, f), mul(c, e)),
                        c01, sub(mul(a, i), mul(c, g)), sub(mul(c, d), mul(a, f)),
                        c02, sub(mul(b, g), mul(a, h)), sub(mul(a, e), mul(b, d))]
        if method_name == "determinant":
            return det
        det = self._capture_value(det, self._pre_stmts)
        return VectorValue([ir.IRBinOp(op="/", left=entry, right=det) for entry in adjugate],
                           (rows, cols))

    def _assign_vector(self, name: str, ast_elts: list) -> list:
        """Assign a vector construction to a variable: v = tack.vec3(a, b, c)."""
        return self._assign_vector_from_ir(name, self._visit_ordered(ast_elts))

    def _assign_vector_from_ir(self, name: str, components: list) -> list:
        """Assign a vector expression result to a variable."""
        ndim = len(components)
        self._bind_vector(name, components)
        names = [self._component_name(name, c) for c in range(ndim)]
        stmts = []
        components = self._settle_components(components, stmts, names)
        for target, comp in zip(names, components):
            stmts.append(ir.IRAssign(target=target, value=comp))
        return stmts

    def _settle_components(self, components: list, statements: list, targets=None) -> list:
        """Components that can be assigned one at a time, in order.

        A vector assignment lowers to one assignment per component, but its
        right side is a single value that Python evaluates completely before
        binding anything. A component that reads what an earlier component's
        assignment has already written would see the new value, so it is
        evaluated into a temporary first: for a vector variable, a read of
        an earlier target component (``v = v.cross(w)``); for an array
        element, any array load, since arrays may share storage
        (``a[i] = a[i].cross(b[i])``). ``targets`` names the variable's
        components, or is None for a store to an array.
        """
        settled = list(components)
        for c in range(1, len(settled)):
            if targets is None:
                hazard = any(isinstance(n, ir.IRArrayLoad) for n in walk_ir(settled[c]))
            else:
                written = set(targets[:c])
                hazard = any(isinstance(n, ir.IRName) and n.name in written
                             for n in walk_ir(settled[c]))
            if hazard:
                settled[c] = self._capture_value(settled[c], statements, freeze_name=True)
        return _like(components, settled)

    def _componentwise(self, node, construct: str, args: list) -> list:
        """Per-component argument lists for a scalar operation given vectors.

        Vector arguments must have one width; scalar arguments are repeated
        for every component.
        """
        self._common_shape(node, construct, args)
        width = next(len(a) for a in args if isinstance(a, list))
        return [[a[c] if isinstance(a, list) else a for a in args] for c in range(width)]

    def _common_shape(self, node, construct: str, args: list) -> tuple:
        """The one shape the vectors among ``args`` have, or a diagnostic."""
        shapes = []
        for a in args:
            if isinstance(a, TupleValue):
                raise self._source_error(
                    node, construct, f"is given {a.describe()} where a value is required")
            if isinstance(a, list):
                shape = getattr(a, 'shape', (len(a),))
                if shape not in shapes:
                    shapes.append(shape)
        if len(shapes) != 1:
            kinds = sorted(VectorValue([None] * _size(shape), shape).describe()[2:]
                           for shape in shapes)
            raise self._source_error(node, construct, "combines a " + " and ".join(kinds))
        return shapes[0]

    def _elementwise(self, node, construct: str, args: list, build) -> VectorValue:
        """Apply ``build`` to each component's arguments; the result keeps the shape."""
        shape = self._common_shape(node, construct, args)
        return VectorValue(
            (build(row) for row in self._componentwise(node, construct, args)), shape)

    def _visit_as_vector(self, value_node, ndim: int) -> list:
        """Visit an expression and ensure it returns vector components."""
        result = self.visit(value_node)
        if isinstance(result, list):
            if len(result) != ndim:
                raise TypeError(f"Vector dimension mismatch: expected {ndim}, got {len(result)}")
            return result
        # Scalar → broadcast to all components
        return [result] * ndim

    def _try_visit_vector_method(self, node: ast.Call):
        """Try to handle vector method calls: v.normalized(), v.cross(w), etc."""
        if not isinstance(node.func, ast.Attribute):
            return None

        method_name = node.func.attr
        obj_node = node.func.value

        # Check if the object is a known vector variable
        if isinstance(obj_node, ast.Name) and obj_node.id in self._vector_vars:
            components = self._variable_value(obj_node.id)
            return self._emit_vector_method(method_name, components, len(components), node.args)

        # Also handle chained expressions: (expr).normalized()
        obj_result = self.visit(obj_node)
        if isinstance(obj_result, list):
            ndim = len(obj_result)
            return self._emit_vector_method(method_name, obj_result, ndim, node.args)

        return None

    def _emit_vector_method(self, method_name: str, components: list, ndim: int, arg_nodes: list):
        """Emit IR for a vector method call."""
        is_matrix = len(getattr(components, 'shape', ())) == 2
        if method_name in ('transpose', 'trace', 'determinant', 'inverse'):
            if not is_matrix:
                raise NotImplementedError(f"{method_name}() is a method of a matrix")
            return self._emit_matrix_method(method_name, components, arg_nodes)
        if is_matrix and method_name in ('dot', 'cross', 'outer_product'):
            raise NotImplementedError(
                f"{method_name}() is a method of a vector; multiply matrices with '@'")
        if method_name == "outer_product":
            if len(arg_nodes) != 1:
                raise TypeError("outer_product() takes exactly one argument")
            other = self.visit(arg_nodes[0])
            if not isinstance(other, VectorValue) or len(other.shape) != 1:
                raise TypeError("outer_product() argument must be a vector")
            if max(ndim, len(other)) > MAX_MATRIX_EXTENT:
                raise NotImplementedError(
                    f"outer_product() of a {ndim}-vector and a {len(other)}-vector is larger "
                    f"than {MAX_MATRIX_EXTENT}x{MAX_MATRIX_EXTENT}")
            left = self._capture_value(VectorValue(components), self._pre_stmts)
            other = self._capture_value(other, self._pre_stmts)
            return VectorValue([ir.IRBinOp(op="*", left=a, right=b) for a in left for b in other],
                               (ndim, len(other)))
        if method_name in ('norm_sqr', 'sum', 'min', 'max') and arg_nodes:
            raise NotImplementedError(f"{method_name}() takes no arguments")
        if method_name == "normalized":
            # length = sqrt(sum(c*c for c in components)), plus an optional
            # eps that keeps a zero vector from dividing by zero
            if len(arg_nodes) > 1:
                raise NotImplementedError("normalized() takes at most one argument (eps)")
            sum_sq = components[0]
            sum_sq = ir.IRBinOp(op="*", left=components[0], right=components[0])
            for c in range(1, ndim):
                sum_sq = ir.IRBinOp(op="+", left=sum_sq,
                                    right=ir.IRBinOp(op="*", left=components[c], right=components[c]))
            length = ir.IRCall(func_name="sqrt", args=[sum_sq])
            if arg_nodes:
                eps = self.visit(arg_nodes[0])
                if isinstance(eps, list):
                    raise TypeError("normalized() eps must be a scalar")
                length = ir.IRBinOp(op="+", left=length, right=eps)
            return VectorValue(ir.IRBinOp(op="/", left=c, right=length) for c in components)

        if method_name == "sum":
            total = components[0]
            for c in components[1:]:
                total = ir.IRBinOp(op="+", left=total, right=c)
            return total

        if method_name in ("min", "max"):
            extreme = components[0]
            for c in components[1:]:
                extreme = ir.IRCall(func_name=method_name, args=[extreme, c])
            return extreme

        if method_name == "norm":
            # norm(eps) is sqrt(norm_sqr() + eps): the eps goes under the
            # root, where it keeps a zero vector's length from being zero.
            if len(arg_nodes) > 1:
                raise NotImplementedError("norm() takes at most one argument (eps)")
            sum_sq = ir.IRBinOp(op="*", left=components[0], right=components[0])
            for c in range(1, ndim):
                sum_sq = ir.IRBinOp(op="+", left=sum_sq,
                                    right=ir.IRBinOp(op="*", left=components[c], right=components[c]))
            if arg_nodes:
                eps = self.visit(arg_nodes[0])
                if isinstance(eps, list):
                    raise TypeError("norm() eps must be a scalar")
                sum_sq = ir.IRBinOp(op="+", left=sum_sq, right=eps)
            return ir.IRCall(func_name="sqrt", args=[sum_sq])

        if method_name == "norm_sqr":
            sum_sq = ir.IRBinOp(op="*", left=components[0], right=components[0])
            for c in range(1, ndim):
                sum_sq = ir.IRBinOp(op="+", left=sum_sq,
                                    right=ir.IRBinOp(op="*", left=components[c], right=components[c]))
            return sum_sq

        if method_name == "dot":
            if len(arg_nodes) != 1:
                raise TypeError("dot() takes exactly one argument")
            other = self.visit(arg_nodes[0])
            if not isinstance(other, list) or len(other) != ndim:
                raise TypeError("dot() argument must be a vector of same dimension")
            result = ir.IRBinOp(op="*", left=components[0], right=other[0])
            for c in range(1, ndim):
                result = ir.IRBinOp(op="+", left=result,
                                    right=ir.IRBinOp(op="*", left=components[c], right=other[c]))
            return result

        if method_name == "cross":
            if ndim != 3:
                raise TypeError("cross() only defined for 3D vectors")
            if len(arg_nodes) != 1:
                raise TypeError("cross() takes exactly one argument")
            other = self.visit(arg_nodes[0])
            if not isinstance(other, list) or len(other) != 3:
                raise TypeError("cross() argument must be a 3D vector")
            a0, a1, a2 = components
            b0, b1, b2 = other
            return VectorValue([
                ir.IRBinOp(op="-",
                           left=ir.IRBinOp(op="*", left=a1, right=b2),
                           right=ir.IRBinOp(op="*", left=a2, right=b1)),
                ir.IRBinOp(op="-",
                           left=ir.IRBinOp(op="*", left=a2, right=b0),
                           right=ir.IRBinOp(op="*", left=a0, right=b2)),
                ir.IRBinOp(op="-",
                           left=ir.IRBinOp(op="*", left=a0, right=b1),
                           right=ir.IRBinOp(op="*", left=a1, right=b0)),
            ])

        raise NotImplementedError(f"Vector method '{method_name}' not supported")

    def _vector_component(self, node: ast.Subscript, components: list):
        """``components[index]`` as one scalar expression.

        A literal index picks the component at lowering. A runtime index
        selects through a chain of conditional expressions, so the result is
        a scalar wherever a scalar is expected and no local array is needed;
        an index past the last component yields the last component (Python
        would raise, and a kernel cannot). The index is evaluated once.
        """
        selector = self._component_selector(
            node, len(components), getattr(components, 'shape', None))
        if isinstance(selector, int):
            return components[selector]
        table = getattr(components, 'table', None)
        if table is not None:
            return ir.IRTableLoad(*table, selector)
        return self._select_component(selector, components)

    def _component_selector(self, node: ast.Subscript, n: int, shape=None):
        """The component a subscript of an n-vector, or of a matrix, names.

        An ``int`` for a literal index, checked against the width;
        otherwise the index as an IR value, evaluated once. A matrix takes
        a row and a column, and the component is their row-major position.
        """
        if shape is not None and len(shape) == 2:
            return self._matrix_selector(node, shape)
        index = node.slice
        if isinstance(index, ast.Tuple):
            raise self._source_error(
                node, "subscript", f"gives {len(index.elts)} indices to a {n}-vector")
        if isinstance(index, ast.Constant) and isinstance(index.value, int):
            c = index.value
            if not -n <= c < n:
                raise self._source_error(
                    node, "subscript", f"component {c} of a {n}-vector is out of range")
            return c % n
        idx, statements = self._visit_expression(index)
        self._pre_stmts.extend(statements)
        if isinstance(idx, list):
            raise self._source_error(node, "subscript", "a vector cannot index a vector")
        return self._capture_value(self._narrow_selector(idx), self._pre_stmts)

    @staticmethod
    def _narrow_selector(selector):
        """The selector as an i32, which the component tests compare against.

        A component index lies in ``[0, n)`` (the language contract), so
        nothing is lost; and IGC 2.7.11 miscompiled a chain of 64-bit
        equality tests on a selector that a conditional expression defined
        (``rows[f][j + 1 if j + 1 < size else 0]`` in a sequential loop):
        the first component for an even selector, the last for an odd one
        (TA-48, ``docs/design/vendor-workarounds.md``). Compared as an i32
        the chain compiles right.
        """
        return ir.IRCast(value=selector, dtype=i32)

    def _matrix_selector(self, node: ast.Subscript, shape: tuple):
        """The row-major position ``m[i, j]`` names in a matrix of ``shape``."""
        rows, cols = shape
        index = node.slice
        if not (isinstance(index, ast.Tuple) and len(index.elts) == 2):
            raise self._source_error(
                node, "subscript", f"of a {rows}x{cols} matrix needs a row and a column: m[i, j]")
        literal = [e.value if isinstance(e, ast.Constant) and isinstance(e.value, int)
                   and not isinstance(e.value, bool) else None for e in index.elts]
        if None not in literal:
            i, j = literal
            if not (-rows <= i < rows and -cols <= j < cols):
                raise self._source_error(
                    node, "subscript", f"entry [{i}, {j}] of a {rows}x{cols} matrix is out of range")
            return (i % rows) * cols + (j % cols)
        row, col = self._visit_ordered(index.elts)
        if isinstance(row, list) or isinstance(col, list):
            raise self._source_error(node, "subscript", "a vector cannot index a matrix")
        position = ir.IRBinOp(
            op="+", left=ir.IRBinOp(op="*", left=row, right=ir.IRConstant(cols)), right=col)
        return self._capture_value(self._narrow_selector(position), self._pre_stmts)

    def _select_component(self, selector, components: list):
        """The component a runtime ``selector`` picks; past the last, the last."""
        result = components[-1]
        for c in range(len(components) - 2, -1, -1):
            result = ir.IRIfExp(
                ir.IRCompare(op="==", left=selector, right=ir.IRConstant(c)),
                components[c], result)
        return result

    def _store_vector_component(self, target: ast.Subscript, value):
        """``v[index] = value`` for a vector variable ``v``.

        A literal index assigns one component. A runtime index guards an
        assignment to each component with a comparison, so exactly the
        selected component changes; out of range changes nothing.
        """
        name = target.value.id
        n = self._vector_vars[name]
        if isinstance(value, list):
            raise self._source_error(
                target, "assignment", "a component of a vector must be a scalar")
        idx = self._component_selector(target, n, self._matrix_vars.get(name))
        if isinstance(idx, int):
            return ir.IRAssign(target=self._component_name(name, idx), value=value)
        value = self._capture_value(value, self._pre_stmts)
        return [
            ir.IRIf(ir.IRCompare(op="==", left=idx, right=ir.IRConstant(c)),
                    [ir.IRAssign(target=self._component_name(name, c), value=value)], [])
            for c in range(n)
        ]
