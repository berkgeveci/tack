"""Array accesses, as lowering emits them.

An index is linearized row-major over the array's dimensions (each a
`IRDimSize` that resolution folds, or, for the parallel loop's bound, dispatch
evaluates) and checked in debug mode. An element of a vector or matrix
array is one access per component: ``element * n + c`` of an interleaved
array, or ``element`` of the component's own parameter for a
structure-of-arrays array, whose name is never an IR value.
"""

import ast

from tack.lang import ir
from tack.lang.array import soa_param_names
from tack.lang.ir_traversal import walk_ir
from tack.lang.lowering.values import VectorValue, _describe_value, _size
from tack.lang.types import i64


class ArrayAccess:
    """Lowers array indexing and the elements of vector and matrix arrays.

    Owns the array metadata the caller passes (`vector_arrays`):
    `_vector_arrays` (name -> component count), `_matrix_arrays` (name ->
    (rows, columns)) and `_soa_arrays` (name -> its component parameters).
    Inlining extends them for a device function's renamed parameters.
    """

    def _init_arrays(self, vector_arrays):
        # Vector array metadata: param_name -> number of components. The
        # caller gives a matrix array as its (rows, columns); those are
        # kept beside the component count.
        self._vector_arrays: dict[str, int] = {}
        self._matrix_arrays: dict[str, tuple] = {}
        # An array stored one component per array (``("soa", width)``):
        # name -> the parameters standing for its components, in order.
        # The name itself is no parameter and never a value in the IR.
        self._soa_arrays: dict[str, tuple] = {}
        for name, width in (vector_arrays or {}).items():
            if isinstance(width, tuple) and width[0] == "soa":
                width = width[1]
                self._soa_arrays[name] = soa_param_names(
                    name, _size(width) if isinstance(width, tuple) else width)
            if isinstance(width, tuple):
                self._matrix_arrays[name] = width
                width = _size(width)
            self._vector_arrays[name] = width

    def _as_dim_size(self, node: ast.Subscript) -> ir.IRDimSize | None:
        """`f.shape[k]` on an array → IRDimSize, which resolve_ir folds.

        Emitting a generic IRAttribute here is what limited `.shape` to the
        outermost loop bound: dispatch evaluated that one expression
        numerically, and every other position reached codegen unresolved.
        IRDimSize is the node the resolve pass already knows how to fold,
        and it recurses through every body.
        """
        value = node.value
        if not (isinstance(value, ast.Attribute) and value.attr == "shape"):
            return None
        if not isinstance(value.value, ast.Name):
            return None
        index = node.slice
        if not (isinstance(index, ast.Constant) and isinstance(index.value, int)):
            raise NotImplementedError(
                f"'{value.value.id}.shape[...]' needs a constant dimension "
                f"index; the shape is resolved when the kernel is compiled."
            )
        return ir.IRDimSize(array_name=self._shape_name(value.value.id), dim=index.value)

    def _array_shape(self, name: str) -> tuple:
        """The shape of one element of the vector or matrix array ``name``."""
        return self._matrix_arrays.get(name) or (self._vector_arrays[name],)

    def _store_vector_array(self, target: ast.Subscript, components: list) -> list:
        """Store a vector into a vector array: array[i] = vec."""
        ndim = len(components)
        name = target.value.id if isinstance(target.value, ast.Name) else None
        width = self._vector_arrays.get(name)
        if width is None:
            raise self._source_error(
                target, "assignment", f"stores {_describe_value(components)} into an "
                f"element of scalar storage; '{name}' is not a vector array")
        shape = self._array_shape(name)
        if getattr(components, 'shape', (ndim,)) != shape:
            element = VectorValue([None] * width, shape).describe()[2:]
            raise self._source_error(
                target, "assignment",
                f"stores {_describe_value(components)} into an array of {element}"
                + ("s" if len(shape) == 1 else " elements"))
        array_node, index_node, components = self._visit_store_location(target, components)
        stmts = []
        # Each component store repeats the index, so one that loads from a
        # array is read before the first store can change it.
        if any(isinstance(n, ir.IRArrayLoad) for n in walk_ir(index_node)):
            index_node = self._capture_value(index_node, stmts)
        locations = self._element_locations(name, self._element_index(index_node), array_node)
        components = self._settle_components(components, stmts)
        for (array, at), component in zip(locations, components, strict=True):
            stmts.append(ir.IRArrayStore(array=array, index=at, value=component))
        return stmts

    @staticmethod
    def _element_index(index):
        """A vector array element's index as an integer.

        The cast makes a float index name an element (127.7 is element
        127) before it is scaled by the width. It is 64-bit: a narrower
        one would wrap the index, or its product with the width, for a
        array of more than 2**31 components.
        """
        return ir.IRCast(value=index, dtype=i64)

    def _shape_name(self, name):
        """The parameter whose shape is array ``name``'s: an SOA array's first
        component, which has the array's shape; ``name`` itself otherwise."""
        components = self._soa_arrays.get(name)
        return components[0] if components else name

    def _element_locations(self, name: str, element, array=None) -> list:
        """``(array, index)`` of each component of element ``element`` of the
        vector array ``name``: ``element * n + c`` of the array, or for an SOA
        array ``element`` of the component's own parameter. ``array`` is the
        array's lowered name, if visited already."""
        components = self._soa_arrays.get(name)
        if components is not None:
            return [(ir.IRName(component), element) for component in components]
        if array is None:
            array = ir.IRName(name)
        ndim = self._vector_arrays[name]
        return [(array, self._component_index(element, ndim, c)) for c in range(ndim)]

    def _component_index(self, element, ndim: int, c: int):
        """Where component ``c`` of element ``element`` is stored: element * ndim + c."""
        return ir.IRBinOp(
            op="+",
            left=ir.IRBinOp(op="*", left=element, right=ir.IRConstant(ndim)),
            right=ir.IRConstant(c),
        )

    def _is_array_component(self, target) -> bool:
        """``vf[i][c]``: one component of an element of a vector array."""
        return (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Subscript)
                and isinstance(target.value.value, ast.Name)
                and target.value.value.id in self._vector_arrays)

    def _array_element(self, node: ast.Subscript) -> list:
        """Where each component of the vector array element ``vf[i]`` is:
        ``(array, index)`` pairs (`_element_locations`).

        The element index is evaluated once, so a load and a store of the
        element can share it.
        """
        index = self._capture_value(self._visit_subscript_index(node), self._pre_stmts)
        return self._element_locations(node.value.id, self._element_index(index))

    def _component_stores(self, locations: list, selector, value):
        """Store ``value`` to the component of an array element ``selector`` picks."""
        if isinstance(selector, int):
            array, index = locations[selector]
            return ir.IRArrayStore(array=array, index=index, value=value)
        value = self._capture_value(value, self._pre_stmts)
        return [
            ir.IRIf(ir.IRCompare(op="==", left=selector, right=ir.IRConstant(c)),
                    [ir.IRArrayStore(array=array, index=index, value=value)], [])
            for c, (array, index) in enumerate(locations)
        ]

    def _assign_array_component(self, target: ast.Subscript, value):
        """``vf[i][c] = value``.

        A literal component is one store. A runtime one guards a store to
        each component, as for a vector variable: out of range stores
        nothing.
        """
        if isinstance(value, list):
            raise self._source_error(
                target, "assignment", "a component of a vector must be a scalar")
        # The value is evaluated before the target's indices.
        saved = self._pre_stmts
        self._pre_stmts = []
        locations = self._array_element(target.value)
        selector = self._component_selector(
            target, len(locations), self._matrix_arrays.get(target.value.value.id))
        statements = self._pre_stmts
        self._pre_stmts = saved
        if statements:
            value = self._capture_value(value, self._pre_stmts)
            self._pre_stmts.extend(statements)
        return self._component_stores(locations, selector, value)

    def _augment_array_component(self, target: ast.Subscript, op: str, value_node):
        """``vf[i][c] op= value``: indices once, the old value read first."""
        locations = self._array_element(target.value)
        selector = self._component_selector(
            target, len(locations), self._matrix_arrays.get(target.value.value.id))
        loads = [ir.IRArrayLoad(array=array, index=index) for array, index in locations]
        if isinstance(selector, int):
            read = loads[selector]
        else:
            read = self._select_component(selector, loads)
        right, statements = self._visit_expression(value_node)
        if isinstance(right, list):
            raise self._source_error(
                target, "augmented assignment", "combines a component of a vector with a vector")
        if statements:
            read = self._capture_value(read, self._pre_stmts)
            self._pre_stmts.extend(statements)
        return self._component_stores(
            locations, selector, ir.IRBinOp(op=op, left=read, right=right))

    def _augment_vector_array(self, target: ast.Subscript, op: str, value_node):
        """``vf[i] op= value`` for a vector or a scalar ``value``."""
        locations = self._array_element(target)
        loads = VectorValue((ir.IRArrayLoad(array=array, index=index)
                             for array, index in locations),
                            self._matrix_arrays.get(target.value.id))
        right, statements = self._visit_expression(value_node)
        if statements:
            loads = self._capture_value(loads, self._pre_stmts)
            self._pre_stmts.extend(statements)
        rows = self._componentwise(target, "augmented assignment", [loads, right])
        stmts = []
        values = self._settle_components(
            [ir.IRBinOp(op=op, left=old, right=new) for old, new in rows], stmts)
        stmts.extend(ir.IRArrayStore(array=array, index=index, value=value)
                     for (array, index), value in zip(locations, values))
        return stmts

    def _visit_store_location(self, target, value):
        """Evaluate an assignment RHS before side effects in its target."""
        saved = self._pre_stmts
        self._pre_stmts = []
        soa = isinstance(target.value, ast.Name) and target.value.id in self._soa_arrays
        array = None if soa else self.visit(target.value)
        index = self._visit_subscript_index(target)
        statements = self._pre_stmts
        self._pre_stmts = saved
        if statements:
            value = self._capture_value(value, self._pre_stmts)
            self._pre_stmts.extend(statements)
        return array, index, value

    def _visit_subscript_index(self, node: ast.Subscript):
        """Extract and linearize index from a subscript.

        Handles:
        - array[i] → i
        - array[()] → 0, the one element of a zero-dimensional array
        - array[i, j] → i * DimSize(array, 1) + j
        - array[i, j, k] → (i * DimSize(array, 1) + j) * DimSize(array, 2) + k
        """
        if isinstance(node.slice, ast.Tuple) and not node.slice.elts:
            return ir.IRConstant(0)

        # Multi-dimensional: array[i, j, k]
        if isinstance(node.slice, ast.Tuple):
            array_name = node.value.id if isinstance(node.value, ast.Name) else None
            if array_name is None:
                raise NotImplementedError("Multi-dim indexing requires a named array")

            return self._linearize_index(node, array_name, self._visit_ordered(node.slice.elts))

        index = self.visit(node.slice)
        if isinstance(index, list):
            # A vector indexes one dimension per component: g[I] is g[I[0], I[1]]
            if not isinstance(node.value, ast.Name):
                raise self._source_error(node, "subscript", "indexes an unnamed array with a vector")
            return self._linearize_index(node, node.value.id, [index])
        name = node.value.id if isinstance(node.value, ast.Name) else None
        return self._checked(node, name, [index], index)

    def _linearize_index(self, node, array_name: str, indices: list):
        """Row-major index from one index per dimension: (i * dim1 + j) * dim2 + k.

        A vector among the indices supplies one dimension per component.
        """
        flat = []
        for index in indices:
            flat.extend(index if isinstance(index, list) else [index])
        # The array's shape is known only at dispatch, where its dimension
        # sizes are resolved; each size carries what that check needs. One
        # index alone has no size to resolve: it is a flat index.
        source = (getattr(node, 'lineno', None), getattr(node, 'col_offset', 0) + 1,
                  self._inline_stack[-1] if self._inline_stack else None)
        result = flat[0]
        for d in range(1, len(flat)):
            dim_size = ir.IRDimSize(array_name=self._shape_name(array_name), dim=d)
            dim_size.index_count = len(flat)
            dim_size.index_source = source
            result = ir.IRBinOp(
                op="+",
                left=ir.IRBinOp(op="*", left=result, right=dim_size),
                right=flat[d],
            )
        return self._checked(node, array_name, flat, result, source=source)
