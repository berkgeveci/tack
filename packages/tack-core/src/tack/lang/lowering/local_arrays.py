"""Workgroup (shared) and per-iteration (local) arrays, as lowering emits them.

`tack.shared(dtype, n)`, `tack.shared_like(array, n)`,
`tack.local_array(dtype, n)` and `tack.local_array_like(array, n)` bound to
a name become `IRSharedAlloc` / `IRLocalAlloc`; the name is then an array
the kernel indexes, and its size is kept for debug mode's bounds checks.
"""

import ast

from tack.lang import ir


class LocalArrays:
    """Recognizes shared and local array allocations (the `_try_parse_*` methods).

    Owns `_shared_vars` (the names bound to such arrays) and `_array_sizes`
    (each one's size expression, which `BoundsChecks` reads).
    """

    def _init_local_arrays(self):
        self._shared_vars: set[str] = set()    # names bound to shared or local arrays
        self._array_sizes: dict = {}   # local/shared array name -> its size expression

    def _try_parse_shared_alloc(self, target_name: str, value_node: ast.Call):
        """Try to parse smem = tack.shared(tack.f32, 256). Returns IRSharedAlloc or None."""
        try:
            name = self._resolve_call_name(value_node)
        except (NotImplementedError, AttributeError):
            return None
        if name != "shared":
            return None
        if len(value_node.args) != 2:
            raise NotImplementedError("tack.shared() takes exactly 2 arguments (dtype, size)")

        # Parse dtype argument — tack.f32, tack.i32, etc.
        dtype_arg = value_node.args[0]
        if isinstance(dtype_arg, ast.Attribute):
            dtype_str = dtype_arg.attr
        elif isinstance(dtype_arg, ast.Name):
            dtype_str = dtype_arg.id
        else:
            raise NotImplementedError("tack.shared() dtype must be tack.f32, tack.i32, etc.")

        from tack.lang.types import _NAME_TO_TYPE
        scalar_type = _NAME_TO_TYPE.get(dtype_str)
        if scalar_type is None:
            raise NotImplementedError(f"Unsupported shared memory dtype: {dtype_str}")

        size = self.visit(value_node.args[1])
        self._shared_vars.add(target_name)
        self._array_sizes[target_name] = size
        return ir.IRSharedAlloc(name=target_name, dtype=scalar_type, size=size)

    def _try_parse_shared_like(self, target_name: str, value_node: ast.Call):
        """Try to parse smem = tack.shared_like(array, 256). Returns IRSharedAlloc or None."""
        try:
            name = self._resolve_call_name(value_node)
        except (NotImplementedError, AttributeError):
            return None
        if name != "shared_like":
            return None
        if len(value_node.args) != 2:
            raise NotImplementedError("tack.shared_like() takes exactly 2 arguments (array, size)")

        # First arg is the array reference
        array_arg = value_node.args[0]
        if isinstance(array_arg, ast.Name):
            array_name = array_arg.id
        elif isinstance(array_arg, ast.Attribute):
            array_name = array_arg.attr
        else:
            raise NotImplementedError("tack.shared_like() first argument must be an array name")

        size = self.visit(value_node.args[1])
        self._shared_vars.add(target_name)
        self._array_sizes[target_name] = size
        return ir.IRSharedAlloc(name=target_name, dtype=None, size=size, array_name=array_name)

    def _try_parse_local_alloc(self, target_name: str, value_node: ast.Call):
        """Try to parse arr = tack.local_array(tack.f32, 8). Returns IRLocalAlloc or None."""
        try:
            name = self._resolve_call_name(value_node)
        except (NotImplementedError, AttributeError):
            return None
        if name != "local_array":
            return None
        if len(value_node.args) != 2:
            raise NotImplementedError("tack.local_array() takes exactly 2 arguments (dtype, size)")

        dtype_arg = value_node.args[0]
        if isinstance(dtype_arg, ast.Attribute):
            dtype_str = dtype_arg.attr
        elif isinstance(dtype_arg, ast.Name):
            dtype_str = dtype_arg.id
        else:
            raise NotImplementedError("tack.local_array() dtype must be tack.f32, tack.i32, etc.")

        from tack.lang.types import _NAME_TO_TYPE
        scalar_type = _NAME_TO_TYPE.get(dtype_str)
        if scalar_type is None:
            raise NotImplementedError(f"Unsupported local array dtype: {dtype_str}")

        size = self.visit(value_node.args[1])
        self._shared_vars.add(target_name)  # treat as array for indexing purposes
        self._array_sizes[target_name] = size
        return ir.IRLocalAlloc(name=target_name, dtype=scalar_type, size=size)

    def _try_parse_local_array_like(self, target_name: str, value_node: ast.Call):
        """Try to parse arr = tack.local_array_like(array, 8). Returns IRLocalAlloc or None."""
        try:
            name = self._resolve_call_name(value_node)
        except (NotImplementedError, AttributeError):
            return None
        if name != "local_array_like":
            return None
        if len(value_node.args) != 2:
            raise NotImplementedError("tack.local_array_like() takes exactly 2 arguments (array, size)")

        # First arg is the array reference
        array_arg = value_node.args[0]
        if isinstance(array_arg, ast.Name):
            array_name = array_arg.id
        elif isinstance(array_arg, ast.Attribute):
            array_name = array_arg.attr
        else:
            raise NotImplementedError("tack.local_array_like() first argument must be an array name")

        size = self.visit(value_node.args[1])
        self._shared_vars.add(target_name)  # treat as array for indexing purposes
        self._array_sizes[target_name] = size
        return ir.IRLocalAlloc(name=target_name, dtype=None, size=size, array_name=array_name)
