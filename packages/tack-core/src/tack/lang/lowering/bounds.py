"""Debug mode's bounds checks, as lowering emits them.

With `tack.set_debug(True)` every array access's index goes through
`_checked`, which wraps it in an `IRBoundsCheck` and records an
`ir_bounds.BoundsSite` describing the access for the error message (see
`tack.lang.ir_bounds` and `tack.runtime.debug_bounds`). Out of debug mode
it returns the index unchanged.
"""

import ast
import copy

from tack.lang import ir, ir_bounds


class BoundsChecks:
    """Wraps array indices in bounds checks in debug mode (`_checked`).

    Owns `_debug`, `_bounds_sites` (one `BoundsSite` per check, in site
    order) and `_sources` (each Python function's source, for the text of
    an access). Reads the local-array sizes (`_array_sizes`, `_shared_vars`)
    and `_shape_name` from the array mixins, and the position state the
    transformer keeps (`_function_name`, `_inline_stack`, `_python_func`).
    """

    def _init_bounds(self, debug):
        # Debug mode: every array index is checked (IRBoundsCheck),
        # and each checked access is described in _bounds_sites.
        self._debug = debug
        self._bounds_sites: list = []
        self._sources: dict = {}       # Python function -> its source, for _source_text

    def _checked(self, node, name, indices, flat, kind=ir.IRArrayExtent.ELEMENTS,
                 source=None):
        """``flat``, the index of an access to ``name``, as debug mode checks it.

        Several ``indices`` are checked against one dimension of the array
        each; one is a flat index, checked against the ``kind`` of extent.
        An array is checked against its size. Out of debug mode, or for an
        array this function cannot size, ``flat`` is returned unchecked.
        """
        if not self._debug or name is None:
            return flat
        name = self._shape_name(name)
        if name in self._shared_vars:
            size = self._array_sizes.get(name)
            if size is None or len(indices) != 1:
                return flat
            extents = [copy.deepcopy(size)]
        elif len(indices) == 1:
            extents = [ir.IRArrayExtent(name, kind)]
        else:
            extents = [ir.IRArrayExtent(name, ir.IRArrayExtent.LEADING)]
            for d in range(1, len(indices)):
                # Tagged as linearization's are, so too many indices is
                # the same TypeError whichever node resolve meets first.
                dim_size = ir.IRDimSize(array_name=name, dim=d)
                dim_size.index_count = len(indices)
                dim_size.index_source = source
                extents.append(dim_size)
        site = len(self._bounds_sites)
        # Whose source the access is in (func.tag_origin): a call's
        # arguments are the caller's although they are lowered while the
        # callee is inlined. The line in the file is where the function's
        # source, parsed alone, starts at line 1 with the decorator.
        origin = getattr(node, '_tack_origin', None)
        python = origin() if origin is not None else None
        if python is not None:
            function = None if python is self._python_func else python.__name__
        else:   # a node lowering made: where the other diagnostics say
            function = self._inline_stack[-1] if self._inline_stack else None
        code = getattr(python, '__code__', None)
        line = getattr(node, 'lineno', None)
        self._bounds_sites.append(ir_bounds.BoundsSite(
            kernel=self._function_name, function=function,
            file=code.co_filename if code is not None and line is not None else None,
            line=code.co_firstlineno + line - 1 if code is not None and line is not None
            else line,
            column=getattr(node, 'col_offset', 0) + 1,
            text=self._source_text(python, node), dimensions=len(indices)))
        # The check evaluates its own copy of the indices: IR is a tree.
        return ir.IRBoundsCheck(indices=[copy.deepcopy(i) for i in indices], extents=extents,
                                flat=flat, site=site, errors=ir.IRName(ir_bounds.ERRORS_PARAM))

    def _source_text(self, python, node) -> str:
        """``node`` as written in ``python``'s source, before inlining renamed it."""
        if python is not None and getattr(node, 'end_col_offset', None) is not None:
            source = self._sources.get(python)
            if source is None:
                import textwrap

                from tack.lang.func import read_source
                try:
                    source = textwrap.dedent(read_source(python, "kernel"))
                except Exception:   # no source to read: the AST's text below
                    source = ""
                self._sources[python] = source
            try:
                text = ast.get_source_segment(source, node)
            except (IndexError, ValueError):     # positions from elsewhere
                text = None
            if text:
                return text
        return ast.unparse(node)
