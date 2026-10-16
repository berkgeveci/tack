"""Values that exist only during lowering, and the names they use.

A vector or matrix is lowered to one scalar IR expression per component
(`VectorValue`), and a source tuple to its items (`TupleValue`); neither
enters the IR.
"""

# Math builtins that map to LLVM intrinsics / libm calls
MATH_BUILTINS = {
    "sqrt", "sin", "cos", "tan",
    "asin", "acos", "atan", "atan2",
    "sinh", "cosh", "tanh",
    "exp", "log", "log2", "log10",
    "floor", "ceil",
    "abs", "min", "max", "pow",
    "fabs",
    "exp2",
}

# Names for the first four components of a vector: v.x is v[0].
COMPONENT_NAMES = "xyzw"

# Matrices are scalarized like vectors, so their size is the size of the
# expressions they expand to: a 4x4 product is 64 multiplications.
MAX_MATRIX_EXTENT = 4


class VectorValue(list):
    """A vector-valued expression during lowering.

    Tack scalarizes vectors: the value is one scalar IR expression per
    component, in storage order, and it never enters the IR itself. It is
    a ``list`` so code that maps over components stays plain. What the
    class adds is an identity distinct from a tuple of values and from a
    list of statements, and a ``shape``: ``(n,)`` for a vector.
    """

    def __init__(self, components, shape=None):
        super().__init__(components)
        self.shape = (len(self),) if shape is None else tuple(shape)

    def describe(self) -> str:
        if len(self.shape) == 1:
            return f"a {self.shape[0]}-vector"
        return "a " + "x".join(str(d) for d in self.shape) + " matrix"


class TupleValue(list):
    """Several values side by side: a tuple in the source.

    The indices of ``f[i, j]``, the right side of ``a, b = b, a``, and what
    a device function with ``return a, b`` yields. An item may itself be a
    ``VectorValue``.
    """

    def describe(self) -> str:
        return f"a tuple of {len(self)} values"


def _size(shape) -> int:
    total = 1
    for extent in shape:
        total *= extent
    return total


def _like(original, components):
    """``components`` as the kind of value ``original`` is."""
    if isinstance(original, VectorValue):
        return VectorValue(components, original.shape)
    if isinstance(original, TupleValue):
        return TupleValue(components)
    return list(components)


def _describe_value(value) -> str:
    describe = getattr(value, 'describe', None)
    return describe() if describe is not None else f"a list of {len(value)} values"


# Kernel value constructors and the shape each builds.
_VECTOR_CONSTRUCTORS = {"vec2": 2, "vec3": 3, "vec4": 4}
_MATRIX_CONSTRUCTORS = {"mat": None, "mat2": 2, "mat3": 3, "mat4": 4}
