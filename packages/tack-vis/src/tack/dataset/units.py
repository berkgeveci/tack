"""Physical units of fields: plain strings, checked and combined by unyt.

A field's unit (``Field.unit``) is meaning, not storage: arrays and kernels
never see one. ``None`` is unknown and ``"dimensionless"`` (``"1"`` on the way
in) is a pure number; nothing derived from an unknown unit is guessed. A
dataset's spatial unit (``DataSet.unit``) is its geometry field's, and its
``origin`` is in it.

Units are stored as unyt's canonical spelling, so ``"m/s"``, ``"m*s**-1"`` and
``"m/s^1"`` are one unit: ``^`` is read as a power, as LLNL units and VTK
write it. Code units are scaled units (``"3.0857e21*cm"``), which need no
registry of their own. Tack's registry is unyt's plus the US survey foot,
``ftUS`` (1200/3937 m), which State Plane coordinates and older LiDAR use and
which differs from the international foot by 2 parts per million.

Derived units (``derived``) are products of powers of units; ``None`` in gives
``None`` out. Units with an offset zero (degrees Celsius and Fahrenheit) are
not multiplied: a difference of them, as a gradient, a jump or a variance
takes, is in kelvin (Rankine), the same size of degree with zero at zero
(``delta``). ``named`` shows a derived unit by its SI name (``Pa*m**3`` as
``J``) where a filter names what it computes.

unyt (with sympy) is imported on first use, so a program that gives no unit
pays nothing. Tack hands out strings only, never unyt objects.
"""

import math
from functools import lru_cache

_registry = None


def _unyt():
    """The unyt module and Tack's registry, made on first use."""
    global _registry
    import unyt

    if _registry is None:
        registry = unyt.UnitRegistry()
        registry.add("ftUS", 1200 / 3937, unyt.dimensions.length, tex_repr=r"\rm{ftUS}")
        _registry = registry
    return unyt, _registry


@lru_cache(maxsize=1024)
def _parsed(text):
    unyt, registry = _unyt()
    try:
        return unyt.Unit(text.replace("^", "**"), registry=registry)
    except Exception as error:                # unyt raises several kinds
        raise ValueError(f"{text!r} is not a unit: {error}") from None


def canonical(unit):
    """``unit`` (a string, or ``None`` for unknown) in unyt's canonical spelling;
    a string that is no unit raises ``ValueError``."""
    if unit is None:
        return None
    if not isinstance(unit, str):
        raise TypeError(f"a unit is a string such as 'm/s', or None, not {unit!r}")
    if not unit.strip():
        raise ValueError("an empty unit: give 'dimensionless' for a pure number, or None "
                         "for unknown")
    return str(_parsed(unit))


def conversion(source, target):
    """``(factor, offset)``: a value in ``source`` is ``factor * value + offset``
    in ``target``. Both must be known and of the same dimensions."""
    if source is None:
        raise ValueError("the unit is unknown: give the field a unit before converting it")
    if target is None:
        raise ValueError("convert to which unit? None is unknown")
    a, b = _parsed(source), _parsed(target)
    if not a.same_dimensions_as(b):
        raise ValueError(f"cannot convert {source!r} to {target!r}: {a.dimensions} and "
                         f"{b.dimensions} differ")
    factor, offset = a.get_conversion_factor(b)
    # unyt's offset is subtracted after scaling.
    return float(factor), 0.0 if offset is None else -float(offset)


def same(a, b):
    """Whether units ``a`` and ``b`` (canonical strings or ``None``) are one unit:
    equal, or both known and one a relabelling of the other (factor 1, no
    offset). Unknown is the same only as unknown."""
    if a == b:
        return True
    if a is None or b is None:
        return False
    a, b = _parsed(a), _parsed(b)
    if not a.same_dimensions_as(b):
        return False
    factor, offset = a.get_conversion_factor(b)
    return factor == 1.0 and offset is None


def compatible(a, b):
    """Whether units ``a`` and ``b`` may meet: one unit, or either unknown (which
    cannot be checked, so it is not refused)."""
    return a is None or b is None or same(a, b)


def common(what, units):
    """The one unit of ``units`` (canonical strings or ``None``), or ``ValueError``
    naming ``what`` (``"the pieces' geometry"``, say) when two known units
    differ. ``None`` if any is unknown."""
    units = list(units)
    known = [unit for unit in units if unit is not None]
    for unit in known[1:]:
        if not same(known[0], unit):
            raise ValueError(f"{what} is in {known[0]!r} and {unit!r}: convert one first "
                             "(tack.dataset.convert)")
    return known[0] if known and len(known) == len(units) else None


def check(what, a, b):
    """``ValueError`` naming ``what`` unless units ``a`` and ``b`` may meet
    (``compatible``)."""
    if not compatible(a, b):
        raise ValueError(f"{what}: {a!r} and {b!r} differ; convert one first "
                         "(tack.dataset.convert)")


def delta(unit):
    """The unit of a difference of values in ``unit``: itself, or for a unit with an
    offset zero, the absolute unit of the same size (kelvin for degrees Celsius,
    Rankine for Fahrenheit). ``None`` stays ``None``."""
    if unit is None:
        return None
    parsed = _parsed(unit)
    if not parsed.base_offset:
        return unit
    for absolute in ("K", "R"):
        if math.isclose(parsed.base_value, _parsed(absolute).base_value, rel_tol=1e-12):
            return absolute
    return canonical(f"{parsed.base_value!r}*K")


def derived(*terms):
    """The unit of a product of powers: ``terms`` are ``(unit, exponent)`` pairs (a
    bare unit is to the first power). ``None`` if any unit is unknown; a unit
    with an offset zero is refused (take its ``delta`` first, or convert)."""
    terms = [(t, 1) if not isinstance(t, tuple) else t for t in terms]
    if any(unit is None for unit, _ in terms):
        return None
    result = None
    for unit, exponent in terms:
        parsed = _parsed(unit)
        if parsed.base_offset:
            raise ValueError(f"{unit!r} has an offset zero, so its values do not multiply: "
                             "convert to an absolute unit (K) first")
        if parsed.is_dimensionless and parsed.base_value == 1.0:
            continue
        power = parsed if exponent == 1 else parsed ** exponent
        result = power if result is None else result * power
    if result is None or (result.is_dimensionless and result.base_value == 1.0):
        return "dimensionless"
    return canonical(str(result))


#: SI derived units a filter may name its result by (``named``), leaving out
#: names that share dimensions with others (Hz and Bq, Gy and Sv).
_NAMED = ("N", "Pa", "J", "W", "C", "V", "Ω", "F", "Wb", "T", "H")


def named(unit):
    """``unit`` shown by the SI derived unit of its dimensions and size, if one is
    (``"Pa*m**3"`` as ``"J"``, ``"kg*m/s**2"`` as ``"N"``), else unchanged. Only
    the dimensions and the size decide, never how the unit was written."""
    if unit is None:
        return None
    parsed = _parsed(unit)
    for name in _NAMED:
        candidate = _parsed(name)
        if parsed.dimensions == candidate.dimensions and \
                math.isclose(parsed.base_value, candidate.base_value, rel_tol=1e-12):
            return name
    return unit


def dimensionless_or_none(what, unit):
    """``"dimensionless"`` for a pure number, ``None`` for unknown, and
    ``ValueError`` naming ``what`` for anything with dimensions or a scale (a
    logarithm of metres, or of kilometres per metre, depends on the unit)."""
    if unit is None:
        return None
    parsed = _parsed(unit)
    if parsed.is_dimensionless and parsed.base_value == 1.0:
        return "dimensionless"
    raise ValueError(f"{what} takes a pure number, not values in {unit!r}: convert to "
                     "'dimensionless' (divide by a reference) first")

