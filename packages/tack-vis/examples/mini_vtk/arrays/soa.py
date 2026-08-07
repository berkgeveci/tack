"""Struct-of-Arrays array types, generated dynamically."""

import linecache

import tack

_soa_cache = {}


def make_soa_type(nc):
    """Generate a @tack.data_oriented SOA class with nc component fields.

    Each component is a separate tack.field (c0, c1, ...).
    The get_value/set_value interface dispatches to the right component
    via a conditional chain that compiles to efficient GPU code.
    """
    if nc in _soa_cache:
        return _soa_cache[nc]

    params = ", ".join(f"c{i}" for i in range(nc))
    init_body = "\n".join(f"        self.c{i} = c{i}" for i in range(nc))

    get_lines = ["        result = self.c0[i]"]
    for i in range(1, nc):
        get_lines.append(f"        if c == {i}:")
        get_lines.append(f"            result = self.c{i}[i]")
    get_lines.append("        return result")
    get_body = "\n".join(get_lines)

    set_lines = []
    for i in range(nc):
        set_lines.append(f"        if c == {i}:")
        set_lines.append(f"            self.c{i}[i] = val")
    set_body = "\n".join(set_lines)

    source = f"""@tack.data_oriented
class SOATupleArray{nc}:
    def __init__(self, {params}):
{init_body}

    @tack.func
    def get_value(self, i, c):
{get_body}

    @tack.func
    def set_value(self, i, c, val):
{set_body}
"""
    filename = f"<soa_tuple_array_{nc}>"
    lines = source.splitlines(True)
    linecache.cache[filename] = (len(source), None, lines, filename)

    code = compile(source, filename, "exec")
    ns = {"tack": tack}
    exec(code, ns)

    cls = ns[f"SOATupleArray{nc}"]
    _soa_cache[nc] = cls
    return cls
