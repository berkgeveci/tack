"""Generate tack/data/_clip_tables.py from Viskores' clip tables.

Usage:
  python packages/tack-vis/tools/make_clip_tables.py \
      <viskores>/viskores/filter/contour/worklet/clip/ClipTables.h \
      packages/tack-vis/src/tack/data/_clip_tables.py

Parses the edge table and both case tables (kept and inverted) with their
per-case offsets, checks that every case of every shape decodes within the
table, and writes them as Python lists with Viskores' license notice.
"""
import re
import sys

src = open(sys.argv[1]).read()
names = {f"P{i}": i for i in range(8)}
names.update({f"E{i:02d}": 8 + i for i in range(12)})
names.update({"N0": 20, "X": 255, "ST_VTX": 1, "ST_LIN": 3, "ST_TRI": 5, "ST_QUA": 9, "ST_TET": 10,
              "ST_HEX": 12, "ST_PYR": 14, "ST_WDG": 13, "ST_PNT": 0})
def block(start):
    i = src.index("{", start) + 1
    j = src.index("};", i)
    body = re.sub(r"//[^\n]*|/\*.*?\*/", "", src[i:j], flags=re.S)
    return [names[t] if t in names else int(t) for t in (x.strip() for x in body.split(",")) if t]
edges = block(src.index("CellEdges[CELL_EDGES_SIZE]"))
data_starts = [m.start() for m in re.finditer(r"ClipTablesData\[\] = \{", src)]
index_starts = [m.start() for m in re.finditer(r"ClipTablesIndices\[\] = \{", src)]
assert len(data_starts) == 2 and len(index_starts) == 2
tables = {inv: (block(data_starts[k]), block(index_starts[k])) for k, inv in enumerate((False, True))}
lookup = {1: 0, 3: 2, 5: 6, 9: 14, 10: 30, 12: 46, 13: 302, 14: 366}
npts = {1: 1, 3: 2, 5: 3, 9: 4, 10: 4, 12: 8, 13: 6, 14: 5}
out = ['"""Clip case tables, generated from Viskores\' ClipTables.h',
       '(viskores/filter/contour/worklet/clip, Viskores 1.1.9999, ab6b17965) by',
       'packages/tack-vis/tools/make_clip_tables.py; do not edit. Copyright (c) Kitware, Inc.; covered by the Viskores license',
       '(BSD-3-Clause; https://github.com/Viskores/viskores/blob/main/LICENSE.txt).',
       'The tables descend from VisIt\'s.',
       '',
       'Per shape (VTK cell type id): ``EDGES[shape]``, the corners of each local edge',
       '(E00, E01, ...), flattened; ``CASES[invert][shape]``, for each case (bit k set',
       'when corner k is at or above the clip value) the offset of its record in',
       '``DATA[invert]``. A record is a count of shapes, then for each the shape',
       '(``POINT`` for the centroid N0, else a cell type), its number of points and',
       'those points: corners 0-7, edge points 8-19 (E00-E11), the centroid 20.',
       '"""', '',
       'POINT = 0', 'CENTROID = 20', 'FIRST_EDGE = 8', '']
out.append("EDGES = {")
for shape in sorted(lookup):
    row = edges[shape * 24:(shape + 1) * 24]
    pairs = [v for v in row if v != 255]
    out.append(f"    {shape}: {pairs},")
out.append("}")
for inv in (False, True):
    data, index = tables[inv]
    # Check every record decodes and stays in bounds.
    for shape, start in lookup.items():
        for case in range(1 << npts[shape]):
            at = index[start + case]
            n = data[at]; at += 1
            for _ in range(n):
                kind, count = data[at], data[at + 1]
                at += 2 + count
                assert at <= len(data), (inv, shape, case)
    out.append(f"\nDATA_{'INVERTED' if inv else 'KEPT'} = {data}")
    out.append(f"\nCASES_{'INVERTED' if inv else 'KEPT'} = {{")
    for shape, start in sorted(lookup.items()):
        out.append(f"    {shape}: {index[start:start + (1 << npts[shape])]},")
    out.append("}")
out.append("\nDATA = {False: DATA_KEPT, True: DATA_INVERTED}")
out.append("CASES = {False: CASES_KEPT, True: CASES_INVERTED}")
open(sys.argv[2], "w").write("\n".join(out) + "\n")
print("cases per shape:", {s: 1 << npts[s] for s in lookup}, "data sizes:", len(tables[False][0]), len(tables[True][0]))
