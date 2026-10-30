# Copyright (c) 2026 Kitware, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""What the gallery's scripts share: the datasets of each kind of cell, the field
they cut, a mesh's outline, and a picture drawn through VTK.

Tack makes everything drawn: the result's polygons, their normals, and the
outline (each cell edge as lines, a curved one subdivided along the geometry).
VTK only draws linear polygons and lines (``dataset_to_vtk``), never
higher-order cells.
"""

import argparse
import math
from pathlib import Path

import numpy as np

import tack
import tack.dataset as td

#: Where the pictures go: the docs' assets.
IMAGES = Path(__file__).resolve().parents[1] / "assets" / "gallery"

#: The kinds of cells, the gallery's columns, in order.
KINDS = ("polyhedra", "linear", "high-order", "curved")

TITLES = {"polyhedra": "Polyhedra", "linear": "Tetrahedra",
          "high-order": "Order 4 on straight cells", "curved": "Order 3 on curved cells"}


def arguments(description, kinds=KINDS):
    """The command line every script takes: ``--arch`` (the device) and the kinds
    to draw (all by default); the device is made the default."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--arch", default="cpu", help="the device (cpu, metal, cuda, ...)")
    parser.add_argument("kinds", nargs="*", choices=[*kinds, []], default=[])
    parser.add_argument("--output", type=Path, default=IMAGES)
    args = parser.parse_args()
    args.kinds = args.kinds or list(kinds)
    tack.set_default_device(args.arch)
    return args


# ── The field ───────────────────────────────────────────────────────

@tack.func
def schwarz(x):
    """Schwarz's P surface at zero over the unit cube: three cosines."""
    k = 2.0 * math.pi
    return tack.cos(k * x[0]) + tack.cos(k * x[1]) + tack.cos(k * x[2])


@tack.func
def _twist(x):
    # The unit cube bent: each z level turned about the cube's axis, and bowed.
    a = 0.6 * (x[2] - 0.5)
    c = tack.cos(a)
    s = tack.sin(a)
    u = x[0] - 0.5
    v = x[1] - 0.5
    return [0.5 + c * u - s * v, 0.5 + s * u + c * v,
            x[2] + 0.12 * tack.sin(math.pi * x[0]) * tack.sin(math.pi * x[1])]


# ── The datasets ────────────────────────────────────────────────────

def _voronoi_columns(n=60, layers=6, seed=3):
    """Voronoi cells of random points over the unit square, the bounded ones,
    extruded through the unit cube in ``layers``: polygonal prisms, faces of up to
    a dozen points."""
    from scipy.spatial import Voronoi

    rng = np.random.default_rng(seed)
    vor = Voronoi(rng.uniform(0, 1, (n, 2)))
    regions = [r for r in (vor.regions[i] for i in vor.point_region)
               if r and -1 not in r and all((0 <= vor.vertices[v]).all()
                                            and (vor.vertices[v] <= 1).all() for v in r)]
    used = sorted({v for r in regions for v in r})
    renumber = {v: i for i, v in enumerate(used)}
    nv = len(used)
    xy = vor.vertices[used]
    points = np.array([[x, y, k / layers] for k in range(layers + 1) for x, y in xy])
    cells = []
    for r in regions:
        ring = [renumber[v] for v in r]
        area = sum(xy[a][0] * xy[b][1] - xy[b][0] * xy[a][1]
                   for a, b in zip(ring, ring[1:] + ring[:1], strict=True))
        if area < 0:
            ring = ring[::-1]                         # counter-clockwise from above
        for k in range(layers):
            lo, hi = k * nv, (k + 1) * nv
            faces = [[lo + v for v in ring[::-1]], [hi + v for v in ring]]
            faces += [[lo + a, lo + b, hi + b, hi + a]
                      for a, b in zip(ring, ring[1:] + ring[:1], strict=True)]
            cells.append(faces)
    return td.DataSet(td.PolyhedralTopology.from_cell_faces(cells, len(points)), points)


# --8<-- [start:datasets]
def dataset(kind):
    """The dataset of ``kind`` (``KINDS``), with the field ``u`` (``schwarz``)."""
    if kind == "polyhedra":
        data = _voronoi_columns()
        data.fields["u"] = td.interpolate(data, schwarz, td.H1(data))
    elif kind == "linear":
        grid = td.uniform_grid((9, 9, 9), spacing=(1 / 8,) * 3)
        data = td.tetrahedralize(grid)
        data.fields["u"] = td.interpolate(data, schwarz, td.H1(data))
    elif kind == "high-order":
        # Few cells, a field of order 4: its surfaces curve inside the cells.
        grid = td.uniform_grid((4, 4, 4), spacing=(1 / 3,) * 3)
        data = td.extract_cells(grid, np.arange(grid.num_cells))
        data.fields["u"] = td.interpolate(data, schwarz, td.H1(data, order=4))
    elif kind == "curved":
        grid = td.uniform_grid((4, 4, 4), spacing=(1 / 3,) * 3)
        data = td.extract_cells(grid, np.arange(grid.num_cells))
        data.geometry = td.interpolate(data, _twist, td.H1(data, order=3), dtype=tack.vec3())
        data.fields["u"] = td.interpolate(data, schwarz, td.H1(data, order=3))
    else:
        raise ValueError(f"no dataset of kind {kind!r}")
    return data
# --8<-- [end:datasets]


def outline(data):
    """``data``'s cell edges as lines, a curved geometry's subdivided along it."""
    return td.extract_edges(data, subdivision=12)


def level(data):
    """The sub-cells' steps a cut of ``data`` takes: 6 a side for a curved geometry or
    a high-order field, ``None`` (no subdivision) for linear cells."""
    curved = data.geometry.space.order > 1
    high = any(getattr(f.space, "order", 1) > 1 for f in data.fields.values())
    return 6 if curved or high else None


def linear(data):
    """``data`` as linear cells: a curved or high-order dataset subdivided, any other
    as it is."""
    steps = level(data)
    return td.subdivide(data, steps) if steps else data


def normals(surface):
    """``surface`` with point normals for shading, weighted by area so the cut's
    slivers do not swing them. Its polygons are wound alike already (a contour's
    toward lower values, a boundary's outward)."""
    return td.surface_normals(surface, consistent=False, weights="area")


def boundary(data):
    """``data``'s boundary as polygons with normals; a curved or high-order dataset
    subdivided first, so it is linear."""
    return normals(td.extract_surface(linear(data)))


# ── Drawing ─────────────────────────────────────────────────────────

#: ParaView's "Cool to Warm", on [0, 1].
_COOL_TO_WARM = ((0.0, (0.23, 0.299, 0.754)), (0.5, (0.865, 0.865, 0.865)),
                 (1.0, (0.706, 0.016, 0.15)))


def draw(path, parts, outline=None, view=(30.0, 20.0), size=(640, 560)):
    """Draw ``parts`` -- ``(dataset, field name, range)``, a field on points or cells,
    or ``(dataset, None, None)`` for plain grey -- and ``outline`` (lines, thin and
    dark) to a PNG at ``path``, seen from ``view`` (azimuth and elevation in
    degrees, from the -y side of the unit cube)."""
    import vtkmodules.vtkCommonDataModel  # registers the pythonic constructors
    import vtkmodules.vtkRenderingOpenGL2  # noqa: F401  # registers the rendering backend
    from vtkmodules.vtkIOImage import vtkPNGWriter
    from vtkmodules.vtkRenderingCore import (
        VTK_GET_ARRAY_BY_NAME,
        VTK_SCALAR_MODE_USE_CELL_FIELD_DATA,
        VTK_SCALAR_MODE_USE_POINT_FIELD_DATA,
        vtkActor,
        vtkColorTransferFunction,
        vtkDataSetMapper,
        vtkRenderer,
        vtkRenderWindow,
        vtkWindowToImageFilter,
    )

    from tack.interop.vtk import dataset_to_vtk

    renderer = vtkRenderer(background=(1.0, 1.0, 1.0))
    for data, name, (low, high) in ((d, n, r or (0.0, 1.0)) for d, n, r in parts):
        grid = dataset_to_vtk(data)
        if "normals" in grid.point_data.keys():
            grid.point_data.SetNormals(grid.point_data["normals"])
        mapper = vtkDataSetMapper(input_data=grid, scalar_visibility=name is not None)
        if name is not None:
            colors = vtkColorTransferFunction()
            for fraction, rgb in _COOL_TO_WARM:
                colors.AddRGBPoint(low + fraction * (high - low), *rgb)
            on_points = name in grid.point_data.keys()
            mapper.SetLookupTable(colors)
            mapper.SetScalarMode(VTK_SCALAR_MODE_USE_POINT_FIELD_DATA if on_points
                                 else VTK_SCALAR_MODE_USE_CELL_FIELD_DATA)
            mapper.SetArrayAccessMode(VTK_GET_ARRAY_BY_NAME)
            mapper.SetArrayName(name)
            mapper.SetScalarRange(low, high)
            mapper.SetInterpolateScalarsBeforeMapping(True)
        actor = vtkActor(mapper=mapper)
        if name is None:
            actor.property.color = (0.8, 0.8, 0.8)
        actor.property.ambient = 0.2
        actor.property.diffuse = 0.8
        actor.property.specular = 0.2
        renderer.AddActor(actor)
    if outline is not None:
        lines = vtkActor(mapper=vtkDataSetMapper(input_data=dataset_to_vtk(outline),
                                                 scalar_visibility=False))
        lines.property.color = (0.25, 0.25, 0.3)
        lines.property.line_width = 1.0
        lines.property.opacity = 0.5
        renderer.AddActor(lines)
    window = vtkRenderWindow(off_screen_rendering=True, size=size, multi_samples=8)
    window.AddRenderer(renderer)
    camera = renderer.active_camera
    camera.position = (0.5, -2.5, 0.5)
    camera.focal_point = (0.5, 0.5, 0.5)
    camera.view_up = (0.0, 0.0, 1.0)
    camera.Azimuth(view[0])
    camera.Elevation(view[1])
    renderer.ResetCamera()
    camera.Zoom(1.15)
    window.Render()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # A PNG writer called on its input writes raw pixels (VTK 9.7.1): connect, Write().
    png = vtkPNGWriter(file_name=str(path))
    vtkWindowToImageFilter(input=window) >> png
    png.Write()
    print(f"wrote {path}")
