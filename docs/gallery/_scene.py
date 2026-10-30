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
from tack.dataset import shapes
from tack.dataset.topology import uniform_cells

#: Where the pictures go: the docs' assets.
IMAGES = Path(__file__).resolve().parents[1] / "assets" / "gallery"

#: The kinds of cells, the gallery's columns, in order.
KINDS = ("polyhedra", "linear", "high-order", "curved")

TITLES = {"polyhedra": "Polyhedra", "linear": "Tetrahedra",
          "high-order": "Order 4 on straight cells", "curved": "Order 3 on curved cells",
          "mixed": "Hexahedra, wedges and pyramids", "hexahedra": "Hexahedra",
          "curved-tetrahedra": "Order 2 on curved tetrahedra",
          "curved-wedges": "Order 2 on curved wedges"}


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


# --8<-- [start:swirl]
@tack.func
def swirl(x):
    """A swirl inside the unit cube: divergence-free, its velocity along every face
    (none through it), so paths wind round inside rather than leave."""
    k = math.pi
    sx, cx = tack.sin(k * x[0]), tack.cos(k * x[0])
    sy, cy = tack.sin(k * x[1]), tack.cos(k * x[1])
    sz, cz = tack.sin(k * x[2]), tack.cos(k * x[2])
    return [sx * cy - 0.5 * sx * cz, -cx * sy + 0.5 * sy * cz, 0.5 * sz * (cx - cy)]


@tack.func
def swirl_base(x):
    """The swirl's part about z (``swirl`` is it plus half of ``swirl_turn``)."""
    k = math.pi
    return [tack.sin(k * x[0]) * tack.cos(k * x[1]), -tack.cos(k * x[0]) * tack.sin(k * x[1]),
            0.0 * x[2]]


@tack.func
def swirl_turn(x):
    """The swirl's other part, divergence-free alone: weighted in time, the swirl
    unsteady."""
    k = math.pi
    sx, cx = tack.sin(k * x[0]), tack.cos(k * x[0])
    sy, cy = tack.sin(k * x[1]), tack.cos(k * x[1])
    sz, cz = tack.sin(k * x[2]), tack.cos(k * x[2])
    return [-sx * cz, sy * cz, sz * (cx - cy)]
# --8<-- [end:swirl]


def with_velocity(data):
    """``data`` with ``v``, the swirl, in the space of its field ``u``."""
    data.fields["v"] = td.interpolate(data, swirl, data.fields["u"].space, dtype=tack.vec3())
    return data


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

def _voronoi(seeds):
    """The Voronoi diagram of ``seeds`` in the unit square, and each seed's region:
    the seeds' mirror images across the square's sides make the regions tile it
    exactly."""
    from scipy.spatial import Voronoi

    mirrored = [seeds]
    for axis in (0, 1):
        for side in (0.0, 1.0):
            image = seeds.copy()
            image[:, axis] = 2 * side - image[:, axis]
            mirrored.append(image)
    vor = Voronoi(np.concatenate(mirrored))
    return vor, [vor.regions[vor.point_region[i]] for i in range(len(seeds))]


def _voronoi_columns(n=150, layers=8, relax=20, seed=3):
    """Voronoi cells of ``n`` points in the unit square, relaxed toward their
    regions' centroids ``relax`` times (Lloyd's method: even cells, no very short
    edges, as a polyhedral mesher makes them), extruded through the unit cube in
    ``layers``: polygonal prisms."""
    seeds = np.random.default_rng(seed).uniform(0, 1, (n, 2))
    for _ in range(relax):
        vor, regions = _voronoi(seeds)
        for i, r in enumerate(regions):
            ring = vor.vertices[r]
            x, y = ring[:, 0], ring[:, 1]
            cross = x * np.roll(y, -1) - np.roll(x, -1) * y
            area = cross.sum() / 2
            seeds[i] = [((x + np.roll(x, -1)) * cross).sum() / (6 * area),
                        ((y + np.roll(y, -1)) * cross).sum() / (6 * area)]
    vor, regions = _voronoi(seeds)
    # The regions' vertices on the square's sides are on them to rounding: snap.
    xy = np.clip(vor.vertices, 0.0, 1.0)
    xy[np.abs(xy) < 1e-12] = 0.0
    xy[np.abs(xy - 1) < 1e-12] = 1.0
    used = sorted({v for r in regions for v in r})
    renumber = {v: i for i, v in enumerate(used)}
    nv = len(used)
    xy = xy[used]
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


def _mixed(n=6):
    """The unit cube's n^3 hexahedra: a third kept, a third each split into two
    wedges, a third into six pyramids about its centre (a point of its own). Every
    face between two cells is the same quad or triangle in both."""
    ijk = np.stack(np.meshgrid(*(np.arange(n + 1),) * 3, indexing="ij"), -1)
    points = list(ijk.transpose(2, 1, 0, 3).reshape(-1, 3) / n)

    def at(i, j, k):
        return i + (n + 1) * (j + (n + 1) * k)

    hexes, wedges, pyramids = [], [], []
    for k in range(n):
        for j in range(n):
            for i in range(n):
                h = [at(i, j, k), at(i + 1, j, k), at(i + 1, j + 1, k), at(i, j + 1, k),
                     at(i, j, k + 1), at(i + 1, j, k + 1), at(i + 1, j + 1, k + 1),
                     at(i, j + 1, k + 1)]
                if i < n // 3:
                    hexes.append(h)
                elif i < 2 * n // 3:
                    wedges += [[h[0], h[1], h[3], h[4], h[5], h[7]],
                               [h[1], h[2], h[3], h[5], h[6], h[7]]]
                else:
                    points.append(np.asarray([i + 0.5, j + 0.5, k + 0.5]) / n)
                    apex = len(points) - 1
                    # Each face's corners reversed from outward: a base toward the apex.
                    for face in ((0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2),
                                 (2, 6, 7, 3), (3, 7, 4, 0)):
                        pyramids.append([h[f] for f in face] + [apex])
    cells = ([(12, c) for c in hexes] + [(13, c) for c in wedges]
             + [(14, c) for c in pyramids])
    types = np.array([t for t, _ in cells], np.uint8)
    sizes = np.array([len(c) for _, c in cells])
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    connectivity = np.concatenate([c for _, c in cells])
    topology = td.UnstructuredTopology(tack.array(types), tack.array(offsets),
                                       tack.array(connectivity), num_points=len(points))
    return td.DataSet(topology, np.asarray(points))


def _wedges(grid):
    """``grid``'s hexahedra, each split into two wedges across a diagonal of its
    bottom and top faces, the same diagonal in every column (so they meet face to
    face)."""
    hexes = td.extract_cells(grid, np.arange(grid.num_cells)).topology.connectivity.numpy()
    conn = []
    for h in hexes.reshape(-1, 8):
        conn += [[h[0], h[1], h[3], h[4], h[5], h[7]], [h[1], h[2], h[3], h[5], h[6], h[7]]]
    types, offsets = uniform_cells(int(shapes.WEDGE), len(conn), tack.i32)
    topology = td.UnstructuredTopology(types, offsets,
                                       tack.array(np.array(conn, np.int32).reshape(-1)),
                                       num_points=grid.num_points)
    return td.DataSet(topology, grid.positions())


def jittered(data, amount=0.25, seed=5):
    """``data`` with its points inside the unit cube moved at random by up to
    ``amount`` of the mesh spacing along each axis (the boundary's stay), so its
    cells differ in shape and quality."""
    x = data.positions().astype(np.float64)
    inside = ((x > 1e-9) & (x < 1 - 1e-9)).all(axis=1)
    edges = data.topology.edges().rows.numpy()
    spacing = np.median(np.linalg.norm(x[edges[:, 1]] - x[edges[:, 0]], axis=1))
    rng = np.random.default_rng(seed)
    x[inside] += rng.uniform(-amount, amount, (inside.sum(), 3)) * spacing
    out = td.DataSet(data.topology, x.astype(np.float32))
    for name, field in data.fields.items():
        out.fields[name] = td.Field(field.space, field.values)
    return out


def twisted(data):
    """``data`` (straight hexahedra) on an order-3 geometry, the cube twisted and
    bowed."""
    data = td.DataSet(data.topology, data.geometry.values)
    data.geometry = td.interpolate(data, _twist, td.H1(data, order=3), dtype=tack.vec3())
    return data


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
    elif kind in ("curved-tetrahedra", "curved-wedges"):
        grid = td.uniform_grid((4, 4, 4), spacing=(1 / 3,) * 3)
        straight = td.tetrahedralize(grid) if kind == "curved-tetrahedra" else _wedges(grid)
        data = td.DataSet(straight.topology, straight.geometry.values)
        data.geometry = td.interpolate(data, _twist, td.H1(data, order=2), dtype=tack.vec3())
        data.fields["u"] = td.interpolate(data, schwarz, td.H1(data, order=2))
    elif kind == "hexahedra":
        grid = td.uniform_grid((7, 7, 7), spacing=(1 / 6,) * 3)
        data = td.extract_cells(grid, np.arange(grid.num_cells))
        data.fields["u"] = td.interpolate(data, schwarz, td.H1(data))
    elif kind == "mixed":
        data = _mixed()
        data.fields["u"] = td.interpolate(data, schwarz, td.H1(data))
    elif kind == "curved":
        grid = td.uniform_grid((4, 4, 4), spacing=(1 / 3,) * 3)
        data = twisted(td.extract_cells(grid, np.arange(grid.num_cells)))
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
    """``data``'s boundary as polygons, a curved or high-order dataset subdivided
    first, so it is linear. Shaded flat: a solid's boundary has sharp edges, which
    point normals would smear."""
    return td.extract_surface(linear(data))


# ── Drawing ─────────────────────────────────────────────────────────

#: ParaView's "Cool to Warm", on [0, 1].
_COOL_TO_WARM = ((0.0, (0.23, 0.299, 0.754)), (0.5, (0.865, 0.865, 0.865)),
                 (1.0, (0.706, 0.016, 0.15)))


def draw(path, parts, outline=None, view=(30.0, 20.0), size=(640, 560), lit=True,
         flat=False):
    """Draw ``parts`` -- ``(dataset, field name, range)``, a field on points or cells,
    or ``(dataset, None, None)`` for plain grey, and optionally a fourth entry, a
    dict of the actor's properties (``color``, ``line_width``, ...) -- and
    ``outline`` (lines, thin and
    dark) to a progressive JPEG at ``path`` (a third of a PNG's size, so the page
    loads quickly), seen from ``view`` (azimuth and elevation in
    degrees, from the -y side of the unit cube). Lit less (``lit=False``), mostly
    ambient, the parts show their colors, as a slice is read by them, with
    enough shade to tell planes apart. ``flat`` looks down the z axis instead, at a
    2D mesh."""
    import vtkmodules.vtkCommonDataModel  # registers the pythonic constructors
    import vtkmodules.vtkRenderingOpenGL2  # noqa: F401  # registers the rendering backend
    from vtkmodules.vtkIOImage import vtkJPEGWriter
    from vtkmodules.vtkRenderingCore import (
        VTK_GET_ARRAY_BY_NAME,
        VTK_SCALAR_MODE_USE_CELL_FIELD_DATA,
        VTK_SCALAR_MODE_USE_POINT_FIELD_DATA,
        vtkActor,
        vtkColorTransferFunction,
        vtkDataSetMapper,
        vtkMapper,
        vtkRenderer,
        vtkRenderWindow,
        vtkWindowToImageFilter,
    )

    from tack.interop.vtk import dataset_to_vtk

    # Lines on the faces they lie on (sub-cell edges, feature edges) drawn over
    # them, not fighting them.
    vtkMapper.SetResolveCoincidentTopologyToPolygonOffset()
    renderer = vtkRenderer(background=(1.0, 1.0, 1.0))
    for data, name, (low, high), properties in (
            (p[0], p[1], p[2] or (0.0, 1.0), p[3] if len(p) > 3 else {}) for p in parts):
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
        actor.property.ambient = 0.2 if lit else 0.6
        actor.property.diffuse = 0.8 if lit else 0.4
        actor.property.specular = 0.2 if lit else 0.0
        for key, value in properties.items():
            setattr(actor.property, key, value)
        renderer.AddActor(actor)
    if outline is not None:
        lines = vtkActor(mapper=vtkDataSetMapper(input_data=dataset_to_vtk(outline),
                                                 scalar_visibility=False))
        lines.property.color = (0.25, 0.25, 0.3)
        lines.property.line_width = 1.0
        # A mesh of many small cells (the tetrahedra) fainter, or it hides the result.
        lines.property.opacity = 0.5 if outline.num_cells < 3000 else 0.12
        renderer.AddActor(lines)
    window = vtkRenderWindow(off_screen_rendering=True, size=size, multi_samples=8)
    window.AddRenderer(renderer)
    camera = renderer.active_camera
    if flat:
        camera.position = (0.0, 0.0, 1.0)
        camera.focal_point = (0.0, 0.0, 0.0)
        camera.view_up = (0.0, 1.0, 0.0)
    else:
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
    # An image writer called on its input writes raw pixels (VTK 9.7.1): connect,
    # Write().
    jpeg = vtkJPEGWriter(file_name=str(path), quality=85, progressive=True)
    vtkWindowToImageFilter(input=window) >> jpeg
    jpeg.Write()
    print(f"wrote {path}")
