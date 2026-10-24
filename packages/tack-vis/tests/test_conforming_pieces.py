"""Rules that depend on more than a cell agree across the pieces of a partitioned dataset.

``tetrahedralize`` splits a shared face along one diagonal on both sides. On a
structured grid it alternates hexahedra by the parity of (i, j, k), which must
be the whole grid's: a piece reads its place from the topology's ``offset``,
which ``split`` sets and ``vtk_to_dataset`` takes from an extent. Elsewhere it
splits from each cell's corner of smallest rank, which must be the same corner
on every piece: the global id where there are global ids. So the pieces'
owned tetrahedra are the whole's, or, for ids that ``generate_global_ids``
made, still meet face to face.
"""

import collections
import types

import numpy as np
import pytest

import tack
import tack.dataset as td
from tack.dataset import Ghosts
from tack.runtime.dispatch import env_flag

try:
    import vtkmodules.vtkCommonDataModel  # noqa: F401  (registers the pythonic API)
    from vtkmodules.vtkImagingCore import vtkRTAnalyticSource

    from tack.interop.vtk import vtk_to_dataset
except ImportError:
    if env_flag("TACK_REQUIRE_VTK"):
        raise
    vtk = None
else:
    vtk = types.SimpleNamespace()

needs_vtk = pytest.mark.skipif(vtk is None, reason="needs VTK, the reference")

pytestmark = pytest.mark.quick


@pytest.fixture(autouse=True)
def cpu():
    tack.set_default_device("cpu")


def _tetrahedra(data):
    """``data``'s owned tetrahedra, each as its sorted corner positions."""
    out = td.tetrahedralize(data)
    positions = np.round(out.positions(), 5)
    rows = tack.arrays.to_host(out.topology.connectivity).reshape(-1, 4)
    if out.ghosts is not None and out.ghosts.cells is not None:
        rows = rows[(out.ghosts.cells.numpy() & Ghosts.DUPLICATECELL) == 0]
    return [tuple(sorted(tuple(positions[p]) for p in row)) for row in rows]


def _renumbered(piece, seed):
    """``piece`` as an unstructured mesh whose points are numbered in a random order,
    its global ids (if any) following its points."""
    piece = td.extract_cells(piece, np.arange(piece.num_cells))
    order = np.random.default_rng(seed).permutation(piece.num_points)
    new_id = np.empty_like(order)
    new_id[order] = np.arange(len(order))
    topology = piece.topology
    with piece.device:
        out = td.DataSet(td.UnstructuredTopology(
            tack.arrays.to_host(topology.types),
            tack.arrays.to_host(topology.offsets),
            new_id[tack.arrays.to_host(topology.connectivity)],
            num_points=piece.num_points), piece.positions()[order])
    marks = piece.ghosts
    if marks is not None:
        out.ghosts = Ghosts(cells=marks.cells, points=None if marks.points is None
                            else marks.points.numpy()[order])
    if piece.global_ids is not None:
        out.global_ids = td.GlobalIds(cells=piece.global_ids.cells,
                                      points=piece.global_ids.points.numpy()[order])
    return out


def test_structured_pieces_alternate_as_the_whole_grid():
    whole = td.wavelet((0, 9, 0, 8, 0, 7))              # odd cell counts: odd offsets
    pieces = td.split(whole, (3, 2, 2), ghost_layers=1)
    assert any(sum(p.topology.offset) % 2 for p in pieces)
    got = [t for piece in pieces for t in _tetrahedra(piece)]
    assert sorted(got) == sorted(_tetrahedra(whole))


def test_unstructured_pieces_split_by_global_ids():
    whole = td.wavelet(4)
    whole.global_ids = td.GlobalIds(cells=np.arange(whole.num_cells),
                                    points=np.arange(whole.num_points))
    expected = sorted(_tetrahedra(td.extract_cells(whole, np.arange(whole.num_cells))))
    pieces = [_renumbered(p, seed) for seed, p in
              enumerate(td.split(whole, (2, 2, 1), ghost_layers=1))]
    assert sorted(t for p in pieces for t in _tetrahedra(p)) == expected
    for piece in pieces:                     # by local ids, the diagonals disagree
        piece.global_ids = None
    assert sorted(t for p in pieces for t in _tetrahedra(p)) != expected


def test_generated_ids_make_pieces_meet_face_to_face():
    """Ids from generate_global_ids rank corners alike on every piece: every
    triangle inside the union of the owned tetrahedra is shared by two of them."""
    whole = td.wavelet(4)
    plain = td.PartitionedDataSet([_renumbered(td.extract_cells(p, np.arange(p.num_cells)), s)
                                   for s, p in enumerate(td.split(whole, 4))])
    for p in plain:
        p.ghosts = None
    pieces = td.generate_global_ids(td.ghost_cells(plain))
    tets = [t for piece in pieces for t in _tetrahedra(piece)]
    faces = collections.Counter(tuple(sorted(f)) for t in tets
                                for f in ((t[0], t[1], t[2]), (t[0], t[1], t[3]),
                                          (t[0], t[2], t[3]), (t[1], t[2], t[3])))
    boundary = sum(1 for n in faces.values() if n == 1)
    assert max(faces.values()) == 2
    assert boundary == 2 * td.extract_surface(whole).num_cells     # two per boundary quad
    assert len(tets) == 6 * whole.num_cells


def test_the_offset_is_checked_and_kept():
    topology = td.StructuredTopology((3, 4, 5), offset=(1, 2, 3))
    assert topology.offset == (1, 2, 3)
    assert td.StructuredTopology((3, 4)).offset == (0, 0)
    with pytest.raises(ValueError, match="one index per axis"):
        td.StructuredTopology((3, 4), offset=(1, 2, 3))
    data = td.DataSet(topology, td.uniform_grid((3, 4, 5)).geometry.values)
    assert data.to_device("cpu").topology.offset == (1, 2, 3)
    pieces = td.split(td.wavelet((0, 5, 0, 5, 0, 5)), (2, 1, 1))
    assert [p.topology.offset for p in pieces] == [(0, 0, 0), (2, 0, 0)]


@needs_vtk
def test_an_images_extent_is_its_offset():
    image = vtkRTAnalyticSource(whole_extent=(-3, 4, -2, 2, 1, 5))()
    data = vtk_to_dataset(image)
    assert data.topology.offset == (-3, -2, 1)
