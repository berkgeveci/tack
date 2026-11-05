# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""``test_empty_ranks``' check under MPI: ``mpiexec -n 3 python _empty_ranks_mpi.py
LAYOUT [CASE ...]``. Every algorithm over pieces runs with ranks that hold none:
four pieces all on rank 0 (``one``), on ranks 1 and 2 (``first-empty``), on ranks 0
and 1 (``last-empty``), or nowhere (``none``). Each case's answer, combined over the
ranks, must be what the same four pieces give in one process; with no pieces
anywhere it must be empty, or the same error on every rank. A case that hangs is
stopped by a deadline (``faulthandler``), which names where. Rank 0 prints ``ok``
or the failures."""

import faulthandler
import os
import sys
import traceback

import numpy as np
from mpi4py import MPI

import tack
import tack.dataset as td
from tack.communication import World

LAYOUTS = {"one": [[0, 1, 2, 3], [], []], "first-empty": [[], [0, 1], [2, 3]],
           "last-empty": [[0, 1], [2, 3], []], "none": [[], [], []]}


# ── The pieces ────────────────────────────────────────────────────────

def _wavelet_pieces():
    return list(td.split(td.wavelet((0, 8, 0, 8, 0, 8)), (2, 2, 1)))


def structured():
    return _wavelet_pieces()


def tetrahedra():
    return [td.tetrahedralize(piece) for piece in _wavelet_pieces()]


def polyhedra():
    return [td.as_polyhedra(piece) for piece in _wavelet_pieces()]


def flowing():
    """The tetrahedra with a velocity ``v``."""
    return _with_velocity(tetrahedra(), 0.0)


def cloud():
    rng = np.random.default_rng(7)
    points = rng.random((2000, 3)) * 8.0
    out = []
    for k in range(4):
        mine = points[(points[:, 0] >= 2 * k) & (points[:, 0] < 2 * k + 2)]
        piece = td.point_cloud(mine)
        piece.fields["h"] = td.Field(td.H1(piece), tack.array(np.full(len(mine), 0.6,
                                                                      np.float32)))
        piece.fields["mass"] = td.Field(td.H1(piece), tack.array(np.ones(len(mine),
                                                                         np.float32)))
        out.append(piece)
    return out


def pds(pieces, world, layout):
    mine = LAYOUTS[layout][world.rank]
    return td.PartitionedDataSet([pieces[k] for k in mine], world=world)


# ── The cases: (pieces, run(data) -> this rank's part, combine(parts) -> answer) ──

def _cells(data):
    return sum(p.num_cells for p in data)


def _owned_cells(data):
    return sum(int((p.ghosts.cells.numpy() == 0).sum()) if p.ghosts is not None
               else p.num_cells for p in data)


def _total(parts):
    return sum(parts)


def _union(parts):
    return sorted(set().union(*parts))


def _first(parts):
    found = [p for p in parts if p is not None]
    return found[0] if found else None


def _same_everywhere(parts):
    assert all(repr(p) == repr(parts[0]) for p in parts), parts
    return parts[0]


SEEDS = np.array([[0.5, 0.5, 0.5], [6.5, 6.5, 4.0], [3.0, 7.0, 7.5]])
WHERE = np.array([[0.25, 0.25, 0.25], [4.0, 4.0, 4.0], [7.9, 7.9, 7.9], [20.0, 0.0, 0.0]])
GRID = td.uniform_grid((5, 5, 5), spacing=(2.0, 2.0, 2.0))


def _ids(data):
    return set().union(*[set(p.global_ids.points.numpy().tolist()) for p in data]) \
        if len(data) else set()


CASES = {
    "ghost_cells/tetrahedra": (tetrahedra, lambda d: _owned_cells(td.ghost_cells(d)), _total),
    "ghost_cells/structured": (structured, lambda d: _owned_cells(td.ghost_cells(d)), _total),
    "ghost_cells/polyhedra": (polyhedra, lambda d: _owned_cells(td.ghost_cells(d)), _total),
    "update_ghosts": (tetrahedra, lambda d: _owned_cells(td.update_ghosts(td.ghost_cells(d),
                                                                          "RTData")
                                                         or td.ghost_cells(d)), _total),
    "generate_global_ids/tetrahedra": (tetrahedra, lambda d: _ids(td.generate_global_ids(d)),
                                       _union),
    "generate_global_ids/structured": (structured, lambda d: _ids(td.generate_global_ids(d)),
                                       _union),
    "merge_pieces/tetrahedra": (tetrahedra, lambda d: (lambda m: None if m is None
                                                       else m.num_points)(td.merge_pieces(d)),
                                _first),
    "merge_pieces/structured": (structured, lambda d: (lambda m: None if m is None
                                                       else m.num_points)(td.merge_pieces(d)),
                                _first),
    "extract_surface": (tetrahedra, lambda d: _cells(td.extract_surface(d)), _total),
    "statistics": (tetrahedra, lambda d: round(td.statistics(td.generate_global_ids(d),
                                                             "RTData").mean, 6),
                   _same_everywhere),
    "entropy": (tetrahedra, lambda d: round(float(td.entropy(td.generate_global_ids(d),
                                                             "RTData")), 6),
                _same_everywhere),
    "integrate": (tetrahedra, lambda d: round(td.integrate(td.generate_global_ids(d)).measure,
                                              9), _same_everywhere),
    "connected_components": (tetrahedra,
                             lambda d: _union_regions(td.connected_components(d)), _union),
    "connected_components/radius": (cloud, lambda d: _union_regions(
        td.connected_components(d, radius=0.3)), _union),
    "probe": (tetrahedra, lambda d: _valid(td.probe(d, WHERE)), _same_everywhere),
    "streamlines/seeds on every rank": (    # each rank releases its share of them
        flowing, lambda d: _paths(td.streamlines(d, "v", SEEDS[d.world.rank::d.world.size],
                                                 0.1, 40)), _total),
    "advect/seeds on rank 0": (
        flowing, lambda d: _paths(td.advect(d, "v", SEEDS if _rank() == 0
                                            else np.zeros((0, 3)), 0.1, 40)), _total),
    "ghost_points": (cloud, lambda d: _cells(td.ghost_points(d, radius=0.5)), _total),
    "voxel_downsample": (cloud, lambda d: _cells(td.voxel_downsample(d, 1.0)), _total),
    "outliers": (cloud, lambda d: sum(int(p.fields["outlier"].values.numpy().sum())
                                      for p in td.outliers(d, radius=0.3, min_neighbors=2)),
                 _total),
    "deposit": (cloud, lambda d: round(float(td.deposit(d, GRID).values.numpy().sum()), 3),
                _same_everywhere),
    "voxelize": (tetrahedra, lambda d: round(float(td.voxelize(d, GRID, samples=1)
                                                   .fields["fraction"].values.numpy().sum()),
                                             6), _same_everywhere),
}


def _rank():
    return MPI.COMM_WORLD.rank


# ── Hierarchies, octrees, steps and VTK, spread by the same layouts ──────────

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class Spread:
    """What a case spreads itself: ``spread(world, layout)`` gives this rank's share,
    and ``alone()`` the whole in one process."""

    def __init__(self, spread, alone):
        self.spread, self.alone = spread, alone


def _buckets(n, world, layout):
    """Which of ``n`` items this rank holds: item i in bucket i % 4, the buckets
    spread by ``layout``."""
    mine = set(LAYOUTS[layout][world.rank])
    return [i for i in range(n) if i % 4 in mine]


def _amr(world, layout):
    import test_amr

    every = [list(level) for level in test_amr.hierarchy().levels]
    items = [(lv, k) for lv, boxes in enumerate(every) for k in range(len(boxes))]
    chosen = {items[i] for i in _buckets(len(items), world, layout)}
    mine = [[test_amr._unmarked(box) for k, box in enumerate(boxes) if (lv, k) in chosen]
            for lv, boxes in enumerate(every)]
    return td.AMRDataSet(mine, [2, 2], world=world)


def _amr_alone():
    import test_amr

    return test_amr.hierarchy()


def _octree(world, layout):
    import test_octrees

    roots, level, centre, rho, temperature = test_octrees.ramses_like(3, trees=(3, 2, 2))
    whole = td.OctreeDataSet.from_cells(roots, level, centre, fields={"T": temperature})
    trees = _buckets(roots.num_cells, world, layout)
    mine = np.isin(whole._tree, trees)
    return td.OctreeDataSet(roots, whole._tree[mine], whole._level[mine], whole._key[mine],
                            fields={"T": whole.fields["T"].numpy()[mine]}, world=world)


def _octree_alone():
    import test_octrees

    roots, level, centre, rho, temperature = test_octrees.ramses_like(3, trees=(3, 2, 2))
    return td.OctreeDataSet.from_cells(roots, level, centre, fields={"T": temperature})


def _with_velocity(pieces, t):
    out = []
    for piece in pieces:
        piece.time = t
        v = np.zeros((piece.num_points, 3), np.float32)
        v[:, 0], v[:, 1] = 1.0, 0.3 * t
        piece.fields["v"] = td.Field(td.H1(piece), tack.array(v, tack.vec3()))
        out.append(piece)
    return out


def _advected(data_at):
    """Particles from SEEDS (rank 0's) carried over three steps: how many, and where
    (summed)."""
    advection = td.Advection("v", seeds=SEEDS if _rank() == 0 else np.zeros((0, 3)),
                             step_size=0.1)
    for t in (0.0, 0.5, 1.0):
        cloud = advection.update(data_at(t))
    clouds = list(cloud) if isinstance(cloud, td.PartitionedDataSet) else [cloud]
    return sum(c.num_points for c in clouds)


def _vtk_pieces(world, layout):
    from vtkmodules.vtkCommonDataModel import vtkPartitionedDataSet

    from tack.interop.vtk import dataset_to_vtk

    out = vtkPartitionedDataSet()
    for k in LAYOUTS[layout][world.rank]:
        out.append(dataset_to_vtk(tetrahedra()[k]))
    return out


def _vtk_alone():
    from vtkmodules.vtkCommonDataModel import vtkPartitionedDataSet

    from tack.interop.vtk import dataset_to_vtk

    out = vtkPartitionedDataSet()
    for piece in tetrahedra():
        out.append(dataset_to_vtk(piece))
    return out


def _vtk(algorithm_of, read):
    """A VTK algorithm (made by ``algorithm_of(world)``) over the VTK pieces."""
    def run(vtk_data, world):
        algorithm = algorithm_of(world)
        out = algorithm(vtk_data)
        if algorithm.error is not None:
            raise algorithm.error
        return read(out)
    return run


def _vtk_probe(vtk_data, world):
    """VTK's Probe of WHERE (the same points on every rank) over the pieces, as its
    source: how many points a piece holds."""
    import vtkmodules.vtkCommonDataModel as dm

    from tack.interop.vtk.algorithms import Probe

    where = dm.vtkPolyData()
    where.points = __import__("vtkmodules.vtkCommonCore", fromlist=["x"]).vtkPoints(
        data=WHERE.copy())
    probe = Probe(world=world)
    probe.SetInputDataObject(0, where)
    probe.SetSourceData(vtk_data)
    probe.Update()
    if probe.error is not None:
        raise probe.error
    return int(np.asarray(probe.GetOutputDataObject(0).point_data["valid"]).view(np.int8).sum())


def _leaf_cells(out):
    if hasattr(out, "GetNumberOfPartitions"):
        return sum(out.GetPartition(i).GetNumberOfCells()
                   for i in range(out.GetNumberOfPartitions()))
    return out.GetNumberOfCells()


SPREAD_CASES = {
    "amr/statistics": (Spread(_amr, _amr_alone),
                       lambda a, w: td.statistics(a, "c").n, _same_everywhere),
    "amr/probe": (Spread(_amr, _amr_alone),
                  lambda a, w: round(float(td.probe(a, WHERE * 0.9).fields["c"].values
                                           .numpy().sum()), 4), _same_everywhere),
    "amr/ghost_cells": (Spread(_amr, _amr_alone),
                        lambda a, w: sum(b.num_cells for _, b in td.ghost_cells(a)), _total),
    "amr/average_down": (Spread(_amr, _amr_alone),
                         lambda a, w: (td.average_down(a, "c"), td.statistics(a, "c").n)[1],
                         _same_everywhere),
    "octree/statistics": (Spread(_octree, _octree_alone),
                          lambda o, w: round(td.statistics(o, "T").mean, 6), _same_everywhere),
    "octree/probe": (Spread(_octree, _octree_alone),
                     lambda o, w: int(td.probe(o, WHERE * 0.1).fields["valid"].values.numpy()
                                      .sum()), _same_everywhere),
    "advection": (Spread(lambda w, layout: (w, layout), lambda: None),
                  lambda wl, w: _advected(
                      (lambda t: td.PartitionedDataSet(
                          _with_velocity([tetrahedra()[k] for k in LAYOUTS[wl[1]][w.rank]], t),
                          world=w)) if wl is not None else
                      (lambda t: td.PartitionedDataSet(_with_velocity(tetrahedra(), t),
                                                       world=World()))), _total),
    "vtk/GenerateGlobalIds": (Spread(_vtk_pieces, _vtk_alone), _vtk(
        lambda w: __import__("tack.interop.vtk.algorithms", fromlist=["x"])
        .GenerateGlobalIds(world=w), _leaf_cells), _total),
    "vtk/Statistics": (Spread(_vtk_pieces, _vtk_alone), _vtk(
        lambda w: __import__("tack.interop.vtk.algorithms", fromlist=["x"])
        .Statistics(fields=["RTData"], world=w),
        lambda t: t.GetColumnByName("n").GetValue(0)), _same_everywhere),
    "vtk/MergePieces": (Spread(_vtk_pieces, _vtk_alone), _vtk(
        lambda w: __import__("tack.interop.vtk.algorithms", fromlist=["x"])
        .MergePieces(world=w), lambda m: m.GetNumberOfPoints() or None), _first),
    "vtk/Integrate": (Spread(_vtk_pieces, _vtk_alone), _vtk(
        lambda w: __import__("tack.interop.vtk.algorithms", fromlist=["x"])
        .Integrate(fields=["RTData"], world=w),
        lambda t: round(float(np.asarray(t.cell_data["Volume"])[0]), 9)), _same_everywhere),
    "vtk/GhostCells": (Spread(_vtk_pieces, _vtk_alone), _vtk(
        lambda w: __import__("tack.interop.vtk.algorithms", fromlist=["x"])
        .GhostCells(world=w), _leaf_cells), _total),
    "vtk/Probe": (Spread(_vtk_pieces, _vtk_alone), _vtk_probe, _same_everywhere),
    "vtk/ExtractSurface": (Spread(_vtk_pieces, _vtk_alone), _vtk(
        lambda w: __import__("tack.interop.vtk.algorithms", fromlist=["x"])
        .ExtractSurface(world=w), _leaf_cells), _total),
}


def _union_regions(data):
    return {int(r) for p in data for r in np.unique(p.fields["region"].values.numpy())
            if r >= 0}


def _valid(result):
    if isinstance(result, td.PartitionedDataSet):
        return sum(int(p.fields["valid"].values.numpy().sum()) for p in result)
    return int(result.fields["valid"].values.numpy().sum())


def _paths(result):
    return sum(p.num_cells for p in result) if isinstance(result, td.PartitionedDataSet) \
        else result.num_cells


# ── Running ───────────────────────────────────────────────────────────

def main():
    comm = MPI.COMM_WORLD
    layout, names = sys.argv[1], sys.argv[2:] or [*CASES, *SPREAD_CASES]
    try:
        import vtkmodules.vtkCommonDataModel  # noqa: F401
    except ImportError:                        # the VTK cases need VTK (interop extra)
        names = [n for n in names if not n.startswith("vtk/")]
    tack.set_default_device("cpu")
    world = World(comm)
    failures = []
    for name in names:
        spread = name in SPREAD_CASES
        make, run, combine = SPREAD_CASES[name] if spread else CASES[name]
        traces = os.environ.get("TACK_EMPTY_RANKS_TRACES")
        out = open(os.path.join(traces, f"rank{comm.rank}.txt"), "w") if traces else sys.stderr
        faulthandler.dump_traceback_later(
            int(os.environ.get("TACK_EMPTY_RANKS_DEADLINE", "120")) + 5 * comm.rank,
            exit=True, file=out)
        try:
            if spread:
                part = run(make.spread(world, layout), world)
            else:
                part = run(pds(make(), world, layout))
            error = None
        except Exception as e:
            part, error = None, f"{type(e).__name__}: {e}"
            if os.environ.get("TACK_EMPTY_RANKS_TRACE"):
                traceback.print_exc()
        faulthandler.cancel_dump_traceback_later()
        parts, errors = comm.allgather(part), comm.allgather(error)
        # The same error on every rank with no pieces anywhere is a refusal, and the
        # ranks go on together; any other error ends the run on every rank alike,
        # since they may disagree from there.
        refused = layout == "none" and any(errors) and len(set(errors)) == 1
        if any(errors) and not refused:
            if comm.rank == 0:
                failures.append(f"{name}: {errors}")
            break
        if comm.rank == 0 and not refused:
            got = combine(parts)
            if layout == "none":
                continue
            alone = World()
            expected = combine([run(make.alone(), alone) if spread else
                                run(td.PartitionedDataSet(make(), world=alone))])
            if repr(got) != repr(expected):
                failures.append(f"{name}: {got!r} against {expected!r} in one process")
    if comm.rank == 0:
        print("ok" if not failures else "FAILED\n" + "\n".join(failures), flush=True)


if __name__ == "__main__":
    main()
