"""Particles that cross the pieces of a partitioned dataset.

``advect`` and ``streamlines`` over a ``PartitionedDataSet`` route each
particle to the piece owning the cell that holds it, advance it there, and
hand it on when a whole step ends in a ghost cell. With a ghost layer wider
than a step, the paths are the whole mesh's up to rounding: here a helix
through a 2x2x2 split of a grid, against the same advection of the whole
grid. Without ghost cells they cross by the short step to the boundary and
differ by about a step. The same runs across MPI ranks, each rank with its
own seeds, give what one process gives (``_particles_mpi.py``).
"""

import os

import numpy as np
import pytest

import tack
import tack.dataset as td
from tack.dataset import Ghosts, ParticleStatus
from tack.runtime import dispatch
from tack.runtime.cpu import CPUBackend

# The quick tier takes the fast ones; across MPI ranks and over generated ghost
# cells (20 s and 5 s) run in the whole suite and CI's mpi job.

SEEDS = np.array([[1.5, 1.0, 0.2], [0.4, 1.2, 0.5], [1.9, 1.9, 1.9], [1.0, 0.3, 0.1],
                  [5.0, 5.0, 5.0]], np.float32)       # the last outside the grid


class _SecondCPU(CPUBackend):
    """A CPU backend posing as another device's: its arrays are not the default's."""

    display_name = "second device"

    @property
    def memory_domain(self):
        return ("cpu", 1)


@pytest.fixture(autouse=True)
def cpu():
    tack.set_default_device("cpu")
    yield
    tack.set_default_device("cpu")


def helix():
    """A grid with a velocity turning about (1, 1) and rising: steps of 0.02 move
    a particle at most 0.04, under the 0.1 cell of a ghost layer."""
    grid = td.uniform_grid((21, 21, 21), spacing=(0.1, 0.1, 0.1))
    x = grid.positions()
    v = np.stack([-(x[:, 1] - 1), x[:, 0] - 1, np.full(len(x), 0.3)], 1).astype(np.float32)
    grid.fields["v"] = td.Field(td.H1(grid), tack.array(v, tack.vec3()))
    return grid


def by_seed(pieces):
    """Each particle by seed: position, steps, status, time; refuses duplicates."""
    out = {}
    for piece in pieces:
        f = {n: piece.fields[n].values.numpy() for n in ("seed", "steps", "status", "time")}
        for i, s in enumerate(f["seed"].tolist()):
            assert s not in out, f"particle {s} is on two pieces"
            out[s] = (piece.positions()[i], int(f["steps"][i]), int(f["status"][i]),
                      float(f["time"][i]))
    return out


@pytest.mark.quick
def test_advect_goes_on_from_its_own_output():
    grid = helix()
    once = td.advect(grid, "v", SEEDS, 0.05, 40)
    again = td.advect(grid, "v", td.advect(grid, "v", SEEDS, 0.05, 20), 0.05, 40)
    np.testing.assert_array_equal(again.positions(), once.positions())
    for name in ("steps", "time", "status"):
        np.testing.assert_array_equal(again.fields[name].values.numpy(),
                                      once.fields[name].values.numpy())
    done = td.advect(grid, "v", once, 0.05, 40)       # nothing left to do
    np.testing.assert_array_equal(done.positions(), once.positions())
    with pytest.raises(ValueError, match="steps, time and status"):
        td.advect(grid, "v", td.extract_cells(grid, [0]), 0.05, 40)


@pytest.mark.parametrize("make", [pytest.param("split", marks=pytest.mark.quick),
                                  "generated"])
def test_pieces_with_a_ghost_layer_give_the_wholes_particles(make):
    grid = helix()
    whole = td.advect(grid, "v", SEEDS, 0.02, 200)
    if make == "split":
        pieces = td.split(grid, (2, 2, 2), ghost_layers=1)
    else:
        plain = [td.extract_cells(p, np.arange(p.num_cells)) for p in td.split(grid, (2, 2, 2))]
        for p in plain:
            p.ghosts = None
        pieces = td.ghost_cells(td.PartitionedDataSet(plain))
    got = by_seed(td.advect(pieces, "v", SEEDS, 0.02, 200))
    assert sorted(got) == list(range(len(SEEDS)))
    for q in range(len(SEEDS)):
        x, n, status, t = got[q]
        assert n == whole.fields["steps"].values.numpy()[q]
        assert status == whole.fields["status"].values.numpy()[q]
        np.testing.assert_allclose(x, whole.positions()[q], atol=1e-5)
        assert t == pytest.approx(float(whole.fields["time"].values.numpy()[q]), rel=1e-5)
    assert got[4][2] == ParticleStatus.SPATIAL_BOUNDS              # seeded outside
    assert got[2][2] & ParticleStatus.SPATIAL_BOUNDS              # left through the top
    assert got[2][1] < 200


@pytest.mark.quick
def test_rk45_steps_go_with_the_particles_across_pieces():
    """RK45 adapts each particle's step, and the step goes with it from piece to
    piece (as does its path's length, for max_length): the pieces' particles
    are the whole's, as with fixed steps."""
    grid = helix()
    options = {"integrator": "rk45", "tolerance": 1e-5, "max_length": 3.0}
    whole = td.advect(grid, "v", SEEDS, 0.02, 400, **options)
    got = by_seed(td.advect(td.split(grid, (2, 2, 2), ghost_layers=1), "v", SEEDS, 0.02, 400,
                            **options))
    for q in range(len(SEEDS)):
        x, n, status, t = got[q]
        assert n == whole.fields["steps"].values.numpy()[q]
        assert status == whole.fields["status"].values.numpy()[q]
        np.testing.assert_allclose(x, whole.positions()[q], atol=1e-5)
        assert t == pytest.approx(float(whole.fields["time"].values.numpy()[q]), rel=1e-5)


@pytest.mark.quick
def test_a_terminal_speed_stops_particles_on_any_piece_as_on_the_whole():
    """A helix that slows as it rises: particles stop below the terminal speed
    wherever that happens, on the piece they are on, as on the whole grid."""
    grid = helix()
    x = grid.positions()
    v = grid.fields["v"].values.numpy() * (1.2 - 0.5 * x[:, 2])[:, None]
    grid.fields["v"] = td.Field(td.H1(grid), tack.array(v.astype(np.float32), tack.vec3()))
    options = {"integrator": "rk45", "tolerance": 1e-5, "terminal_speed": 0.25}
    whole = td.advect(grid, "v", SEEDS, 0.02, 400, **options)
    got = by_seed(td.advect(td.split(grid, (2, 2, 2), ghost_layers=1), "v", SEEDS, 0.02, 400,
                            **options))
    slow = whole.fields["status"].values.numpy() & ParticleStatus.SLOW_SPEED
    assert slow.sum() >= 2 * ParticleStatus.SLOW_SPEED      # some stop, mid-path
    for q in range(len(SEEDS)):
        x, n, status, t = got[q]
        assert n == whole.fields["steps"].values.numpy()[q]
        assert status == whole.fields["status"].values.numpy()[q]
        np.testing.assert_allclose(x, whole.positions()[q], atol=1e-5)


@pytest.mark.parametrize("budget", [1, 3])
def test_launches_of_a_few_steps_hand_particles_on_alike(budget, monkeypatch):
    """Advection in launches of a few steps (a GPU's) pauses particles after whole
    steps; a hand-off needs a step taken in this call, which a pause must not
    forget. Over ghosted pieces the particles and the streamlines' stretches
    are one launch's, bit for bit."""
    from tack.dataset.algorithms import _advection

    pieces = td.split(helix(), (2, 2, 2), ghost_layers=1)
    monkeypatch.setattr(_advection, "_CHUNKED_ON_CPU", True)
    monkeypatch.setattr(_advection, "_GROWTH", 1)
    runs = []
    for first in (10**9, budget):
        monkeypatch.setattr(_advection, "_FIRST_STEPS", first)
        particles = by_seed(td.advect(pieces, "v", SEEDS, 0.02, 200))
        paths = td.streamlines(pieces, "v", SEEDS[:4], 0.02, 200)
        runs.append((particles, [(p.positions(), p.fields["time"].values.numpy(),
                                  p.fields["seed"].values.numpy(),
                                  p.fields["segment"].values.numpy()) for p in paths]))
    (one, one_paths), (few, few_paths) = runs
    assert sorted(one) == sorted(few)
    for q in one:
        for a, b in zip(one[q], few[q]):
            np.testing.assert_array_equal(a, b)
    for a, b in zip(one_paths, few_paths):
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)


@pytest.mark.quick
def test_pieces_without_ghost_cells_cross_by_the_short_step():
    grid = helix()
    whole = td.advect(grid, "v", SEEDS, 0.02, 200)
    got = by_seed(td.advect(td.split(grid, (2, 2, 2)), "v", SEEDS, 0.02, 200))
    assert sorted(got) == list(range(len(SEEDS)))
    worst = max(float(np.abs(got[q][0] - whole.positions()[q]).max()) for q in range(4))
    assert 1e-4 < worst < 0.1           # off by about a step, not lost


@pytest.mark.quick
def test_streamlines_over_pieces_are_the_wholes_paths():
    grid = helix()
    whole = td.streamlines(grid, "v", SEEDS[:4], 0.02, 200)
    offsets = tack.arrays.to_host(whole.topology.line_offsets)
    members = tack.arrays.to_host(whole.topology.line_points)
    expected = {int(s): whole.positions()[members[offsets[c]:offsets[c + 1]]]
                for c, s in enumerate(whole.fields["seed"].values.numpy())}
    pieces = td.streamlines(td.split(grid, (2, 2, 2), ghost_layers=1), "v", SEEDS[:4],
                            0.02, 200)
    owned, stretches = {}, {}
    for piece in pieces:
        offsets = tack.arrays.to_host(piece.topology.line_offsets)
        members = tack.arrays.to_host(piece.topology.line_points)
        marks = piece.ghosts.points.numpy()
        times = piece.fields["time"].values.numpy()
        for c, (s, k) in enumerate(zip(piece.fields["seed"].values.numpy().tolist(),
                                       piece.fields["segment"].values.numpy().tolist())):
            stretches.setdefault(s, []).append(k)
            rows = members[offsets[c]:offsets[c + 1]]
            assert bool(marks[rows[0]] & Ghosts.DUPLICATEPOINT) == (k > 0)
            for r in rows:
                if not marks[r] & Ghosts.DUPLICATEPOINT:
                    owned.setdefault(s, []).append((times[r], piece.positions()[r]))
    for s, path in expected.items():
        got = np.array([x for _, x in sorted(owned[s], key=lambda e: e[0])])
        np.testing.assert_allclose(got, path, atol=1e-5)
        assert sorted(stretches[s]) == list(range(len(stretches[s])))
    assert max(len(v) for v in stretches.values()) > 2            # they do cross


@pytest.mark.quick
def test_pieces_on_two_devices():
    grid = helix()
    second = dispatch.as_device(_SecondCPU())
    pieces = td.PartitionedDataSet([p.to_device(second) if g % 2 else p for g, p in
                                    enumerate(td.split(grid, (2, 2, 2), ghost_layers=1))])
    out = td.advect(pieces, "v", SEEDS, 0.02, 200)
    assert [p.device for p in out] == [p.device for p in pieces]
    expected = by_seed(td.advect(td.split(grid, (2, 2, 2), ghost_layers=1), "v", SEEDS,
                                 0.02, 200))
    got = by_seed([p.to_device("cpu") for p in out])
    for q in expected:
        np.testing.assert_array_equal(got[q][0], expected[q][0])


def test_particles_across_mpi_ranks(mpiexec):
    script = os.path.join(os.path.dirname(__file__), "_particles_mpi.py")
    for ranks in (2, 3):
        run = mpiexec(ranks, script)
        assert run.returncode == 0 and "ok" in run.stdout, run.stdout + run.stderr


def particles_across(world):
    """Advection and streamlines over this rank's share of eight pieces, from this
    rank's share of the seeds: each particle by seed, and each seed's stretches."""
    grid = helix()
    every = list(td.split(grid, (2, 2, 2), ghost_layers=1))
    mine = np.array_split(np.arange(len(every)), world.size)[world.rank]
    seeds = np.array_split(SEEDS, world.size)[world.rank]
    pieces = td.PartitionedDataSet([every[i] for i in mine], world)
    particles = {s: (x.tolist(), n, status, t) for s, (x, n, status, t)
                 in by_seed(td.advect(pieces, "v", seeds, 0.02, 200)).items()}
    lines = {}
    for piece in td.streamlines(pieces, "v", seeds, 0.02, 200):
        for s, k in zip(piece.fields["seed"].values.numpy().tolist(),
                        piece.fields["segment"].values.numpy().tolist()):
            lines.setdefault(s, []).append(k)
    return particles, {s: sorted(v) for s, v in lines.items()}
