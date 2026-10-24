"""Cases of communication across ranks, run on threads and under MPI.

Each case is ``(ranks, body, check)``: ``body(world)`` runs on every rank and
returns something picklable, and ``check(results)`` asserts on the results by
rank. ``test_communication.py`` runs them on simulated ranks (threads), and
``_communication_mpi.py`` under ``mpiexec`` with ``tack.communication.World``
over MPI, host-staged and GPU-aware, so every transport answers alike.
"""

import numpy as np

import tack
from tack import communication as tc


def _contiguous_ids(world):
    blocks = tc.Blocks(["cpu"] * (world.rank + 1), world)      # 1, 2 and 3 blocks
    return blocks.count, blocks.local, [blocks.rank_of(g) for g in range(blocks.count)]


def _check_contiguous_ids(results):
    assert [r[1] for r in results] == [range(1), range(1, 3), range(3, 6)]
    assert all(r[0] == 6 and r[2] == [0, 1, 1, 2, 2, 2] for r in results)


def _no_blocks(world):
    blocks = tc.Blocks([] if world.rank == 0 else ["cpu"], world)
    offsets, total = blocks.exclusive_scan(dict.fromkeys(blocks.local, 5))
    incoming = blocks.exchange({})
    return (blocks.local, offsets, total, blocks.all_reduce(dict.fromkeys(blocks.local, 1.0)),
            incoming)


def _check_no_blocks(results):
    (l0, o0, t0, s0, i0), (l1, o1, t1, s1, i1) = results
    assert l0 == range(0) and l1 == range(1) and i0 == {} and i1 == {0: {}}
    assert o0 == {} and o1 == {0: 0} and t0 == t1 == 5 and s0 == s1 == 1.0


def _ring(world):
    blocks = tc.Blocks(["cpu", "cpu"], world)
    outgoing = {g: {(g + 1) % blocks.count: {"from": tack.array(np.array([g], np.int32))}}
                for g in blocks.local}
    incoming = blocks.exchange(outgoing)
    return {g: {s: int(m["from"].numpy()[0]) for s, m in r.items()} for g, r in incoming.items()}


def _check_ring(results):
    assert results == [{0: {5: 5}, 1: {0: 0}}, {2: {1: 1}, 3: {2: 2}}, {4: {3: 3}, 5: {4: 4}}]


def _kinds(g):
    """A message of every kind of array, its values from the sending block's id."""
    return {
        "vec": tack.array(np.arange(6, dtype=np.float64).reshape(2, 3) + g,
                          tack.vec3(tack.f64)),
        "mat": tack.array(np.arange(12, dtype=np.float32).reshape(2, 2, 3) - g,
                          tack.mat(2, 3)),
        "grid": tack.array(np.arange(6, dtype=np.int32).reshape(2, 3) * g),
        "bytes": tack.array(np.array([g, 255], np.uint8)),
        "none": tack.empty(0, tack.f32),
    }


def _every_kind(world):
    blocks = tc.Blocks(["cpu", "cpu"], world)
    outgoing = {g: {d: (_kinds(g) if d % 2 == 0 else {}) for d in range(blocks.count) if d != g}
                for g in blocks.local}
    incoming = blocks.exchange(outgoing)
    return {g: {s: {n: (str(a.dtype), a.shape, a.numpy().tolist()) for n, a in m.items()}
                for s, m in r.items()} for g, r in incoming.items()}


def _check_every_kind(results):
    every = {g: received for r in results for g, received in r.items()}
    assert sorted(every) == [0, 1, 2, 3]
    for g, received in every.items():
        assert sorted(received) == [s for s in range(4) if s != g]
        for s, message in received.items():
            if g % 2:
                assert message == {}
                continue
            expected = {n: (str(a.dtype), a.shape, a.numpy().tolist())
                        for n, a in _kinds(s).items()}
            assert message == expected


def _sum_values():
    rng = np.random.default_rng(3)
    return rng.standard_normal(6) * 10.0 ** rng.integers(-8, 8, 6)


def _sum_bits(world):
    values = _sum_values()
    blocks = tc.Blocks(["cpu"] * (1 if world.rank < 2 else 4), world)
    return blocks.all_reduce({g: float(values[g]) for g in blocks.local})


def _check_sum_bits(results):
    one = tc.Blocks(["cpu"] * 6).all_reduce({g: float(v) for g, v in enumerate(_sum_values())})
    assert set(results) == {one}


def _gather_and_scan(world):
    blocks = tc.Blocks(["cpu"] * 2, world)
    every = blocks.all_gather({g: ("block", g) for g in blocks.local})
    offsets, total = blocks.exclusive_scan({g: g + 1 for g in blocks.local})
    low = blocks.all_reduce({g: np.array([g, -g]) for g in blocks.local}, "min")
    return every, offsets, total, low.tolist()


def _check_gather_and_scan(results):
    assert all(r[0] == [("block", g) for g in range(4)] and r[2] == 10 for r in results)
    assert results[0][1] == {0: 0, 1: 1} and results[1][1] == {2: 3, 3: 6}
    assert all(r[3] == [0, -3] for r in results)


def _links(world):
    blocks = tc.Blocks(["cpu"] * 2, world)
    boxes = {g: ((g, 0.0), (g + 1.0, 1.0)) for g in blocks.local}     # a row of squares
    return blocks.link_by_bounds(boxes), blocks.link_by_bounds(boxes, margin=1.5)


def _check_links(results):
    touching = {g: n for r in results for g, n in r[0].items()}
    near = {g: n for r in results for g, n in r[1].items()}
    assert touching == {0: [1], 1: [0, 2], 2: [1, 3], 3: [2]}
    assert near == {0: [1, 2], 1: [0, 2, 3], 2: [0, 1, 3], 3: [1, 2]}


def _plan_ring(world):
    blocks = tc.Blocks(["cpu"], world)
    g = blocks.local.start
    values = {g: tack.array(np.full((3, 3), float(g), np.float32), tack.vec3())}
    ids = {g: tack.array(np.full(3, 10 * g, np.int64))}
    right, left = (g + 1) % blocks.count, (g - 1) % blocks.count
    plan = tc.Plan(blocks, sends={g: {right: [1]}}, receives={g: {left: [0]}})
    plan.run({g: {"v": values[g], "id": ids[g]}})
    return values[g].numpy()[:, 0].tolist(), ids[g].numpy().tolist()


def _check_plan_ring(results):
    assert results == [([2, 0, 0], [20, 0, 0]), ([0, 1, 1], [0, 10, 10]),
                       ([1, 2, 2], [10, 20, 20])]


CASES = {
    "contiguous ids": (3, _contiguous_ids, _check_contiguous_ids),
    "a rank with no blocks": (2, _no_blocks, _check_no_blocks),
    "exchange in a ring": (3, _ring, _check_ring),
    "every kind of array": (2, _every_kind, _check_every_kind),
    "a sum's bits": (3, _sum_bits, _check_sum_bits),
    "gather and scan": (2, _gather_and_scan, _check_gather_and_scan),
    "links": (2, _links, _check_links),
    "a plan in a ring": (3, _plan_ring, _check_plan_ring),
}
