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



def _sort_input(gid):
    """Block ``gid``'s keys for the sort: some repeated, block 3's none."""
    rng = np.random.default_rng(gid)
    n = (0, 50, 7, 0, 300)[gid % 5] if gid != 0 else 120
    return rng.integers(-20, 40, n).astype(np.int64)


def _sort(world):
    blocks = tc.Blocks(["cpu"] * 2, world)
    keys = {g: tack.array(_sort_input(g), tack.i64) for g in blocks.local}
    values = {g: {"origin": tack.array(np.full(len(_sort_input(g)), g, np.int32)),
                  "index": tack.array(np.arange(len(_sort_input(g)), dtype=np.int64)),
                  "vec": tack.array(np.repeat(_sort_input(g)[:, None], 3, 1).astype(np.float32),
                                    tack.vec3())}
              for g in blocks.local}
    out_keys, out_values = tc.sort_by_key(blocks, keys, values)
    return {g: (out_keys[g].numpy().tolist(),
                {n: a.numpy().tolist() for n, a in out_values[g].items()})
            for g in blocks.local}


def _check_sort(results):
    every = {g: r for result in results for g, r in result.items()}
    keys = [k for g in sorted(every) for k in every[g][0]]
    origin = [o for g in sorted(every) for o in every[g][1]["origin"]]
    index = [i for g in sorted(every) for i in every[g][1]["index"]]
    pairs = sorted(((k, g, i) for g in range(4) for i, k in enumerate(_sort_input(g))))
    assert list(zip(keys, origin, index)) == pairs          # stable, block then index
    last = [every[g][0][-1] for g in sorted(every) if every[g][0]]
    first = [every[g][0][0] for g in sorted(every) if every[g][0]]
    assert all(a < b for a, b in zip(last, first[1:]))     # a key on one block only
    assert all(v == [k] * 3 for g in every for k, v in zip(every[g][0], every[g][1]["vec"]))
    assert sum(1 for g in every if every[g][0]) >= 3       # shared out


CASES = {
    "contiguous ids": (3, _contiguous_ids, _check_contiguous_ids),
    "a rank with no blocks": (2, _no_blocks, _check_no_blocks),
    "exchange in a ring": (3, _ring, _check_ring),
    "every kind of array": (2, _every_kind, _check_every_kind),
    "a sum's bits": (3, _sum_bits, _check_sum_bits),
    "gather and scan": (2, _gather_and_scan, _check_gather_and_scan),
    "links": (2, _links, _check_links),
    "a plan in a ring": (3, _plan_ring, _check_plan_ring),
    "a sort": (2, _sort, _check_sort),
}
