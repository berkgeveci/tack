"""Communication between blocks: exchange, the collectives, links and plans.

Blocks are put on a simulated second device -- a CPU backend whose memory
domain says otherwise, as ``test_devices.py`` does -- so messages are seen
to arrive on their destination's device on any machine. Several ranks are
simulated by threads sharing a hub (`_ThreadWorld`), each with its own
`Blocks`, which checks the paths that cross ranks without MPI.
"""

import threading

import numpy as np
import pytest

import tack
from tack import communication as tc
from tack.runtime import dispatch
from tack.runtime.cpu import CPUBackend

pytestmark = pytest.mark.quick


class _SecondCPU(CPUBackend):
    """A CPU backend posing as another device's: its arrays are not the default's."""

    display_name = "second device"

    @property
    def memory_domain(self):
        return ("cpu", 1)


@pytest.fixture
def cpu():
    tack.set_default_device("cpu")
    yield tack.device("cpu")
    tack.set_default_device("cpu")


@pytest.fixture
def second(cpu):
    return dispatch.as_device(_SecondCPU())


def _on(device, array):
    return array._backend.memory_domain == device._backend.memory_domain


class _Hub:
    """What the threads of a simulated run share: a barrier and a slot per rank."""

    def __init__(self, size):
        self.size = size
        self.barrier = threading.Barrier(size, timeout=60)
        self.slots = [None] * size


class _ThreadWorld:
    """One rank of a run simulated by threads: the transport `World` is, over a hub."""

    def __init__(self, hub, rank):
        self.hub, self.rank, self.size = hub, rank, hub.size

    def _allgather(self, value):
        self.hub.slots[self.rank] = value
        self.hub.barrier.wait()
        every = list(self.hub.slots)
        self.hub.barrier.wait()
        return every

    def _exchange(self, messages, rank_of, device_of):
        every = self._allgather(messages)
        return [(source, destination, {name: values.to_device(device_of(destination))
                                       for name, values in message.items()})
                for sent in every for source, destination, message in sent
                if rank_of(destination) == self.rank]


def _run_ranks(size, body):
    """``body(world)`` on ``size`` simulated ranks at once; their results by rank."""
    hub = _Hub(size)
    results, errors = [None] * size, []

    def run(rank):
        try:
            results[rank] = body(_ThreadWorld(hub, rank))
        except BaseException as error:          # report it, and free the others
            errors.append(error)
            hub.barrier.abort()

    threads = [threading.Thread(target=run, args=(r,)) for r in range(size)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    real = [e for e in errors if not isinstance(e, threading.BrokenBarrierError)]
    if real or errors:
        raise (real or errors)[0]
    return results


# -- blocks ------------------------------------------------------------------

def test_blocks_number_this_process_blocks_from_zero(cpu, second):
    blocks = tc.Blocks([cpu, second, "cpu"])
    assert blocks.count == 3 and blocks.local == range(3)
    assert blocks.device(1) == second and blocks.device(2) == cpu
    assert [blocks.rank_of(g) for g in range(3)] == [0, 0, 0]
    with pytest.raises(ValueError, match="not a block id"):
        blocks.rank_of(3)


def test_global_ids_are_contiguous_by_rank(cpu):
    def body(world):
        blocks = tc.Blocks(["cpu"] * (world.rank + 1), world)      # 1, 2 and 3 blocks
        return blocks.count, blocks.local, [blocks.rank_of(g) for g in range(blocks.count)]

    results = _run_ranks(3, body)
    assert [r[1] for r in results] == [range(1), range(1, 3), range(3, 6)]
    assert all(r[0] == 6 and r[2] == [0, 1, 1, 2, 2, 2] for r in results)


def test_a_rank_may_hold_no_blocks(cpu):
    def body(world):
        blocks = tc.Blocks([] if world.rank == 0 else ["cpu"], world)
        offsets, total = blocks.exclusive_scan(dict.fromkeys(blocks.local, 5))
        return blocks.local, offsets, total, blocks.all_reduce(dict.fromkeys(blocks.local, 1.0))

    (l0, o0, t0, s0), (l1, o1, t1, s1) = _run_ranks(2, body)
    assert l0 == range(0) and l1 == range(1)
    assert o0 == {} and o1 == {0: 0} and t0 == t1 == 5 and s0 == s1 == 1.0


# -- exchange ----------------------------------------------------------------

def test_messages_arrive_on_their_destinations_device(cpu, second):
    blocks = tc.Blocks([cpu, second, cpu])
    x = tack.array(np.arange(4, dtype=np.float32), device=cpu)
    ids = tack.array(np.array([7, 8], dtype=np.int32), device=cpu)
    y = tack.array(np.ones(3, dtype=np.float32), device=second)
    incoming = blocks.exchange({0: {1: {"x": x, "ids": ids}, 2: {"x": x}},
                                1: {0: {"y": y}}})
    assert list(incoming) == [0, 1, 2]
    assert set(incoming[1][0]) == {"ids", "x"}
    moved = incoming[1][0]["x"]
    assert _on(second, moved) and moved.numpy().tolist() == [0, 1, 2, 3]
    assert incoming[1][0]["ids"].dtype == tack.i32
    # On a device that addresses the source's memory, the arrays themselves.
    assert incoming[2][0]["x"] is x
    assert _on(cpu, incoming[0][1]["y"]) and incoming[0][1]["y"].numpy().tolist() == [1] * 3


def test_every_local_block_has_an_entry_in_source_order(cpu):
    blocks = tc.Blocks(["cpu"] * 4)
    a = tack.zeros(1)
    incoming = blocks.exchange({3: {0: {"a": a}}, 1: {0: {"a": a}}, 2: {}})
    assert list(incoming) == [0, 1, 2, 3]
    assert list(incoming[0]) == [1, 3] and incoming[1] == incoming[2] == incoming[3] == {}


def test_exchange_refuses_what_it_cannot_send(cpu, second):
    blocks = tc.Blocks([cpu, second])
    a = tack.zeros(2, device=cpu)
    with pytest.raises(ValueError, match="not on this rank"):
        blocks.exchange({2: {0: {"a": a}}})
    with pytest.raises(ValueError, match="not a block id"):
        blocks.exchange({0: {5: {"a": a}}})
    with pytest.raises(TypeError, match="not a dict"):
        blocks.exchange({0: {1: a}})
    with pytest.raises(TypeError, match="not a Tack array"):
        blocks.exchange({0: {1: {"a": np.zeros(2)}}})
    with pytest.raises(ValueError, match="not on its source block's device"):
        blocks.exchange({1: {0: {"a": a}}})


def test_exchange_crosses_ranks(cpu):
    def body(world):
        blocks = tc.Blocks(["cpu", "cpu"], world)
        outgoing = {g: {(g + 1) % blocks.count: {"from": tack.array(np.array([g], np.int32))}}
                    for g in blocks.local}
        incoming = blocks.exchange(outgoing)
        return {g: {s: int(m["from"].numpy()[0]) for s, m in r.items()}
                for g, r in incoming.items()}

    results = _run_ranks(3, body)
    assert results == [{0: {5: 5}, 1: {0: 0}}, {2: {1: 1}, 3: {2: 2}}, {4: {3: 3}, 5: {4: 4}}]


# -- collectives -------------------------------------------------------------

def test_all_reduce_combines_in_global_id_order(cpu):
    blocks = tc.Blocks(["cpu"] * 3)
    assert blocks.all_reduce({0: 1e16, 1: 1.0, 2: -1e16}) == (1e16 + 1.0) - 1e16
    assert blocks.all_reduce({0: 3, 1: 1, 2: 2}, "min") == 1
    assert blocks.all_reduce({0: 3, 1: 1, 2: 2}, "max") == 3
    total = blocks.all_reduce({g: np.array([g, 1.0]) for g in range(3)})
    np.testing.assert_array_equal(total, [3.0, 3.0])
    np.testing.assert_array_equal(
        blocks.all_reduce({0: np.array([1, 5]), 1: np.array([4, 2]), 2: 3}, "min"), [1, 2])
    with pytest.raises(ValueError, match="op 'sum', 'min' or 'max'"):
        blocks.all_reduce({0: 1, 1: 1, 2: 1}, "mean")
    with pytest.raises(ValueError, match="every block on this rank"):
        blocks.all_reduce({0: 1, 2: 1})


def test_a_sum_has_the_same_bits_however_blocks_are_assigned(cpu):
    rng = np.random.default_rng(3)
    values = rng.standard_normal(6) * 10.0 ** rng.integers(-8, 8, 6)
    one = tc.Blocks(["cpu"] * 6).all_reduce(dict(enumerate(values)))

    def body(world):
        blocks = tc.Blocks(["cpu"] * (1 if world.rank < 2 else 4), world)
        return blocks.all_reduce({g: values[g] for g in blocks.local})

    assert set(_run_ranks(3, body)) == {one}


def test_all_gather_and_exclusive_scan(cpu):
    def body(world):
        blocks = tc.Blocks(["cpu"] * 2, world)
        every = blocks.all_gather({g: ("block", g) for g in blocks.local})
        offsets, total = blocks.exclusive_scan({g: g + 1 for g in blocks.local})
        return every, offsets, total

    results = _run_ranks(2, body)
    assert all(r[0] == [("block", g) for g in range(4)] and r[2] == 10 for r in results)
    assert results[0][1] == {0: 0, 1: 1} and results[1][1] == {2: 3, 3: 6}
    with pytest.raises(ValueError, match="integers of at least 0"):
        tc.Blocks(["cpu"]).exclusive_scan({0: -1})


def test_link_by_bounds(cpu):
    blocks = tc.Blocks(["cpu"] * 4)
    boxes = {0: ((0, 0, 0), (1, 1, 1)), 1: ((1, 0, 0), (2, 1, 1)),     # share a face
             2: ((2, 1, 1), (3, 2, 2)),                                # an edge with 1
             3: ((3.5, 0, 0), (4, 1, 1))}                              # apart
    assert blocks.link_by_bounds(boxes) == {0: [1], 1: [0, 2], 2: [1], 3: []}
    assert blocks.link_by_bounds(boxes, margin=0.5) == {0: [1], 1: [0, 2], 2: [1, 3],
                                                        3: [2]}
    with pytest.raises(ValueError, match="lower corner above"):
        blocks.link_by_bounds({**boxes, 3: ((1, 1, 1), (0, 0, 0))})
    with pytest.raises(ValueError, match="one dimension"):
        blocks.link_by_bounds({**boxes, 3: ((0, 0), (1, 1))})
    with pytest.raises(ValueError, match="at least 0"):
        blocks.link_by_bounds(boxes, margin=-1)


# -- plans -------------------------------------------------------------------

def test_a_plan_copies_elements_into_place(cpu, second):
    blocks = tc.Blocks([cpu, second, cpu])
    values = {g: tack.array(np.arange(5, dtype=np.float32) + 10 * g, device=blocks.device(g))
              for g in blocks.local}
    plan = tc.Plan(blocks, sends={0: {1: [4, 0]}, 2: {1: np.array([1], np.int32), 0: [3]}},
                   receives={1: {0: [1, 3], 2: [0]}, 0: {2: [2]}})
    plan.run(values)
    assert values[1].numpy().tolist() == [21, 4, 12, 0, 14]
    assert values[0].numpy().tolist() == [0, 1, 23, 3, 4]
    assert values[2].numpy().tolist() == [20, 21, 22, 23, 24]
    plan.run(values)                    # again, from the updated values
    assert values[0].numpy().tolist() == [0, 1, 23, 3, 4]


def test_a_plan_moves_whole_elements_and_several_arrays(cpu, second):
    blocks = tc.Blocks([cpu, second])
    vec = {g: tack.array(np.arange(12, dtype=np.float64).reshape(4, 3) + 100 * g,
                         tack.vec3(tack.f64), device=blocks.device(g)) for g in blocks.local}
    ids = {g: tack.array(np.arange(4, dtype=np.int64) + 10 * g, device=blocks.device(g))
           for g in blocks.local}
    indices = tack.array(np.array([2, 0], np.int64), device=second)
    plan = tc.Plan(blocks, sends={1: {0: indices}}, receives={0: {1: [3, 1]}})
    plan.run({g: {"vec": vec[g], "ids": ids[g]} for g in blocks.local})
    np.testing.assert_array_equal(vec[0].numpy()[[3, 1]], [[106, 107, 108], [100, 101, 102]])
    assert ids[0].numpy().tolist() == [0, 10, 2, 12]
    assert vec[1].numpy()[0].tolist() == [100, 101, 102]


def test_a_plan_crosses_ranks(cpu):
    def body(world):
        blocks = tc.Blocks(["cpu"], world)
        g = blocks.local.start
        values = {g: tack.array(np.full(3, float(g), np.float32))}
        right, left = (g + 1) % blocks.count, (g - 1) % blocks.count
        plan = tc.Plan(blocks, sends={g: {right: [1]}}, receives={g: {left: [0]}})
        plan.run(values)
        return values[g].numpy().tolist()

    assert _run_ranks(3, body) == [[2, 0, 0], [0, 1, 1], [1, 2, 2]]


def test_a_plan_refuses_mismatched_sides(cpu, second):
    blocks = tc.Blocks([cpu, second])
    with pytest.raises(ValueError, match="sends 2 elements to block 1, which receives 1"):
        tc.Plan(blocks, {0: {1: [0, 1]}}, {1: {0: [0]}})
    with pytest.raises(ValueError, match="receives 1 elements from block 0, which sends 0"):
        tc.Plan(blocks, {}, {1: {0: [0]}})
    with pytest.raises(ValueError, match="with itself"):
        tc.Plan(blocks, {0: {0: [0]}}, {0: {0: [0]}})
    with pytest.raises(TypeError, match="not integers"):
        tc.Plan(blocks, {0: {1: [0.5]}}, {1: {0: [0]}})
    with pytest.raises(ValueError, match="not on block 0's device"):
        tc.Plan(blocks, {0: {1: tack.array(np.array([0]), device=second)}}, {1: {0: [0]}})
    plan = tc.Plan(blocks, {0: {1: [0]}}, {1: {0: [0]}})
    a, b = tack.zeros(2, device=cpu), tack.zeros(2, device=second)
    with pytest.raises(ValueError, match="needs values for blocks \\[1\\]"):
        plan.run({0: a})
    with pytest.raises(ValueError, match="not on the block's device"):
        plan.run({0: a, 1: a})
    with pytest.raises(TypeError, match="sent 'values' as"):
        plan.run({0: a, 1: tack.zeros(2, tack.f64, device=second)})
    with pytest.raises(TypeError, match="not both"):
        plan.run({0: a, 1: {"values": b}})
    with pytest.raises(ValueError, match="same names"):
        plan.run({0: {"a": a}, 1: {"b": b}})


def test_an_empty_plan_does_nothing(cpu):
    blocks = tc.Blocks(["cpu", "cpu"])
    plan = tc.Plan(blocks, {0: {1: []}}, {1: {0: []}})
    a = tack.array(np.arange(3, dtype=np.float32))
    plan.run({0: a, 1: a})
    assert a.numpy().tolist() == [0, 1, 2]
