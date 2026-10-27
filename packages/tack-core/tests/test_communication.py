"""Communication between blocks: exchange, the collectives, links and plans.

Blocks are put on a simulated second device -- a CPU backend whose memory
domain says otherwise, as ``test_devices.py`` does -- so messages are seen
to arrive on their destination's device on any machine. Several ranks are
simulated by threads sharing a hub (`_ThreadWorld`), each with its own
`Blocks`, which checks the paths that cross ranks without MPI.
"""

import os
import threading

import _communication_cases as cases
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
    with pytest.raises(TypeError, match="structure of arrays"):
        blocks.exchange({0: {1: {"a": tack.soa(a, a)}}})


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


def test_exclusive_scan_takes_counts(cpu):
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


def test_a_plan_sends_a_block_its_own_elements(cpu, second):
    # A piece in a periodic box is its own neighbour: what a block sends itself
    # is read before anything is written.
    blocks = tc.Blocks([cpu, second])
    plan = tc.Plan(blocks, {0: {0: [0, 1], 1: [2]}}, {0: {0: [1, 2]}, 1: {0: [0]}})
    values = {0: tack.array(np.array([5.0, 6.0, 7.0]), device=cpu),
              1: tack.array(np.array([1.0]), device=second)}
    plan.run(values)
    assert values[0].numpy().tolist() == [5.0, 5.0, 6.0]
    assert values[1].numpy().tolist() == [7.0]


def test_a_plan_refuses_mismatched_sides(cpu, second):
    blocks = tc.Blocks([cpu, second])
    with pytest.raises(ValueError, match="sends 2 elements to block 1, which receives 1"):
        tc.Plan(blocks, {0: {1: [0, 1]}}, {1: {0: [0]}})
    with pytest.raises(ValueError, match="receives 1 elements from block 0, which sends 0"):
        tc.Plan(blocks, {}, {1: {0: [0]}})
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


# -- across ranks ------------------------------------------------------------

@pytest.mark.parametrize("name", list(cases.CASES))
def test_across_simulated_ranks(cpu, name):
    ranks, body, check = cases.CASES[name]
    check(_run_ranks(ranks, body))


@pytest.mark.parametrize("mode", [[], ["--gpu-aware"], ["--gpu-aware", "--not-direct"]],
                         ids=["staged", "gpu-aware", "gpu-aware-not-direct"])
def test_across_mpi_ranks(mpiexec, mode):
    """Every case under ``mpiexec -n 3``, through MPI; GPU-aware hands the
    arrays themselves to MPI through DLPack (CPU arrays here), or, where the
    backend says MPI may not take them as they are (HIP's pool), device copies
    in ``allocate_mpi_array``'s memory."""
    script = os.path.join(os.path.dirname(__file__), "_communication_mpi.py")
    run = mpiexec(3, script, *mode)
    report = run.stdout + run.stderr
    assert run.returncode == 0, report
    for name in cases.CASES:
        assert f"ok {name}" in run.stdout, report


# -- sort --------------------------------------------------------------------

def test_a_sort_spans_devices_and_keeps_equal_keys_together(cpu, second):
    blocks = tc.Blocks([cpu, second, cpu])
    rng = np.random.default_rng(1)
    host = {g: rng.integers(0, 2**40, 200 * (g + 1)).astype(np.uint64) % np.uint64(97)
            for g in blocks.local}
    keys = {g: tack.array(host[g], tack.u64, device=blocks.device(g)) for g in blocks.local}
    out, values = tc.sort_by_key(blocks, keys)
    assert values is None
    assert all(_on(blocks.device(g), out[g]) for g in blocks.local)
    got = [out[g].numpy() for g in blocks.local]
    np.testing.assert_array_equal(np.concatenate(got), np.sort(np.concatenate(list(host.values()))))
    sets = [set(k.tolist()) for k in got]
    assert not (sets[0] & sets[1]) and not (sets[1] & sets[2]) and not (sets[0] & sets[2])
    sizes = [len(k) for k in got]
    assert min(sizes) > 0.5 * sum(sizes) / 3                 # roughly even


def test_blocks_in_one_place_sort_without_messages(cpu, monkeypatch):
    """Every block on this rank and one device: the blocks' keys sorted once and cut
    at the splitters (``_in_one_place``), not sent block to block (P squared
    messages; 3.6M kernel launches for 512 pieces). Each block holds what the
    messages give it, keys and values, equal keys and empty blocks included."""
    from tack.communication import _sort

    rng = np.random.default_rng(5)
    sizes = [0, 300, 1, 0, 250, 40, 0, 600]
    blocks = tc.Blocks(["cpu"] * len(sizes))
    keys = {g: tack.array(rng.integers(0, 50, n).astype(np.int64), tack.i64)
            for g, n in enumerate(sizes)}
    values = {g: {"at": tack.array(np.arange(n, dtype=np.int32) + 1000 * g, tack.i32)}
              for g, n in enumerate(sizes)}
    once = tc.sort_by_key(blocks, keys, values)
    assert _sort._in_one_place(blocks)
    monkeypatch.setattr(_sort, "_in_one_place", lambda blocks: False)
    sent = tc.sort_by_key(blocks, keys, values)
    for g in range(len(sizes)):
        np.testing.assert_array_equal(once[0][g].numpy(), sent[0][g].numpy())
        np.testing.assert_array_equal(once[1][g]["at"].numpy(), sent[1][g]["at"].numpy())


def test_a_sort_of_one_block_and_of_nothing(cpu):
    one = tc.Blocks(["cpu"])
    out, values = tc.sort_by_key(one, {0: tack.array(np.array([3, 1, 2], np.int32))},
                                 {0: {"v": tack.array(np.array([30., 10., 20.], np.float32))}})
    assert out[0].numpy().tolist() == [1, 2, 3] and values[0]["v"].numpy().tolist() == [10, 20, 30]
    blocks = tc.Blocks(["cpu"] * 3)
    out, values = tc.sort_by_key(blocks, {g: tack.empty(0, tack.i64) for g in blocks.local},
                                 {g: {"v": tack.empty(0, tack.vec3())} for g in blocks.local})
    assert all(out[g].size == 0 and out[g].dtype == tack.i64 for g in blocks.local)
    assert all(values[g]["v"].dtype == tack.vec3() for g in blocks.local)


def test_a_sort_refuses_what_it_cannot_sort(cpu, second):
    blocks = tc.Blocks([cpu, second])
    keys = {0: tack.zeros(3, tack.i32, device=cpu), 1: tack.zeros(2, tack.i32, device=second)}
    with pytest.raises(TypeError, match="i32, u32, i64 or u64"):
        tc.sort_by_key(blocks, {0: tack.zeros(3, device=cpu), 1: keys[1]})
    with pytest.raises(TypeError, match="one key type"):
        tc.sort_by_key(blocks, {0: tack.zeros(3, tack.i64, device=cpu), 1: keys[1]})
    with pytest.raises(ValueError, match="one element per key"):
        tc.sort_by_key(blocks, keys, {0: {"v": tack.zeros(2, device=cpu)},
                                      1: {"v": tack.zeros(2, device=second)}})
    with pytest.raises(ValueError, match="not on the block's device"):
        tc.sort_by_key(blocks, {0: keys[1], 1: keys[1]})
    with pytest.raises(ValueError, match="every block on this rank"):
        tc.sort_by_key(blocks, {0: keys[0]})


# -- routing -----------------------------------------------------------------

def test_routing_in_one_process(cpu, second):
    blocks = tc.Blocks([cpu, second])
    x = {0: np.array([[0.2, 0.0], [1.5, 0.0], [5.0, 5.0]]), 1: np.zeros((0, 2))}
    boxes = {0: ((0.0, 0.0), (1.0, 1.0)), 1: ((1.0, 0.0), (2.0, 1.0))}
    asked = []

    def holds(g, points):
        asked.append((g, len(points)))
        return np.ones(len(points), bool)

    routing = tc.Routing(blocks, x, boxes, holds)
    assert routing.destination[0].tolist() == [0, 1, -1] and routing.destination[1].tolist() == []
    assert sorted(asked) == [(0, 1), (1, 1)]                  # by box, the outsider asked of none
    values = {0: {"w": tack.array(np.array([1.0, 2.0, 3.0], np.float32), device=cpu)}, 1: {}}
    arrived = routing.send(values)
    assert arrived[1][0].tolist() == [[1.5, 0.0]] and _on(second, arrived[1][1]["w"])
    with pytest.raises(ValueError, match="send first"):
        tc.Routing(blocks, x, boxes, holds).reply({}, {"w": tack.f32})
    back = routing.reply({g: {"w": arrived[g][1]["w"]} for g in (0, 1) if len(arrived[g][0])},
                         {"w": tack.f32})
    assert back[0]["w"].numpy().tolist() == [1.0, 2.0, 0.0] and back[1]["w"].size == 0


def test_routing_prefers_the_higher_rank_then_the_lower_block(cpu, second):
    """A block that holds a point (rank 2) beats a lower block that only comes
    close to it (rank 1), as a locator ranks holding ahead of its tolerance."""
    blocks = tc.Blocks([cpu, second, cpu])
    x = {0: np.array([[0.5], [1.0], [2.0]]), 1: np.zeros((0, 1)), 2: np.zeros((0, 1))}
    boxes = dict.fromkeys(range(3), ((0.0,), (3.0,)))
    ranks = {0: [1, 1, 0], 1: [2, 1, 0], 2: [2, 2, 0]}       # per point, by block
    routing = tc.Routing(blocks, x, boxes, lambda g, p: np.array(ranks[g]))
    assert routing.destination[0].tolist() == [1, 2, -1]


def test_gpu_aware_sends_copy_what_mpi_cannot_take(cpu, second):
    """``_direct``: an array whose memory the backend says MPI may not take as it
    is (HIP's stream-ordered pool) goes as a device copy in
    ``allocate_mpi_array``'s memory; one it may take goes as it is."""
    from tack.communication._world import _direct

    backend = second._backend
    made = []
    backend.mpi_direct = lambda buffer: getattr(buffer, "_for_mpi", False)
    plain = backend.allocate_mpi_array

    def counted(dtype, shape):
        buffer = plain(dtype, shape)
        buffer._for_mpi = True
        made.append(shape)
        return buffer
    backend.allocate_mpi_array = counted
    values = tack.array(np.arange(12, dtype=np.float32).reshape(4, 3), tack.vec3(),
                        device=second)
    sent = _direct(values)
    assert sent is not values and made == [(12,)]
    assert sent.dtype == values.dtype and _on(second, sent)
    np.testing.assert_array_equal(sent.numpy(), values.numpy())
    assert _direct(sent) is sent                     # already MPI's memory
    other = tack.zeros(3, device=cpu)
    assert _direct(other) is other                   # the default: as it is
