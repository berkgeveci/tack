"""``runtime.buffer_cache.BufferCache``, the cache of dropped arrays' buffers HIP
allocates through (ROCm 7.1.1's pool slowed with every allocation it made)."""

import pytest

from tack.runtime.buffer_cache import BufferCache

pytestmark = pytest.mark.quick


def _cache(limit):
    freed = []
    return BufferCache(limit, freed.append), freed


def test_capacities_round_up_to_a_quarter_of_a_power_of_two():
    assert BufferCache.capacity(1) == BufferCache.MIN
    assert BufferCache.capacity(256) == 256
    assert BufferCache.capacity(257) == 320          # 256 in quarters of 64
    assert BufferCache.capacity(1000) == 1024
    assert BufferCache.capacity(5 << 20) == 5 << 20
    assert BufferCache.capacity((5 << 20) + 1) == 6 << 20
    for n in (300, 4097, 10 ** 6, 3 * 10 ** 9):
        assert n <= BufferCache.capacity(n) <= n * 1.25 + BufferCache.MIN


def test_a_given_buffer_comes_back_for_its_capacity_alone():
    cache, freed = _cache(1 << 20)
    cache.give(0x1000, 256)
    cache.give(0x2000, 512)
    assert cache.held == 768
    assert cache.take(320) is None
    assert cache.take(512) == 0x2000 and cache.take(512) is None
    assert cache.take(256) == 0x1000
    assert cache.held == 0 and freed == []


def test_past_the_limit_the_oldest_are_freed():
    cache, freed = _cache(1024)
    for k in range(5):
        cache.give(0x1000 * (k + 1), 256)            # the fifth passes the limit
    assert freed == [0x1000] and cache.held == 1024
    cache.give(0x9000, 2048)                         # larger than the limit: freed at once
    assert freed == [0x1000, 0x9000] and cache.held == 1024
    assert [cache.take(256) for _ in range(5)] == [0x5000, 0x4000, 0x3000, 0x2000, None]


def test_clear_frees_everything_held():
    cache, freed = _cache(1 << 20)
    cache.give(0x1000, 256)
    cache.give(0x2000, 1024)
    cache.clear()
    assert sorted(freed) == [0x1000, 0x2000] and cache.held == 0
    assert cache.take(256) is None
