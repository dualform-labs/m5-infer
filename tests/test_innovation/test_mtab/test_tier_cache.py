"""MTAB tier_cache tests."""

from app.innovation.mtab.tier_cache import TierCache, TierEntry


def _make_entry(hash_: str, tokens: list[int], boundary: int) -> TierEntry:
    return TierEntry(
        prompt_hash=hash_,
        prompt_tokens=list(tokens),
        layer_boundary=boundary,
        hidden_states=None,
    )


def test_store_and_lookup_exact():
    tc = TierCache(max_entries=4)
    entry = _make_entry("h1", [1, 2, 3], 16)
    tc.store(entry)
    found, matched = tc.lookup_best([1, 2, 3])
    assert found is entry
    assert matched == 3


def test_lookup_returns_partial_match():
    tc = TierCache(max_entries=4)
    tc.store(_make_entry("h1", [1, 2, 3, 4], 16))
    found, matched = tc.lookup_best([1, 2, 3, 99])
    assert found is not None
    assert matched == 3


def test_lookup_miss():
    tc = TierCache(max_entries=4)
    tc.store(_make_entry("h1", [1, 2, 3], 16))
    found, matched = tc.lookup_best([7, 8, 9])
    assert found is None
    assert matched == 0


def test_eviction_when_full():
    tc = TierCache(max_entries=2)
    for i in range(3):
        tc.store(_make_entry(f"h{i}", [i], 16))
    stats = tc._stats
    assert stats["evictions"] >= 1
