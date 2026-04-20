"""Unit tests for TPC (Token Prefix Compiler) cache and background compiler."""
from __future__ import annotations

import threading
import time

import pytest

from app.innovation.tpc.fast_lookup import TPCCache, TPCEntry
from app.innovation.tpc.background_compiler import BackgroundCompiler


def test_content_hash_stable():
    c = TPCCache()
    h1 = c.content_hash("hello world")
    h2 = c.content_hash("hello world")
    h3 = c.content_hash("hello World")  # case-sensitive
    assert h1 == h2
    assert h1 != h3
    assert len(h1) == 32


def test_store_and_lookup():
    c = TPCCache(max_entries=4)
    tokens = [1, 2, 3, 4, 5]
    entry = c.store("hello", tokens, ctrsp_token_hash="abc")
    assert entry.token_count == 5
    assert entry.ctrsp_token_hash == "abc"

    hit = c.lookup("hello")
    assert hit is not None
    assert hit.token_ids == tuple(tokens)
    assert hit.hits == 1

    miss = c.lookup("world")
    assert miss is None

    stats = c.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["stores"] == 1


def test_lru_eviction():
    c = TPCCache(max_entries=3)
    for i in range(5):
        c.store(f"content_{i}", [i])
    stats = c.stats()
    assert stats["size"] == 3
    assert stats["evictions"] == 2
    # content_0, content_1 should be evicted
    assert c.lookup("content_0") is None
    assert c.lookup("content_4") is not None


def test_mark_flags():
    c = TPCCache()
    c.store("foo", [1, 2, 3])
    assert c.lookup("foo").mtab_observed is False
    c.mark_mtab_observed("foo")
    assert c.lookup("foo").mtab_observed is True
    c.mark_redundancy_scanned("foo")
    assert c.lookup("foo").redundancy_scanned is True


def test_duplicate_store_updates_ctrsp_hash():
    c = TPCCache()
    c.store("foo", [1, 2], ctrsp_token_hash=None)
    c.store("foo", [1, 2], ctrsp_token_hash="xyz")
    assert c.lookup("foo").ctrsp_token_hash == "xyz"


def test_background_compiler_runs_task():
    bg = BackgroundCompiler()
    bg.start()
    done = threading.Event()

    def work():
        time.sleep(0.01)
        done.set()

    ok = bg.submit("test_task", work)
    assert ok
    assert done.wait(timeout=2.0)
    # Give stats a moment to update
    time.sleep(0.05)
    stats = bg.stats()
    assert stats["completed"] >= 1
    bg.stop()


def test_background_compiler_survives_exception():
    bg = BackgroundCompiler()
    bg.start()
    marker = threading.Event()

    def bad():
        raise RuntimeError("boom")

    def good():
        marker.set()

    bg.submit("bad", bad)
    bg.submit("good", good)
    assert marker.wait(timeout=2.0)
    time.sleep(0.05)
    stats = bg.stats()
    assert stats["errors"] >= 1
    assert stats["completed"] >= 1
    bg.stop()
