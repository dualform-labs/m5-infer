"""MMRS registry unit tests."""

import pytest

from app.engine.mmrs_registry import MMRSRegistry


def test_starts_inactive():
    r = MMRSRegistry()
    assert r.is_active() is False
    assert r.stats()["total_gb"] == 0


def test_activate_sets_budget():
    r = MMRSRegistry()
    r.activate(total_resident_gb_budget=30.0)
    assert r.is_active() is True
    assert r.stats()["budget_gb"] == 30.0


def test_register_within_budget():
    r = MMRSRegistry()
    r.activate(20.0)
    r.register("main", "path/a", object(), estimated_gb=8.0)
    r.register("heavy", "path/b", object(), estimated_gb=8.0)
    assert "main" in r.stats()["resident"]
    assert r.stats()["total_gb"] == pytest.approx(16.0)


def test_register_rejects_overflow():
    r = MMRSRegistry()
    r.activate(10.0)
    r.register("main", "p", object(), estimated_gb=8.0)
    # This would exceed budget; silently rejected.
    r.register("heavy", "p2", object(), estimated_gb=5.0)
    assert "heavy" not in r.stats()["resident"]


def test_evict_pops_entry():
    r = MMRSRegistry()
    r.activate(20.0)
    r.register("vision", "p", object(), estimated_gb=4.0)
    assert "vision" in r.stats()["resident"]
    r.evict("vision")
    assert "vision" not in r.stats()["resident"]
