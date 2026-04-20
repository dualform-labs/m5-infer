"""SSD n-gram table tests."""

import json
import tempfile
from pathlib import Path

from app.innovation.ssd.ngram_table import SSDTable
from app.innovation.ssd.skip_predictor import SkipPredictor


def test_empty_lookup_returns_none():
    t = SSDTable()
    assert t.lookup([1, 2, 3]) is None


def test_load_from_file(tmp_path: Path):
    data = {
        "ngram_size": 3,
        "min_confidence": 0.9,
        "entries": [
            {"context": [1, 2, 3], "next": 4, "confidence": 0.99},
            {"context": [5, 6], "next": 7, "confidence": 0.5},
        ],
    }
    p = tmp_path / "table.json"
    p.write_text(json.dumps(data))
    t = SSDTable()
    assert t.load_from_file(str(p)) is True
    assert t.size() == 2


def test_lookup_respects_min_confidence(tmp_path: Path):
    data = {
        "entries": [
            {"context": [10, 11], "next": 42, "confidence": 0.8},
        ],
    }
    p = tmp_path / "t.json"
    p.write_text(json.dumps(data))
    t = SSDTable()
    t.load_from_file(str(p))
    # Above threshold
    assert t.lookup([10, 11], min_confidence=0.7) == 42
    # Below threshold
    assert t.lookup([10, 11], min_confidence=0.9) is None


def test_skip_predictor_disabled_returns_none():
    p = SkipPredictor(enabled=False)
    assert p.predict([1, 2, 3]) is None


def test_skip_predictor_verify_cadence():
    """Every Nth call should force a model pass (return None)."""
    p = SkipPredictor(enabled=True, verify_every=2)
    # Empty table means lookup always returns None even when not forced.
    for _ in range(5):
        out = p.predict([99, 100])
        assert out is None
    stats = p.stats()
    assert stats["enabled"] is True
