"""MRPB scheduler unit tests."""

import asyncio

import pytest

from app.engine.mrpb_scheduler import MRPBScheduler, classify_request


def test_classify_light():
    assert classify_request(max_tokens=30, prompt_tokens_estimate=100) == "light"


def test_classify_heavy_by_output():
    assert classify_request(max_tokens=2048, prompt_tokens_estimate=100) == "heavy"


def test_classify_heavy_by_prompt():
    assert classify_request(max_tokens=128, prompt_tokens_estimate=10000) == "heavy"


def test_classify_medium():
    assert classify_request(max_tokens=256, prompt_tokens_estimate=500) == "medium"


def test_parallel_defaults_off():
    s = MRPBScheduler()
    assert s.is_parallel_enabled() is False


def test_enable_parallel_flag():
    s = MRPBScheduler()
    s.enable_parallel()
    assert s.is_parallel_enabled() is True


def test_acquire_serializes_when_off():
    s = MRPBScheduler()

    async def run():
        async with s.acquire("light"):
            # Master lock should be held.
            assert s._master.locked()

    asyncio.run(run())


def test_acquire_class_when_on():
    s = MRPBScheduler()
    s.enable_parallel()

    async def run():
        async with s.acquire("heavy"):
            assert s._class_locks["heavy"].locked()
            # Master should NOT be held.
            assert not s._master.locked()

    asyncio.run(run())
