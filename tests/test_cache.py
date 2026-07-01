# -*- coding: utf-8 -*-
"""Unit tests for _cache.TTLCache — pure stdlib LRU + TTL."""
from __future__ import annotations

import pytest

from _cache import TTLCache


class TestTTLCacheBasics:
    def test_get_missing_returns_none(self):
        c = TTLCache(maxsize=4, ttl_s=60)
        assert c.get("nope") is None

    def test_set_then_get(self):
        c = TTLCache(maxsize=4, ttl_s=60)
        c.set("k", 42)
        assert c.get("k") == 42

    def test_clear_resets(self):
        c = TTLCache(maxsize=4, ttl_s=60)
        c.set("k", 1)
        c.clear()
        assert c.get("k") is None
        assert c.stats()["size"] == 0

    def test_stats_track_hits_and_misses(self):
        c = TTLCache(maxsize=4, ttl_s=60)
        c.set("a", 1)
        c.get("a")          # hit
        c.get("b")          # miss
        s = c.stats()
        assert s["hits"] == 1
        assert s["misses"] == 1
        assert s["size"] == 1


class TestTTLCacheLRU:
    def test_evicts_oldest_over_capacity(self):
        c = TTLCache(maxsize=2, ttl_s=60)
        c.set("a", 1)
        c.set("b", 2)
        c.set("c", 3)               # forces eviction
        assert c.get("a") is None   # a was oldest
        assert c.get("b") == 2
        assert c.get("c") == 3

    def test_get_marks_entry_recent(self):
        c = TTLCache(maxsize=2, ttl_s=60)
        c.set("a", 1)
        c.set("b", 2)
        c.get("a")                  # a becomes MRU
        c.set("c", 3)               # b is oldest now → evicted
        assert c.get("b") is None
        assert c.get("a") == 1
        assert c.get("c") == 3


class TestTTLCacheExpiry:
    def test_expired_entry_yields_none(self, monkeypatch):
        import _cache
        clock = {"t": 1000.0}
        monkeypatch.setattr(_cache.time, "monotonic", lambda: clock["t"])

        c = TTLCache(maxsize=4, ttl_s=10)
        c.set("k", "v")
        clock["t"] += 11            # past TTL
        assert c.get("k") is None
        assert c.stats()["size"] == 0   # expired entry evicted lazily
