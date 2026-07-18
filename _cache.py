# -*- coding: utf-8 -*-
"""In-process LRU cache with per-entry TTL. Pure stdlib, thread-safe.

Purpose-built for pora_llm — categorize_llm hits repeat aggressively across
users, parse_recipe hits repeat when several users import the same recipe URL.
Both wrap this cache so the LLM (or web) is not called twice for the same
input within the TTL window.

Time source is time.monotonic so the cache is immune to wall-clock jumps.
Expiry is lazy: an expired entry is evicted only when `get` touches it.
Stats are plain counters, not atomic — best-effort.
"""
from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict
from typing import Any, Optional

import constants as C


def cache_enabled_from_env() -> bool:
    """Read the PORA_CACHE_ENABLED switch. Shared by every cache owner."""
    raw = os.getenv(C.CACHE_ENABLED_ENV)
    if raw is None:
        return C.CACHE_ENABLED_DEFAULT
    return raw.strip().lower() not in C.ENV_FALSY


class TTLCache:
    """Thread-safe LRU cache with per-entry TTL.

    Backed by an ordered dict where each value is `(payload, expires_at_monotonic)`.
    On `get`, an expired entry is removed and treated as a miss; a live entry is
    moved to the MRU position. On `set`, entries beyond `maxsize` are evicted
    from the LRU end.
    """

    def __init__(self, maxsize: int, ttl_s: float):
        if maxsize <= 0:
            raise ValueError("maxsize must be positive")
        if ttl_s <= 0:
            raise ValueError("ttl_s must be positive")
        self._maxsize = maxsize
        self._ttl = float(ttl_s)
        self._data: "OrderedDict[Any, tuple[Any, float]]" = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: Any) -> Optional[Any]:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                self._misses += 1
                return None
            payload, expires_at = entry
            if time.monotonic() >= expires_at:
                self._data.pop(key, None)
                self._misses += 1
                return None
            self._data.move_to_end(key)
            self._hits += 1
            return payload

    def set(self, key: Any, value: Any) -> None:
        with self._lock:
            self._data[key] = (value, time.monotonic() + self._ttl)
            self._data.move_to_end(key)
            while len(self._data) > self._maxsize:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> dict:
        with self._lock:
            return {"size": len(self._data), "hits": self._hits, "misses": self._misses}
