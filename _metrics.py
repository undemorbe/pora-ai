# -*- coding: utf-8 -*-
"""In-process LLM telemetry. Pure stdlib, thread-safe, zero overhead when idle.

Collects per-model-kind counters for every LLM round trip made through
``pora_llm._chat``: calls, errors, latency, and token usage (when the
provider returns ``response.usage``). Exposed read-only via ``GET /metrics``
together with cache stats.

Deliberately NOT Prometheus format — the Go backend consumes JSON everywhere
else; scraping can be added later without touching this module's callers.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Optional


class Metrics:
    """Thread-safe counter bag for LLM calls.

    All numbers are cumulative since process start. ``snapshot()`` returns a
    plain dict safe to serialize; per-kind latency is exposed as total ms and
    call count so the consumer computes averages without float drift here.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._started = time.monotonic()
        self._calls: dict[str, int] = defaultdict(int)
        self._errors: dict[str, int] = defaultdict(int)
        self._latency_ms: dict[str, float] = defaultdict(float)
        self._prompt_tokens = 0
        self._completion_tokens = 0

    def record(self, kind: str, latency_s: float, ok: bool,
               usage: Optional[object] = None) -> None:
        """Record one LLM round trip.

        ``usage`` is the OpenAI SDK usage object (or None); read defensively —
        local providers (Ollama) may omit it or return partial fields.
        """
        with self._lock:
            self._calls[kind] += 1
            if not ok:
                self._errors[kind] += 1
            self._latency_ms[kind] += latency_s * 1000.0
            if usage is not None:
                self._prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
                self._completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)

    def snapshot(self) -> dict:
        with self._lock:
            kinds = sorted(set(self._calls) | set(self._errors))
            return {
                "uptime_s": round(time.monotonic() - self._started, 1),
                "llm_calls": {k: self._calls[k] for k in kinds},
                "llm_errors": {k: self._errors[k] for k in kinds},
                "llm_latency_ms_total": {k: round(self._latency_ms[k], 1) for k in kinds},
                "tokens": {"prompt": self._prompt_tokens,
                           "completion": self._completion_tokens},
            }

    def reset(self) -> None:
        """Test helper — wipe counters, keep start time."""
        with self._lock:
            self._calls.clear()
            self._errors.clear()
            self._latency_ms.clear()
            self._prompt_tokens = 0
            self._completion_tokens = 0


METRICS = Metrics()
