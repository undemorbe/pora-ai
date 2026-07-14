# -*- coding: utf-8 -*-
"""Tests for _metrics.Metrics and the /metrics endpoint."""
from __future__ import annotations

import pytest

from _metrics import METRICS, Metrics

import pora_llm as ai


@pytest.fixture(autouse=True)
def _reset_metrics():
    METRICS.reset()
    yield
    METRICS.reset()


class TestMetricsUnit:
    def test_empty_snapshot(self):
        m = Metrics()
        snap = m.snapshot()
        assert snap["llm_calls"] == {}
        assert snap["tokens"] == {"prompt": 0, "completion": 0}
        assert snap["uptime_s"] >= 0

    def test_record_success_with_usage(self):
        m = Metrics()

        class _Usage:
            prompt_tokens = 120
            completion_tokens = 30

        m.record("fast", 0.25, ok=True, usage=_Usage())
        snap = m.snapshot()
        assert snap["llm_calls"] == {"fast": 1}
        assert snap["llm_errors"] == {"fast": 0}
        assert snap["llm_latency_ms_total"]["fast"] == 250.0
        assert snap["tokens"] == {"prompt": 120, "completion": 30}

    def test_record_failure_counts_error(self):
        m = Metrics()
        m.record("main", 1.0, ok=False)
        snap = m.snapshot()
        assert snap["llm_calls"] == {"main": 1}
        assert snap["llm_errors"] == {"main": 1}

    def test_usage_read_defensively(self):
        m = Metrics()

        class _Partial:
            prompt_tokens = None   # Ollama-style partial usage

        m.record("fast", 0.1, ok=True, usage=_Partial())
        assert m.snapshot()["tokens"] == {"prompt": 0, "completion": 0}

    def test_reset(self):
        m = Metrics()
        m.record("fast", 0.1, ok=True)
        m.reset()
        assert m.snapshot()["llm_calls"] == {}


class TestChatRecordsMetrics:
    def test_success_recorded_with_kind_and_usage(self, monkeypatch):
        monkeypatch.setattr(ai, "API_KEY", "test-key")
        monkeypatch.setattr(ai, "_transient_llm_errors", lambda: ())

        class _Usage:
            prompt_tokens = 10
            completion_tokens = 5

        class _Msg:
            content = "ok"

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]
            usage = _Usage()

        class _Cli:
            class chat:
                class completions:
                    @staticmethod
                    def create(**kw):
                        return _Resp()

        monkeypatch.setattr(ai, "client", lambda: _Cli())
        ai._chat("s", "u", model_kind="fast")
        snap = METRICS.snapshot()
        assert snap["llm_calls"] == {"fast": 1}
        assert snap["llm_errors"] == {"fast": 0}
        assert snap["tokens"] == {"prompt": 10, "completion": 5}

    def test_terminal_failure_recorded_as_error(self, monkeypatch):
        monkeypatch.setattr(ai, "API_KEY", "test-key")
        monkeypatch.setattr(ai, "_transient_llm_errors", lambda: ())

        class _Cli:
            class chat:
                class completions:
                    @staticmethod
                    def create(**kw):
                        raise ValueError("boom")

        monkeypatch.setattr(ai, "client", lambda: _Cli())
        assert ai._chat("s", "u") is None
        snap = METRICS.snapshot()
        assert snap["llm_calls"] == {"main": 1}
        assert snap["llm_errors"] == {"main": 1}

    def test_disabled_llm_records_nothing(self):
        assert ai._chat("s", "u") is None  # API_KEY empty in tests
        assert METRICS.snapshot()["llm_calls"] == {}


class TestMetricsEndpoint:
    def test_shape(self, client):
        r = client.get("/metrics")
        assert r.status_code == 200
        body = r.json()
        assert {"llm", "caches"} <= set(body)
        assert {"uptime_s", "llm_calls", "llm_errors",
                "llm_latency_ms_total", "tokens"} <= set(body["llm"])
        assert {"categorize", "recipe"} <= set(body["caches"])
        for cache in body["caches"].values():
            assert {"size", "hits", "misses"} <= set(cache)

    def test_reflects_recorded_calls(self, client):
        METRICS.record("fast", 0.2, ok=True)
        body = client.get("/metrics").json()
        assert body["llm"]["llm_calls"] == {"fast": 1}
