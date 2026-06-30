# -*- coding: utf-8 -*-
"""Shared fixtures: TestClient + LLM mocking.

The OpenAI client is module-level lazy (`pora_llm._client`). We don't make real
network calls in tests — instead we set `LLM_API_KEY` to enable `llm_enabled()`
and monkey-patch `pora_llm._chat` (the single LLM choke point) to return scripted
JSON or text per-test.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# allow `import main`, `import brain`, `import pora_llm` from tests/
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def client():
    """FastAPI TestClient. Importing `main` triggers Categorizer.fit() once."""
    from fastapi.testclient import TestClient
    from main import app
    return TestClient(app)


@pytest.fixture
def enable_llm(monkeypatch):
    """Force `llm_enabled()` to True so endpoints take the LLM path."""
    import pora_llm
    monkeypatch.setattr(pora_llm, "API_KEY", "test-key")
    return pora_llm


@pytest.fixture
def mock_chat(monkeypatch):
    """Replace `pora_llm._chat` with a scripted responder.

    Usage:
        mock_chat({"section": "produce"})              # single value reused
        mock_chat(["resp1", "resp2"])                  # consumed in order
        mock_chat(callable)                            # full control: (system, user, **kw) -> str|None
    """
    import pora_llm

    def install(script):
        if callable(script):
            responder = script
        elif isinstance(script, list):
            queue = list(script)

            def responder(system, user, **kw):
                return queue.pop(0) if queue else None
        else:
            value = script

            def responder(system, user, **kw):
                import json as _json
                return value if isinstance(value, str) else _json.dumps(value)

        monkeypatch.setattr(pora_llm, "_chat", responder)
        # also force API_KEY non-empty so llm_enabled() is True
        monkeypatch.setattr(pora_llm, "API_KEY", "test-key")
        return responder

    return install
