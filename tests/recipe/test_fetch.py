# -*- coding: utf-8 -*-
"""Fetching and HTML→text reduction (recipe.fetch)."""
from __future__ import annotations

import httpx as _httpx
import pytest

import recipe
from recipe import fetch as rf
from recipe.fetch import (accept_language, extract_main_content,
                          html_to_text, web_fetch)


# --------------------------------------------------------------------------
# HTML stripping (anti-hallucination prep)
# --------------------------------------------------------------------------
class TestHtmlToText:
    def test_strips_script_and_style(self):
        html = "<html><script>var x=1;</script><style>p{color:red}</style><p>Hello</p></html>"
        text = rf.html_to_text(html)
        assert "var x" not in text
        assert "color:red" not in text
        assert "Hello" in text

    def test_decodes_entities(self):
        assert "salt & pepper" in rf.html_to_text("<p>salt &amp; pepper</p>")

    def test_collapses_whitespace(self):
        assert rf.html_to_text("<p>a\n\n\n  b</p>") == "a b"

    def test_handles_empty(self):
        assert rf.html_to_text("") == ""


# --------------------------------------------------------------------------
# validate_against_source — anti-hallucination guard
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# extract_jsonld
# --------------------------------------------------------------------------
class TestExtractMainContent:
    def test_picks_article(self):
        html = "<html><body><nav>menu</nav><article>real content</article><footer>x</footer></body></html>"
        assert "real content" in rf.extract_main_content(html)

    def test_picks_main(self):
        html = "<html><body><header>x</header><main>recipe content</main></body></html>"
        out = rf.extract_main_content(html)
        assert "recipe content" in out
        # main wins over nav/header stripping
        assert "recipe content" in rf.html_to_text(out)

    def test_falls_back_to_body_minus_boilerplate(self):
        html = "<html><body><nav>menu items</nav><div>content X</div><footer>fine print</footer></body></html>"
        text = rf.html_to_text(rf.extract_main_content(html))
        assert "content X" in text
        assert "menu items" not in text
        assert "fine print" not in text


class TestWebFetch:
    def test_returns_url_status_html_text(self, monkeypatch):
        class _Resp:
            status_code = 200
            url = "http://example.com/r"
            text = "<html><body><main>Hello world</main></body></html>"

            def raise_for_status(self):
                pass

        class _Client:
            def __init__(self, *a, **kw): pass
            def __enter__(self): return self
            def __exit__(self, *a): pass
            def get(self, url): return _Resp()

        monkeypatch.setattr(rf.httpx, "Client", _Client)
        out = rf.web_fetch("http://example.com/r", lang="ru")
        assert out["status"] == 200
        assert "Hello world" in out["text"]
        assert out["html"].startswith("<html>")

    def test_retries_on_5xx_then_succeeds(self, monkeypatch):
        calls = {"n": 0}

        class _Bad:
            status_code = 503
            url = "http://x"
            text = ""

            def raise_for_status(self):
                raise _httpx.HTTPStatusError("bad", request=None, response=None)

        class _Good:
            status_code = 200
            url = "http://x"
            text = "<main>OK</main>"

            def raise_for_status(self):
                pass

        class _Client:
            def __init__(self, *a, **kw): pass
            def __enter__(self): return self
            def __exit__(self, *a): pass
            def get(self, url):
                calls["n"] += 1
                return _Bad() if calls["n"] < 2 else _Good()

        monkeypatch.setattr(rf.httpx, "Client", _Client)
        out = rf.web_fetch("http://x", retries=2)
        assert out["status"] == 200
        assert calls["n"] >= 2

    def test_propagates_after_max_retries(self, monkeypatch):
        class _Client:
            def __init__(self, *a, **kw): pass
            def __enter__(self): return self
            def __exit__(self, *a): pass
            def get(self, url):
                raise _httpx.RequestError("connection refused")

        monkeypatch.setattr(rf.httpx, "Client", _Client)
        with pytest.raises(_httpx.HTTPError):
            rf.web_fetch("http://x", retries=1)

    def test_accept_language_built_from_lang(self):
        # private helper but worth a smoke test — verifies lang flows into header
        assert rf.accept_language("ru").startswith("ru")
        assert rf.accept_language(None).startswith("en")
        assert rf.accept_language("pt-BR").startswith("pt")
