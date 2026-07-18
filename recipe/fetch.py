# -*- coding: utf-8 -*-
"""Browser-like page fetching and HTML→text reduction.

Recipe sites are hostile to naive clients: some 403 anything without a
realistic User-Agent, some rate-limit, and most bury the recipe under
navigation. This module handles the transport and hands the rest of the
pipeline both the raw HTML (for the structured tiers) and readable text
(for the LLM tier).
"""
from __future__ import annotations

import re
from typing import Optional

import httpx

from . import constants as RC

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_STYLE_RE = re.compile(r"<(script|style|noscript)[^>]*>.*?</\1>", re.S | re.I)
_BOILER_RE = re.compile(r"<(nav|header|footer|aside|form)\b[^>]*>.*?</\1>", re.S | re.I)
_MAIN_CONTENT_RE = re.compile(r"<(main|article)\b[^>]*>(.*?)</\1>", re.S | re.I)
_WS_RE = re.compile(r"\s+")


def html_to_text(html: str) -> str:
    """Strip <script>/<style>, tags, decode common entities, collapse whitespace."""
    s = _SCRIPT_STYLE_RE.sub(" ", html)
    s = _HTML_TAG_RE.sub(" ", s)
    for k, v in RC.HTML_ENTITIES.items():
        s = s.replace(k, v)
    return _WS_RE.sub(" ", s).strip()


def extract_main_content(html: str) -> str:
    """Pick the main content: first <main>/<article>, else strip boilerplate.

    Old table-based layouts have neither, so the boilerplate strip is the
    common path — it is deliberately conservative and keeps everything it is
    not sure about.
    """
    m = _MAIN_CONTENT_RE.search(html)
    if m:
        return m.group(2)
    return _BOILER_RE.sub(" ", html)


def accept_language(lang: Optional[str]) -> str:
    """Build an Accept-Language header from the caller's locale."""
    if not lang:
        return RC.ACCEPT_LANGUAGE_DEFAULT
    lang = lang.split("-")[0].lower()
    return f"{lang},{lang};q=0.9,en;q=0.5"


def web_fetch(url: str, lang: Optional[str] = None,
              timeout: float = RC.FETCH_TIMEOUT_S,
              max_bytes: int = RC.FETCH_MAX_BYTES,
              retries: int = RC.FETCH_RETRIES) -> dict:
    """Fetch a page with realistic headers and retries.

    Returns ``{"url", "status", "html", "text"}``: the final URL after
    redirects, the status code, the truncated raw HTML, and readable text
    from the main content area.

    Raises ``httpx.HTTPError`` if every attempt fails — ``/v1/parse-recipe``
    turns that into a 502.
    """
    headers = {
        "User-Agent": RC.DEFAULT_USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": accept_language(lang),
        "Accept-Encoding": "gzip, deflate",
        "DNT": "1",
        "Upgrade-Insecure-Requests": "1",
    }
    last_exc: Optional[Exception] = None
    with httpx.Client(timeout=timeout, follow_redirects=True, headers=headers) as cli:
        for attempt in range(retries + 1):
            try:
                r = cli.get(url)
                if r.status_code in RC.FETCH_RETRY_STATUSES and attempt < retries:
                    continue
                r.raise_for_status()
                html = r.text[:max_bytes]
                return {"url": str(r.url), "status": r.status_code,
                        "html": html, "text": html_to_text(extract_main_content(html))}
            except httpx.HTTPError as e:
                last_exc = e
                if attempt == retries:
                    raise
    assert last_exc is not None
    raise last_exc
