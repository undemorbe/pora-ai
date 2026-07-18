# -*- coding: utf-8 -*-
"""Tier 1 — schema.org JSON-LD. Free, exact, and the fastest path.

Sites that publish `<script type="application/ld+json">` with a Recipe node
hand us the ingredient list verbatim; no heuristics and no LLM needed.
"""
from __future__ import annotations

import json
import re
from typing import Optional

from . import constants as RC

_LD_JSON_RE = re.compile(
    r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>', re.S | re.I)


def _iter_recipe_nodes(data):
    """Yield every Recipe node, walking @graph containers and @type arrays.

    Publishers nest the recipe in wildly different ways — top level, inside
    ``@graph``, or with ``@type: ["Recipe", "Article"]`` — so the walk is
    iterative and tolerant.
    """
    stack = [data]
    while stack:
        node = stack.pop()
        if isinstance(node, list):
            stack.extend(node)
        elif isinstance(node, dict):
            if "@graph" in node:
                g = node["@graph"]
                stack.extend(g if isinstance(g, list) else [g])
            t = node.get("@type")
            for x in (t if isinstance(t, list) else [t]):
                if x and str(x).lower() == "recipe":
                    yield node


def extract_jsonld(html: str) -> Optional[dict]:
    """Extract a recipe from JSON-LD markup, or None when the page has none."""
    for block in _LD_JSON_RE.findall(html):
        try:
            data = json.loads(block.strip())
        except Exception:
            continue                              # one bad block must not kill the page
        for node in _iter_recipe_nodes(data):
            ings = node.get("recipeIngredient") or node.get("ingredients")
            if ings:
                ings = [ings] if isinstance(ings, str) else ings
                return {
                    "title": node.get("name"),
                    "ingredients": [{"raw": str(i).strip(), "name": None,
                                     "qty": None, "unit": None} for i in ings],
                    "source": RC.SOURCE_JSONLD,
                }
    return None
