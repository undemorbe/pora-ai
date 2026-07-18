# -*- coding: utf-8 -*-
"""URL → Recipe orchestration: fetch, then three extraction tiers, then tag.

Tier order is by cost, and the LLM is genuinely a last resort — on most sites
it never runs at all.
"""
from __future__ import annotations

from typing import Optional

import brain

from _cache import TTLCache, cache_enabled_from_env

from . import constants as RC
from .extract import extract_recipe_from_text
from .fetch import web_fetch
from .jsonld import extract_jsonld
from .models import Recipe
from .parser import parse_ingredients_html
from .sections import tag_ingredients

# Per-process cache, keyed by everything that changes the output.
_CACHE_ENABLED = cache_enabled_from_env()
_recipe_cache = TTLCache(RC.RECIPE_CACHE_SIZE, RC.RECIPE_CACHE_TTL_S)


def _cache_key(url: str, sections: Optional[list[str]], lang: Optional[str]) -> tuple:
    return ("recipe", url, tuple(sorted(sections or ())), lang or "")


def run_tiers(html: str, text: str) -> dict:
    """Run the three extraction tiers, cheapest first, and return the winner.

      1. ``extract_jsonld``          schema.org JSON-LD — free, exact
      2. ``parse_ingredients_html``  Python parser      — free, fast
      3. ``extract_recipe_from_text`` LLM               — slow, paid

    The result always carries ``source`` naming the tier that produced it
    (``jsonld`` | ``parser`` | ``llm`` | ``none``).
    """
    return extract_jsonld(html) or parse_ingredients_html(html) or extract_recipe_from_text(text)


def parse_recipe(url: str, categorizer: brain.Categorizer,
                 sections: Optional[list[str]] = None,
                 lang: Optional[str] = None) -> Recipe:
    """Fetch ``url`` and return a fully tagged :class:`Recipe`.

    Cached by (url, sections, lang). Failed extractions are not cached, so a
    transient site or LLM outage stays retryable.
    """
    key = _cache_key(url, sections, lang)
    if _CACHE_ENABLED:
        cached = _recipe_cache.get(key)
        if cached is not None:
            return Recipe.model_validate(cached)

    fetched = web_fetch(url, lang=lang)
    data = run_tiers(fetched["html"], fetched["text"])
    tag_ingredients(data.get("ingredients") or [], categorizer, sections)

    recipe = Recipe.model_validate(data)
    if _CACHE_ENABLED and data.get("source") in RC.CACHEABLE_SOURCES:
        _recipe_cache.set(key, recipe.model_dump())
    return recipe
