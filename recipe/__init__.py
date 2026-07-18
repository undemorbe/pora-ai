# -*- coding: utf-8 -*-
"""Recipe fetcher — URL → structured, section-tagged ingredient list.

Self-contained feature. Everything about *getting a recipe off a web page*
lives here; the rest of Pora only needs :func:`parse_recipe`.

Layout
------
``constants``  feature-owned knobs (fetch limits, parser thresholds, synonyms)
``models``     ``Recipe`` / ``Ingredient`` — part of the Go wire contract
``fetch``      browser-like HTTP + HTML→text reduction
``jsonld``     tier 1 — schema.org JSON-LD        (free, exact)
``parser``     tier 2 — pure-Python HTML parser   (free, heuristic)
``extract``    tier 3 — LLM + anti-hallucination  (slow, paid)
``sections``   store-section tagging for the extracted ingredients
``pipeline``   orchestration + per-URL cache

Tiers run cheapest-first and stop at the first confident result, so the LLM
never runs for sites that publish JSON-LD or parseable markup.

Dependency direction is one-way: ``recipe`` → ``pora_llm`` (for the LLM
plumbing) and ``recipe`` → ``brain`` (for the fast classifier). Neither
imports ``recipe`` back.

    >>> from recipe import parse_recipe
    >>> parse_recipe("https://example.com/borsch", categorizer).source
    'parser'
"""
from __future__ import annotations

from .extract import (extract_recipe_from_text, recipe_window,
                      validate_against_source)
from .fetch import extract_main_content, html_to_text, web_fetch
from .jsonld import extract_jsonld
from .models import Ingredient, Recipe
from .parser import parse_ingredients_html, split_quantity
from .pipeline import parse_recipe, run_tiers
from .sections import tag_ingredients

__all__ = [
    # main entry point
    "parse_recipe",
    # wire shapes
    "Recipe", "Ingredient",
    # tiers, individually usable and testable
    "run_tiers", "extract_jsonld", "parse_ingredients_html",
    "extract_recipe_from_text",
    # building blocks
    "web_fetch", "html_to_text", "extract_main_content",
    "split_quantity", "recipe_window", "validate_against_source",
    "tag_ingredients",
]
