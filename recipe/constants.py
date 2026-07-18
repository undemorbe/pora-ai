# -*- coding: utf-8 -*-
"""Knobs owned by the recipe feature.

Split out of the service-wide ``constants`` so the feature can be tuned (or
lifted into another project) without touching global config. Anything here is
about *getting a recipe off a web page*; anything shared with the rest of Pora
(store sections, LLM plumbing, prompts) stays in the root ``constants``.
"""
from __future__ import annotations

# ==========================================================================
# HTTP fetch (fetch.web_fetch)
# ==========================================================================
FETCH_TIMEOUT_S = 20.0
FETCH_RETRIES = 2                   # total attempts = FETCH_RETRIES + 1
FETCH_MAX_BYTES = 400_000
FETCH_RETRY_STATUSES: frozenset[int] = frozenset({429, 500, 502, 503, 504})
ACCEPT_LANGUAGE_DEFAULT = "en-US,en;q=0.9"

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 PoraBot/2.0"
)

HTML_ENTITIES: dict[str, str] = {
    "&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"',
    "&#39;": "'", "&apos;": "'", "&nbsp;": " ",
}

# ==========================================================================
# Tier 2 — pure-Python parser (parser.parse_ingredients_html)
# ==========================================================================
# CSS class / id / itemprop fragments that mark an ingredient element.
PARSER_INGREDIENT_CLASS_MARKERS: tuple[str, ...] = (
    "ingredient", "ingr", "recipe-item", "product-item", "sostav",
)
# Headings that introduce the ingredient block (lowercase, substring match).
PARSER_INGREDIENT_HEADINGS: tuple[str, ...] = (
    "ингредиент", "продукты", "состав", "нам понадобится", "потребуется",
    "ingredient", "you will need", "what you need",
)
PARSER_HEADING_MAX_LEN = 60         # a heading is short; a paragraph is not
PARSER_MIN_LINE_LEN = 2
PARSER_MAX_LINE_LEN = 200           # longer ⇒ a prose block, not an ingredient
PARSER_MIN_INGREDIENTS = 2          # fewer ⇒ not confident, hand over to the LLM
PARSER_MAX_INGREDIENTS = 60         # cap runaway matches
# Share of candidate lines that must carry a quantity+unit for an untrusted
# (heuristic) strategy to be believed. Guards against parsing a nav menu.
PARSER_MIN_QTY_RATIO = 0.5

# ==========================================================================
# Tier 3 — LLM extraction (extract.py)
# ==========================================================================
# Quantity+unit signature used to locate the ingredient block inside a long
# page (see extract._recipe_window) and to sanity-check parser output.
# Feeding the LLM the first N chars of a page is wrong: old-school layouts put
# navigation first and the ingredients thousands of chars down.
QTY_UNIT_PATTERN = (
    r"\d+[.,]?\d*\s*"
    r"(?:г|кг|мл|л|шт|зубчик|щепотк|ст\.?\s*л|ч\.?\s*л|стакан|пучок|банк|"
    r"g|kg|ml|l|oz|lb|cup|cups|tbsp|tsp|pcs|clove|pinch|can)\b"
)
# Fraction of the window kept as context BEFORE the first ingredient hit
# (the block usually starts with a heading like "Продукты" / "Ingredients").
RECIPE_WINDOW_LEAD = 0.25

# Chars of page text fed to the LLM. Deliberately modest: the window targets
# the ingredient block precisely, so a bigger cap buys nothing but tokens. It
# also has to fit the model's context — Cyrillic costs ~2 chars/token, so
# 4000 chars ≈ 2000 tokens, leaving room for the few-shot block and the system
# prompt inside a 4096-token default (local Ollama).
LLM_TEXT_CAP = 4_000

# Cross-lingual bridge for validate_against_source. The LLM sometimes
# normalizes or translates the name it extracts (page says "молоко", model
# returns "milk") — a verbatim check would wrongly drop a real ingredient.
# Each pair is bidirectional.
INGREDIENT_SYNONYM_PAIRS: tuple[tuple[str, str], ...] = (
    ("молоко", "milk"), ("яйца", "eggs"), ("яйцо", "egg"),
    ("сыр", "cheese"), ("мука", "flour"), ("сахар", "sugar"),
    ("соль", "salt"), ("масло", "butter"), ("вода", "water"),
    ("курица", "chicken"), ("говядина", "beef"), ("свинина", "pork"),
    ("рис", "rice"), ("паста", "pasta"), ("спагетти", "spaghetti"),
    ("хлеб", "bread"), ("помидоры", "tomatoes"), ("помидор", "tomato"),
    ("лук", "onion"), ("чеснок", "garlic"), ("морковь", "carrot"),
    ("картофель", "potato"), ("перец", "pepper"), ("сливки", "cream"),
    ("сметана", "sour cream"), ("творог", "cottage cheese"),
    ("бекон", "bacon"), ("лосось", "salmon"), ("креветки", "shrimp"),
    ("грибы", "mushrooms"), ("гриб", "mushroom"), ("лимон", "lemon"),
    ("мёд", "honey"), ("орехи", "nuts"), ("шоколад", "chocolate"),
    ("корица", "cinnamon"), ("ваниль", "vanilla"), ("дрожжи", "yeast"),
    ("уксус", "vinegar"), ("капуста", "cabbage"),
)
MIN_SYNONYM_NAME_LEN = 3            # shorter names are too generic to match on

# ==========================================================================
# Pipeline
# ==========================================================================
RECIPE_CACHE_SIZE = 256
RECIPE_CACHE_TTL_S = 86_400          # 24 hours
# Extraction outcomes worth caching. "none" is excluded on purpose: a failed
# parse is often transient (site hiccup, LLM outage) and must be retryable.
CACHEABLE_SOURCES: frozenset[str] = frozenset({"jsonld", "parser", "llm"})

SOURCE_JSONLD = "jsonld"
SOURCE_PARSER = "parser"
SOURCE_LLM = "llm"
SOURCE_NONE = "none"
