# -*- coding: utf-8 -*-
"""Tier 3 — LLM extraction with an anti-hallucination guard.

Only reached when JSON-LD and the Python parser both come up empty: it is the
slow, paid path. Two safeguards make it trustworthy:

  * the model sees a *window* around the densest quantity+unit cluster, not
    the first N chars of the page (navigation lives there);
  * every ingredient it returns must actually appear in the source text,
    otherwise it is dropped as invented.
"""
from __future__ import annotations

import re
from typing import Optional

import constants as C
import pora_llm as llm

from . import constants as RC
from .models import LLM_RECIPE_SCHEMA, LLMRecipeResponse

_WS_RE = re.compile(r"\s+")
_QTY_UNIT_RE = re.compile(RC.QTY_UNIT_PATTERN, re.I)


def _norm(s: str) -> str:
    return _WS_RE.sub(" ", s.lower()).strip()


# --------------------------------------------------------------------------
# Window selection
# --------------------------------------------------------------------------
def recipe_window(text: str, cap: int = RC.LLM_TEXT_CAP) -> str:
    """Pick the cap-sized slice of ``text`` most likely to hold the ingredients.

    Naively sending ``text[:cap]`` breaks on real pages: old table-based
    layouts (and any site without <main>/<article>) put navigation first and
    the ingredient block thousands of characters down — the model then reads a
    menu and correctly reports "no recipe", after burning a full inference.

    Heuristic: ingredients are where quantity+unit pairs cluster ("550 г",
    "2 cups"). Slide a cap-sized window over those hits and keep the densest
    one, with a little lead-in so the "Продукты"/"Ingredients" heading and the
    first line survive. No hits (e.g. a prose page) → fall back to the head.
    """
    if len(text) <= cap:
        return text
    hits = [m.start() for m in _QTY_UNIT_RE.finditer(text)]
    if not hits:
        return text[:cap]

    lead = int(cap * RC.RECIPE_WINDOW_LEAD)
    best_start, best_count = 0, 0
    for h in hits:
        start = max(0, h - lead)
        end = start + cap
        count = sum(1 for x in hits if start <= x < end)
        if count > best_count:
            best_start, best_count = start, count
    return text[best_start:best_start + cap]


# --------------------------------------------------------------------------
# Anti-hallucination validation
# --------------------------------------------------------------------------
def _build_synonym_lookup() -> dict:
    """word → tuple of alternatives, both directions, built once at import."""
    lookup: dict[str, tuple[str, ...]] = {}
    for a, b in RC.INGREDIENT_SYNONYM_PAIRS:
        lookup[a] = lookup.get(a, ()) + (b,)
        lookup[b] = lookup.get(b, ()) + (a,)
    return lookup


_SYNONYMS = _build_synonym_lookup()


def name_in_source(name: str, haystack: str) -> bool:
    """Match an ingredient name against the source with graceful degradation.

    1. verbatim substring;
    2. singular-strip: "eggs" matches a source that only has "egg";
    3. cross-lingual synonym bridge — the LLM sometimes translates the name.
    """
    if len(name) < RC.MIN_SYNONYM_NAME_LEN:
        return False
    if name in haystack:
        return True
    if name.endswith("s") and len(name) >= 4 and name[:-1] in haystack:
        return True
    return any(alt in haystack for alt in _SYNONYMS.get(name, ()))


def validate_against_source(ingredients: list[dict], source_text: str) -> list[dict]:
    """Drop ingredients that do not appear in the source.

    The LLM may invent plausible-looking ingredients. We require the full
    ``raw`` line OR the ``name`` to appear in the page — verbatim,
    singular-stripped, or through the RU↔EN synonym bridge.
    """
    haystack = _norm(source_text)
    kept: list[dict] = []
    for ing in ingredients:
        raw = _norm(ing.get("raw") or "")
        name = _norm(ing.get("name") or "")
        if raw and raw in haystack:
            kept.append(ing)
        elif name and name_in_source(name, haystack):
            kept.append(ing)
    return kept


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def extract_recipe_from_text(text: str) -> dict:
    """Ask the LLM for the ingredients, then verify them against the page."""
    resp = llm._chat_model(
        C.RECIPE_EXTRACT_SYSTEM, recipe_window(text), LLMRecipeResponse,
        examples=C.FEW_SHOT_EXAMPLES.get("recipe_extract"),
        response_format={"type": "json_schema",
                         "json_schema": {"name": "recipe", "strict": True,
                                         "schema": LLM_RECIPE_SCHEMA}},
        model_kind=C.LLM_MODEL_ROUTING["recipe_extract"],
    )
    if resp is None:
        return {"title": None, "ingredients": [], "source": RC.SOURCE_NONE}
    validated = validate_against_source(resp.ingredients, text)
    return {
        "title": resp.title,
        "ingredients": validated,
        "source": RC.SOURCE_LLM if validated else RC.SOURCE_NONE,
    }
