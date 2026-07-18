# -*- coding: utf-8 -*-
"""Assign a store section to every extracted ingredient.

Two modes, both ending with one section key per ingredient:

  * default taxonomy — fast classifier, escalating only its weak guesses to
    the LLM (same rule as ``/v1/categorize``);
  * caller-supplied taxonomy — the fast classifier knows nothing about those
    keys, so everything goes to the LLM in one batched call.
"""
from __future__ import annotations

from typing import Optional

import brain
import constants as C
import pora_llm as llm


def tag_default_taxonomy(labels: list[str], categorizer: brain.Categorizer) -> list[str]:
    """Fast classifier first, LLM only for the labels it is unsure about.

    Ingredient lines are noisy ("1 (16 ounce) container Cool Whip") and often
    unlike the training data, so without the escalation the fast classifier
    ships confident-looking nonsense. Weak labels are escalated together in a
    single batched call, and the fast guess survives whenever the LLM is
    disabled or fails.
    """
    fast = categorizer.predict_batch(labels) if labels else []
    result = [sec if label else C.DEFAULT_FALLBACK_SECTION
              for label, (sec, _conf) in zip(labels, fast)]

    weak = [i for i, (label, (_sec, conf)) in enumerate(zip(labels, fast))
            if label and conf < C.FAST_ESCALATE_CONF_BELOW]
    if not weak or not llm.llm_enabled():
        return result

    for i, (sec, conf) in zip(weak, llm.categorize_llm_batch([labels[i] for i in weak])):
        if conf >= C.LLM_CONF_HIGH:      # keep the fast guess if the LLM failed
            result[i] = sec
    return result


def tag_custom_taxonomy(labels: list[str], sections: list[str]) -> list[str]:
    """Batched LLM tagging against a caller-supplied section list."""
    fallback = llm._fallback_section(sections)
    tagged = llm.categorize_llm_batch(labels, sections)
    return [sec if label else fallback
            for label, (sec, _conf) in zip(labels, tagged)]


def tag_ingredients(ingredients: list[dict], categorizer: brain.Categorizer,
                    sections: Optional[list[str]] = None) -> None:
    """Set ``section`` on every ingredient in place."""
    if not ingredients:
        return
    labels = [ing.get("name") or ing.get("raw") or "" for ing in ingredients]
    tagged = (tag_custom_taxonomy(labels, sections) if sections
              else tag_default_taxonomy(labels, categorizer))
    for ing, sec in zip(ingredients, tagged):
        ing["section"] = sec
