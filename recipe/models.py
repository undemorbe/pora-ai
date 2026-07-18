# -*- coding: utf-8 -*-
"""Wire shapes for the recipe feature.

``Recipe`` is what ``/v1/parse-recipe`` returns, so field names here are part
of the Go contract — renaming one is a breaking change.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

import constants as C


class Ingredient(BaseModel):
    raw: str                                    # verbatim line from the page
    name: Optional[str] = None                  # canonical food noun
    qty: Optional[float] = None
    unit: Optional[str] = None
    section: str = C.DEFAULT_FALLBACK_SECTION   # store section key


class Recipe(BaseModel):
    title: Optional[str] = None
    ingredients: list[Ingredient] = Field(default_factory=list)
    source: str = "none"                        # jsonld | parser | llm | none


class LLMRecipeResponse(BaseModel):
    """Shape the LLM must return in tier 3 (validated by _chat_model).

    ``ingredients`` stays ``list[dict]`` on purpose: the anti-hallucination
    pass and the section tagger both work on dicts, and ``Recipe`` enforces
    the full shape at the end of the pipeline.
    """
    title: Optional[str] = None
    ingredients: list[dict] = Field(default_factory=list)


# JSON schema handed to the provider's structured-output mode.
LLM_RECIPE_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["title", "ingredients"],
    "properties": {
        "title": {"type": ["string", "null"]},
        "ingredients": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["raw", "name", "qty", "unit"],
            "properties": {"raw": {"type": "string"}, "name": {"type": ["string", "null"]},
                           "qty": {"type": ["number", "null"]}, "unit": {"type": ["string", "null"]}},
        }},
    },
}
