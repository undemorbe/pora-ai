# -*- coding: utf-8 -*-
"""Tier 1 — schema.org JSON-LD (recipe.jsonld)."""
from __future__ import annotations

from recipe.jsonld import extract_jsonld


class TestExtractJsonld:
    def test_simple_recipe(self):
        html = '<script type="application/ld+json">{"@type":"Recipe","name":"Carbonara","recipeIngredient":["Spaghetti 400g","Eggs 4"]}</script>'
        out = extract_jsonld(html)
        assert out["title"] == "Carbonara"
        assert len(out["ingredients"]) == 2
        assert out["source"] == "jsonld"

    def test_walks_graph(self):
        html = '<script type="application/ld+json">{"@graph":[{"@type":"WebPage"},{"@type":"Recipe","name":"X","recipeIngredient":["A"]}]}</script>'
        out = extract_jsonld(html)
        assert out["title"] == "X"

    def test_type_array(self):
        html = '<script type="application/ld+json">{"@type":["Recipe","Article"],"name":"Y","recipeIngredient":["B"]}</script>'
        out = extract_jsonld(html)
        assert out["title"] == "Y"

    def test_no_recipe_returns_none(self):
        html = '<script type="application/ld+json">{"@type":"Article","name":"X"}</script>'
        assert extract_jsonld(html) is None

    def test_malformed_json_skipped(self):
        html = '<script type="application/ld+json">{not json</script>'
        assert extract_jsonld(html) is None

    def test_ingredients_as_string_normalized_to_list(self):
        html = '<script type="application/ld+json">{"@type":"Recipe","name":"X","recipeIngredient":"Only one"}</script>'
        out = extract_jsonld(html)
        assert len(out["ingredients"]) == 1


# --------------------------------------------------------------------------
# extract_recipe_from_text — LLM mocked
# --------------------------------------------------------------------------
