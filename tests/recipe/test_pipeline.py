# -*- coding: utf-8 -*-
"""URL → Recipe orchestration (recipe.pipeline)."""
from __future__ import annotations

import pytest

import brain
import recipe
import pora_llm
from recipe import pipeline as rp
from recipe.pipeline import _recipe_cache


def _fake_fetch(html: str, url: str = "http://example.com/r"):
    """Build a web_fetch return shape from a chunk of HTML."""
    return {"url": url, "status": 200, "html": html,
            "text": recipe.html_to_text(recipe.extract_main_content(html))}


class TestParseRecipe:
    def test_jsonld_path_no_llm_call(self, monkeypatch):
        html = '<html><script type="application/ld+json">{"@type":"Recipe","name":"Carbonara","recipeIngredient":["Spaghetti 400g","Eggs 4"]}</script></html>'
        monkeypatch.setattr(rp, "web_fetch", lambda *a, **kw: _fake_fetch(html))
        # _chat must NOT be called for the JSON-LD path
        monkeypatch.setattr(pora_llm, "_chat",
                            lambda *a, **kw: (_ for _ in ()).throw(AssertionError("_chat called on JSON-LD path")))

        import brain
        recipe = rp.parse_recipe("http://example.com/r", brain.Categorizer().fit())
        assert recipe.title == "Carbonara"
        assert recipe.source == "jsonld"
        assert len(recipe.ingredients) == 2
        for ing in recipe.ingredients:
            assert ing.section in brain.SECTIONS

    def test_llm_fallback_path(self, monkeypatch, mock_chat):
        html = "<html><body>Recipe: take Spaghetti 400g and Eggs 4.</body></html>"
        monkeypatch.setattr(rp, "web_fetch", lambda *a, **kw: _fake_fetch(html))
        mock_chat({
            "title": "Pasta",
            "ingredients": [{"raw": "Spaghetti 400g", "name": "spaghetti", "qty": 400, "unit": "g"}],
        })
        import brain
        recipe = rp.parse_recipe("http://x", brain.Categorizer().fit())
        assert recipe.source == "llm"
        assert recipe.title == "Pasta"
        assert recipe.ingredients[0].section in brain.SECTIONS

    def test_parser_tier_used_when_no_jsonld_and_llm_never_called(self, monkeypatch):
        # No JSON-LD, but the markup is parseable → tier 2 wins, no LLM cost.
        html = """<html><head><title>Оладьи</title></head><body>
          <li class="ingredient">Мука - 200 г</li>
          <li class="ingredient">Молоко - 250 мл</li>
          <li class="ingredient">Яйцо - 2 шт</li></body></html>"""
        monkeypatch.setattr(rp, "web_fetch", lambda *a, **kw: _fake_fetch(html))
        monkeypatch.setattr(pora_llm, "_chat",
                            lambda *a, **kw: (_ for _ in ()).throw(AssertionError("LLM must not run")))
        import brain
        recipe = rp.parse_recipe("http://x", brain.Categorizer().fit())
        assert recipe.source == "parser"
        assert [i.raw for i in recipe.ingredients] == [
            "Мука - 200 г", "Молоко - 250 мл", "Яйцо - 2 шт"]
        assert recipe.ingredients[0].qty == 200.0
        assert all(i.section in brain.SECTIONS for i in recipe.ingredients)

    def test_jsonld_still_wins_over_parser(self, monkeypatch):
        html = ('<html><script type="application/ld+json">{"@type":"Recipe","name":"JL",'
                '"recipeIngredient":["Сахар - 100 г"]}</script>'
                '<li class="ingredient">Мука - 200 г</li>'
                '<li class="ingredient">Соль - 5 г</li>'
                '<li class="ingredient">Вода - 1 л</li></html>')
        monkeypatch.setattr(rp, "web_fetch", lambda *a, **kw: _fake_fetch(html))
        import brain
        recipe = rp.parse_recipe("http://x", brain.Categorizer().fit())
        assert recipe.source == "jsonld"
        assert recipe.title == "JL"

    def test_llm_tier_only_when_parser_finds_nothing(self, monkeypatch, mock_chat):
        html = "<html><body><p>Просто текст про еду: возьмите Мука 200 г и смешайте.</p></body></html>"
        monkeypatch.setattr(rp, "web_fetch", lambda *a, **kw: _fake_fetch(html))
        mock_chat('{"title": "Из текста", "ingredients": '
                  '[{"raw":"Мука 200 г","name":"Мука","qty":200,"unit":"г"}]}')
        import brain
        recipe = rp.parse_recipe("http://x", brain.Categorizer().fit())
        assert recipe.source == "llm"

    def test_low_confidence_ingredients_escalate_to_llm(self, monkeypatch, mock_chat):
        # "Cool Whip" is nothing like the RU/EN training data → the fast
        # classifier is unsure → the LLM must be asked instead of shipping a
        # low-confidence guess.
        html = ('<html><script type="application/ld+json">{"@type":"Recipe","name":"X",'
                '"recipeIngredient":["молоко","Cool Whip"]}</script></html>')
        monkeypatch.setattr(rp, "web_fetch", lambda *a, **kw: _fake_fetch(html))

        seen = {"names": None}

        def responder(system, user, **kw):
            seen["names"] = user
            return '{"results": [{"name": "Cool Whip", "section": "dairy"}]}'

        mock_chat(responder)
        import brain
        recipe = rp.parse_recipe("http://x", brain.Categorizer().fit())
        by_raw = {i.raw: i.section for i in recipe.ingredients}
        assert by_raw["Cool Whip"] == "dairy"          # LLM answer used
        assert by_raw["молоко"] == "dairy"             # confident fast answer kept
        # only the weak label was sent — the confident one costs no tokens
        assert "Cool Whip" in seen["names"]
        assert "молоко" not in seen["names"]

    def test_no_escalation_when_fast_classifier_is_confident(self, monkeypatch):
        html = ('<html><script type="application/ld+json">{"@type":"Recipe","name":"X",'
                '"recipeIngredient":["молоко"]}</script></html>')
        monkeypatch.setattr(rp, "web_fetch", lambda *a, **kw: _fake_fetch(html))
        monkeypatch.setattr(pora_llm, "_chat",
                            lambda *a, **kw: (_ for _ in ()).throw(AssertionError("must not call LLM")))
        import brain
        recipe = rp.parse_recipe("http://x", brain.Categorizer().fit())
        assert recipe.ingredients[0].section == "dairy"

    def test_escalation_skipped_when_llm_disabled(self, monkeypatch):
        # LLM off → keep the fast guess, never raise
        html = ('<html><script type="application/ld+json">{"@type":"Recipe","name":"X",'
                '"recipeIngredient":["Cool Whip"]}</script></html>')
        monkeypatch.setattr(rp, "web_fetch", lambda *a, **kw: _fake_fetch(html))
        monkeypatch.setattr(pora_llm, "API_KEY", "")
        import brain
        recipe = rp.parse_recipe("http://x", brain.Categorizer().fit())
        assert recipe.ingredients[0].section in brain.SECTIONS

    def test_failed_llm_escalation_keeps_fast_guess(self, monkeypatch, mock_chat):
        html = ('<html><script type="application/ld+json">{"@type":"Recipe","name":"X",'
                '"recipeIngredient":["Cool Whip"]}</script></html>')
        monkeypatch.setattr(rp, "web_fetch", lambda *a, **kw: _fake_fetch(html))
        mock_chat("garbage not json")     # LLM fails → fall back to fast guess
        import brain
        cat = brain.Categorizer().fit()
        expected = cat.predict("Cool Whip")[0]
        recipe = rp.parse_recipe("http://x", cat)
        assert recipe.ingredients[0].section == expected

    def test_custom_sections_route_to_batched_llm(self, monkeypatch, mock_chat):
        html = '<html><script type="application/ld+json">{"@type":"Recipe","name":"R","recipeIngredient":["bacon","eggs"]}</script></html>'
        monkeypatch.setattr(rp, "web_fetch", lambda *a, **kw: _fake_fetch(html))
        # only the batched LLM call should happen — return a per-ingredient mapping
        mock_chat({"results": [{"name": "bacon", "section": "meat"},
                                {"name": "eggs", "section": "protein"}]})
        import brain
        recipe = rp.parse_recipe("http://x", brain.Categorizer().fit(),
                                 sections=["meat", "protein", "other"])
        sections = sorted(i.section for i in recipe.ingredients)
        assert sections == ["meat", "protein"]


# --------------------------------------------------------------------------
# categorize_llm
# --------------------------------------------------------------------------


class TestRecipeCacheWiring:
    @pytest.fixture(autouse=True)
    def _clear(self):
        _recipe_cache.clear()
        yield
        _recipe_cache.clear()

    def test_second_call_skips_web_fetch(self, monkeypatch):
        html = '<html><script type="application/ld+json">{"@type":"Recipe","name":"X","recipeIngredient":["a"]}</script></html>'
        calls = {"n": 0}

        def fake_fetch(url, *a, **kw):
            calls["n"] += 1
            return {"url": url, "status": 200, "html": html,
                    "text": recipe.html_to_text(recipe.extract_main_content(html))}

        monkeypatch.setattr(rp, "web_fetch", fake_fetch)
        import brain
        cat = brain.Categorizer().fit()
        r1 = rp.parse_recipe("http://example.com/x", cat)
        r2 = rp.parse_recipe("http://example.com/x", cat)
        assert calls["n"] == 1
        assert r1.title == r2.title
