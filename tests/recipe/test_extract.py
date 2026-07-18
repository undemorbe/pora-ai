# -*- coding: utf-8 -*-
"""Tier 3 — LLM extraction and the anti-hallucination guard (recipe.extract)."""
from __future__ import annotations

import pytest

import recipe
from recipe import extract as rx
from recipe.extract import (extract_recipe_from_text, recipe_window,
                            validate_against_source)


# --------------------------------------------------------------------------
# validate_against_source — anti-hallucination guard
# --------------------------------------------------------------------------
class TestValidateAgainstSource:
    def test_keeps_ingredient_present_in_source(self):
        ings = [{"raw": "Spaghetti 400g", "name": "spaghetti"}]
        out = rx.validate_against_source(ings, "Recipe: Spaghetti 400g, eggs")
        assert out == ings

    def test_drops_invented_ingredient(self):
        ings = [{"raw": "Truffle 500g", "name": "truffle"}]
        out = rx.validate_against_source(ings, "Recipe: Spaghetti, eggs, bacon")
        assert out == []

    def test_falls_back_to_name_when_raw_differs(self):
        # LLM reformatted raw but name appears in source
        ings = [{"raw": "400 g spaghetti pasta", "name": "spaghetti"}]
        out = rx.validate_against_source(ings, "Ingredients list contains spaghetti and eggs")
        assert out == ings

    def test_case_insensitive_match(self):
        ings = [{"raw": "EGGS", "name": "EGGS"}]
        out = rx.validate_against_source(ings, "Two eggs needed")
        assert out == ings

    def test_short_names_rejected(self):
        # 2-char names too generic, only `raw` match accepted
        ings = [{"raw": "salt", "name": "sa"}]
        out = rx.validate_against_source(ings, "Pinch of pepper, sa hidden")
        assert out == []  # raw "salt" not in source; name "sa" too short

    def test_mixed_drops_only_hallucinated(self):
        ings = [
            {"raw": "Eggs 4", "name": "eggs"},
            {"raw": "Dragon scales 100g", "name": "dragon"},  # hallucinated
        ]
        out = rx.validate_against_source(ings, "Eggs 4, flour 200g")
        assert len(out) == 1
        assert out[0]["raw"] == "Eggs 4"

    def test_translated_name_kept_via_synonym_bridge(self):
        # RU page, LLM normalized the name to English — must survive
        ings = [{"raw": "milk 200 ml", "name": "milk"}]
        out = rx.validate_against_source(ings, "Рецепт: молоко 200 мл, мука 3 ст.л.")
        assert out == ings

    def test_synonym_bridge_works_both_directions(self):
        ings = [{"raw": "сыр 100 г", "name": "сыр"}]
        out = rx.validate_against_source(ings, "Recipe: cheese 100 g, bread")
        assert out == ings

    def test_plural_name_matches_singular_source(self):
        ings = [{"raw": "2 large eggs", "name": "eggs"}]
        out = rx.validate_against_source(ings, "take one egg and whisk")
        assert out == ings

    def test_unrelated_translation_still_dropped(self):
        # synonym bridge must not let arbitrary words through
        ings = [{"raw": "unicorn 1", "name": "unicorn"}]
        out = rx.validate_against_source(ings, "молоко, мука, сахар")
        assert out == []


# --------------------------------------------------------------------------
# extract_jsonld
# --------------------------------------------------------------------------


class TestRecipeWindow:
    def test_short_text_returned_whole(self):
        assert rx.recipe_window("молоко 200 мл", 8000) == "молоко 200 мл"

    def test_picks_window_around_ingredients_not_the_start(self):
        # Real-world shape: long nav/menu first, ingredients far down the page.
        noise = "Главная страница Рецепты Статьи Наша Кухня Поиск Рассылки " * 200
        recipe = "Продукты: Кабачок - 550 г, Брынза - 190 г, Творог - 60 г, Ветчина - 100 г"
        text = noise + recipe + noise
        window = rx.recipe_window(text, 400)
        assert "Кабачок - 550 г" in window
        assert "Брынза - 190 г" in window
        assert len(window) <= 400

    def test_no_quantity_signals_falls_back_to_head(self):
        text = "a" * 100 + "b" * 100
        assert rx.recipe_window(text, 50) == "a" * 50

    def test_english_units_recognized(self):
        noise = "navigation link home about contact " * 100
        recipe = "Ingredients: 2 cups flour, 3 tbsp butter, 400 g sugar"
        window = rx.recipe_window(noise + recipe + noise, 300)
        assert "2 cups flour" in window

    def test_window_never_exceeds_cap(self):
        text = "молоко 200 мл " * 5000
        assert len(rx.recipe_window(text, 1000)) <= 1000


class TestExtractRecipeUsesWindow:
    def test_ingredients_far_down_the_page_reach_the_llm(self, mock_chat):
        noise = "Меню Рецепты Статьи Поиск Войти Регистрация " * 300
        recipe = "Продукты: Кабачок - 550 г, Брынза - 190 г"
        seen = {"user": None}

        def responder(system, user, **kw):
            seen["user"] = user
            return ('{"title": "Рулетики", "ingredients": ['
                    '{"raw":"Кабачок - 550 г","name":"Кабачок","qty":550,"unit":"г"}]}')

        mock_chat(responder)
        out = rx.extract_recipe_from_text(noise + recipe + noise)
        # the prompt the model actually received must contain the ingredients
        assert "Кабачок - 550 г" in seen["user"]
        assert out["source"] == "llm"
        assert out["ingredients"][0]["name"] == "Кабачок"


# --------------------------------------------------------------------------
# extract_recipe_from_text — LLM mocked
# --------------------------------------------------------------------------
class TestExtractRecipeFromText:
    def test_disabled_llm_returns_empty(self):
        # API_KEY empty by default in test env unless `enable_llm` used
        out = rx.extract_recipe_from_text("any text")
        assert out == {"title": None, "ingredients": [], "source": "none"}

    def test_valid_extraction_validated_against_source(self, mock_chat):
        mock_chat({
            "title": "Pasta",
            "ingredients": [
                {"raw": "Spaghetti 400g", "name": "spaghetti", "qty": 400, "unit": "g"},
                {"raw": "Eggs 4", "name": "eggs", "qty": 4, "unit": None},
            ],
        })
        out = rx.extract_recipe_from_text("Cook with Spaghetti 400g and Eggs 4 — done.")
        assert out["title"] == "Pasta"
        assert len(out["ingredients"]) == 2
        assert out["source"] == "llm"

    def test_hallucinated_items_dropped(self, mock_chat):
        # LLM returns dragon scales — not in source. Must be dropped.
        mock_chat({
            "title": "Recipe",
            "ingredients": [
                {"raw": "Eggs 4", "name": "eggs", "qty": 4, "unit": None},
                {"raw": "Dragon scales 100g", "name": "dragon", "qty": 100, "unit": "g"},
            ],
        })
        out = rx.extract_recipe_from_text("Take Eggs 4 and mix.")
        assert len(out["ingredients"]) == 1
        assert out["ingredients"][0]["name"] == "eggs"

    def test_all_hallucinated_returns_source_none(self, mock_chat):
        mock_chat({
            "title": "Fake",
            "ingredients": [{"raw": "Unicorn horn", "name": "unicorn", "qty": None, "unit": None}],
        })
        out = rx.extract_recipe_from_text("Lorem ipsum dolor sit amet.")
        assert out["ingredients"] == []
        assert out["source"] == "none"

    def test_malformed_llm_json_returns_empty(self, mock_chat):
        mock_chat("not a json {")
        out = rx.extract_recipe_from_text("text")
        assert out["source"] == "none"
        assert out["ingredients"] == []

    def test_code_fences_stripped(self, mock_chat):
        mock_chat('```json\n{"title": null, "ingredients": [{"raw": "salt", "name": "salt", "qty": null, "unit": null}]}\n```')
        out = rx.extract_recipe_from_text("pinch of salt please")
        assert out["ingredients"][0]["raw"] == "salt"


# --------------------------------------------------------------------------
# parse_recipe (full flow: httpx mocked)
# --------------------------------------------------------------------------
def _fake_fetch(html: str, url: str = "http://example.com/r"):
    """Helper: build a web_fetch return shape from a chunk of HTML."""
    return {"url": url, "status": 200, "html": html, "text": ai.html_to_text(ai.extract_main_content(html))}
