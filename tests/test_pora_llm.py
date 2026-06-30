# -*- coding: utf-8 -*-
"""Unit tests for pora_llm.py — uses mock_chat fixture to avoid real LLM calls."""
from __future__ import annotations

import pora_llm as ai


# --------------------------------------------------------------------------
# detect_lang
# --------------------------------------------------------------------------
class TestDetectLang:
    def test_cyrillic_is_ru(self):
        assert ai.detect_lang("дай рецепт борща") == "ru"

    def test_ascii_default_en(self):
        assert ai.detect_lang("give me a recipe") == "en"

    def test_cjk_is_zh(self):
        assert ai.detect_lang("给我一个食谱") == "zh"

    def test_hangul_is_ko(self):
        assert ai.detect_lang("레시피 주세요") == "ko"

    def test_spanish_diacritics(self):
        assert ai.detect_lang("¿qué receta?") == "es"

    def test_german_umlaut(self):
        assert ai.detect_lang("Brötchen") == "de"

    def test_default_overrideable(self):
        assert ai.detect_lang("xyz", default="zh") == "zh"


# --------------------------------------------------------------------------
# REFUSALS / refusal()
# --------------------------------------------------------------------------
class TestRefusal:
    def test_known_lang(self):
        assert "🙂" in ai.refusal("ru")
        assert ai.refusal("en") == "I only help with food and shopping 🙂"

    def test_unknown_lang_falls_back_to_en(self):
        assert ai.refusal("xx") == ai.refusal("en")


# --------------------------------------------------------------------------
# guard_on_topic
# --------------------------------------------------------------------------
class TestGuardOnTopic:
    def test_food_topic_passes(self):
        assert ai.guard_on_topic("как сварить борщ?")

    def test_python_code_blocked(self):
        assert not ai.guard_on_topic("write me python code please")

    def test_legal_blocked(self):
        assert not ai.guard_on_topic("нужна консультация юриста")

    def test_medical_blocked(self):
        assert not ai.guard_on_topic("какое medication принимать?")


# --------------------------------------------------------------------------
# HTML stripping (anti-hallucination prep)
# --------------------------------------------------------------------------
class TestHtmlToText:
    def test_strips_script_and_style(self):
        html = "<html><script>var x=1;</script><style>p{color:red}</style><p>Hello</p></html>"
        text = ai.html_to_text(html)
        assert "var x" not in text
        assert "color:red" not in text
        assert "Hello" in text

    def test_decodes_entities(self):
        assert "salt & pepper" in ai.html_to_text("<p>salt &amp; pepper</p>")

    def test_collapses_whitespace(self):
        assert ai.html_to_text("<p>a\n\n\n  b</p>") == "a b"

    def test_handles_empty(self):
        assert ai.html_to_text("") == ""


# --------------------------------------------------------------------------
# validate_against_source — anti-hallucination guard
# --------------------------------------------------------------------------
class TestValidateAgainstSource:
    def test_keeps_ingredient_present_in_source(self):
        ings = [{"raw": "Spaghetti 400g", "name": "spaghetti"}]
        out = ai.validate_against_source(ings, "Recipe: Spaghetti 400g, eggs")
        assert out == ings

    def test_drops_invented_ingredient(self):
        ings = [{"raw": "Truffle 500g", "name": "truffle"}]
        out = ai.validate_against_source(ings, "Recipe: Spaghetti, eggs, bacon")
        assert out == []

    def test_falls_back_to_name_when_raw_differs(self):
        # LLM reformatted raw but name appears in source
        ings = [{"raw": "400 g spaghetti pasta", "name": "spaghetti"}]
        out = ai.validate_against_source(ings, "Ingredients list contains spaghetti and eggs")
        assert out == ings

    def test_case_insensitive_match(self):
        ings = [{"raw": "EGGS", "name": "EGGS"}]
        out = ai.validate_against_source(ings, "Two eggs needed")
        assert out == ings

    def test_short_names_rejected(self):
        # 2-char names too generic, only `raw` match accepted
        ings = [{"raw": "salt", "name": "sa"}]
        out = ai.validate_against_source(ings, "Pinch of pepper, sa hidden")
        assert out == []  # raw "salt" not in source; name "sa" too short

    def test_mixed_drops_only_hallucinated(self):
        ings = [
            {"raw": "Eggs 4", "name": "eggs"},
            {"raw": "Dragon scales 100g", "name": "dragon"},  # hallucinated
        ]
        out = ai.validate_against_source(ings, "Eggs 4, flour 200g")
        assert len(out) == 1
        assert out[0]["raw"] == "Eggs 4"


# --------------------------------------------------------------------------
# extract_jsonld
# --------------------------------------------------------------------------
class TestExtractJsonld:
    def test_simple_recipe(self):
        html = '<script type="application/ld+json">{"@type":"Recipe","name":"Carbonara","recipeIngredient":["Spaghetti 400g","Eggs 4"]}</script>'
        out = ai.extract_jsonld(html)
        assert out["title"] == "Carbonara"
        assert len(out["ingredients"]) == 2
        assert out["source"] == "jsonld"

    def test_walks_graph(self):
        html = '<script type="application/ld+json">{"@graph":[{"@type":"WebPage"},{"@type":"Recipe","name":"X","recipeIngredient":["A"]}]}</script>'
        out = ai.extract_jsonld(html)
        assert out["title"] == "X"

    def test_type_array(self):
        html = '<script type="application/ld+json">{"@type":["Recipe","Article"],"name":"Y","recipeIngredient":["B"]}</script>'
        out = ai.extract_jsonld(html)
        assert out["title"] == "Y"

    def test_no_recipe_returns_none(self):
        html = '<script type="application/ld+json">{"@type":"Article","name":"X"}</script>'
        assert ai.extract_jsonld(html) is None

    def test_malformed_json_skipped(self):
        html = '<script type="application/ld+json">{not json</script>'
        assert ai.extract_jsonld(html) is None

    def test_ingredients_as_string_normalized_to_list(self):
        html = '<script type="application/ld+json">{"@type":"Recipe","name":"X","recipeIngredient":"Only one"}</script>'
        out = ai.extract_jsonld(html)
        assert len(out["ingredients"]) == 1


# --------------------------------------------------------------------------
# extract_recipe_from_text — LLM mocked
# --------------------------------------------------------------------------
class TestExtractRecipeFromText:
    def test_disabled_llm_returns_empty(self):
        # API_KEY empty by default in test env unless `enable_llm` used
        out = ai.extract_recipe_from_text("any text")
        assert out == {"title": None, "ingredients": [], "source": "none"}

    def test_valid_extraction_validated_against_source(self, mock_chat):
        mock_chat({
            "title": "Pasta",
            "ingredients": [
                {"raw": "Spaghetti 400g", "name": "spaghetti", "qty": 400, "unit": "g"},
                {"raw": "Eggs 4", "name": "eggs", "qty": 4, "unit": None},
            ],
        })
        out = ai.extract_recipe_from_text("Cook with Spaghetti 400g and Eggs 4 — done.")
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
        out = ai.extract_recipe_from_text("Take Eggs 4 and mix.")
        assert len(out["ingredients"]) == 1
        assert out["ingredients"][0]["name"] == "eggs"

    def test_all_hallucinated_returns_source_none(self, mock_chat):
        mock_chat({
            "title": "Fake",
            "ingredients": [{"raw": "Unicorn horn", "name": "unicorn", "qty": None, "unit": None}],
        })
        out = ai.extract_recipe_from_text("Lorem ipsum dolor sit amet.")
        assert out["ingredients"] == []
        assert out["source"] == "none"

    def test_malformed_llm_json_returns_empty(self, mock_chat):
        mock_chat("not a json {")
        out = ai.extract_recipe_from_text("text")
        assert out["source"] == "none"
        assert out["ingredients"] == []

    def test_code_fences_stripped(self, mock_chat):
        mock_chat('```json\n{"title": null, "ingredients": [{"raw": "salt", "name": "salt", "qty": null, "unit": null}]}\n```')
        out = ai.extract_recipe_from_text("pinch of salt please")
        assert out["ingredients"][0]["raw"] == "salt"


# --------------------------------------------------------------------------
# parse_recipe (full flow: httpx mocked)
# --------------------------------------------------------------------------
class TestParseRecipe:
    def test_jsonld_path_no_llm_call(self, monkeypatch):
        import httpx as _httpx
        html = '<html><script type="application/ld+json">{"@type":"Recipe","name":"Carbonara","recipeIngredient":["Spaghetti 400g","Eggs 4"]}</script></html>'

        class _Resp:
            text = html

        monkeypatch.setattr(_httpx, "get", lambda *a, **kw: _Resp())
        # _chat must NOT be called — guarantee by raising
        monkeypatch.setattr(ai, "_chat", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("_chat must not be called for JSON-LD path")))

        import brain
        recipe = ai.parse_recipe("http://example.com/r", brain.Categorizer().fit())
        assert recipe.title == "Carbonara"
        assert recipe.source == "jsonld"
        assert len(recipe.ingredients) == 2
        # each ingredient gets a section assigned
        for ing in recipe.ingredients:
            assert ing.section in brain.SECTIONS

    def test_llm_fallback_path(self, monkeypatch, mock_chat):
        import httpx as _httpx

        class _Resp:
            text = "<html><body>Recipe: take Spaghetti 400g and Eggs 4.</body></html>"

        monkeypatch.setattr(_httpx, "get", lambda *a, **kw: _Resp())
        mock_chat({
            "title": "Pasta",
            "ingredients": [{"raw": "Spaghetti 400g", "name": "spaghetti", "qty": 400, "unit": "g"}],
        })
        import brain
        recipe = ai.parse_recipe("http://x", brain.Categorizer().fit())
        assert recipe.source == "llm"
        assert recipe.title == "Pasta"
        assert recipe.ingredients[0].section in brain.SECTIONS


# --------------------------------------------------------------------------
# categorize_llm
# --------------------------------------------------------------------------
class TestCategorizeLLM:
    def test_disabled_returns_other(self):
        assert ai.categorize_llm("anything") == ("other", 0.0)

    def test_valid_response(self, mock_chat):
        mock_chat({"section": "produce"})
        key, conf = ai.categorize_llm("авокадо")
        assert key == "produce"
        assert conf == 0.9

    def test_malformed_returns_other(self, mock_chat):
        mock_chat("nope")
        assert ai.categorize_llm("x") == ("other", 0.0)


# --------------------------------------------------------------------------
# suggest_dish_llm
# --------------------------------------------------------------------------
class TestSuggestDishLLM:
    def test_disabled_returns_none(self):
        assert ai.suggest_dish_llm("Italian", ["pasta"], "en") is None

    def test_returns_parsed_dict(self, mock_chat):
        mock_chat({"dish": "Carbonara", "reason": "matches your pasta basket"})
        out = ai.suggest_dish_llm("Italian", ["pasta"], "en")
        assert out == {"dish": "Carbonara", "reason": "matches your pasta basket"}

    def test_malformed_returns_none(self, mock_chat):
        mock_chat("garbage")
        assert ai.suggest_dish_llm("Italian", ["pasta"], "en") is None


# --------------------------------------------------------------------------
# chat
# --------------------------------------------------------------------------
class TestChat:
    def test_offtopic_refused_without_llm_call(self):
        out = ai.chat("write me python code")
        assert out["refused"] is True
        assert out["lang"] == "en"

    def test_offtopic_refused_in_ru(self):
        out = ai.chat("дай мне sql запрос")
        assert out["refused"] is True
        assert out["lang"] == "ru"

    def test_llm_disabled_falls_back_to_refusal(self):
        out = ai.chat("как сварить борщ?")
        assert out["refused"] is False
        assert "note" in out and out["note"] == "llm_disabled"

    def test_llm_enabled_returns_answer(self, mock_chat):
        mock_chat("Сначала сварите бульон.")
        out = ai.chat("как сварить борщ?")
        assert out["text"] == "Сначала сварите бульон."
        assert out["refused"] is False


# --------------------------------------------------------------------------
# generate_tip
# --------------------------------------------------------------------------
class TestGenerateTip:
    def test_fallback_when_llm_disabled_ru(self):
        out = ai.generate_tip("Итальянская", ["паста"], "ru")
        assert out["source"] == "fallback"
        assert "Итальянская" in out["tip"]

    def test_fallback_when_llm_disabled_en(self):
        out = ai.generate_tip("Italian", ["pasta"], "en")
        assert out["source"] == "fallback"
        assert "Italian" in out["tip"]

    def test_llm_enabled(self, mock_chat):
        mock_chat("Любите итальянскую — попробуйте оссобуко!")
        out = ai.generate_tip("Итальянская", ["паста"], "ru")
        assert out["source"] == "llm"
        assert "оссобуко" in out["tip"]
