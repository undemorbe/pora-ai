# -*- coding: utf-8 -*-
"""Unit tests for pora_llm.py — uses mock_chat fixture to avoid real LLM calls."""
from __future__ import annotations

import pytest

import pora_llm as ai


# --------------------------------------------------------------------------
# detect_lang
# --------------------------------------------------------------------------
class TestDetectLang:
    def test_cyrillic_is_ru(self):
        assert ai.detect_lang("дай рецепт борща") == "ru"

    def test_ascii_default_en(self):
        assert ai.detect_lang("give me a recipe") == "en"

    def test_japanese_kana_beats_zh(self):
        assert ai.detect_lang("レシピを教えて") == "ja"

    def test_zh_pure_han(self):
        assert ai.detect_lang("给我一个食谱") == "zh"

    def test_hangul_is_ko(self):
        assert ai.detect_lang("레시피 주세요") == "ko"

    def test_arabic(self):
        assert ai.detect_lang("أعطني وصفة") == "ar"

    def test_hindi(self):
        assert ai.detect_lang("मुझे एक रेसिपी दो") == "hi"

    def test_hebrew(self):
        assert ai.detect_lang("תן לי מתכון") == "he"

    def test_polish_diacritics(self):
        assert ai.detect_lang("zażółć gęślą jaźń") == "pl"

    def test_turkish_diacritics(self):
        assert ai.detect_lang("yoğurt ve çay") == "tr"

    def test_portuguese_tilde(self):
        assert ai.detect_lang("pão e leite, açúcar") == "pt"

    def test_spanish_inverted_marks(self):
        assert ai.detect_lang("¿qué receta?") == "es"

    def test_french_oe_ligature(self):
        assert ai.detect_lang("œufs et beurre français") == "fr"

    def test_german_umlaut(self):
        assert ai.detect_lang("Brötchen mit Süß") == "de"

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

    def test_15_languages_covered(self):
        expected = {"ru", "en", "es", "pt", "de", "fr", "it", "pl", "tr",
                    "zh", "ja", "ko", "ar", "hi", "he"}
        assert expected <= set(ai.REFUSALS), f"missing: {expected - set(ai.REFUSALS)}"

    def test_all_have_emoji(self):
        for code, text in ai.REFUSALS.items():
            assert "🙂" in text, f"{code} has no emoji"


# --------------------------------------------------------------------------
# guard_on_topic
# --------------------------------------------------------------------------
class TestSafeJsonLoad:
    def test_none_input(self):
        assert ai._safe_json_load(None) is None

    def test_empty_string(self):
        assert ai._safe_json_load("") is None

    def test_valid_json(self):
        assert ai._safe_json_load('{"a": 1}') == {"a": 1}

    def test_stripped_code_fences(self):
        assert ai._safe_json_load('```json\n{"a": 1}\n```') == {"a": 1}

    def test_malformed_returns_none(self):
        assert ai._safe_json_load("not json {") is None


class TestChatRetry:
    def test_returns_none_when_disabled(self, monkeypatch):
        monkeypatch.setattr(ai, "API_KEY", "")
        assert ai._chat("s", "u") is None

    def test_non_transient_returns_none_no_retry(self, monkeypatch):
        monkeypatch.setattr(ai, "API_KEY", "test")
        calls = {"n": 0}

        class _Cli:
            class chat:
                class completions:
                    @staticmethod
                    def create(**kw):
                        calls["n"] += 1
                        raise ValueError("permanent bad request")

        monkeypatch.setattr(ai, "client", lambda: _Cli())
        # transient list is empty (no openai import in test) → any exception is non-transient
        monkeypatch.setattr(ai, "_transient_llm_errors", lambda: ())
        result = ai._chat("s", "u")
        assert result is None
        assert calls["n"] == 1  # no retry for non-transient

    def test_transient_retries_then_succeeds(self, monkeypatch):
        monkeypatch.setattr(ai, "API_KEY", "test")
        monkeypatch.setattr(ai, "LLM_RETRY_BACKOFF_S", 0, raising=False)
        # patch constants module used by pora_llm
        import constants
        monkeypatch.setattr(constants, "LLM_RETRY_BACKOFF_S", 0)

        class TransientErr(Exception):
            pass

        monkeypatch.setattr(ai, "_transient_llm_errors", lambda: (TransientErr,))
        calls = {"n": 0}

        class _Msg:
            content = "OK"

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]

        class _Cli:
            class chat:
                class completions:
                    @staticmethod
                    def create(**kw):
                        calls["n"] += 1
                        if calls["n"] < 3:
                            raise TransientErr("network")
                        return _Resp()

        monkeypatch.setattr(ai, "client", lambda: _Cli())
        result = ai._chat("s", "u")
        assert result == "OK"
        assert calls["n"] == 3

    def test_transient_exhausts_retries_returns_none(self, monkeypatch):
        monkeypatch.setattr(ai, "API_KEY", "test")
        import constants
        monkeypatch.setattr(constants, "LLM_RETRY_BACKOFF_S", 0)

        class TransientErr(Exception):
            pass

        monkeypatch.setattr(ai, "_transient_llm_errors", lambda: (TransientErr,))
        calls = {"n": 0}

        class _Cli:
            class chat:
                class completions:
                    @staticmethod
                    def create(**kw):
                        calls["n"] += 1
                        raise TransientErr("permanently down")

        monkeypatch.setattr(ai, "client", lambda: _Cli())
        assert ai._chat("s", "u") is None
        # LLM_MAX_RETRIES + 1 attempts total
        assert calls["n"] == constants.LLM_MAX_RETRIES + 1


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
class TestExtractMainContent:
    def test_picks_article(self):
        html = "<html><body><nav>menu</nav><article>real content</article><footer>x</footer></body></html>"
        assert "real content" in ai.extract_main_content(html)

    def test_picks_main(self):
        html = "<html><body><header>x</header><main>recipe content</main></body></html>"
        out = ai.extract_main_content(html)
        assert "recipe content" in out
        # main wins over nav/header stripping
        assert "recipe content" in ai.html_to_text(out)

    def test_falls_back_to_body_minus_boilerplate(self):
        html = "<html><body><nav>menu items</nav><div>content X</div><footer>fine print</footer></body></html>"
        text = ai.html_to_text(ai.extract_main_content(html))
        assert "content X" in text
        assert "menu items" not in text
        assert "fine print" not in text


class TestWebFetch:
    def test_returns_url_status_html_text(self, monkeypatch):
        import httpx as _httpx

        class _Resp:
            status_code = 200
            url = "http://example.com/r"
            text = "<html><body><main>Hello world</main></body></html>"

            def raise_for_status(self):
                pass

        class _Client:
            def __init__(self, *a, **kw): pass
            def __enter__(self): return self
            def __exit__(self, *a): pass
            def get(self, url): return _Resp()

        monkeypatch.setattr(_httpx, "Client", _Client)
        out = ai.web_fetch("http://example.com/r", lang="ru")
        assert out["status"] == 200
        assert "Hello world" in out["text"]
        assert out["html"].startswith("<html>")

    def test_retries_on_5xx_then_succeeds(self, monkeypatch):
        import httpx as _httpx

        calls = {"n": 0}

        class _Bad:
            status_code = 503
            url = "http://x"
            text = ""

            def raise_for_status(self):
                raise _httpx.HTTPStatusError("bad", request=None, response=None)

        class _Good:
            status_code = 200
            url = "http://x"
            text = "<main>OK</main>"

            def raise_for_status(self):
                pass

        class _Client:
            def __init__(self, *a, **kw): pass
            def __enter__(self): return self
            def __exit__(self, *a): pass
            def get(self, url):
                calls["n"] += 1
                return _Bad() if calls["n"] < 2 else _Good()

        monkeypatch.setattr(_httpx, "Client", _Client)
        out = ai.web_fetch("http://x", retries=2)
        assert out["status"] == 200
        assert calls["n"] >= 2

    def test_propagates_after_max_retries(self, monkeypatch):
        import httpx as _httpx

        class _Client:
            def __init__(self, *a, **kw): pass
            def __enter__(self): return self
            def __exit__(self, *a): pass
            def get(self, url):
                raise _httpx.RequestError("connection refused")

        monkeypatch.setattr(_httpx, "Client", _Client)
        with pytest.raises(_httpx.HTTPError):
            ai.web_fetch("http://x", retries=1)

    def test_accept_language_built_from_lang(self):
        # private helper but worth a smoke test — verifies lang flows into header
        assert ai._accept_language("ru").startswith("ru")
        assert ai._accept_language(None).startswith("en")
        assert ai._accept_language("pt-BR").startswith("pt")


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
def _fake_fetch(html: str, url: str = "http://example.com/r"):
    """Helper: build a web_fetch return shape from a chunk of HTML."""
    return {"url": url, "status": 200, "html": html, "text": ai.html_to_text(ai.extract_main_content(html))}


class TestParseRecipe:
    def test_jsonld_path_no_llm_call(self, monkeypatch):
        html = '<html><script type="application/ld+json">{"@type":"Recipe","name":"Carbonara","recipeIngredient":["Spaghetti 400g","Eggs 4"]}</script></html>'
        monkeypatch.setattr(ai, "web_fetch", lambda *a, **kw: _fake_fetch(html))
        # _chat must NOT be called for the JSON-LD path
        monkeypatch.setattr(ai, "_chat",
                            lambda *a, **kw: (_ for _ in ()).throw(AssertionError("_chat called on JSON-LD path")))

        import brain
        recipe = ai.parse_recipe("http://example.com/r", brain.Categorizer().fit())
        assert recipe.title == "Carbonara"
        assert recipe.source == "jsonld"
        assert len(recipe.ingredients) == 2
        for ing in recipe.ingredients:
            assert ing.section in brain.SECTIONS

    def test_llm_fallback_path(self, monkeypatch, mock_chat):
        html = "<html><body>Recipe: take Spaghetti 400g and Eggs 4.</body></html>"
        monkeypatch.setattr(ai, "web_fetch", lambda *a, **kw: _fake_fetch(html))
        mock_chat({
            "title": "Pasta",
            "ingredients": [{"raw": "Spaghetti 400g", "name": "spaghetti", "qty": 400, "unit": "g"}],
        })
        import brain
        recipe = ai.parse_recipe("http://x", brain.Categorizer().fit())
        assert recipe.source == "llm"
        assert recipe.title == "Pasta"
        assert recipe.ingredients[0].section in brain.SECTIONS

    def test_custom_sections_route_to_batched_llm(self, monkeypatch, mock_chat):
        html = '<html><script type="application/ld+json">{"@type":"Recipe","name":"R","recipeIngredient":["bacon","eggs"]}</script></html>'
        monkeypatch.setattr(ai, "web_fetch", lambda *a, **kw: _fake_fetch(html))
        # only the batched LLM call should happen — return a per-ingredient mapping
        mock_chat({"results": [{"name": "bacon", "section": "meat"},
                                {"name": "eggs", "section": "protein"}]})
        import brain
        recipe = ai.parse_recipe("http://x", brain.Categorizer().fit(),
                                 sections=["meat", "protein", "other"])
        sections = sorted(i.section for i in recipe.ingredients)
        assert sections == ["meat", "protein"]


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

    def test_custom_sections_enum_used(self, mock_chat):
        mock_chat({"section": "meat"})
        key, conf = ai.categorize_llm("стейк", sections=["meat", "veg", "other"])
        assert key == "meat" and conf == 0.9

    def test_custom_sections_fallback_no_other(self, mock_chat):
        # LLM disabled by default — fallback is the first section when 'other' is absent
        ai.API_KEY = ""  # ensure disabled
        key, conf = ai.categorize_llm("x", sections=["a", "b"])
        assert key == "a" and conf == 0.0


class TestCategorizeLLMBatch:
    def test_empty_input(self):
        assert ai.categorize_llm_batch([]) == []

    def test_disabled_returns_fallbacks(self):
        out = ai.categorize_llm_batch(["a", "b"], sections=["meat", "veg", "other"])
        assert out == [("other", 0.0), ("other", 0.0)]

    def test_disabled_no_other_falls_back_to_first(self):
        out = ai.categorize_llm_batch(["a"], sections=["meat", "veg"])
        assert out == [("meat", 0.0)]

    def test_valid_batch_response_aligned_to_input_order(self, mock_chat):
        mock_chat({"results": [
            {"name": "молоко", "section": "dairy"},
            {"name": "хлеб", "section": "bakery"},
        ]})
        out = ai.categorize_llm_batch(["молоко", "хлеб"])
        assert out == [("dairy", 0.9), ("bakery", 0.9)]

    def test_partial_response_unmapped_get_fallback(self, mock_chat):
        mock_chat({"results": [{"name": "молоко", "section": "dairy"}]})
        out = ai.categorize_llm_batch(["молоко", "хлеб"])
        assert out[0] == ("dairy", 0.9)
        assert out[1] == ("other", 0.0)

    def test_invalid_section_in_response_treated_as_unmapped(self, mock_chat):
        mock_chat({"results": [{"name": "x", "section": "bogus"}]})
        out = ai.categorize_llm_batch(["x"], sections=["a", "b", "other"])
        assert out == [("other", 0.0)]

    def test_malformed_returns_fallbacks(self, mock_chat):
        mock_chat("not json")
        assert ai.categorize_llm_batch(["a", "b"]) == [("other", 0.0)] * 2


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
