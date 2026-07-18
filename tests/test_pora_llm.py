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


class TestCategorizeCacheWiring:
    @pytest.fixture(autouse=True)
    def _clear(self):
        import pora_llm
        pora_llm._categorize_cache.clear()
        yield
        pora_llm._categorize_cache.clear()

    def test_second_call_hits_cache_and_skips_llm(self, mock_chat):
        calls = {"n": 0}

        def responder(system, user, **kw):
            calls["n"] += 1
            return '{"section": "dairy"}'

        mock_chat(responder)
        ai.categorize_llm("молоко")
        ai.categorize_llm("молоко")
        assert calls["n"] == 1

    def test_batch_partial_cache_hit_only_misses_reach_llm(self, mock_chat):
        # Prime cache with молоко → dairy
        mock_chat('{"section": "dairy"}')
        ai.categorize_llm("молоко")

        # Batch call for [молоко, чай] — only чай should be sent
        seen = {"names": None}

        def responder(system, user, **kw):
            seen["names"] = user
            return '{"results": [{"name": "чай", "section": "drinks"}]}'

        mock_chat(responder)
        out = ai.categorize_llm_batch(["молоко", "чай"])
        assert out == [("dairy", 0.9), ("drinks", 0.9)]
        assert "чай" in seen["names"]
        assert "молоко" not in seen["names"]

    def test_custom_sections_have_separate_cache_key(self, mock_chat):
        mock_chat(['{"section": "dairy"}', '{"section": "other"}'])
        r1 = ai.categorize_llm("молоко")                     # default sections
        r2 = ai.categorize_llm("молоко", sections=["a", "b", "other"])
        assert r1 != r2                                       # both missed


class TestCacheDisabledByEnv:
    def test_disabled_bypasses_all_lookups(self, monkeypatch, mock_chat):
        import pora_llm
        pora_llm._categorize_cache.clear()
        monkeypatch.setattr(pora_llm, "_CACHE_ENABLED", False)
        calls = {"n": 0}

        def responder(system, user, **kw):
            calls["n"] += 1
            return '{"section": "dairy"}'

        mock_chat(responder)
        ai.categorize_llm("молоко")
        ai.categorize_llm("молоко")
        assert calls["n"] == 2
        pora_llm._categorize_cache.clear()


class TestChatModel:
    class _Toy(__import__("pydantic").BaseModel):
        section: str

    def test_happy_path(self, mock_chat):
        mock_chat('{"section": "produce"}')
        out = ai._chat_model("sys", "user", self._Toy)
        assert out is not None and out.section == "produce"

    def test_retry_on_validation_error(self, mock_chat):
        calls = {"n": 0, "prompts": []}

        def responder(system, user, **kw):
            calls["n"] += 1
            calls["prompts"].append(user)
            return '{"wrong": "shape"}' if calls["n"] == 1 else '{"section": "dairy"}'

        mock_chat(responder)
        out = ai._chat_model("sys", "user text", self._Toy)
        assert out is not None and out.section == "dairy"
        assert calls["n"] == 2
        assert "failed validation" in calls["prompts"][1].lower()

    def test_exhausts_retries_returns_none(self, mock_chat):
        mock_chat(['{"bad": 1}', '{"still": "bad"}'])
        assert ai._chat_model("sys", "user", self._Toy) is None

    def test_returns_none_when_llm_disabled(self):
        assert ai._chat_model("sys", "user", self._Toy) is None

    def test_examples_kwarg_accepted(self, mock_chat):
        mock_chat('{"section": "dairy"}')
        out = ai._chat_model(
            "sys", "user",
            self._Toy,
            examples=[{"user": "x", "assistant": '{"section": "dairy"}'}],
        )
        assert out is not None


class TestChatExamplesInjection:
    def test_chat_injects_examples_between_system_and_user(self, monkeypatch):
        monkeypatch.setattr(ai, "API_KEY", "test-key")
        captured = {"messages": None}

        class _Cli:
            class chat:
                class completions:
                    @staticmethod
                    def create(**kw):
                        captured["messages"] = kw["messages"]

                        class _M:
                            content = '{"ok": true}'

                        class _C:
                            message = _M()

                        class _R:
                            choices = [_C()]

                        return _R()

        monkeypatch.setattr(ai, "client", lambda: _Cli())
        monkeypatch.setattr(ai, "_transient_llm_errors", lambda: ())

        ai._chat(
            "SYSTEM",
            "REAL_USER",
            examples=[
                {"user": "ex_u1", "assistant": "ex_a1"},
                {"user": "ex_u2", "assistant": "ex_a2"},
            ],
        )
        roles = [m["role"] for m in captured["messages"]]
        assert roles == ["system", "user", "assistant", "user", "assistant", "user"]
        contents = [m["content"] for m in captured["messages"]]
        assert contents == ["SYSTEM", "ex_u1", "ex_a1", "ex_u2", "ex_a2", "REAL_USER"]


class TestCategorizeInvalidSectionFallsBack:
    def test_out_of_enum_section_returns_fallback(self, mock_chat):
        mock_chat('{"section": "bogus_not_in_enum"}')
        key, conf = ai.categorize_llm("x")
        assert key == "other"
        assert conf == 0.0


class TestResolveModel:
    def test_fast_returns_fast_env(self, monkeypatch):
        monkeypatch.setattr(ai, "MODEL_MAIN", "big-model")
        monkeypatch.setattr(ai, "MODEL_FAST", "small-model")
        assert ai._resolve_model("fast") == "small-model"
        assert ai._resolve_model("main") == "big-model"

    def test_fast_falls_back_to_main_when_unset(self, monkeypatch):
        monkeypatch.setattr(ai, "MODEL_MAIN", "big-model")
        monkeypatch.setattr(ai, "MODEL_FAST", "big-model")
        assert ai._resolve_model("fast") == "big-model"


class TestChatUsesModelKind:
    @staticmethod
    def _capture_client(captured):
        class _Cli:
            class chat:
                class completions:
                    @staticmethod
                    def create(**kw):
                        captured["model"] = kw["model"]

                        class _M:
                            content = "ok"

                        class _C:
                            message = _M()

                        class _R:
                            choices = [_C()]

                        return _R()

        return _Cli

    def test_default_kind_is_main(self, monkeypatch):
        monkeypatch.setattr(ai, "API_KEY", "test-key")
        monkeypatch.setattr(ai, "MODEL_MAIN", "MAIN_M")
        monkeypatch.setattr(ai, "MODEL_FAST", "FAST_M")
        captured = {"model": None}
        monkeypatch.setattr(ai, "client", lambda: self._capture_client(captured)())
        monkeypatch.setattr(ai, "_transient_llm_errors", lambda: ())
        ai._chat("s", "u")
        assert captured["model"] == "MAIN_M"

    def test_fast_kind_uses_fast_model(self, monkeypatch):
        monkeypatch.setattr(ai, "API_KEY", "test-key")
        monkeypatch.setattr(ai, "MODEL_MAIN", "MAIN_M")
        monkeypatch.setattr(ai, "MODEL_FAST", "FAST_M")
        captured = {"model": None}
        monkeypatch.setattr(ai, "client", lambda: self._capture_client(captured)())
        monkeypatch.setattr(ai, "_transient_llm_errors", lambda: ())
        ai._chat("s", "u", model_kind="fast")
        assert captured["model"] == "FAST_M"


class TestCategorizeUsesFastModel:
    def test_categorize_llm_routes_fast(self, monkeypatch):
        captured = {"kind": None}

        def tracing_chat(system, user, temperature=0.4, response_format=None,
                         examples=None, model_kind="main"):
            captured["kind"] = model_kind
            return '{"section": "dairy"}'

        monkeypatch.setattr(ai, "_chat", tracing_chat)
        ai.categorize_llm("молоко")
        assert captured["kind"] == "fast"

    def test_chat_routes_main(self, monkeypatch):
        captured = {"kind": None}

        def tracing_chat(system, user, temperature=0.4, response_format=None,
                         examples=None, model_kind="main"):
            captured["kind"] = model_kind
            return "ответ"

        monkeypatch.setattr(ai, "_chat", tracing_chat)
        ai.chat("как сварить борщ?")
        assert captured["kind"] == "main"
