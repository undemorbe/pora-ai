# -*- coding: utf-8 -*-
"""Sanity tests for constants.py — verify the config table stays coherent and
that brain/pora_llm/main use it as the single source of truth.
"""
from __future__ import annotations

import constants as C


class TestSectionsCoherence:
    def test_default_fallback_is_present(self):
        assert C.DEFAULT_FALLBACK_SECTION in C.DEFAULT_SECTIONS

    def test_default_sections_covered_by_labels(self):
        for lang in ("ru", "en"):
            for section in C.DEFAULT_SECTIONS:
                assert section in C.SECTION_LABELS[lang], \
                    f"missing label for {section} in {lang}"

    def test_reason_labels_cover_all_suggestion_types(self):
        expected_types = {"basket_fit", "replenish", "recipe", "dish"}
        for lang in ("ru", "en"):
            assert expected_types <= set(C.REASON_LABELS[lang])


class TestScoringInvariants:
    def test_urgency_multipliers_ordered(self):
        u = C.URGENCY_MULTIPLIERS
        assert u["overdue"] > u["due"] > u["soon"]

    def test_urgency_statuses_match_multipliers(self):
        assert set(C.URGENT_STATUSES) == set(C.URGENCY_MULTIPLIERS)

    def test_basket_fit_scores_reasonable(self):
        # Max possible basket_fit score should stay under 1.5
        max_score = C.BASKET_FIT_BASE_SCORE + C.BASKET_FIT_REGULAR_BONUS + C.BASKET_FIT_UNTRIED_BONUS
        assert 0 <= max_score <= 1.5

    def test_replenishment_thresholds_ordered(self):
        assert C.OVERDUE_DAYS_LEFT < C.DUE_DAYS_LEFT < C.SOON_DAYS_LEFT


class TestLLMPlumbing:
    def test_confidence_thresholds(self):
        assert 0 < C.FAST_ESCALATE_CONF_BELOW < 1
        assert 0 <= C.LLM_CONF_LOW < C.LLM_CONF_HIGH <= 1

    def test_fetch_retry_statuses_are_5xx_or_429(self):
        for s in C.FETCH_RETRY_STATUSES:
            assert s == 429 or 500 <= s < 600

    def test_temperatures_in_range(self):
        for t in (C.TEMPERATURE_STRICT, C.TEMPERATURE_CHAT,
                  C.TEMPERATURE_TIP, C.TEMPERATURE_DISH):
            assert 0.0 <= t <= 2.0


class TestLanguageTables:
    def test_refusals_have_emoji(self):
        for code, text in C.REFUSALS.items():
            assert "🙂" in text, f"{code} refusal missing emoji"

    def test_script_and_latin_do_not_overlap_by_key(self):
        script_langs = {lang for lang, _ in C.SCRIPT_PATTERNS}
        assert not (script_langs & C.LATIN_MARKERS.keys()), \
            "same language should not appear in both tables"


class TestModulesUseConstants:
    def test_brain_sections_matches_default(self):
        import brain
        assert brain.SECTIONS == list(C.DEFAULT_SECTIONS)

    def test_brain_reason_labels_matches(self):
        import brain
        assert brain.REASON_LABELS is C.REASON_LABELS

    def test_pora_llm_refusals_matches(self):
        import pora_llm
        assert pora_llm.REFUSALS is C.REFUSALS

    def test_pora_llm_scope_system_matches(self):
        import pora_llm
        assert pora_llm.SCOPE_SYSTEM is C.SCOPE_SYSTEM

    def test_main_fast_langs_matches(self):
        import main
        assert main.FAST_LANGS is C.FAST_LANGS


class TestFewShot:
    def test_categorize_examples_present(self):
        assert "categorize" in C.FEW_SHOT_EXAMPLES
        assert len(C.FEW_SHOT_EXAMPLES["categorize"]) >= 5

    def test_recipe_extract_examples_present(self):
        assert "recipe_extract" in C.FEW_SHOT_EXAMPLES
        assert len(C.FEW_SHOT_EXAMPLES["recipe_extract"]) >= 3

    def test_all_assistant_payloads_are_valid_json(self):
        import json
        for group in C.FEW_SHOT_EXAMPLES.values():
            for pair in group:
                assert {"user", "assistant"} <= set(pair)
                json.loads(pair["assistant"])  # raises if malformed

    def test_categorize_examples_use_default_sections(self):
        import json
        for pair in C.FEW_SHOT_EXAMPLES["categorize"]:
            data = json.loads(pair["assistant"])
            assert data["section"] in C.DEFAULT_SECTIONS


class TestModelRouting:
    def test_kinds_defined(self):
        assert C.LLM_MODEL_KIND_MAIN == "main"
        assert C.LLM_MODEL_KIND_FAST == "fast"

    def test_routing_table_covers_all_callers(self):
        expected = {"categorize", "categorize_batch", "dish", "tip",
                    "chat", "recipe_extract"}
        assert expected == set(C.LLM_MODEL_ROUTING)

    def test_routing_values_are_valid_kinds(self):
        valid = {C.LLM_MODEL_KIND_MAIN, C.LLM_MODEL_KIND_FAST}
        for kind in C.LLM_MODEL_ROUTING.values():
            assert kind in valid


class TestDeHardcode:
    def test_recipe_catalog_shape(self):
        assert len(C.RECIPE_CATALOG) >= 3
        for r in C.RECIPE_CATALOG:
            assert r["name"] and r["cuisine"]
            assert len(r["ingredients"]) >= 2

    def test_default_cuisine_appears_in_catalog(self):
        assert C.DEFAULT_CUISINE in {r["cuisine"] for r in C.RECIPE_CATALOG}

    def test_prompt_templates_have_lang_placeholder(self):
        assert "{lang}" in C.DISH_SYSTEM_TEMPLATE
        assert "{lang}" in C.TIP_SYSTEM_TEMPLATE

    def test_tip_fallbacks_have_cuisine_placeholder(self):
        assert set(C.TIP_FALLBACKS) >= {"ru", "en"}
        for tpl in C.TIP_FALLBACKS.values():
            assert "{cuisine}" in tpl

    def test_env_names_and_defaults(self):
        assert C.LLM_BASE_URL_ENV == "LLM_BASE_URL"
        assert C.LLM_API_KEY_ENV == "LLM_API_KEY"
        assert C.LLM_MODEL_ENV == "LLM_MODEL"
        assert C.LLM_MODEL_FAST_ENV == "LLM_MODEL_FAST"
        assert C.LLM_BASE_URL_DEFAULT.startswith("http")
        assert C.LLM_MODEL_DEFAULT

    def test_env_falsy_lowercase(self):
        assert all(v == v.lower() for v in C.ENV_FALSY)


class TestCacheDefaults:
    def test_categorize_cache_defaults(self):
        assert C.CATEGORIZE_CACHE_SIZE > 0
        assert C.CATEGORIZE_CACHE_TTL_S > 0

    def test_recipe_cache_defaults(self):
        assert C.RECIPE_CACHE_SIZE > 0
        assert C.RECIPE_CACHE_TTL_S > 0

    def test_cache_env_name(self):
        assert C.CACHE_ENABLED_ENV == "PORA_CACHE_ENABLED"
        assert C.CACHE_ENABLED_DEFAULT is True
