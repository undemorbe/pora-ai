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
