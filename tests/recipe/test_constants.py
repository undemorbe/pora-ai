# -*- coding: utf-8 -*-
"""Sanity checks for the recipe feature's own config table."""
from __future__ import annotations

import re

import constants as C
from recipe import constants as RC


class TestFetch:
    def test_retry_statuses_are_5xx_or_429(self):
        for s in RC.FETCH_RETRY_STATUSES:
            assert s == 429 or 500 <= s < 600

    def test_limits_positive(self):
        assert RC.FETCH_TIMEOUT_S > 0
        assert RC.FETCH_RETRIES >= 0
        assert RC.FETCH_MAX_BYTES > 0

    def test_user_agent_looks_like_a_browser(self):
        assert "Mozilla/5.0" in RC.DEFAULT_USER_AGENT
        assert "PoraBot" in RC.DEFAULT_USER_AGENT   # stay honest about who we are


class TestParserThresholds:
    def test_ordering(self):
        assert RC.PARSER_MIN_LINE_LEN < RC.PARSER_MAX_LINE_LEN
        assert RC.PARSER_MIN_INGREDIENTS < RC.PARSER_MAX_INGREDIENTS

    def test_qty_ratio_is_a_fraction(self):
        assert 0 < RC.PARSER_MIN_QTY_RATIO <= 1

    def test_markers_and_headings_lowercase(self):
        for m in RC.PARSER_INGREDIENT_CLASS_MARKERS + RC.PARSER_INGREDIENT_HEADINGS:
            assert m == m.lower(), f"{m!r} must be lowercase — matching is case-folded"


class TestExtractionTier:
    def test_qty_unit_pattern_compiles_and_matches(self):
        rx = re.compile(RC.QTY_UNIT_PATTERN, re.I)
        assert rx.search("Кабачок - 550 г")
        assert rx.search("2 cups flour")
        assert not rx.search("просто текст без количеств")

    def test_window_lead_is_a_fraction(self):
        assert 0 <= RC.RECIPE_WINDOW_LEAD < 1

    def test_text_cap_fits_a_small_context(self):
        # Cyrillic ≈ 2 chars/token, so the cap must leave room for the
        # few-shot block and system prompt inside a 4096-token context.
        assert 0 < RC.LLM_TEXT_CAP <= 6000

    def test_synonym_pairs_are_distinct_and_lowercase(self):
        for a, b in RC.INGREDIENT_SYNONYM_PAIRS:
            assert a != b
            assert a == a.lower() and b == b.lower()

    def test_synonym_names_long_enough_to_match(self):
        for a, b in RC.INGREDIENT_SYNONYM_PAIRS:
            assert min(len(a), len(b)) >= RC.MIN_SYNONYM_NAME_LEN


class TestPipeline:
    def test_cache_defaults(self):
        assert RC.RECIPE_CACHE_SIZE > 0
        assert RC.RECIPE_CACHE_TTL_S > 0

    def test_failed_extraction_is_not_cacheable(self):
        assert RC.SOURCE_NONE not in RC.CACHEABLE_SOURCES

    def test_every_successful_source_is_cacheable(self):
        assert RC.CACHEABLE_SOURCES == {RC.SOURCE_JSONLD, RC.SOURCE_PARSER, RC.SOURCE_LLM}


class TestFeatureBoundary:
    def test_recipe_knobs_left_the_root_config(self):
        """Moved constants must not linger in the root table — one home each."""
        for name in ("FETCH_TIMEOUT_S", "DEFAULT_USER_AGENT", "QTY_UNIT_PATTERN",
                     "PARSER_MIN_INGREDIENTS", "INGREDIENT_SYNONYM_PAIRS",
                     "RECIPE_CACHE_SIZE", "HTML_ENTITIES"):
            assert not hasattr(C, name), f"constants.{name} should live in recipe.constants"

    def test_shared_knobs_stay_in_the_root_config(self):
        """The feature must not fork service-wide config."""
        for name in ("DEFAULT_SECTIONS", "FAST_ESCALATE_CONF_BELOW",
                     "LLM_MODEL_ROUTING", "RECIPE_EXTRACT_SYSTEM"):
            assert hasattr(C, name)
            assert not hasattr(RC, name), f"recipe.constants.{name} duplicates the root table"
