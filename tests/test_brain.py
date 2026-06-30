# -*- coding: utf-8 -*-
"""Pure-logic tests for brain.py — no network, no LLM."""
from __future__ import annotations

import datetime as dt

import pytest

import brain


# --------------------------------------------------------------------------
# Sections / labels
# --------------------------------------------------------------------------
class TestSections:
    def test_sections_are_canonical_english_keys(self):
        assert brain.SECTIONS == ["dairy", "produce", "bakery", "pantry", "drinks", "meat_fish", "other"]

    def test_section_label_known_ru(self):
        assert brain.section_label("dairy", "ru") == "Молочное"

    def test_section_label_unknown_lang_falls_back_to_en(self):
        assert brain.section_label("dairy", "xx") == "Dairy"

    def test_section_label_unknown_key_returns_key(self):
        assert brain.section_label("nonsense", "en") == "nonsense"


# --------------------------------------------------------------------------
# predict_replenishment
# --------------------------------------------------------------------------
class TestPredictReplenishment:
    def _build(self, product: str, every_days: int, count: int, end: dt.date):
        return [{"product": product, "date": end - dt.timedelta(days=every_days * i)}
                for i in reversed(range(count))]

    def test_skips_products_with_fewer_than_three_events(self):
        purchases = self._build("Молоко", 7, 2, dt.date(2026, 6, 18))
        assert brain.predict_replenishment(purchases, dt.date(2026, 6, 18)) == []

    def test_regular_purchases_give_due_status(self):
        purchases = self._build("Молоко", 7, 5, dt.date(2026, 6, 18))
        out = brain.predict_replenishment(purchases, dt.date(2026, 6, 25))
        assert len(out) == 1
        p = out[0]
        assert p["product"] == "Молоко"
        assert p["every_days"] == 7.0
        assert p["confidence"] == 1.0  # CV=0 → 1.0
        assert p["status"] in ("overdue", "due")

    def test_status_buckets(self):
        end = dt.date(2026, 6, 18)
        purchases = self._build("X", 5, 5, end)
        # next due = end + 5 = 2026-06-23
        assert brain.predict_replenishment(purchases, dt.date(2026, 6, 24))[0]["status"] == "overdue"
        assert brain.predict_replenishment(purchases, dt.date(2026, 6, 23))[0]["status"] == "due"
        assert brain.predict_replenishment(purchases, dt.date(2026, 6, 21))[0]["status"] == "soon"
        assert brain.predict_replenishment(purchases, dt.date(2026, 6, 18))[0]["status"] == "ok"

    def test_results_sorted_by_days_left(self):
        end = dt.date(2026, 6, 18)
        purchases = self._build("A", 3, 5, end) + self._build("B", 10, 5, end)
        out = brain.predict_replenishment(purchases, end)
        assert [p["product"] for p in out] == ["A", "B"]

    def test_irregular_intervals_lower_confidence(self):
        purchases = [
            {"product": "X", "date": dt.date(2026, 5, 1)},
            {"product": "X", "date": dt.date(2026, 5, 3)},
            {"product": "X", "date": dt.date(2026, 5, 20)},
            {"product": "X", "date": dt.date(2026, 6, 1)},
        ]
        out = brain.predict_replenishment(purchases, dt.date(2026, 6, 18))
        assert out[0]["confidence"] < 0.7


# --------------------------------------------------------------------------
# Categorizer (RU + EN training)
# --------------------------------------------------------------------------
class TestCategorizer:
    @pytest.fixture(scope="class")
    def cat(self):
        return brain.Categorizer().fit()

    @pytest.mark.parametrize("name,section", [
        ("молоко", "dairy"), ("кефир", "dairy"), ("milk", "dairy"),
        ("моцарелла", "dairy"), ("mozzarella", "dairy"),
        ("банан", "produce"), ("avocado", "produce"),
        ("шпинат", "produce"), ("spinach", "produce"),
        ("хлеб", "bakery"), ("baguette", "bakery"),
        ("курица", "meat_fish"), ("chicken", "meat_fish"),
        ("лосось", "meat_fish"), ("salmon", "meat_fish"),
        ("креветки", "meat_fish"), ("shrimp", "meat_fish"),
        ("вода", "drinks"), ("juice", "drinks"),
        ("вино", "drinks"), ("wine", "drinks"),
        ("рис", "pantry"), ("rice", "pantry"),
        ("оливковое масло", "pantry"), ("olive oil", "pantry"),
    ])
    def test_known_items_classified_correctly(self, cat, name, section):
        key, conf = cat.predict(name)
        assert key == section
        assert 0.0 <= conf <= 1.0

    def test_predict_batch_aligned_to_input(self, cat):
        names = ["молоко", "banana", "wine"]
        out = cat.predict_batch(names)
        assert len(out) == len(names)
        assert [k for k, _ in out] == ["dairy", "produce", "drinks"]

    def test_predict_batch_empty(self, cat):
        assert cat.predict_batch([]) == []


# --------------------------------------------------------------------------
# best_notify_hour
# --------------------------------------------------------------------------
class TestBestNotifyHour:
    def test_empty_returns_default_18(self):
        assert brain.best_notify_hour([]) == {"hour": 18, "window_share": 0.0, "samples": 0}

    def test_picks_evening_peak(self):
        hours = [19, 19, 19, 8, 8, 12]
        out = brain.best_notify_hour(hours)
        assert out["hour"] == 19
        assert out["samples"] == 6

    def test_no_evening_data_falls_back_to_overall_peak(self):
        hours = [9, 9, 9, 10]
        assert brain.best_notify_hour(hours)["hour"] == 9

    def test_ignores_out_of_range_hours(self):
        hours = [19, 19, 99, -1, 25]
        assert brain.best_notify_hour(hours)["samples"] == 2


# --------------------------------------------------------------------------
# recommend (existing baseline)
# --------------------------------------------------------------------------
class TestRecommend:
    def test_returns_top_cuisine_from_imports(self):
        out = brain.recommend(["Карбонара", "Лазанья"], ["паста", "сыр"])
        assert out["top_cuisine"] == "Итальянская"

    def test_default_cuisine_when_no_imports(self):
        out = brain.recommend([], [])
        assert "recipe" in out
        assert "cuisine" in out

    def test_excludes_already_tried_recipes(self):
        out = brain.recommend(["Карбонара"], ["паста", "сыр"])
        assert out["recipe"] != "Карбонара"


# --------------------------------------------------------------------------
# Suggest engine — basket_fit, replenish, recipes, merge
# --------------------------------------------------------------------------
class TestSuggestEngine:
    def test_basket_fit_recommends_missing_ingredient_from_recipe(self):
        # Cart has спагетти+бекон → matches Карбонара → suggest пармезан/яйца
        out = brain.suggest_basket_fit(
            current_cart=["спагетти", "бекон"],
            regular_products=["пармезан", "яйца"],
            recipe_imports=[],
            lang="ru",
        )
        assert out, "expected at least one basket_fit suggestion"
        products = [s["product"] for s in out]
        assert "пармезан" in products or "яйца" in products
        for s in out:
            assert s["type"] == "basket_fit"
            assert s["recipe"] == "Карбонара"
            assert s["reason"] == "Подходит к корзине!"

    def test_basket_fit_empty_cart_returns_nothing(self):
        out = brain.suggest_basket_fit([], ["сыр"], [], "ru")
        assert out == []

    def test_basket_fit_no_overlap_returns_nothing(self):
        # cart item that no recipe in catalog uses
        out = brain.suggest_basket_fit(["несуществующее"], [], [], "en")
        assert out == []

    def test_basket_fit_prefers_items_user_regularly_buys(self):
        out = brain.suggest_basket_fit(
            current_cart=["спагетти", "бекон"],
            regular_products=["пармезан"],
            recipe_imports=[],
            lang="ru",
        )
        first = out[0]
        assert first["product"] == "пармезан"
        assert first["meta"]["in_regular"] is True
        assert first["score"] >= 0.9

    def test_basket_fit_dedupes_across_recipes(self):
        # паста matches Mac-n-cheese and Lasagna both; should suggest each product once
        out = brain.suggest_basket_fit(["паста"], ["сыр", "молоко"], [], "ru", max_items=10)
        products = [s["product"] for s in out]
        assert len(products) == len(set(products))

    def test_replenish_filters_status_and_uses_urgency_score(self):
        end = dt.date(2026, 6, 18)
        purchases = [{"product": "Молоко", "date": end - dt.timedelta(days=7 * i)} for i in reversed(range(5))]
        out = brain.suggest_replenish(purchases, end + dt.timedelta(days=8), lang="ru")
        assert out and out[0]["type"] == "replenish"
        assert out[0]["meta"]["status"] == "overdue"
        assert out[0]["reason"] == "Скоро закончится — пора пополнить"

    def test_replenish_empty_when_nothing_urgent(self):
        end = dt.date(2026, 6, 18)
        purchases = [{"product": "Х", "date": end - dt.timedelta(days=30 * i)} for i in reversed(range(5))]
        # last bought 0 days ago, every ~30 days → status=ok
        out = brain.suggest_replenish(purchases, end, "en")
        assert out == []

    def test_recipes_excludes_tried(self):
        out = brain.suggest_recipes(["Карбонара"], ["сыр", "паста"], "en")
        names = [s["recipe"] for s in out]
        assert "Карбонара" not in names

    def test_recipes_top_cuisine_boost(self):
        out = brain.suggest_recipes(["Лазанья"], ["паста", "сыр", "молоко", "помидоры", "яйца"], "ru")
        assert out
        # cuisine bias toward Итальянская since user tried one
        assert any(s["meta"]["cuisine"] == "Итальянская" for s in out)

    def test_merge_dedupes_keeps_first_occurrence(self):
        # X appears in both groups: the first occurrence (score 0.5 from `a`) is kept,
        # the higher-score duplicate from `b` is discarded.
        a = [{"type": "recipe", "product": None, "recipe": "X", "reason": "r", "score": 0.5, "meta": {}}]
        b = [{"type": "recipe", "product": None, "recipe": "X", "reason": "r", "score": 0.9, "meta": {}},
             {"type": "recipe", "product": None, "recipe": "Y", "reason": "r", "score": 0.7, "meta": {}}]
        out = brain.merge_suggestions(a, b, limit=5)
        assert {s["recipe"] for s in out} == {"X", "Y"}
        x_entry = next(s for s in out if s["recipe"] == "X")
        assert x_entry["score"] == 0.5  # not 0.9 — first occurrence wins
        # final order is by score desc → Y (0.7) before X (0.5)
        assert [s["recipe"] for s in out] == ["Y", "X"]

    def test_merge_respects_limit(self):
        many = [{"type": "recipe", "product": None, "recipe": f"R{i}", "reason": "",
                 "score": i / 10, "meta": {}} for i in range(10)]
        assert len(brain.merge_suggestions(many, limit=3)) == 3

    def test_reason_label_localized(self):
        assert brain.reason_label("basket_fit", "ru") == "Подходит к корзине!"
        assert brain.reason_label("basket_fit", "en") == "Pairs with your cart!"
        assert brain.reason_label("basket_fit", "xx") == "Pairs with your cart!"  # fallback to en
