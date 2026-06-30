# -*- coding: utf-8 -*-
"""Pora AI — локальная ML/стат-логика (без сети, мультиязычно по данным).

Разделы магазина — языко-независимые КЛЮЧИ (dairy, produce, …).
Отображаемые названия локализует приложение (SECTION_LABELS — для справки/совместимости).
"""
from __future__ import annotations

import datetime as dt
from collections import Counter, defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# ============================================================
# Разделы — канонические ключи + локализация (для справки)
# ============================================================
SECTIONS = ["dairy", "produce", "bakery", "pantry", "drinks", "meat_fish", "other"]

SECTION_LABELS = {
    "ru": {"dairy": "Молочное", "produce": "Овощи и фрукты", "bakery": "Хлеб и выпечка",
           "pantry": "Бакалея", "drinks": "Напитки", "meat_fish": "Мясо и рыба", "other": "Другое"},
    "en": {"dairy": "Dairy", "produce": "Produce", "bakery": "Bakery",
           "pantry": "Pantry", "drinks": "Drinks", "meat_fish": "Meat & Fish", "other": "Other"},
}


def section_label(key: str, lang: str = "en") -> str:
    return SECTION_LABELS.get(lang, SECTION_LABELS["en"]).get(key, key)


# ============================================================
# 1. ПОПОЛНЕНИЕ — когда продукт закончится (язык не важен: работаем по датам)
# ============================================================
def predict_replenishment(purchases: list[dict], today: dt.date, min_events: int = 3) -> list[dict]:
    by: dict[str, list[dt.date]] = defaultdict(list)
    for p in purchases:
        by[p["product"]].append(p["date"])

    out = []
    for product, dates in by.items():
        dates = sorted(dates)
        if len(dates) < min_events:
            continue
        intervals = np.diff([d.toordinal() for d in dates])
        median = float(np.median(intervals))
        mean = float(np.mean(intervals))
        cv = float(np.std(intervals) / mean) if mean > 0 else 1.0
        due = dates[-1] + dt.timedelta(days=round(median))
        days_left = (due - today).days
        confidence = max(0.0, min(1.0, 1.0 - cv))
        status = ("overdue" if days_left < 0 else
                  "due" if days_left <= 1 else
                  "soon" if days_left <= 3 else "ok")
        out.append({
            "product": product, "every_days": round(median, 1), "due_date": due.isoformat(),
            "days_left": days_left, "confidence": round(confidence, 2), "status": status, "events": len(dates),
        })
    out.sort(key=lambda r: r["days_left"])
    return out


# ============================================================
# 2. КАТЕГОРИЗАТОР — быстрый, мультиязычный (RU + EN), выдаёт КЛЮЧ раздела
#    Для остальных языков main делегирует в LLM (ai.categorize_llm).
# ============================================================
TRAINING = {
    "dairy": ["молоко", "кефир", "йогурт", "творог", "сметана", "сыр", "сливки",
              "milk", "kefir", "yogurt", "curd", "sour cream", "cheese", "cream", "butter"],
    "produce": ["бананы", "помидоры", "огурцы", "авокадо", "яблоки", "брокколи", "лимон", "картофель", "лук", "морковь",
                "banana", "tomato", "cucumber", "avocado", "apple", "broccoli", "lemon", "potato", "onion", "carrot"],
    "bakery": ["хлеб", "багет", "булочки", "лаваш", "батон", "круассан",
               "bread", "baguette", "bun", "loaf", "croissant", "roll"],
    "pantry": ["паста", "спагетти", "рис", "гречка", "мука", "сахар", "соль", "кофе", "макароны",
               "pasta", "spaghetti", "rice", "flour", "sugar", "salt", "coffee", "tea", "noodles", "oil"],
    "drinks": ["вода", "минералка", "сок", "чай", "лимонад", "газировка", "морс",
               "water", "juice", "soda", "lemonade", "sparkling water"],
    "meat_fish": ["курица", "куриное филе", "фарш", "говядина", "лосось", "креветки", "бекон", "колбаса",
                  "chicken", "beef", "mince", "salmon", "shrimp", "bacon", "sausage", "fish"],
}


class Categorizer:
    def __init__(self):
        self.pipe = make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)),
            LogisticRegression(max_iter=1000, C=8.0),
        )

    def fit(self):
        X, y = [], []
        for key, names in TRAINING.items():
            for n in names:
                X.append(n.lower())
                y.append(key)
        self.pipe.fit(X, y)
        return self

    def predict(self, name: str) -> tuple[str, float]:
        proba = self.pipe.predict_proba([name.lower()])[0]
        i = int(np.argmax(proba))
        return self.pipe.classes_[i], float(proba[i])


# ============================================================
# 3. ВРЕМЯ ПУША — из часов заходов (язык не важен)
# ============================================================
def best_notify_hour(hours: list[int]) -> dict:
    c = Counter(h for h in hours if 0 <= h <= 23)
    total = sum(c.values())
    if not total:
        return {"hour": 18, "window_share": 0.0, "samples": 0}
    evening = {h: c.get(h, 0) for h in range(16, 22)}
    peak = max(evening, key=evening.get) if sum(evening.values()) else max(c, key=c.get)
    window = sum(c.get(h, 0) for h in (peak - 1, peak, peak + 1))
    return {"hour": peak, "window_share": round(window / total, 2), "samples": total}


# ============================================================
# 4. РЕКОМЕНДАЦИЯ ПО ВКУСУ
# ============================================================
RECIPE_CATALOG = [
    {"name": "Карбонара", "cuisine": "Итальянская", "ingredients": {"спагетти", "бекон", "яйца", "пармезан"}},
    {"name": "Мак-н-чиз", "cuisine": "Итальянская", "ingredients": {"паста", "сыр", "молоко", "масло"}},
    {"name": "Лазанья", "cuisine": "Итальянская", "ingredients": {"паста", "фарш", "сыр", "помидоры"}},
    {"name": "Том ям", "cuisine": "Азиатская", "ingredients": {"креветки", "грибы", "лайм", "кокос"}},
    {"name": "Сырники", "cuisine": "Завтраки", "ingredients": {"творог", "яйца", "мука", "сахар"}},
]


def recommend(recipe_imports: list[str], regular_products: list[str]) -> dict:
    tried = set(recipe_imports)
    regular = {p.lower().split()[0] for p in regular_products if p}
    cats = [r["cuisine"] for r in RECIPE_CATALOG if r["name"] in tried]
    top_cuisine = Counter(cats).most_common(1)[0][0] if cats else "Итальянская"

    best, best_score = None, -1.0
    for r in RECIPE_CATALOG:
        if r["name"] in tried:
            continue
        bonus = 1.0 if r["cuisine"] == top_cuisine else 0.0
        match = len(r["ingredients"] & regular) / len(r["ingredients"])
        if bonus + match > best_score:
            best, best_score = r, bonus + match
    if best is None:
        best = RECIPE_CATALOG[0]
    match = len(best["ingredients"] & regular) / len(best["ingredients"])
    return {"top_cuisine": top_cuisine, "recipe": best["name"],
            "cuisine": best["cuisine"], "pantry_match": round(match, 2)}
