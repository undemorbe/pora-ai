# -*- coding: utf-8 -*-
"""Pora AI — локальная ML/стат-логика (без сети, мультиязычно по данным).

Разделы магазина — языко-независимые КЛЮЧИ (`constants.DEFAULT_SECTIONS`).
Отображаемые названия локализует приложение (`SECTION_LABELS` — для справки/совместимости).

Все пороги/скоринг/лейблы читаются из `constants` — крутится один файл конфигурации,
логика остаётся чистой.
"""
from __future__ import annotations

import datetime as dt
from collections import Counter, defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

import constants as C

# ============================================================
# Разделы — реэкспорт из constants для обратной совместимости.
# Внешний код (Go client, тесты) обращается к brain.SECTIONS.
# ============================================================
SECTIONS: list[str] = list(C.DEFAULT_SECTIONS)
SECTION_LABELS = C.SECTION_LABELS
REASON_LABELS = C.REASON_LABELS


def section_label(key: str, lang: str = "en") -> str:
    return SECTION_LABELS.get(lang, SECTION_LABELS["en"]).get(key, key)


def reason_label(kind: str, lang: str = "en") -> str:
    return REASON_LABELS.get(lang, REASON_LABELS["en"]).get(kind, kind)


# ============================================================
# 1. ПОПОЛНЕНИЕ — когда продукт закончится (язык не важен: работаем по датам)
# ============================================================
def _bucket_status(days_left: int) -> str:
    if days_left < C.OVERDUE_DAYS_LEFT:
        return "overdue"
    if days_left <= C.DUE_DAYS_LEFT:
        return "due"
    if days_left <= C.SOON_DAYS_LEFT:
        return "soon"
    return "ok"


def predict_replenishment(purchases: list[dict], today: dt.date,
                          min_events: int = C.MIN_PURCHASES_FOR_FORECAST) -> list[dict]:
    by: dict[str, list[dt.date]] = defaultdict(list)
    for p in purchases:
        by[p["product"]].append(p["date"])

    out: list[dict] = []
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
        out.append({
            "product": product, "every_days": round(median, 1), "due_date": due.isoformat(),
            "days_left": days_left, "confidence": round(confidence, 2),
            "status": _bucket_status(days_left), "events": len(dates),
        })
    out.sort(key=lambda r: r["days_left"])
    return out


# ============================================================
# 2. КАТЕГОРИЗАТОР — быстрый, мультиязычный (RU + EN), выдаёт КЛЮЧ раздела
#    Для остальных языков main делегирует в LLM (ai.categorize_llm).
# ============================================================
TRAINING: dict[str, list[str]] = {
    "dairy": [
        # ru
        "молоко", "молоко безлактозное", "кефир", "ряженка", "айран",
        "йогурт", "греческий йогурт", "творог", "творожок", "сметана",
        "сыр", "твёрдый сыр", "плавленый сыр", "моцарелла", "пармезан",
        "фета", "брынза", "рикотта", "маскарпоне", "сливки", "сливочное масло",
        "масло сливочное", "сгущёнка", "сгущённое молоко",
        # en
        "milk", "lactose-free milk", "kefir", "buttermilk", "yogurt",
        "greek yogurt", "curd", "cottage cheese", "sour cream", "cheese",
        "hard cheese", "processed cheese", "mozzarella", "parmesan", "feta",
        "ricotta", "mascarpone", "cream", "heavy cream", "whipping cream",
        "butter", "condensed milk", "ghee",
    ],
    "produce": [
        # ru — frequent grocery items
        "бананы", "помидоры", "томаты", "огурцы", "авокадо", "яблоки",
        "груши", "виноград", "брокколи", "цветная капуста", "капуста",
        "лимон", "лайм", "апельсины", "мандарины", "картофель", "лук",
        "лук репчатый", "лук зелёный", "морковь", "свёкла", "редис",
        "перец", "перец болгарский", "чеснок", "имбирь", "грибы",
        "шампиньоны", "зелень", "укроп", "петрушка", "базилик", "руккола",
        "шпинат", "салат", "клубника", "малина", "черника",
        # en
        "banana", "tomato", "cherry tomato", "cucumber", "avocado",
        "apple", "pear", "grapes", "broccoli", "cauliflower", "cabbage",
        "lemon", "lime", "orange", "mandarin", "potato", "sweet potato",
        "onion", "red onion", "green onion", "scallion", "carrot",
        "beetroot", "radish", "bell pepper", "garlic", "ginger",
        "mushroom", "champignon", "herbs", "dill", "parsley", "basil",
        "arugula", "spinach", "salad", "lettuce", "strawberry",
        "raspberry", "blueberry",
    ],
    "bakery": [
        "хлеб", "хлеб ржаной", "хлеб бородинский", "багет", "булочки",
        "лаваш", "тортилья", "батон", "круассан", "сдоба", "пирожки",
        "печенье", "пряники", "вафли", "пита", "фокачча",
        "bread", "rye bread", "sourdough", "baguette", "bun", "burger bun",
        "loaf", "croissant", "roll", "pastry", "cookies", "biscuits",
        "waffles", "pita", "focaccia", "tortilla",
    ],
    "pantry": [
        "паста", "спагетти", "пенне", "лазанья листы", "рис", "рис басмати",
        "гречка", "перловка", "овсянка", "мука", "сахар", "соль", "кофе",
        "кофе молотый", "чай", "чай зелёный", "макароны", "лапша",
        "оливковое масло", "подсолнечное масло", "уксус", "соевый соус",
        "томатная паста", "консервы", "тунец консервированный", "фасоль",
        "горох", "чечевица", "крупа", "мёд", "варенье", "джем",
        "шоколад", "какао", "специи", "приправы",
        "pasta", "spaghetti", "penne", "lasagna sheets", "rice",
        "basmati rice", "buckwheat", "oats", "oatmeal", "flour", "sugar",
        "brown sugar", "salt", "coffee", "ground coffee", "tea",
        "green tea", "noodles", "olive oil", "sunflower oil", "vegetable oil",
        "vinegar", "soy sauce", "tomato paste", "canned beans", "canned tuna",
        "beans", "peas", "lentils", "honey", "jam", "chocolate", "cocoa",
        "spices", "seasoning",
    ],
    "drinks": [
        "вода", "минералка", "минеральная вода", "сок", "сок апельсиновый",
        "сок яблочный", "чай холодный", "лимонад", "газировка", "кола",
        "морс", "квас", "компот", "пиво", "вино", "вино красное",
        "вино белое", "шампанское", "энергетик",
        "water", "still water", "sparkling water", "mineral water",
        "juice", "orange juice", "apple juice", "iced tea", "lemonade",
        "soda", "cola", "kombucha", "beer", "wine", "red wine",
        "white wine", "champagne", "energy drink",
    ],
    "meat_fish": [
        "курица", "куриное филе", "куриные грудки", "куриные крылья",
        "куриные бёдра", "фарш", "фарш говяжий", "фарш куриный",
        "говядина", "говядина вырезка", "свинина", "свинина шейка",
        "стейк", "рёбрышки", "баранина", "индейка", "утка", "лосось",
        "сёмга", "форель", "тунец", "креветки", "мидии", "осьминог",
        "кальмар", "бекон", "ветчина", "колбаса", "сосиски", "пельмени",
        "котлеты",
        "chicken", "chicken breast", "chicken thigh", "chicken wing",
        "ground chicken", "mince", "ground beef", "beef", "beef tenderloin",
        "pork", "pork shoulder", "steak", "ribs", "lamb", "turkey", "duck",
        "salmon", "trout", "tuna", "shrimp", "prawn", "mussels", "octopus",
        "squid", "bacon", "ham", "sausage", "hot dog", "fish", "cod",
    ],
}


class Categorizer:
    """Character-ngram TF-IDF + LogisticRegression over bilingual RU/EN TRAINING.

    Hyperparameters (n-gram range, C, max_iter, class_weight) live in
    `constants` so the model config stays greppable. Predictions are calibrated
    only by `predict_proba`'s raw output — no extra calibration pass.
    """

    def __init__(self):
        self.pipe = make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=C.NGRAM_RANGE,
                            min_df=C.TFIDF_MIN_DF, sublinear_tf=C.TFIDF_SUBLINEAR),
            LogisticRegression(max_iter=C.LOGREG_MAX_ITER, C=C.LOGREG_C,
                               class_weight=C.LOGREG_CLASS_WEIGHT),
        )

    def fit(self):
        X, y = [], []
        for key, names in TRAINING.items():
            for n in names:
                X.append(n.lower().strip())
                y.append(key)
        self.pipe.fit(X, y)
        return self

    def predict(self, name: str) -> tuple[str, float]:
        proba = self.pipe.predict_proba([name.lower().strip()])[0]
        i = int(np.argmax(proba))
        return self.pipe.classes_[i], float(proba[i])

    def predict_batch(self, names: list[str]) -> list[tuple[str, float]]:
        """Vectorized batch prediction — single fit() reuse, no per-call retraining."""
        if not names:
            return []
        clean = [n.lower().strip() for n in names]
        probs = self.pipe.predict_proba(clean)
        out = []
        for row in probs:
            i = int(np.argmax(row))
            out.append((self.pipe.classes_[i], float(row[i])))
        return out


# ============================================================
# 3. ВРЕМЯ ПУША — из часов заходов (язык не важен)
# ============================================================
def best_notify_hour(hours: list[int]) -> dict:
    c = Counter(h for h in hours if 0 <= h <= 23)
    total = sum(c.values())
    if not total:
        return {"hour": C.DEFAULT_NOTIFY_HOUR, "window_share": 0.0, "samples": 0}
    evening_range = range(C.EVENING_WINDOW_START, C.EVENING_WINDOW_END)
    evening = {h: c.get(h, 0) for h in evening_range}
    peak = max(evening, key=evening.get) if sum(evening.values()) else max(c, key=c.get)
    window = sum(c.get(h, 0) for h in (peak - 1, peak, peak + 1))
    return {"hour": peak, "window_share": round(window / total, 2), "samples": total}


# ============================================================
# 4. РЕКОМЕНДАЦИЯ ПО ВКУСУ
# ============================================================
def _first_token(s: str) -> str:
    return s.lower().split()[0] if s and s.strip() else ""


def _tokens(items: list[str]) -> set[str]:
    return {_first_token(p) for p in items if p and p.strip()}


def _normalize_catalog(catalog) -> list[dict]:
    """Turn a caller-supplied (or the built-in) catalog into the internal shape.

    Accepts entries with `ingredients` as any iterable of strings; normalizes
    each ingredient to its lowercase first token so matching against user
    product names stays consistent. Entries without a name are dropped;
    missing cuisine falls back to constants.DEFAULT_CUISINE.
    """
    out: list[dict] = []
    for r in catalog:
        name = (r.get("name") or "").strip()
        if not name:
            continue
        out.append({
            "name": name,
            "cuisine": r.get("cuisine") or C.DEFAULT_CUISINE,
            "ingredients": {_first_token(i) for i in (r.get("ingredients") or ()) if i and i.strip()},
        })
    return [r for r in out if r["ingredients"]]


# Built-in catalog, pre-normalized once at import. Kept as a module attribute
# for backward compatibility with existing callers/tests.
RECIPE_CATALOG: list[dict] = _normalize_catalog(C.RECIPE_CATALOG)


def _catalog_or_default(catalog) -> list[dict]:
    if not catalog:
        return RECIPE_CATALOG
    normalized = _normalize_catalog(catalog)
    return normalized or RECIPE_CATALOG


def recommend(recipe_imports: list[str], regular_products: list[str],
              catalog: list[dict] | None = None) -> dict:
    cat = _catalog_or_default(catalog)
    tried = set(recipe_imports)
    regular = _tokens(regular_products)
    cats = [r["cuisine"] for r in cat if r["name"] in tried]
    top_cuisine = Counter(cats).most_common(1)[0][0] if cats else C.DEFAULT_CUISINE

    best, best_score = None, -1.0
    for r in cat:
        if r["name"] in tried:
            continue
        bonus = 1.0 if r["cuisine"] == top_cuisine else 0.0
        match = len(r["ingredients"] & regular) / len(r["ingredients"])
        if bonus + match > best_score:
            best, best_score = r, bonus + match
    if best is None:
        best = cat[0]
    match = len(best["ingredients"] & regular) / len(best["ingredients"])
    return {"top_cuisine": top_cuisine, "recipe": best["name"],
            "cuisine": best["cuisine"], "pantry_match": round(match, 2)}


# ============================================================
# 5. SUGGEST — гибридный движок советов (корзина + пополнение + рецепты)
#    Тип suggestion: basket_fit | replenish | recipe | dish
# ============================================================
def suggest_replenish(purchases: list[dict], today: dt.date, lang: str = "en",
                      max_items: int = C.DEFAULT_REPLENISH_MAX) -> list[dict]:
    """Top urgent replenishments. Reuses predict_replenishment, keeps only overdue/due/soon."""
    preds = predict_replenishment(purchases, today)
    urgent = [p for p in preds if p["status"] in C.URGENT_STATUSES]
    out: list[dict] = []
    for p in urgent[:max_items]:
        urgency = C.URGENCY_MULTIPLIERS[p["status"]]
        out.append({
            "type": "replenish", "product": p["product"], "recipe": None,
            "reason": reason_label("replenish", lang),
            "score": round(urgency * p["confidence"], 2),
            "meta": {"status": p["status"], "days_left": p["days_left"],
                     "due_date": p["due_date"], "every_days": p["every_days"]},
        })
    return out


def suggest_basket_fit(current_cart: list[str], regular_products: list[str],
                       recipe_imports: list[str], lang: str = "en",
                       max_items: int = C.DEFAULT_BASKET_MAX,
                       catalog: list[dict] | None = None) -> list[dict]:
    """For each cart item, find a recipe that uses it and suggest a missing ingredient
    that the user regularly buys (= they'll actually need it)."""
    if not current_cart:
        return []
    cart_tokens = _tokens(current_cart)
    regular = _tokens(regular_products)
    tried = set(recipe_imports)

    out: list[dict] = []
    seen_products: set[str] = set()
    for recipe in _catalog_or_default(catalog):
        overlap_cart = recipe["ingredients"] & cart_tokens
        if not overlap_cart:
            continue
        missing = recipe["ingredients"] - cart_tokens
        for product in sorted(missing, key=lambda x: (x not in regular, x)):
            if product in seen_products:
                continue
            seen_products.add(product)
            in_regular = product in regular
            recipe_bonus = C.BASKET_FIT_UNTRIED_BONUS if recipe["name"] not in tried else 0.0
            regular_bonus = C.BASKET_FIT_REGULAR_BONUS if in_regular else 0.0
            out.append({
                "type": "basket_fit", "product": product, "recipe": recipe["name"],
                "reason": reason_label("basket_fit", lang),
                "score": round(C.BASKET_FIT_BASE_SCORE + regular_bonus + recipe_bonus, 2),
                "meta": {"matched_cart_item": next(iter(overlap_cart)),
                         "cuisine": recipe["cuisine"], "in_regular": in_regular},
            })
            if len(out) >= max_items:
                return out
    return out


def suggest_recipes(recipe_imports: list[str], regular_products: list[str],
                    lang: str = "en", max_items: int = C.DEFAULT_RECIPE_MAX,
                    catalog: list[dict] | None = None) -> list[dict]:
    """Rank catalog recipes by cuisine affinity + pantry overlap, exclude already-tried."""
    cat = _catalog_or_default(catalog)
    tried = set(recipe_imports)
    regular = _tokens(regular_products)
    cats = [r["cuisine"] for r in cat if r["name"] in tried]
    top_cuisine = Counter(cats).most_common(1)[0][0] if cats else None

    ranked = []
    for r in cat:
        if r["name"] in tried:
            continue
        match = len(r["ingredients"] & regular) / len(r["ingredients"])
        bonus = C.RECIPE_CUISINE_BONUS if top_cuisine and r["cuisine"] == top_cuisine else 0.0
        ranked.append((round(match + bonus, 2), r, match))
    ranked.sort(key=lambda x: -x[0])

    out: list[dict] = []
    for score, r, match in ranked[:max_items]:
        out.append({
            "type": "recipe", "product": None, "recipe": r["name"],
            "reason": reason_label("recipe", lang),
            "score": score,
            "meta": {"cuisine": r["cuisine"], "pantry_match": round(match, 2),
                     "missing": sorted(r["ingredients"] - regular)},
        })
    return out


def merge_suggestions(*groups: list[dict], limit: int = C.DEFAULT_SUGGEST_LIMIT) -> list[dict]:
    """Flatten + sort by score desc, drop duplicates by (type, product, recipe)."""
    seen: set[tuple] = set()
    flat: list[dict] = []
    for g in groups:
        for s in g:
            key = (s["type"], s.get("product"), s.get("recipe"))
            if key in seen:
                continue
            seen.add(key)
            flat.append(s)
    flat.sort(key=lambda s: -s["score"])
    return flat[:limit]
