# -*- coding: utf-8 -*-
"""Pora AI — HTTP-сервис (мультиязычный). Go-бэкенд обращается к нему по REST.

Запуск:   uvicorn main:app --port 8000      |   docker compose up
Доки:     http://localhost:8000/docs
"""
from __future__ import annotations

import datetime as dt

from fastapi import FastAPI, HTTPException

import brain
import constants as C
import pora_llm as ai
import recipe
from schemas import (
    BriefRequest, CategorizeRequest, ChatRequest, NotifyTimeRequest,
    ParseRecipeRequest, RecommendRequest, ReplenishmentRequest, SuggestRequest, TipRequest,
)

app = FastAPI(title="Pora AI", version="2.0.0",
              description="Мультиязычный ИИ Pora: пополнение, категоризация, время пуша, "
                          "рекомендации, парсинг рецептов, советы и заскоупленный чат.")

_cat = brain.Categorizer().fit()
FAST_LANGS = C.FAST_LANGS  # backward-compat alias for tests / external code


@app.get("/health")
def health():
    return {"status": "ok", "version": app.version,
            "llm_enabled": ai.llm_enabled(),
            "models": {"main": ai.MODEL_MAIN, "fast": ai.MODEL_FAST},
            "sections": brain.SECTIONS, "fast_langs": sorted(FAST_LANGS),
            "refusal_langs": sorted(ai.REFUSALS)}


@app.get("/metrics")
def metrics():
    """Операционная телеметрия (JSON): LLM-вызовы/ошибки/латенси/токены по
    видам моделей + статистика кэшей. Счётчики кумулятивные с момента старта
    процесса; сервис остаётся stateless по пользовательским данным."""
    from _metrics import METRICS
    return {
        "llm": METRICS.snapshot(),
        "caches": {
            "categorize": ai._categorize_cache.stats(),
            "recipe": recipe.pipeline._recipe_cache.stats(),
        },
    }


def _parse_dates(purchases):
    try:
        return [{"product": p.product, "date": dt.date.fromisoformat(p.date)} for p in purchases]
    except ValueError as e:
        raise HTTPException(400, f"bad date: {e}")


@app.post("/v1/replenishment")
def replenishment(req: ReplenishmentRequest):
    today = dt.date.fromisoformat(req.today) if req.today else dt.date.today()
    return {"today": today.isoformat(), "predictions": brain.predict_replenishment(_parse_dates(req.purchases), today)}


def _label_for(section: str, lang: str, custom_labels):
    """Per-request section label resolution.

    Custom taxonomies (req.sections set) use `req.section_labels` if provided,
    else echo the section key. Default taxonomy uses brain.section_label.
    """
    if custom_labels and section in custom_labels:
        return custom_labels[section]
    if custom_labels is not None:
        return section
    return brain.section_label(section, lang)


@app.post("/v1/categorize")
def categorize(req: CategorizeRequest):
    """Categorize grocery item names into store sections.

    Routing:
      - req.sections is None and lang ∈ FAST_LANGS → fast classifier (LLM escalation
        only when confidence < 0.45 AND LLM enabled)
      - req.sections is None and lang ∉ FAST_LANGS → per-item LLM with brain.SECTIONS
      - req.sections is set → always batched LLM with the custom enum (one LLM call total)
    """
    results = []
    custom_labels = req.section_labels if req.sections else None

    if req.sections:
        # Custom taxonomy — one batched LLM call, regardless of language
        if ai.llm_enabled():
            tagged = ai.categorize_llm_batch(req.names, req.sections)
        else:
            fb = "other" if "other" in req.sections else req.sections[0]
            tagged = [(fb, 0.0)] * len(req.names)
        for name, (key, conf) in zip(req.names, tagged):
            lang = req.lang or ai.detect_lang(name)
            results.append({"name": name, "section": key,
                            "section_label": _label_for(key, lang, custom_labels),
                            "confidence": round(conf, 2), "lang": lang, "method": "llm"})
        return {"results": results}

    for name in req.names:
        lang = req.lang or ai.detect_lang(name)
        if lang in C.FAST_LANGS:
            key, conf = _cat.predict(name)
            method = "fast"
            if conf < C.FAST_ESCALATE_CONF_BELOW and ai.llm_enabled():
                key, conf, method = *ai.categorize_llm(name), "llm"
        else:
            key, conf = ai.categorize_llm(name)
            method = "llm"
        results.append({"name": name, "section": key,
                        "section_label": _label_for(key, lang, None),
                        "confidence": round(conf, 2), "lang": lang, "method": method})
    return {"results": results}


@app.post("/v1/notify-time")
def notify_time(req: NotifyTimeRequest):
    hours = []
    for ts in req.opens:
        try:
            hours.append(dt.datetime.fromisoformat(ts).hour)
        except ValueError:
            continue
    return brain.best_notify_hour(hours)


def _catalog_dicts(catalog):
    """Convert pydantic CatalogRecipe list to plain dicts for brain (None passthrough)."""
    return [r.model_dump() for r in catalog] if catalog else None


@app.post("/v1/recommend")
def recommend(req: RecommendRequest):
    return brain.recommend(req.recipe_imports, req.regular_products,
                           catalog=_catalog_dicts(req.catalog))


@app.post("/v1/parse-recipe")
def parse_recipe(req: ParseRecipeRequest):
    """Fetch a recipe URL, extract ingredients, tag each with a section.

    Anti-hallucination: LLM-extracted ingredients are validated against the
    page source — items that don't appear verbatim are dropped. With custom
    `sections`, ingredient tagging is delegated to a batched LLM call using
    the supplied taxonomy.
    """
    try:
        return recipe.parse_recipe(req.url, _cat, sections=req.sections, lang=req.lang).model_dump()
    except Exception as e:
        raise HTTPException(502, f"fetch/parse failed: {e}")


@app.post("/v1/tip")
def tip(req: TipRequest):
    lang = req.lang or ai.detect_lang(req.top_cuisine + " " + " ".join(req.frequent))
    return ai.generate_tip(req.top_cuisine, req.frequent, lang)


@app.post("/v1/chat")
def chat(req: ChatRequest):
    """Заскоупленный мультиязычный ассистент (кулинарные советы)."""
    return ai.chat(req.message, req.lang)


@app.post("/v1/suggest")
def suggest(req: SuggestRequest):
    """Гибридные советы: «подходит к корзине», «скоро закончится», рецепты, блюдо от LLM.

    Объединяет статистику покупок (пополнение), историю парсинга рецептов
    (предпочтения по кухне), регулярные покупки (что обычно нужно) и текущую
    корзину. Возвращает отсортированный по `score` список разнотипных подсказок.
    """
    today = dt.date.fromisoformat(req.today) if req.today else dt.date.today()
    lang = req.lang or "en"
    catalog = _catalog_dicts(req.catalog)

    basket = brain.suggest_basket_fit(req.current_cart, req.regular_products,
                                      req.recipe_imports, lang, catalog=catalog)
    replenish = brain.suggest_replenish(_parse_dates(req.purchases), today, lang) if req.purchases else []
    recipes = brain.suggest_recipes(req.recipe_imports, req.regular_products, lang, catalog=catalog)

    dish_list: list[dict] = []
    if ai.llm_enabled():
        rec = brain.recommend(req.recipe_imports, req.regular_products, catalog=catalog)
        dish = ai.suggest_dish_llm(rec["top_cuisine"], req.regular_products, lang)
        if dish and dish.get("dish"):
            dish_list = [{
                "type": "dish", "product": None, "recipe": dish["dish"],
                "reason": dish.get("reason") or brain.reason_label("dish", lang),
                "score": C.DISH_DEFAULT_SCORE,
                "meta": {"top_cuisine": rec["top_cuisine"], "source": "llm"},
            }]

    merged = brain.merge_suggestions(basket, replenish, recipes, dish_list, limit=req.limit)
    return {"lang": lang, "today": today.isoformat(), "suggestions": merged}


@app.post("/v1/brief")
def brief(req: BriefRequest):
    """Весь дневной брифинг одним вызовом."""
    today = dt.date.fromisoformat(req.today) if req.today else dt.date.today()
    lang = req.lang or "ru"
    rec = brain.recommend(req.recipe_imports, req.regular_products)
    hours = []
    for ts in req.opens:
        try:
            hours.append(dt.datetime.fromisoformat(ts).hour)
        except ValueError:
            pass
    return {
        "lang": lang,
        "replenishment": brain.predict_replenishment(_parse_dates(req.purchases), today),
        "notify_time": brain.best_notify_hour(hours),
        "recommendation": rec,
        "tip": ai.generate_tip(rec["top_cuisine"], req.regular_products, lang),
    }
