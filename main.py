# -*- coding: utf-8 -*-
"""Pora AI — HTTP-сервис (мультиязычный). Go-бэкенд обращается к нему по REST.

Запуск:   uvicorn main:app --port 8000      |   docker compose up
Доки:     http://localhost:8000/docs
"""
from __future__ import annotations

import datetime as dt

from fastapi import FastAPI, HTTPException

import brain
import pora_llm as ai
from schemas import (
    BriefRequest, CategorizeRequest, ChatRequest, NotifyTimeRequest,
    ParseRecipeRequest, RecommendRequest, ReplenishmentRequest, TipRequest,
)

app = FastAPI(title="Pora AI", version="2.0.0",
              description="Мультиязычный ИИ Pora: пополнение, категоризация, время пуша, "
                          "рекомендации, парсинг рецептов, советы и заскоупленный чат.")

_cat = brain.Categorizer().fit()
FAST_LANGS = {"ru", "en"}  # языки быстрого классификатора; остальное → LLM


@app.get("/health")
def health():
    return {"status": "ok", "version": app.version, "llm_enabled": ai.llm_enabled(),
            "sections": brain.SECTIONS, "fast_langs": sorted(FAST_LANGS),
            "refusal_langs": sorted(ai.REFUSALS)}


def _parse_dates(purchases):
    try:
        return [{"product": p.product, "date": dt.date.fromisoformat(p.date)} for p in purchases]
    except ValueError as e:
        raise HTTPException(400, f"bad date: {e}")


@app.post("/v1/replenishment")
def replenishment(req: ReplenishmentRequest):
    today = dt.date.fromisoformat(req.today) if req.today else dt.date.today()
    return {"today": today.isoformat(), "predictions": brain.predict_replenishment(_parse_dates(req.purchases), today)}


@app.post("/v1/categorize")
def categorize(req: CategorizeRequest):
    results = []
    for name in req.names:
        lang = req.lang or ai.detect_lang(name)
        if lang in FAST_LANGS:
            key, conf = _cat.predict(name)
            method = "fast"
            if conf < 0.45 and ai.llm_enabled():     # неуверенно → уточняем LLM
                key, conf, method = *ai.categorize_llm(name), "llm"
        else:
            key, conf = ai.categorize_llm(name)        # другой язык → LLM
            method = "llm"
        results.append({"name": name, "section": key, "section_label": brain.section_label(key, lang),
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


@app.post("/v1/recommend")
def recommend(req: RecommendRequest):
    return brain.recommend(req.recipe_imports, req.regular_products)


@app.post("/v1/parse-recipe")
def parse_recipe(req: ParseRecipeRequest):
    try:
        return ai.parse_recipe(req.url, _cat).model_dump()
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
