# -*- coding: utf-8 -*-
"""Офлайн-прогон (без LLM-сервера). Проверяет всю детерминированную логику.
Запуск:  python smoke_test.py
LLM-эндпоинты (categorize не-ru/en, chat-онтоп, parse по URL) требуют Ollama/ключа."""
import datetime as dt
from fastapi.testclient import TestClient

import pora_llm as ai
import recipe
from main import app

c = TestClient(app)


def show(t, r):
    print(f"\n— {t} [{r.status_code}] {r.json()}")


show("GET /health", c.get("/health"))

# мультиязычная категоризация (быстрый путь RU + EN)
show("categorize RU+EN", c.post("/v1/categorize", json={"names": ["авокадо", "кефир", "milk", "chicken breast", "baguette"]}))

# пополнение
purchases = []
for prod, every in [("Молоко", 7), ("Хлеб", 3), ("Coffee", 14)]:
    d = dt.date(2026, 6, 18) - dt.timedelta(days=60)
    while d < dt.date(2026, 6, 18):
        purchases.append({"product": prod, "date": d.isoformat()})
        d += dt.timedelta(days=every)
show("replenishment", c.post("/v1/replenishment", json={"today": "2026-06-18", "purchases": purchases}))

show("notify-time", c.post("/v1/notify-time", json={"opens": [f"2026-06-{d:02d}T18:30:00" for d in range(1, 20)]}))
show("recommend", c.post("/v1/recommend", json={"recipe_imports": ["Карбонара"], "regular_products": ["Молоко", "Сыр", "Паста"]}))

# tip — без ключа отдаёт fallback на нужном языке
show("tip (fallback)", c.post("/v1/tip", json={"top_cuisine": "Итальянская", "frequent": ["паста"], "lang": "ru"}))

# chat — офтоп режется роутером БЕЗ LLM (отказ на языке пользователя)
show("chat off-topic RU", c.post("/v1/chat", json={"message": "напиши python код для сортировки"}))
show("chat off-topic EN", c.post("/v1/chat", json={"message": "write me a SQL query"}))

# определение языка
print("\n— detect_lang:",
      ai.detect_lang("дай рецепт борща"), ai.detect_lang("give me a recipe"),
      ai.detect_lang("dame una receta"), ai.detect_lang("レシピを教えて"))

# JSON-LD парсер офлайн
html = '<script type="application/ld+json">{"@type":"Recipe","name":"Carbonara","recipeIngredient":["Spaghetti 400g","Eggs 4"]}</script>'
print("— extract_jsonld (ступень 1):", recipe.extract_jsonld(html))

print("\n✅ офлайн smoke test пройден")
