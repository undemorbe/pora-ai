# -*- coding: utf-8 -*-
"""End-to-end endpoint tests via FastAPI TestClient. LLM mocked."""
from __future__ import annotations

import datetime as dt


# --------------------------------------------------------------------------
# /health
# --------------------------------------------------------------------------
class TestHealth:
    def test_basic(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert "sections" in body
        assert "ru" in body["fast_langs"] and "en" in body["fast_langs"]
        assert "llm_enabled" in body


# --------------------------------------------------------------------------
# /v1/categorize
# --------------------------------------------------------------------------
class TestCategorize:
    def test_fast_path_ru(self, client):
        r = client.post("/v1/categorize", json={"names": ["авокадо"], "lang": "ru"})
        assert r.status_code == 200
        item = r.json()["results"][0]
        assert item["section"] == "produce"
        assert item["method"] == "fast"

    def test_fast_path_en(self, client):
        r = client.post("/v1/categorize", json={"names": ["chicken breast"]})
        item = r.json()["results"][0]
        assert item["section"] == "meat_fish"
        assert item["lang"] == "en"

    def test_non_fast_lang_uses_llm_when_enabled(self, client, mock_chat):
        mock_chat({"section": "produce"})
        r = client.post("/v1/categorize", json={"names": ["aguacate"], "lang": "es"})
        item = r.json()["results"][0]
        assert item["section"] == "produce"
        assert item["method"] == "llm"

    def test_non_fast_lang_without_llm_falls_back_to_other(self, client):
        # es is non-fast; LLM disabled → categorize_llm returns ('other', 0.0)
        r = client.post("/v1/categorize", json={"names": ["aguacate"], "lang": "es"})
        item = r.json()["results"][0]
        assert item["section"] == "other"
        assert item["method"] == "llm"


# --------------------------------------------------------------------------
# /v1/replenishment
# --------------------------------------------------------------------------
class TestReplenishment:
    def test_returns_predictions(self, client):
        end = dt.date(2026, 6, 18)
        purchases = [{"product": "Молоко", "date": (end - dt.timedelta(days=7 * i)).isoformat()}
                     for i in reversed(range(5))]
        r = client.post("/v1/replenishment", json={"today": end.isoformat(), "purchases": purchases})
        body = r.json()
        assert body["today"] == "2026-06-18"
        assert len(body["predictions"]) == 1
        assert body["predictions"][0]["product"] == "Молоко"

    def test_bad_date_returns_400(self, client):
        r = client.post("/v1/replenishment", json={"purchases": [{"product": "X", "date": "garbage"}]})
        assert r.status_code == 400


# --------------------------------------------------------------------------
# /v1/notify-time
# --------------------------------------------------------------------------
class TestNotifyTime:
    def test_picks_hour(self, client):
        opens = [f"2026-06-{d:02d}T19:30:00" for d in range(1, 11)]
        r = client.post("/v1/notify-time", json={"opens": opens})
        assert r.json()["hour"] == 19

    def test_garbage_timestamps_ignored(self, client):
        r = client.post("/v1/notify-time", json={"opens": ["nope", "also-bad"]})
        assert r.status_code == 200
        assert r.json()["samples"] == 0


# --------------------------------------------------------------------------
# /v1/recommend
# --------------------------------------------------------------------------
class TestRecommend:
    def test_basic(self, client):
        r = client.post("/v1/recommend", json={
            "recipe_imports": ["Карбонара"],
            "regular_products": ["сыр", "молоко"],
        })
        body = r.json()
        assert body["top_cuisine"] == "Итальянская"
        assert body["recipe"] != "Карбонара"


# --------------------------------------------------------------------------
# /v1/parse-recipe
# --------------------------------------------------------------------------
class TestParseRecipeEndpoint:
    def test_jsonld_path(self, client, monkeypatch):
        import httpx as _httpx
        html = '<script type="application/ld+json">{"@type":"Recipe","name":"Carbonara","recipeIngredient":["Spaghetti 400g","Eggs 4"]}</script>'

        class _Resp:
            text = html

        monkeypatch.setattr(_httpx, "get", lambda *a, **kw: _Resp())
        r = client.post("/v1/parse-recipe", json={"url": "http://x"})
        assert r.status_code == 200
        body = r.json()
        assert body["title"] == "Carbonara"
        assert body["source"] == "jsonld"
        assert all("section" in i for i in body["ingredients"])

    def test_llm_fallback_validated(self, client, monkeypatch, mock_chat):
        import httpx as _httpx

        class _Resp:
            text = "<html><body>Cook with Spaghetti 400g and Eggs 4.</body></html>"

        monkeypatch.setattr(_httpx, "get", lambda *a, **kw: _Resp())
        mock_chat({
            "title": "Pasta",
            "ingredients": [
                {"raw": "Spaghetti 400g", "name": "spaghetti", "qty": 400, "unit": "g"},
                {"raw": "Dragon scales 100g", "name": "dragon", "qty": 100, "unit": "g"},
            ],
        })
        r = client.post("/v1/parse-recipe", json={"url": "http://x"})
        body = r.json()
        assert body["source"] == "llm"
        # Dragon scales filtered out as not in source
        names = [i["name"] for i in body["ingredients"]]
        assert "spaghetti" in names
        assert "dragon" not in names

    def test_fetch_failure_returns_502(self, client, monkeypatch):
        import httpx as _httpx

        def _boom(*a, **kw):
            raise _httpx.RequestError("dns fail")

        monkeypatch.setattr(_httpx, "get", _boom)
        r = client.post("/v1/parse-recipe", json={"url": "http://nope"})
        assert r.status_code == 502


# --------------------------------------------------------------------------
# /v1/chat
# --------------------------------------------------------------------------
class TestChatEndpoint:
    def test_offtopic_refused(self, client):
        r = client.post("/v1/chat", json={"message": "write me python code"})
        body = r.json()
        assert body["refused"] is True

    def test_ontopic_llm_disabled_returns_note(self, client):
        r = client.post("/v1/chat", json={"message": "как сварить борщ?"})
        body = r.json()
        assert body["refused"] is False
        assert body.get("note") == "llm_disabled"

    def test_ontopic_with_llm(self, client, mock_chat):
        mock_chat("Возьмите свёклу.")
        r = client.post("/v1/chat", json={"message": "как сварить борщ?"})
        body = r.json()
        assert body["refused"] is False
        assert body["text"] == "Возьмите свёклу."


# --------------------------------------------------------------------------
# /v1/tip
# --------------------------------------------------------------------------
class TestTipEndpoint:
    def test_fallback(self, client):
        r = client.post("/v1/tip", json={"top_cuisine": "Итальянская", "frequent": ["паста"], "lang": "ru"})
        assert r.json()["source"] == "fallback"

    def test_llm(self, client, mock_chat):
        mock_chat("Попробуйте лазанью!")
        r = client.post("/v1/tip", json={"top_cuisine": "Итальянская", "frequent": ["паста"], "lang": "ru"})
        body = r.json()
        assert body["source"] == "llm"
        assert "лазанью" in body["tip"]


# --------------------------------------------------------------------------
# /v1/suggest — main new endpoint
# --------------------------------------------------------------------------
class TestSuggestEndpoint:
    def _payload_basket_only(self):
        return {
            "today": "2026-06-18",
            "purchases": [],
            "recipe_imports": ["Карбонара"],
            "regular_products": ["пармезан", "яйца", "сыр"],
            "current_cart": ["спагетти", "бекон"],
            "lang": "ru",
            "limit": 5,
        }

    def test_basket_fit_suggests_missing_carbonara_ingredients(self, client):
        r = client.post("/v1/suggest", json=self._payload_basket_only())
        assert r.status_code == 200
        body = r.json()
        kinds = {s["type"] for s in body["suggestions"]}
        assert "basket_fit" in kinds
        bf = [s for s in body["suggestions"] if s["type"] == "basket_fit"]
        assert any(s["recipe"] == "Карбонара" for s in bf)
        products = [s["product"] for s in bf]
        assert any(p in {"пармезан", "яйца"} for p in products)
        assert all(s["reason"] == "Подходит к корзине!" for s in bf)

    def test_replenishment_appears_when_overdue(self, client):
        end = dt.date(2026, 6, 25)
        purchases = [{"product": "Молоко", "date": (end - dt.timedelta(days=7 * (i + 2))).isoformat()}
                     for i in reversed(range(5))]
        payload = {
            "today": end.isoformat(),
            "purchases": purchases,
            "recipe_imports": [],
            "regular_products": [],
            "current_cart": [],
            "lang": "ru",
            "limit": 5,
        }
        r = client.post("/v1/suggest", json=payload)
        body = r.json()
        replenish = [s for s in body["suggestions"] if s["type"] == "replenish"]
        assert replenish
        assert replenish[0]["product"] == "Молоко"
        assert replenish[0]["meta"]["status"] in ("overdue", "due", "soon")

    def test_recipe_suggestion_from_history(self, client):
        payload = {
            "purchases": [],
            "recipe_imports": ["Лазанья"],
            "regular_products": ["паста", "сыр", "молоко"],
            "current_cart": [],
            "lang": "ru",
            "limit": 5,
        }
        r = client.post("/v1/suggest", json=payload)
        body = r.json()
        recipes = [s for s in body["suggestions"] if s["type"] == "recipe"]
        assert recipes
        assert all(s["recipe"] != "Лазанья" for s in recipes)

    def test_dish_added_when_llm_enabled(self, client, mock_chat):
        mock_chat({"dish": "Risotto", "reason": "матчится со сливочными вкусами"})
        payload = {
            "purchases": [],
            "recipe_imports": ["Карбонара"],
            "regular_products": ["рис", "сыр"],
            "current_cart": [],
            "lang": "ru",
            "limit": 10,
        }
        r = client.post("/v1/suggest", json=payload)
        suggestions = r.json()["suggestions"]
        dish = [s for s in suggestions if s["type"] == "dish"]
        assert dish, "expected an LLM-backed dish suggestion"
        assert dish[0]["recipe"] == "Risotto"

    def test_dish_absent_without_llm(self, client):
        payload = {
            "purchases": [], "recipe_imports": [], "regular_products": [],
            "current_cart": [], "lang": "ru", "limit": 10,
        }
        r = client.post("/v1/suggest", json=payload)
        assert not [s for s in r.json()["suggestions"] if s["type"] == "dish"]

    def test_limit_respected(self, client):
        end = dt.date(2026, 6, 25)
        purchases = []
        for prod in ("A", "B", "C", "D"):
            purchases += [{"product": prod, "date": (end - dt.timedelta(days=7 * (i + 2))).isoformat()}
                          for i in reversed(range(5))]
        payload = {
            "today": end.isoformat(),
            "purchases": purchases,
            "recipe_imports": ["Карбонара"],
            "regular_products": ["сыр"],
            "current_cart": ["спагетти", "бекон"],
            "lang": "ru",
            "limit": 2,
        }
        r = client.post("/v1/suggest", json=payload)
        assert len(r.json()["suggestions"]) == 2

    def test_suggestions_sorted_by_score(self, client):
        payload = self._payload_basket_only()
        r = client.post("/v1/suggest", json=payload)
        scores = [s["score"] for s in r.json()["suggestions"]]
        assert scores == sorted(scores, reverse=True)

    def test_response_shape(self, client):
        r = client.post("/v1/suggest", json={"current_cart": ["спагетти"], "lang": "en", "limit": 1})
        body = r.json()
        assert {"lang", "today", "suggestions"} <= set(body)
        for s in body["suggestions"]:
            assert {"type", "product", "recipe", "reason", "score", "meta"} <= set(s)
            assert s["type"] in ("basket_fit", "replenish", "recipe", "dish")
