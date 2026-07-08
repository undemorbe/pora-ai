# Pora AI — мультиязычный ИИ-сервис

Stateless HTTP-микросервис (FastAPI), который Go-бэкенд вызывает по REST.
Данные живут в Go (БД/история покупок), Pora AI получает их в каждом запросе и
возвращает категории, рецепты, советы, прогнозы и подсказки. Один и тот же код
работает с локальной Ollama и с любым OpenAI-совместимым облаком (OpenAI,
Groq, Together, Fireworks, Mistral La Plateforme и т.д.) — отличие только в env.

```
Flutter ─REST→ Go backend ─REST→ Pora AI ──→ LLM (Ollama / OpenAI-совместимый API)
```

Сервис **stateless** (в БД не лезет), **мультиязычный** (15 языков отказа,
скрипт-определение языка для 14+ языков), **graceful-degrade** (без LLM-ключа
остаются ML-эндпоинты, fast-классификатор RU/EN, JSON-LD парсер рецептов).

---

## Содержание

1. [Быстрый старт](#быстрый-старт)
2. [Конфигурация (env)](#конфигурация-env)
3. [Архитектура](#архитектура)
4. [Эндпоинты — полный справочник](#эндпоинты--полный-справочник)
5. [Парсинг рецепта по URL — usecase](#парсинг-рецепта-по-url--usecase)
6. [Гибридные советы `/v1/suggest` — usecase](#гибридные-советы-v1suggest--usecase)
7. [Кастомные секции (своя таксономия)](#кастомные-секции-своя-таксономия)
8. [Мультиязычность](#мультиязычность)
9. [Тесты](#тесты)
10. [Подключение из Go](#подключение-из-go)
11. [Безопасность и прод](#безопасность-и-прод)

---

## Быстрый старт

### Вариант 1 — всё одной командой через Docker + Ollama

```bash
docker compose up -d
docker compose exec ollama ollama pull qwen3      # один раз — скачать модель
# Документация: http://localhost:8000/docs (Swagger UI)
# Здоровье:     curl http://localhost:8000/health
```

### Вариант 2 — облако (OpenAI / любой OpenAI-совместимый)

В `docker-compose.yml` замени блок переменных Ollama на:
```yaml
environment:
  LLM_BASE_URL: "https://api.openai.com/v1"
  LLM_API_KEY:  "sk-..."
  LLM_MODEL:    "gpt-4o-mini"
```
Запусти `docker compose up -d` — Ollama больше не нужна.

### Вариант 3 — локально без Docker

```bash
pip install -r requirements.txt
LLM_API_KEY=sk-... LLM_MODEL=gpt-4o-mini uvicorn main:app --port 8000 --reload
```

### Офлайн-проверка детерминированной логики

```bash
python smoke_test.py        # быстрый демо-прогон через TestClient
pytest                      # полный suite (162 теста, LLM замокан)
```

---

## Конфигурация (env)

Три обязательные переменные. Пустой `LLM_API_KEY` отключает LLM
(graceful-degrade), сервис остаётся рабочим.

| Переменная     | Ollama (локально)              | OpenAI / cloud              | Без LLM (degraded)        |
|----------------|--------------------------------|-----------------------------|---------------------------|
| `LLM_BASE_URL` | `http://ollama:11434/v1`       | `https://api.openai.com/v1` | (любое)                   |
| `LLM_API_KEY`  | `ollama` (любой непустой)      | `sk-...`                    | **пусто** → LLM выключен  |
| `LLM_MODEL`    | `qwen3` / `pora-chef`          | `gpt-4o-mini`               | (не используется)         |

Что работает без LLM-ключа:
- `/v1/replenishment`, `/v1/notify-time`, `/v1/recommend`, `/v1/suggest` —
  полностью (чистая ML/стат-логика).
- `/v1/categorize` — fast-классификатор RU/EN; другие языки и кастомные
  секции возвращают `other`.
- `/v1/parse-recipe` — только JSON-LD path; если на странице нет JSON-LD —
  пустой результат с `source: "none"`.
- `/v1/chat` — офтоп режется роутером и возвращает локализованный отказ;
  on-topic — отказ с пометкой `note: "llm_disabled"`.
- `/v1/tip` — отдаёт `fallback` на нужном языке.

---

## Архитектура

Три слоя, строгая граница. Каждый импортирует только нижестоящий.

```
main.py        ──┐  FastAPI-эндпоинты, роутинг, локализация лейблов
                 │
pora_llm.py    ──┤  единый LLM-модуль (chat, categorize, parse_recipe,
                 │  tip, dish, html_to_text, web_fetch, validate_against_source)
                 │
brain.py       ──┘  чистая локальная логика (no network, no LLM):
                    predict_replenishment, Categorizer, suggest_*,
                    merge_suggestions, recommend, best_notify_hour
```

| Файл               | Назначение                                                 |
|--------------------|------------------------------------------------------------|
| `main.py`          | FastAPI-эндпоинты + routing rule                           |
| `pora_llm.py`      | LLM-вызовы, web_fetch, language detect, refusals, prompts  |
| `brain.py`         | Чистый ML/стат — без сети и LLM                             |
| `schemas.py`       | Pydantic — wire-контракт с Go                              |
| `tests/`           | pytest suite (162 теста, LLM замокан)                       |
| `smoke_test.py`    | Демо-прогон через TestClient                                |
| `client_example.go`| Референс Go-клиента                                         |
| `llm.py`           | Deprecated shim — оставлен для обратной совместимости       |

Подробности — в [CLAUDE.md](CLAUDE.md).

---

## Эндпоинты — полный справочник

Все ответы — `application/json`, UTF-8. Все ошибки клиента — `400` с `{detail}`,
ошибки апстрима (LLM/HTTP fetch) — `502`. Поле `lang` везде опционально.

### `GET /health`

Сервисная проверка. Возвращает версию, состояние LLM, доступные секции и языки.

```json
{
  "status": "ok",
  "version": "2.0.0",
  "llm_enabled": true,
  "sections": ["dairy", "produce", "bakery", "pantry", "drinks", "meat_fish", "other"],
  "fast_langs": ["en", "ru"],
  "refusal_langs": ["ar", "de", "en", "es", "fr", "he", "hi", "it", "ja", "ko", "pl", "pt", "ru", "tr", "zh"]
}
```

### `POST /v1/replenishment` — прогноз "когда закончится"

**Тело:**
```json
{
  "today": "2026-06-18",
  "purchases": [
    {"product": "Молоко", "date": "2026-06-04"},
    {"product": "Молоко", "date": "2026-06-11"},
    {"product": "Молоко", "date": "2026-06-17"}
  ]
}
```
**Ответ:** медианный интервал, confidence (1 − CV), статус-бакет.
```json
{
  "today": "2026-06-18",
  "predictions": [
    {"product": "Молоко", "every_days": 7.0, "due_date": "2026-06-24",
     "days_left": 6, "confidence": 1.0, "status": "ok", "events": 3}
  ]
}
```
Статусы: `overdue` (просрочено), `due` (≤1 день), `soon` (≤3 дня), `ok`.
Нужно ≥3 закупки на продукт — иначе пропускается.

### `POST /v1/categorize` — раздел магазина для названия

**Тело:**
```json
{
  "names": ["авокадо", "chicken breast"],
  "lang": "ru",
  "sections": null,
  "section_labels": null
}
```

Routing:
- `sections` не задано **и** `lang ∈ {ru, en}` → fast-классификатор; если
  `confidence < 0.45` и LLM включён — эскалация в LLM.
- `sections` не задано и язык не fast → LLM per-item.
- `sections` задано → один **батчевый** LLM-вызов на все имена (см.
  [Кастомные секции](#кастомные-секции-своя-таксономия)).

**Ответ:**
```json
{"results": [
  {"name": "авокадо", "section": "produce", "section_label": "Овощи и фрукты",
   "confidence": 0.75, "lang": "ru", "method": "fast"}
]}
```

### `POST /v1/notify-time` — лучший час уведомления

**Тело:**
```json
{"opens": ["2026-06-18T18:30:00", "2026-06-18T19:01:00", "2026-06-17T19:15:00"]}
```
**Ответ:** `{hour, window_share, samples}` — пик с вечерним приоритетом
(16:00–21:00); `window_share` — доля окна ±1 час вокруг пика.

### `POST /v1/recommend` — один рецепт под вкус

**Тело:** `{recipe_imports, regular_products, lang}`.
**Ответ:** `{top_cuisine, recipe, cuisine, pantry_match}`.

### `POST /v1/parse-recipe` — извлечь рецепт по URL

См. [usecase ниже](#парсинг-рецепта-по-url--usecase).

**Тело:**
```json
{
  "url": "https://example.com/recipe/carbonara",
  "lang": "ru",
  "sections": null
}
```
**Ответ:** `{title, ingredients[], source}`, где `source ∈ {jsonld, llm, none}`.
Каждый ингредиент — `{raw, name, qty, unit, section}`.

### `POST /v1/tip` — короткий совет от LLM (fallback без ключа)

**Тело:** `{top_cuisine, frequent[], lang}`. **Ответ:** `{tip, lang, source}`,
`source ∈ {llm, fallback}`.

### `POST /v1/chat` — мультиязычный заскоупленный чат

Офтоп режется до LLM-вызова (cost guard). Промпт `SCOPE_SYSTEM` дополнительно
запрещает медицинские/юридические/финансовые советы и утечку системного
сообщения.

**Тело:** `{message, lang}`. **Ответ:** `{text, lang, refused}` (+ `note` при
отключённом LLM).

### `POST /v1/brief` — дневной брифинг одним вызовом

Композит из `replenishment + notify_time + recommend + tip`. Удобно дёргать
из утреннего push-сервиса.

### `POST /v1/suggest` — главный новый эндпоинт ⭐

Гибридные подсказки (basket-fit + replenish + recipe + dish). См.
[usecase ниже](#гибридные-советы-v1suggest--usecase).

---

## Парсинг рецепта по URL — usecase

> "Пользователь скинул URL с блюдом — LLM **распарсила**, посмотрела и
> **вернула в конкретном JSON-формате ингредиенты**. Обязательно распарсить,
> **не выдумать**."

Pipeline `/v1/parse-recipe` ([pora_llm.py:parse_recipe](pora_llm.py)):

1. **`web_fetch`** — браузероподобный HTTP-запрос:
   - реалистичный `User-Agent` (Chrome 120 + `PoraBot/2.0` суффикс)
   - `Accept-Language` берётся из `lang` (ru → `ru,ru;q=0.9,en;q=0.5`)
   - редиректы, 2 retry на `429/5xx`, cap 400 КБ
   - `extract_main_content` — берёт `<main>`/`<article>` или вырезает
     `<nav>/<header>/<footer>/<aside>/<form>` из тела
2. **`extract_jsonld`** — бесплатный путь: ищет `<script type="application/ld+json">`
   с `@type: "Recipe"` (включая внутри `@graph` и `@type` массивов). Если нашёл —
   `source: "jsonld"`, LLM не вызывается.
3. **`extract_recipe_from_text`** — LLM-фолбэк по очищенному тексту:
   - строгий system-prompt: "ONLY use ingredients that LITERALLY appear in the
     text. NEVER invent, infer, translate, complete or substitute."
   - `response_format=json_schema, strict=True`
4. **`validate_against_source`** — anti-hallucination guard:
   - каждый ингредиент проверяется на наличие `raw` или `name` (≥3 символа)
     в исходном тексте (case-insensitive, нормализованный whitespace)
   - выдуманное удаляется; если **всё** выдумано — `source` переключается на
     `"none"`, список пустой
5. **Section tagging** — каждому ингредиенту присваивается раздел магазина:
   - дефолтная таксономия → быстрый `Categorizer`
   - кастомные `sections` → батчевый `categorize_llm_batch` (один LLM-вызов
     на весь список)

**Пример запроса:**
```bash
curl -X POST http://localhost:8000/v1/parse-recipe \
  -H 'Content-Type: application/json' \
  -d '{"url": "https://example.com/recipes/carbonara", "lang": "ru"}'
```
**Пример ответа (JSON-LD path):**
```json
{
  "title": "Спагетти карбонара",
  "ingredients": [
    {"raw": "Спагетти 400 г", "name": null, "qty": null, "unit": null, "section": "pantry"},
    {"raw": "Яйца 4 шт",     "name": null, "qty": null, "unit": null, "section": "dairy"},
    {"raw": "Бекон 200 г",   "name": null, "qty": null, "unit": null, "section": "meat_fish"}
  ],
  "source": "jsonld"
}
```

---

## Гибридные советы `/v1/suggest` — usecase

> "Пользователь по статистике часто парсил X рецепты и добавлял в список
> покупок X ингредиенты. LLM должна **советовать взять X продукт**
> ("Подходит к корзине!") или **предложить взять X продукт** (тк тот часто
> берётся с периодом и погрешностью) или **предложить рецепт** или
> **предложить блюдо**."

Один эндпоинт — четыре типа подсказок в одном уровнем поле `type`:

| `type`        | Когда срабатывает                                    | Источник     |
|---------------|------------------------------------------------------|--------------|
| `basket_fit`  | В корзине есть товар, входящий в каталожный рецепт   | brain        |
| `replenish`   | Товар прогнозируется как overdue/due/soon            | brain        |
| `recipe`      | Каталог рецептов под top_cuisine и pantry overlap    | brain        |
| `dish`        | LLM-сгенерированное блюдо под предпочтения           | LLM          |

**Запрос:**
```json
{
  "today": "2026-06-18",
  "purchases": [...],
  "recipe_imports": ["Карбонара"],
  "regular_products": ["пармезан", "молоко", "яйца"],
  "current_cart":     ["спагетти", "бекон"],
  "lang": "ru",
  "limit": 5
}
```

**Ответ** — отсортирован по `score` desc, дубликаты по `(type, product, recipe)`
схлопнуты:
```json
{
  "lang": "ru",
  "today": "2026-06-18",
  "suggestions": [
    {
      "type": "basket_fit", "product": "пармезан", "recipe": "Карбонара",
      "reason": "Подходит к корзине!", "score": 0.9,
      "meta": {"matched_cart_item": "бекон", "cuisine": "Итальянская", "in_regular": true}
    },
    {
      "type": "basket_fit", "product": "яйца", "recipe": "Карбонара",
      "reason": "Подходит к корзине!", "score": 0.9,
      "meta": {"matched_cart_item": "бекон", "cuisine": "Итальянская", "in_regular": true}
    },
    {
      "type": "recipe", "product": null, "recipe": "Мак-н-чиз",
      "reason": "Подойдёт под ваш вкус", "score": 0.8,
      "meta": {"cuisine": "Итальянская", "pantry_match": 0.5, "missing": ["масло", "паста"]}
    },
    {
      "type": "replenish", "product": "Молоко", "recipe": null,
      "reason": "Скоро закончится — пора пополнить", "score": 0.8,
      "meta": {"status": "due", "days_left": 0, "due_date": "2026-06-18", "every_days": 7.0}
    },
    {
      "type": "dish", "product": null, "recipe": "Лазанья болоньезе",
      "reason": "Близко к вашим вкусам и хорошо ляжет на регулярные покупки",
      "score": 0.5, "meta": {"top_cuisine": "Итальянская", "source": "llm"}
    }
  ]
}
```

Scoring:
- `basket_fit`: `0.6 + 0.3·(в regular_products?) + 0.2·(рецепт не пробовали?)`
- `replenish`: `urgency × confidence`, urgency ∈ {overdue: 1.0, due: 0.8, soon: 0.6}
- `recipe`: `pantry_match + 0.3·(совпала top_cuisine?)`
- `dish`: фиксировано `0.5` (предельная новизна)

`limit` ∈ `[1, 20]`. Без `current_cart` нет `basket_fit`. Без `purchases` нет
`replenish`. Без LLM-ключа нет `dish`.

---

## Кастомные секции (своя таксономия)

Не нравится `dairy/produce/bakery/...`? Передавай свой набор ключей в
`/v1/categorize` и `/v1/parse-recipe`:

```bash
curl -X POST http://localhost:8000/v1/categorize \
  -H 'Content-Type: application/json' \
  -d '{
    "names": ["стейк", "шпинат", "молоко"],
    "sections": ["meat_section", "veggies", "dairy_section", "other"],
    "section_labels": {
      "meat_section":  "Мясной отдел",
      "veggies":       "Овощи и зелень",
      "dairy_section": "Молочный отдел",
      "other":         "Прочее"
    },
    "lang": "ru"
  }'
```

Что произойдёт:
- Fast-классификатор пропускается (он обучен на дефолтных ключах).
- Один **батчевый** LLM-вызов (`categorize_llm_batch`) с твоим enum в
  `response_format=json_schema, strict=True`.
- `section_label` каждого результата — из твоего `section_labels`, иначе сам
  ключ.

Если LLM выключен — все имена получат fallback-секцию: `"other"` если она
есть в твоём списке, иначе первая секция. `confidence: 0.0`.

То же поле `sections` поддерживает `/v1/parse-recipe` — каждому ингредиенту
рецепта проставится раздел из твоей таксономии.

---

## Мультиязычность

`detect_lang` (без зависимостей, чистый regex) поддерживает 14 языков:

| Тип       | Языки                                                    | Метод                          |
|-----------|----------------------------------------------------------|--------------------------------|
| Скрипт    | `ru, ar, hi, he, ko, ja, zh`                             | Уникальный Unicode-диапазон    |
| Диакритика| `pl, tr, pt, es, fr, de`                                 | Скоринг по unique markers      |
| Fallback  | `en`                                                     | Иначе                          |

Алгоритм:
1. Если найден кириллический/арабский/деванагари/иврит/хангыль → возврат сразу.
2. Кана/хирагана → `ja`; чистые Han иероглифы → `zh`.
3. Подсчёт уникальных diacritic-символов по каждому языку; побеждает
   язык с **наибольшим** ненулевым счётом. Ничьи решаются порядком регистрации
   (`pl, tr, pt, es, fr, de`).
4. Иначе — `default` (`en`).

`REFUSALS` — 15 локализованных отказов (включая `it` для итальянского, который
по diacritic-эвристике не отличается от Романских — лучше передавай `lang`
явно). Любой неизвестный язык падает на `en`.

LLM-стороне:
- `SCOPE_SYSTEM` явно требует отвечать **на языке пользователя** (по скрипту
  + регистру). Mixed-language → доминирующий.
- `_CATEGORIZE_SYSTEM` интерпретирует имена кросс-лингво **без перевода** —
  модель выбирает раздел по семантике, не по подстроке.

---

## Тесты

Полный suite — pytest, 162 теста, LLM **никогда не вызывается реально**.

```bash
pytest                                  # все тесты, ~1.5 сек
pytest tests/test_brain.py              # один файл
pytest -k suggest                       # по подстроке имени
pytest -x --tb=long                     # стоп на первой ошибке, длинный трейс
```

Fixtures в [tests/conftest.py](tests/conftest.py):

| Fixture       | Назначение                                                       |
|---------------|------------------------------------------------------------------|
| `client`      | FastAPI `TestClient`. Categorizer `_cat` фитится один раз (импорт `main`). |
| `enable_llm`  | Принудительно ставит `pora_llm.API_KEY = "test-key"` (`llm_enabled() == True`). |
| `mock_chat`   | Подменяет `pora_llm._chat` на скриптованный ответчик: dict (auto-JSON) / str / list[...] / callable. |

Покрытие:
- **`tests/test_brain.py`** — sections, predict_replenishment buckets,
  Categorizer (parametrized, расширенный набор), best_notify_hour, recommend,
  все suggest_* engine'ы, merge_suggestions (dedup + sort).
- **`tests/test_pora_llm.py`** — detect_lang (14 языков), REFUSALS (15 + emoji
  check), guard_on_topic, html_to_text, validate_against_source (hallucination
  drop), extract_main_content, web_fetch (retry, max_retries, Accept-Language),
  extract_jsonld (`@graph` + `@type` arrays + malformed), extract_recipe_from_text
  (mocked LLM, source flip), parse_recipe (jsonld + LLM + custom sections),
  categorize_llm, categorize_llm_batch, suggest_dish_llm, chat off-topic,
  generate_tip fallback/LLM.
- **`tests/test_endpoints.py`** — все эндпоинты через TestClient, включая
  `/v1/categorize` с кастомными секциями, `/v1/parse-recipe` со всеми путями,
  `/v1/suggest` shape/limit/sort/dish on-off, fallback при выключенном LLM.

---

## Подключение из Go

См. [client_example.go](client_example.go) — референс-клиент с типизированными
DTO. Базовый паттерн:

```go
client := ai.New("http://pora-ai:8000")        // в docker-compose сети
resp, _ := client.Suggest(ctx, ai.SuggestReq{
    Today:           time.Now().Format("2006-01-02"),
    Purchases:       purchases,                // []ai.Purchase
    RecipeImports:   []string{"Карбонара"},
    RegularProducts: regulars,
    CurrentCart:     cart,
    Lang:            user.Locale,              // "ru" / "en" / ...
    Limit:           5,
})
```

Куда что вешать на стороне Go:
- `/v1/replenishment` — ночной cron-job по каждому домохозяйству, складывать в
  таблицу `predictions`.
- `/v1/categorize`, `/v1/parse-recipe`, `/v1/suggest` — по запросу из UI.
- `/v1/notify-time` — раз в неделю по событиям `app_open`, складывать в
  `user_preferences.notify_hour`.

---

## Безопасность и прод

- **Ключ LLM** — только на сервере. Никогда не выставляй его в Flutter/браузер.
- **Ollama-порт `11434`** — не выпускай в публичный интернет без auth. В
  `docker-compose.yml` оставь его внутри сети Docker (комментарии в файле).
- **Историю покупок** шифруй на стороне Go (at rest). Pora AI стейтлесс — он
  не сохраняет ничего, что ему пришло.
- **PII**: модель не должна получать email/телефон/адрес. Передавай только
  товарные имена и метаданные.
- **Rate-limit и cost cap** — на стороне Go (Pora AI про деньги ничего не
  знает). Особенно важно для `/v1/parse-recipe` (web fetch + LLM) и
  `/v1/chat`.
- **Локализованные отказы** — всегда возвращаются с HTTP 200 и
  `refused: true`. Не маскируй их как ошибку — это product-level decision.
