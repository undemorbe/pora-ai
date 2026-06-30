# Pora AI — мультиязычный ИИ-сервис (v2)

ИИ Pora как отдельный HTTP-сервис. **Go-бэкенд зовёт его по REST.** Сервис
**stateless** (Go присылает данные в запросе, в БД сервис не лезет) и
**мультиязычный**: отвечает на языке пользователя, категоризация работает на любом
языке, отказы локализованы.

```
Flutter ─REST→ Go backend ─REST→ Pora AI ──→ LLM (локальная Ollama ИЛИ облако)
```

Один и тот же код работает с локальной Ollama и с облаком — отличие только в env.

## Запуск

**Всё одной командой (Pora AI + локальная Ollama):**
```bash
docker compose up -d
docker compose exec ollama ollama pull qwen3   # один раз скачать модель
# доки: http://localhost:8000/docs
```

**Локально без Docker:**
```bash
pip install -r requirements.txt
uvicorn main:app --port 8000
```

**Офлайн-проверка логики (без LLM-сервера):**
```bash
python smoke_test.py
```

## Конфиг (env) — Ollama vs облако

| Переменная | Ollama (локально) | Облако (MVP) |
|---|---|---|
| `LLM_BASE_URL` | `http://ollama:11434/v1` | `https://api.openai.com/v1` |
| `LLM_API_KEY`  | `ollama` (любой непустой) | `sk-...` |
| `LLM_MODEL`    | `qwen3` (или `pora-chef`) | `gpt-4o-mini` |

Без ключа сервис тоже рабочий: ML-эндпоинты — полностью, RU/EN-категоризация —
быстрый классификатор, `parse-recipe` — через JSON-LD, `tip`/`chat` — отказ/фолбэк.

## Мультиязычность — как устроено

- **Разделы магазина — ключи** (`dairy, produce, bakery, pantry, drinks, meat_fish, other`),
  не текст. Приложение локализует отображение; сервис также вернёт `section_label`
  для переданного `lang`.
- Поле **`lang`** в запросах опционально: не передал — сервис определит сам по тексту.
  Лучше передавать локаль приложения (точнее, чем автоопределение).
- **Категоризация:** RU/EN — мгновенный классификатор; другие языки и неуверенные
  случаи → LLM (строгий enum ключей). Работает на любом языке.
- **chat/tip/parse-recipe** отвечают/парсят на языке пользователя. Отказы — на 8 языках.

## Контракт (то, что зовёт Go)

`GET /health` → `{status, llm_enabled, sections, fast_langs, refusal_langs}`

`POST /v1/replenishment` `{today?, purchases:[{product,date}]}` → `{today, predictions:[{product,every_days,due_date,days_left,confidence,status}]}`

`POST /v1/categorize` `{names:[...], lang?}` →
```json
{"results":[{"name":"авокадо","section":"produce","section_label":"Овощи и фрукты",
             "confidence":0.75,"lang":"ru","method":"fast"}]}
```

`POST /v1/notify-time` `{opens:[ISO datetime]}` → `{hour, window_share, samples}`

`POST /v1/recommend` `{recipe_imports:[...], regular_products:[...]}` → `{top_cuisine,recipe,cuisine,pantry_match}`

`POST /v1/parse-recipe` `{url, lang?}` → `{title, ingredients:[{raw,name,qty,unit,section}], source}`  *(jsonld|llm|none)*

`POST /v1/tip` `{top_cuisine, frequent:[...], lang?}` → `{tip, lang, source}`

`POST /v1/chat` `{message, lang?}` → `{text, lang, refused}`  *(заскоупленный ассистент, офтоп → отказ)*

`POST /v1/brief` `{today?, purchases, opens, recipe_imports, regular_products, lang?}` →
`{lang, replenishment, notify_time, recommendation, tip}`  *(весь дневной брифинг одним вызовом)*

## Как подключить из Go

1. `docker compose up -d` + `ollama pull qwen3` (или укажи облачные env).
2. В Go используй `client_example.go` (`ai.New("http://pora-ai:8000")`).
3. Прокидывай локаль пользователя в поле `lang` (из Flutter → Go → сюда).
4. Пополнение считай ночным job'ом (`/v1/replenishment` по каждому домохозяйству →
   в таблицу `predictions`), категоризацию/парсинг — по запросу.

## Файлы
- `brain.py` — ML/стат (пополнение, классификатор RU/EN, время пуша, рекомендации, ключи разделов)
- `pora_llm.py` — мультиязычный LLM-модуль (chat/categorize/tip/recipe, detect_lang, отказы)
- `main.py` — FastAPI-эндпоинты
- `schemas.py` — контракт (Pydantic)
- `docker-compose.yml` — Pora AI + Ollama
- `client_example.go` — Go-клиент
- `llm.py` — устаревший шим (используйте `pora_llm`)

Безопасность: ключ LLM только на сервере; не выставляй `:11434` Ollama в публичный
интернет без авторизации; историю покупок шифруй на стороне бэкенда.
