# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Pora AI — stateless multilingual HTTP microservice (FastAPI) called by a Go backend. Data lives in Go; this service receives it per request. Same code runs against local Ollama or any OpenAI-compatible cloud — difference is env only.

```
Flutter → Go backend → Pora AI → LLM (Ollama OR OpenAI-compatible)
```

## Commands

```bash
# Full stack (service + Ollama):
docker compose up -d
docker compose exec ollama ollama pull qwen3   # one-time model pull
# Docs: http://localhost:8000/docs

# Local dev without Docker:
pip install -r requirements.txt
uvicorn main:app --port 8000 --reload

# Offline check of deterministic logic (no LLM server needed, uses FastAPI TestClient):
python smoke_test.py

# Full unit/integration suite (LLM mocked via conftest fixtures):
pytest                       # all tests
pytest tests/test_brain.py   # one file
pytest -k suggest            # filter by name
```

`smoke_test.py` remains as the quick demo. The real regression check is the `tests/` suite — extend it when adding endpoints. LLM is never called for real in tests: `tests/conftest.py` provides `mock_chat`/`enable_llm` fixtures that monkey-patch `pora_llm._chat`.

## Env contract

`LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`. If `LLM_API_KEY` is empty, `llm_enabled()` returns `False` and every LLM path falls back gracefully (RU/EN fast classifier, JSON-LD for recipes, localized refusal for chat/tip). Never raise on missing key — preserve this graceful-degrade behavior.

## Architecture

Three layers, strict separation. Touching one usually means touching the others — keep the boundary clean.

- **`schemas.py`** — Pydantic request contract. This is the Go ↔ Pora wire format. Changing a field name or making a field required is a breaking change for the Go client. `lang` is always optional: caller passes app locale OR service auto-detects via `pora_llm.detect_lang`.

- **`brain.py`** — pure local logic (no network, no LLM):
  - `predict_replenishment` — median interval per product, confidence from coefficient of variation, status bucketed (`overdue|due|soon|ok`). Needs ≥3 events per product.
  - `Categorizer` — char-ngram TF-IDF + LogisticRegression, trained on bilingual RU/EN `TRAINING` dict, fit once in `main.py:_cat = brain.Categorizer().fit()` at import time.
  - `best_notify_hour` — evening-window-biased peak from hour histogram.
  - `recommend` — heuristic cuisine + pantry-overlap score over hardcoded `RECIPE_CATALOG`.
  - `suggest_basket_fit` / `suggest_replenish` / `suggest_recipes` / `merge_suggestions` — hybrid recommendation engine behind `/v1/suggest`. All four return uniform `{type, product, recipe, reason, score, meta}` dicts. `merge_suggestions` dedupes by `(type, product, recipe)` (first occurrence wins) and sorts by score desc.
  - `REASON_LABELS` — RU/EN reason strings for the four suggestion types (`basket_fit`, `replenish`, `recipe`, `dish`).
  - `SECTIONS` — canonical English keys (`dairy, produce, …`). Sections are KEYS, never localized text. `SECTION_LABELS` exists only as fallback for callers that don't localize themselves.

- **`pora_llm.py`** — single multilingual LLM module. Imports `brain` (one-way: brain never imports LLM). OpenAI SDK client is lazy (`client()`), so importing the module never touches the network.
  - `detect_lang` — script-based (Cyrillic/CJK/diacritics), no deps.
  - `chat` flow: `guard_on_topic` regex blocklist runs BEFORE the LLM call; off-topic → localized refusal, no tokens spent.
  - `categorize_llm` / `extract_recipe_from_text` use OpenAI structured-output (`response_format=json_schema, strict=True`). Section enum mirrors `brain.SECTIONS`.
  - `parse_recipe`: HTTP fetch → `extract_jsonld` (free path) → LLM fallback over **stripped HTML text** (`html_to_text`) → `validate_against_source` drops any ingredient whose `raw`/`name` does not literally appear in the source (anti-hallucination guard) → quick classifier tags each ingredient with a section. Walks `@graph` and `@type` arrays for JSON-LD recipes. If LLM hallucinates everything, `source` flips back to `"none"` and ingredients are empty.
  - `suggest_dish_llm` — JSON-schema-constrained LLM dish suggester used by `/v1/suggest` for the `dish` slot. Returns `None` when LLM disabled.
  - `REFUSALS` — 8 languages; unknown lang → English.

- **`main.py`** — thin FastAPI endpoints. `/v1/categorize` is the only place where the routing rule lives: `lang ∈ {ru, en}` → fast classifier (and escalate to LLM only when `confidence < 0.45` AND LLM enabled), other languages → LLM directly. Keep that routing in `main.py`, not inside `brain` or `pora_llm`. `/v1/suggest` is the only endpoint that fuses multiple `brain.suggest_*` engines + an LLM dish slot through `brain.merge_suggestions` — keep the fusion glue here, suggestion engines stay pure in `brain.py`.

- **`llm.py`** — deprecated shim, re-exports from `pora_llm`. Do not add new code here; do not delete (Go-side or older imports may reference it).

- **`client_example.go`** — reference Go client. Update when the wire contract in `schemas.py` changes.

## Conventions that aren't obvious

- Section keys come from `brain.SECTIONS` — when adding a section, update both `brain.SECTIONS`, `brain.SECTION_LABELS` (ru + en), `brain.TRAINING` (or LLM-only sections won't classify), and the LLM enum is auto-derived.
- Adding a refusal language: append to `pora_llm.REFUSALS` AND extend `detect_lang` heuristics, otherwise the language never resolves and falls back to English.
- The off-topic `_OFFTOPIC` tuple is intentionally crude — it's a pre-LLM cost guard, not a security boundary. The real scope rule lives in `SCOPE_SYSTEM` prompt.
- Module-level `_cat` in `main.py` is fit once at import; tests/scripts that import `main` pay that cost. `smoke_test.py` relies on this.
- `_chat()` returns `None` when LLM disabled — every caller must handle `None` and fall back, never raise.
- `pora_llm.MODEL_MAIN` / `MODEL_FAST` are the two model envs (`LLM_MODEL`
  and optional `LLM_MODEL_FAST`). Every LLM call site declares its
  `model_kind` (fast for categorize/dish/tip, main for chat/recipe_extract).
  `pora_llm.MODEL` is retained as a backward-compat alias for external code.
- `pora_llm._categorize_cache` and `_recipe_cache` are per-process TTL caches.
  Set `PORA_CACHE_ENABLED=0` to bypass; call `.clear()` between test runs.
- `pora_llm._chat_model(system, user, model_cls, ...)` is the canonical entry
  for structured JSON calls — it does one retry-on-validation-error with the
  pydantic error carried into the corrective prompt.
- The recipe catalog is NOT hardcoded in logic: `/v1/recommend` and
  `/v1/suggest` accept an optional `catalog` field (list of
  `{name, cuisine, ingredients}`); `brain._catalog_or_default` normalizes it
  (lowercase first-token ingredients) and falls back to
  `constants.RECIPE_CATALOG` when absent/empty. All prompts, env names,
  fallback strings live in `constants.py` — never inline new ones in logic
  modules.
