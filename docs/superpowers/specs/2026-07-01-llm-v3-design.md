# LLM v3 — Design Spec

**Date:** 2026-07-01
**Branch:** `worktree-llm-recipe-suggest`
**Base:** `main`
**Status:** Approved for implementation

## Goal

Ship one coherent iteration ("Approach A") that lifts the LLM subsystem across four axes at once:

1. **Accuracy** — few-shot examples in the two structured-output prompts.
2. **Cost / speed** — in-process LRU + TTL cache around `categorize_llm` / `categorize_llm_batch` / `parse_recipe`; cheap-vs-main model routing.
3. **Robustness** — every JSON-returning LLM call is validated through a Pydantic model with one retry-on-parse.
4. **New capabilities** — none in this iteration (deferred: semantic ingredient matching, dietary restrictions, telemetry).

Nothing in this spec changes the wire contract with the Go backend (`schemas.py` untouched). One new field appears in `GET /health` (`models` object). One optional env var (`LLM_MODEL_FAST`).

## Non-goals

- No new HTTP endpoints.
- No changes to `schemas.py`.
- No new runtime dependencies (Pydantic and httpx already pinned).
- No persistence layer for the cache — Pora AI stays stateless w.r.t. user data. The cache is derived, per-process, and ephemeral.
- Deferred to a later iteration: semantic ingredient matching, dietary restrictions, telemetry, streaming chat.

## Architecture map

```
constants.py           MOD  + FEW_SHOT_EXAMPLES, model routing table, cache defaults
pora_llm.py            MOD  + response models, _chat_model helper, model_kind arg,
                             cache wrapping on categorize / parse_recipe
_cache.py              NEW  TTLCache (thread-safe LRU + per-entry TTL)
main.py                MOD  /health now reports {models: {main, fast}}
docs/superpowers/specs/2026-07-01-llm-v3-design.md   NEW  this file

tests/test_cache.py            NEW
tests/test_constants.py        MOD  + few-shot invariants, routing table coverage
tests/test_pora_llm.py         MOD  + _chat_model retry, routing, cache wiring
tests/test_endpoints.py        MOD  + /health reports both models
```

## Env contract

| Var                | Default                | Purpose                                                |
|--------------------|------------------------|--------------------------------------------------------|
| `LLM_BASE_URL`     | `http://localhost:11434/v1` | unchanged                                          |
| `LLM_API_KEY`      | `""`                   | unchanged (empty → LLM disabled, graceful degrade)     |
| `LLM_MODEL`        | `qwen3`                | Main model (chat, parse-recipe)                        |
| `LLM_MODEL_FAST`   | *unset → equals `LLM_MODEL`* | Cheap model (categorize, tip, dish)              |
| `PORA_CACHE_ENABLED` | `1`                  | Set to `0` to bypass every cache lookup                |

Backward compat: unset `LLM_MODEL_FAST` collapses to a single-model config.

## Item 1 — Few-shot examples

**Where.** New constant `FEW_SHOT_EXAMPLES: dict[str, list[dict]]` in `constants.py`.

Shape:
```python
FEW_SHOT_EXAMPLES = {
    "categorize": [
        {"user": "молоко",         "assistant": '{"section": "dairy"}'},
        {"user": "chicken breast", "assistant": '{"section": "meat_fish"}'},
        {"user": "свежий базилик", "assistant": '{"section": "produce"}'},
        {"user": "багет",          "assistant": '{"section": "bakery"}'},
        {"user": "кока-кола",      "assistant": '{"section": "drinks"}'},
    ],
    "recipe_extract": [
        # positive Russian
        {"user": "Карбонара: спагетти 400 г, бекон 200 г, яйца 4 шт, пармезан 100 г.",
         "assistant": '{"title": "Карбонара", "ingredients": [{"raw":"спагетти 400 г","name":"спагетти","qty":400,"unit":"г"}, {"raw":"бекон 200 г","name":"бекон","qty":200,"unit":"г"}, {"raw":"яйца 4 шт","name":"яйца","qty":4,"unit":"шт"}, {"raw":"пармезан 100 г","name":"пармезан","qty":100,"unit":"г"}]}'},
        # positive English
        {"user": "French Toast: 2 eggs, 1 cup milk, 4 slices bread.",
         "assistant": '{"title": "French Toast", "ingredients": [{"raw":"2 eggs","name":"eggs","qty":2,"unit":null}, {"raw":"1 cup milk","name":"milk","qty":1,"unit":"cup"}, {"raw":"4 slices bread","name":"bread","qty":4,"unit":"slices"}]}'},
        # negative (not a recipe)
        {"user": "How to boil water: turn stove on high, wait until bubbles form.",
         "assistant": '{"title": null, "ingredients": []}'},
    ],
}
```

**Injection.** `_chat` gets an optional `examples: list[dict] | None` parameter. If set, entries are converted into alternating `role: user` / `role: assistant` messages placed between the system prompt and the real user turn:

```
[system, ex1_user, ex1_assistant, ..., real_user]
```

**Custom sections gate.** Few-shot for `categorize` is only injected when the caller uses the default taxonomy (`sections is None` or `sections == list(brain.SECTIONS)`). Under a caller-supplied taxonomy the examples would reference keys that may not exist — they are suppressed and the model relies on the system prompt alone.

**Cost.** ~150 tokens overhead per categorize call, ~350 per recipe_extract call. Amortized to near-zero by the cache layer.

**Tests** (`tests/test_constants.py`, `tests/test_pora_llm.py`):
- `TestFewShot::test_categorize_has_min_examples` — ≥5.
- `TestFewShot::test_all_assistant_payloads_parse_as_json` — every `assistant` string is valid JSON.
- `TestFewShotWiring::test_examples_inserted_between_system_and_user` — capture `messages` via mock_chat; verify order and role alternation.
- `TestFewShotWiring::test_no_examples_kwarg_yields_no_examples_in_messages` — default call has exactly `[system, user]`.
- `TestCategorizeSuppressesFewShotForCustomSections` — `categorize_llm("x", sections=["a","b"])` triggers a `_chat` call without any assistant messages.

## Item 2 — TTLCache + wiring

**New file `_cache.py`** — ~60 lines, pure stdlib.

```python
class TTLCache:
    """Thread-safe LRU with per-entry TTL.

    Backed by collections.OrderedDict + threading.Lock. Time source is
    time.monotonic so we are immune to wall-clock jumps. `get` evicts
    expired entries lazily; `set` evicts LRU on overflow. Stats are
    plain counters, not atomic — best-effort.
    """
    def __init__(self, maxsize: int, ttl_s: float): ...
    def get(self, key) -> Optional[Any]: ...
    def set(self, key, value) -> None: ...
    def clear(self) -> None: ...
    def stats(self) -> dict: ...   # {"size", "hits", "misses"}
```

**Module-level instances in `pora_llm.py`:**
```python
_categorize_cache = TTLCache(C.CATEGORIZE_CACHE_SIZE, C.CATEGORIZE_CACHE_TTL_S)
_recipe_cache     = TTLCache(C.RECIPE_CACHE_SIZE,     C.RECIPE_CACHE_TTL_S)
```

**Cache defaults (constants.py):**
```python
CATEGORIZE_CACHE_SIZE = 2048
CATEGORIZE_CACHE_TTL_S = 3600           # 1 hour
RECIPE_CACHE_SIZE = 256
RECIPE_CACHE_TTL_S = 86_400              # 24 hours
CACHE_ENABLED_ENV = "PORA_CACHE_ENABLED"
CACHE_ENABLED_DEFAULT = True
```

**Cache keys** — deterministic, sorted, normalized:

- `categorize_llm(name, sections)`:
  `("cat", name.lower().strip(), tuple(sorted(sections or ())))`
- `categorize_llm_batch(names, sections)`:
  Uses per-item keys (same as `categorize_llm`). Batch flow:
  1. Look up each name; keep hits with the returned `(section, conf)`.
  2. Send the misses in a single LLM call.
  3. Populate cache with each new `(name, section)` from the response.
  4. Return results in original input order.
- `parse_recipe(url, sections, lang)`:
  `("recipe", url, tuple(sorted(sections or ())), lang or "")` → caches the final `Recipe.model_dump()` dict; a subsequent call rebuilds `Recipe.model_validate(cached_dict)` (fast, no LLM, no HTTP).

**Not cached:** `chat`, `generate_tip`, `suggest_dish_llm` — non-deterministic (`temperature > 0`), caching would be a footgun.

**Cache bypass paths:**
- Env: `PORA_CACHE_ENABLED=0` at import time short-circuits every cache lookup (still exposes `stats()` reading zeros).
- Programmatic: `_categorize_cache.clear()` / `_recipe_cache.clear()` — used by test fixtures.

**Tests** (`tests/test_cache.py`, plus `tests/test_pora_llm.py`):
- `TestTTLCache::test_get_returns_none_when_missing`
- `TestTTLCache::test_set_then_get_returns_value`
- `TestTTLCache::test_expired_entry_evicted_on_get` (monkey-patch `time.monotonic`)
- `TestTTLCache::test_lru_eviction_when_over_capacity`
- `TestTTLCache::test_get_moves_entry_to_recent`
- `TestTTLCache::test_clear_resets_state`
- `TestTTLCache::test_stats_counts_hits_and_misses`
- `TestCategorizeCaching::test_second_call_hits_cache_and_skips_llm`
- `TestCategorizeCaching::test_batch_partial_cache_hit_only_misses_go_to_llm`
- `TestCategorizeCaching::test_custom_sections_have_separate_cache_key`
- `TestRecipeCaching::test_second_call_skips_web_fetch_and_llm`
- `TestCacheDisabledViaEnv` — set env, reload module (or clear caches + skip), verify LLM called every time.

## Item 3 — Pydantic-validated `_chat_model` + retry-on-parse

**Motivation.** Today every JSON-returning caller does:
```python
data = _safe_json_load(out)
if not data or "expected_field" not in data or data["expected_field"] not in allowed_values:
    return <fallback>
return data["expected_field"]
```

The validation is scattered, boilerplate, and any failure falls silently into the fallback. New helper centralizes shape enforcement and enables one self-correcting retry.

**New helper:**

```python
T = TypeVar("T", bound=BaseModel)

def _chat_model(
    system: str, user: str,
    model_cls: type[T],
    *,
    examples: list[dict] | None = None,
    temperature: float = C.TEMPERATURE_STRICT,
    response_format: dict | None = None,
    model_kind: str = C.LLM_MODEL_KIND_MAIN,
) -> Optional[T]:
    """LLM call → parsed pydantic model.

    Attempt 1: normal call → _safe_json_load → model_cls.model_validate.
    On ValidationError, attempt 2 amends user with the pydantic error text
    ("Your last response failed validation: {err}. Return STRICT JSON.").
    On second failure returns None. Never raises.
    """
```

**New Pydantic response models** (private, in `pora_llm.py`):

```python
class _SectionResponse(BaseModel):
    section: str

class _SectionBatchItem(BaseModel):
    name: str
    section: str

class _SectionBatchResponse(BaseModel):
    results: list[_SectionBatchItem]

class _DishResponse(BaseModel):
    dish: str
    reason: str

class _RecipeResponse(BaseModel):
    title: Optional[str] = None
    ingredients: list[dict] = Field(default_factory=list)
```

**Enum enforcement.** LLM structured output already enforces the enum server-side via `response_format=json_schema, strict=True`, so an out-of-enum section from a well-behaved provider is rare. When it happens anyway, callers check the returned section against `allowed_sections` after `model_validate`. Mismatch → callers return the fallback section at `LLM_CONF_LOW` (no extra retry from `_chat_model` — enum drift is treated as "provider ignored strict mode, fall back gracefully"). `_chat_model` retries only on JSON parse failure or Pydantic `ValidationError`.

**Nested ingredient shape.** `_RecipeResponse.ingredients` stays typed as `list[dict]` (not `list[Ingredient]`) — the existing `validate_against_source` pass and the section-tagging step in `parse_recipe` both operate on dicts, and `Recipe.model_validate` at the end enforces the full shape. Tightening this to nested Pydantic here would break that flow without a measurable win.

**Refactored callers** — every one becomes a thin wrapper:
- `categorize_llm(name, sections)` → `_chat_model(..., _SectionResponse, model_kind="fast")`, then enum check.
- `categorize_llm_batch(names, sections)` → `_chat_model(..., _SectionBatchResponse, model_kind="fast")`.
- `suggest_dish_llm(...)` → `_chat_model(..., _DishResponse, model_kind="fast")`.
- `extract_recipe_from_text(text)` → `_chat_model(..., _RecipeResponse, model_kind="main")` then the existing `validate_against_source` anti-hallucination pass.

**`_chat` is not touched** other than accepting new `examples` and `model_kind` kwargs — old call sites remain valid.

**`_safe_json_load` stays** — internally used by `_chat_model`, plus by `chat` which returns free-form text (not JSON).

**Tests:**
- `TestChatModel::test_happy_path` — mock returns valid JSON → parsed model instance.
- `TestChatModel::test_retry_on_validation_error` — first mock output invalid, second valid → `_chat` called 2x, second user prompt contains the phrase `failed validation`, result is parsed.
- `TestChatModel::test_exhausts_retries_returns_none` — both outputs invalid → `None`, 2 `_chat` calls total.
- `TestChatModel::test_returns_none_when_llm_disabled` — no retries attempted.
- `TestCategorizeInvalidSectionRetries` — LLM returns `{"section":"bogus"}` then `{"section":"produce"}` → returns `("produce", HIGH_CONF)`.
- Existing `TestChatRetry` (transient error path) still passes — `_chat` behavior unchanged.

## Item 4 — Model routing (cheap/main split)

**Routing table (constants.py):**

```python
LLM_MODEL_KIND_MAIN = "main"
LLM_MODEL_KIND_FAST = "fast"

# Jump table: which call kind picks which env-configured model.
LLM_MODEL_ROUTING: dict[str, str] = {
    "categorize":       LLM_MODEL_KIND_FAST,
    "categorize_batch": LLM_MODEL_KIND_FAST,
    "dish":             LLM_MODEL_KIND_FAST,
    "tip":              LLM_MODEL_KIND_FAST,
    "chat":             LLM_MODEL_KIND_MAIN,
    "recipe_extract":   LLM_MODEL_KIND_MAIN,
}
```

**Resolver (pora_llm.py):**

```python
MODEL_MAIN = os.getenv("LLM_MODEL", "qwen3")
MODEL_FAST = os.getenv("LLM_MODEL_FAST") or MODEL_MAIN
MODEL = MODEL_MAIN  # backward-compat alias for external code

def _resolve_model(kind: str) -> str:
    return MODEL_FAST if kind == C.LLM_MODEL_KIND_FAST else MODEL_MAIN
```

`_chat` gains `model_kind: str = C.LLM_MODEL_KIND_MAIN` and resolves through `_resolve_model`. Callers pass `model_kind` explicitly:

| Caller                       | model_kind |
|------------------------------|------------|
| `categorize_llm`             | `"fast"`   |
| `categorize_llm_batch`       | `"fast"`   |
| `suggest_dish_llm`           | `"fast"`   |
| `generate_tip`               | `"fast"`   |
| `chat`                       | `"main"`   |
| `extract_recipe_from_text`   | `"main"`   |

**Rationale.** `dish` and `tip` are short, templated one-shots — a cheap model is enough. `chat` needs conversational quality. `recipe_extract` is high-stakes structured extraction where hallucination is expensive — pay for the main model.

**`/health` update (main.py):**

```json
{
  "status": "ok",
  "version": "2.0.0",
  "llm_enabled": true,
  "models": {"main": "gpt-4o-mini", "fast": "gpt-4o-nano"},
  "sections": [...],
  "fast_langs": [...],
  "refusal_langs": [...]
}
```

Additive field. Go client keeps working.

**Tests:**
- `TestModelRouting::test_all_kinds_have_mapping` — every value in `LLM_MODEL_ROUTING` ∈ {`main`, `fast`}.
- `TestResolveModel::test_fast_when_set` — with `LLM_MODEL_FAST=xyz`, `_resolve_model("fast") == "xyz"`.
- `TestResolveModel::test_fast_falls_back_to_main_when_unset`.
- `TestChatModelKindRoutes` — monkey-patch `_chat`, capture `model` kwarg; call `categorize_llm` → uses `MODEL_FAST`; call `chat` → uses `MODEL_MAIN`.
- `TestHealth::test_reports_both_models` — `/health` returns `models.main` and `models.fast`.

## Rollout / commit sequence

Four green-at-every-step commits:

1. `feat(constants): model routing table + few-shot examples + cache defaults` — constants only, no behavior change, test_constants covers invariants.
2. `feat(cache): TTLCache + wire around categorize + parse_recipe` — new file, wire, tests.
3. `feat(llm): _chat_model helper + pydantic response models + retry-on-parse` — refactor all four JSON callers.
4. `feat(llm): model routing (fast/main split) + /health reports both` — env + `_resolve_model` + endpoint update.

Total delta ≈ 700 lines; tests grow from 188 to ~230.

## Invariants preserved

- Wire contract (`schemas.py`) unchanged.
- Without `LLM_API_KEY` the whole graceful-degrade chain still holds (`_chat` returns `None`, `_chat_model` returns `None`, callers fallback).
- `brain.SECTIONS`, `pora_llm.REFUSALS`, `pora_llm.SCOPE_SYSTEM`, `pora_llm.MODEL`, `pora_llm.DEFAULT_USER_AGENT` remain module-level attributes.
- Cache is per-process, ephemeral, opt-out via env.
- No new runtime dependencies.

## Deferred to a later iteration

- Semantic ingredient matching (`сыр` ≈ `cheese`) for `validate_against_source`.
- Dietary restrictions field on `SuggestRequest`/`TipRequest`/`ChatRequest`.
- Telemetry hook (tokens/latency/counts).
- Streaming responses for `/v1/chat`.
- Quantity normalization for parsed recipes.

Each of the above will get its own spec.
