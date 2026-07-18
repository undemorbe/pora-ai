# -*- coding: utf-8 -*-
"""Pora AI — единый мультиязычный LLM-модуль (scope + категоризация + советы + рецепты).

Один код работает с локальной Ollama И с облаком — отличие только в env:
  Ollama:  LLM_BASE_URL=http://localhost:11434/v1  LLM_API_KEY=ollama   LLM_MODEL=qwen3
  Облако:  LLM_BASE_URL=https://api.openai.com/v1   LLM_API_KEY=sk-...    LLM_MODEL=gpt-4o-mini

Разбор рецептов вынесен в пакет ``recipe`` — здесь только LLM-обвязка,
язык, категоризация, чат и советы.

Мультиязычность: определяем язык запроса (или берём из параметра locale), отвечаем
на этом языке, отказы локализованы, категоризация через LLM работает на любом языке.

Все крутилки/лимиты/промпты вынесены в ``constants``. Здесь только логика вызовов.

Зависимости: openai>=1.0, pydantic>=2, httpx>=0.27
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Optional, TypeVar

from pydantic import BaseModel, Field

import brain
import constants as C
from _cache import TTLCache, cache_enabled_from_env
from _metrics import METRICS

# --------------------------------------------------------------------------
# Конфиг + ленивый клиент (не трогаем сеть на импорте)
# --------------------------------------------------------------------------
BASE_URL = os.getenv(C.LLM_BASE_URL_ENV, C.LLM_BASE_URL_DEFAULT)
API_KEY = os.getenv(C.LLM_API_KEY_ENV, "")
MODEL_MAIN = os.getenv(C.LLM_MODEL_ENV, C.LLM_MODEL_DEFAULT)
MODEL_FAST = os.getenv(C.LLM_MODEL_FAST_ENV) or MODEL_MAIN
MODEL = MODEL_MAIN                             # backward-compat alias


def _resolve_model(kind: str) -> str:
    return MODEL_FAST if kind == C.LLM_MODEL_KIND_FAST else MODEL_MAIN

_client = None


# --------------------------------------------------------------------------
# In-process cache for categorization (per name + sections)
# --------------------------------------------------------------------------
_CACHE_ENABLED = cache_enabled_from_env()
_categorize_cache = TTLCache(C.CATEGORIZE_CACHE_SIZE, C.CATEGORIZE_CACHE_TTL_S)


def llm_enabled() -> bool:
    return bool(API_KEY)


def client():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(base_url=BASE_URL, api_key=API_KEY or "noop")
    return _client


# Re-exports so external callers keep the pora_llm.<name> surface stable.
# (User-Agent and other fetch knobs now belong to the `recipe` package.)
REFUSALS = C.REFUSALS
SCOPE_SYSTEM = C.SCOPE_SYSTEM


# --------------------------------------------------------------------------
# Язык — определение
# --------------------------------------------------------------------------
def detect_lang(text: str, default: str = "en") -> str:
    """Best-effort language detection (script → CJK split → Latin diacritic scoring).

    Deterministic and dependency-free. See ``constants.SCRIPT_PATTERNS`` and
    ``constants.LATIN_MARKERS`` for the tables driving this.
    """
    for lang, pat in C.SCRIPT_PATTERNS:
        if re.search(pat, text):
            return lang
    if re.search(C.CJK_KANA_PATTERN, text):
        return "ja"
    if re.search(C.CJK_HAN_PATTERN, text):
        return "zh"
    low = text.lower()
    best_lang, best_score = None, 0
    for lang, pat in C.LATIN_MARKERS.items():
        n = len(re.findall(pat, low))
        if n > best_score:
            best_lang, best_score = lang, n
    return best_lang or default


def refusal(lang: str) -> str:
    return REFUSALS.get(lang, REFUSALS["en"])


# --------------------------------------------------------------------------
# LLM plumbing: safe chat + JSON helpers
# --------------------------------------------------------------------------
def _strip_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*", "", s).strip()
        if s.endswith("```"):
            s = s[:-3].strip()
    return s


def _safe_json_load(s: Optional[str]) -> Optional[dict]:
    """Strip code fences and parse JSON. Return None on any failure (never raise)."""
    if not s:
        return None
    try:
        return json.loads(_strip_fences(s))
    except Exception:
        return None


def _transient_llm_errors() -> tuple:
    """Import OpenAI error classes lazily. Return () if the SDK isn't installed
    so tests / offline paths still work."""
    try:
        import openai as _openai  # type: ignore
    except Exception:
        return ()
    return tuple(
        c for c in (
            getattr(_openai, "APIConnectionError", None),
            getattr(_openai, "APITimeoutError", None),
            getattr(_openai, "RateLimitError", None),
            getattr(_openai, "InternalServerError", None),
        ) if c is not None
    )


def _chat(system: str, user: str, temperature: float = 0.4,
          response_format=None,
          examples: Optional[list[dict]] = None,
          model_kind: str = C.LLM_MODEL_KIND_MAIN) -> Optional[str]:
    """Single LLM entry point.

    Returns None when LLM is disabled OR when a call fails terminally (after
    ``constants.LLM_MAX_RETRIES`` transient retries). Never raises — every
    caller in this module already treats None as "fall back gracefully".

    Optional `examples` are inserted as alternating user/assistant messages
    between the system prompt and the real user turn.

    ``model_kind`` selects between MODEL_MAIN and MODEL_FAST via _resolve_model.
    """
    if not llm_enabled():
        return None
    transient = _transient_llm_errors()

    messages: list[dict] = [{"role": "system", "content": system}]
    for pair in examples or []:
        messages.append({"role": "user", "content": pair["user"]})
        messages.append({"role": "assistant", "content": pair["assistant"]})
    messages.append({"role": "user", "content": user})

    kwargs: dict = dict(model=_resolve_model(model_kind), temperature=temperature, messages=messages)
    if response_format:
        kwargs["response_format"] = response_format

    t0 = time.monotonic()
    for attempt in range(C.LLM_MAX_RETRIES + 1):
        try:
            resp = client().chat.completions.create(**kwargs)
            METRICS.record(model_kind, time.monotonic() - t0, ok=True,
                           usage=getattr(resp, "usage", None))
            return resp.choices[0].message.content
        except Exception as e:
            is_transient = transient and isinstance(e, transient)
            if not is_transient or attempt == C.LLM_MAX_RETRIES:
                METRICS.record(model_kind, time.monotonic() - t0, ok=False)
                return None
            time.sleep(C.LLM_RETRY_BACKOFF_S * (attempt + 1))
    return None


T = TypeVar("T", bound=BaseModel)


def _chat_model(
    system: str,
    user: str,
    model_cls: type[T],
    *,
    examples: Optional[list[dict]] = None,
    temperature: float = C.TEMPERATURE_STRICT,
    response_format: Optional[dict] = None,
    model_kind: str = C.LLM_MODEL_KIND_MAIN,
) -> Optional[T]:
    """LLM call → parsed pydantic model with one retry-on-parse.

    Attempt 1: normal call → _safe_json_load → model_cls.model_validate.
    On ValidationError, attempt 2 re-issues with the pydantic error text
    injected into the user prompt so the model can self-correct.
    On second failure returns None. Never raises.
    """
    from pydantic import ValidationError

    def _call(u: str) -> Optional[T]:
        out = _chat(system, u, temperature=temperature,
                    response_format=response_format, examples=examples,
                    model_kind=model_kind)
        data = _safe_json_load(out)
        if data is None:
            return None
        try:
            return model_cls.model_validate(data)
        except ValidationError as exc:
            _call.last_error = str(exc)          # type: ignore[attr-defined]
            return None

    result = _call(user)
    if result is not None:
        return result

    err = getattr(_call, "last_error", None)
    if err is None:
        return None

    corrective = (
        f"Your last response failed validation: {err}. "
        "Return STRICT JSON matching the schema — no prose, no code fences.\n\n"
        f"Original request:\n{user}"
    )
    return _call(corrective)


# --------------------------------------------------------------------------
# Off-topic guard (pre-LLM cost cutter)
# --------------------------------------------------------------------------
def guard_on_topic(text: str) -> bool:
    low = text.lower()
    return not any(h in low for h in C.OFFTOPIC_MARKERS)


# --------------------------------------------------------------------------
# Публичные функции
# --------------------------------------------------------------------------
def chat(message: str, lang: Optional[str] = None) -> dict:
    """Заскоупленный мультиязычный ответ (кулинарные советы)."""
    lang = lang or detect_lang(message)
    if not guard_on_topic(message):
        return {"text": refusal(lang), "lang": lang, "refused": True}
    out = _chat(C.SCOPE_SYSTEM, message, temperature=C.TEMPERATURE_CHAT,
                model_kind=C.LLM_MODEL_ROUTING["chat"])
    if out is None:
        return {"text": refusal(lang), "lang": lang, "refused": False, "note": "llm_disabled"}
    return {"text": out.strip(), "lang": lang, "refused": False}


# ---- категоризация через LLM (любой язык, любой набор секций) ----
def _section_schema(sections: list[str]) -> dict:
    return {
        "type": "object", "additionalProperties": False, "required": ["section"],
        "properties": {"section": {"type": "string", "enum": list(sections)}},
    }


def _section_batch_schema(sections: list[str]) -> dict:
    return {
        "type": "object", "additionalProperties": False, "required": ["results"],
        "properties": {
            "results": {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "required": ["name", "section"],
                "properties": {
                    "name": {"type": "string"},
                    "section": {"type": "string", "enum": list(sections)},
                },
            }},
        },
    }


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


def _fallback_section(sections: list[str]) -> str:
    return C.DEFAULT_FALLBACK_SECTION if C.DEFAULT_FALLBACK_SECTION in sections else sections[0]


def _sections_or_default(sections: Optional[list[str]]) -> list[str]:
    return list(sections) if sections else list(brain.SECTIONS)


def categorize_llm(name: str, sections: Optional[list[str]] = None) -> tuple[str, float]:
    """Section for a grocery name in ANY language and ANY caller-supplied taxonomy."""
    sections = _sections_or_default(sections)
    key = ("cat", name.lower().strip(), tuple(sorted(sections)))
    if _CACHE_ENABLED:
        cached = _categorize_cache.get(key)
        if cached is not None:
            return cached

    examples = C.FEW_SHOT_EXAMPLES.get("categorize") if sections == list(brain.SECTIONS) else None
    resp = _chat_model(
        f"{C.CATEGORIZE_SYSTEM}\nAllowed keys: {', '.join(sections)}.",
        name, _SectionResponse,
        examples=examples,
        response_format={"type": "json_schema",
                         "json_schema": {"name": "section", "strict": True,
                                         "schema": _section_schema(sections)}},
        model_kind=C.LLM_MODEL_ROUTING["categorize"],
    )
    if resp is None or resp.section not in sections:
        return _fallback_section(sections), C.LLM_CONF_LOW

    result = (resp.section, C.LLM_CONF_HIGH)
    if _CACHE_ENABLED:
        _categorize_cache.set(key, result)
    return result


def categorize_llm_batch(names: list[str], sections: Optional[list[str]] = None) -> list[tuple[str, float]]:
    """Batched LLM categorization with per-item cache + one retry-on-parse."""
    if not names:
        return []
    sections = _sections_or_default(sections)
    fallback = _fallback_section(sections)
    sections_key = tuple(sorted(sections))

    results: list[Optional[tuple[str, float]]] = [None] * len(names)
    misses: list[tuple[int, str]] = []
    for i, n in enumerate(names):
        key = ("cat", n.lower().strip(), sections_key)
        cached = _categorize_cache.get(key) if _CACHE_ENABLED else None
        if cached is not None:
            results[i] = cached
        else:
            misses.append((i, n))

    if misses:
        miss_names = [n for _, n in misses]
        resp = _chat_model(
            f"{C.CATEGORIZE_SYSTEM}\nAllowed keys: {', '.join(sections)}. "
            "Classify EVERY item in the input array. "
            "For each item return the name verbatim and one section key.",
            json.dumps(miss_names, ensure_ascii=False),
            _SectionBatchResponse,
            response_format={"type": "json_schema",
                             "json_schema": {"name": "sections", "strict": True,
                                             "schema": _section_batch_schema(sections)}},
            model_kind=C.LLM_MODEL_ROUTING["categorize_batch"],
        )
        by_name: dict[str, str] = {}
        if resp is not None:
            by_name = {item.name: item.section for item in resp.results}
        for idx, n in misses:
            sec = by_name.get(n)
            if sec in sections:
                results[idx] = (sec, C.LLM_CONF_HIGH)
                if _CACHE_ENABLED:
                    _categorize_cache.set(("cat", n.lower().strip(), sections_key),
                                          (sec, C.LLM_CONF_HIGH))
            else:
                results[idx] = (fallback, C.LLM_CONF_LOW)

    return [r or (fallback, C.LLM_CONF_LOW) for r in results]


# ---- LLM-предложение блюда (для /v1/suggest, dish-тип) ----
_DISH_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["dish", "reason"],
    "properties": {"dish": {"type": "string"}, "reason": {"type": "string"}},
}


def suggest_dish_llm(top_cuisine: str, frequent: list[str], lang: str = "en") -> Optional[dict]:
    """LLM-generated dish suggestion. Returns {dish, reason} or None if LLM disabled/failed."""
    system = C.DISH_SYSTEM_TEMPLATE.format(lang=lang)
    user = f"Favourite cuisine: {top_cuisine or 'n/a'}. Often buys: {', '.join(frequent) or 'n/a'}."
    resp = _chat_model(
        system, user, _DishResponse,
        temperature=C.TEMPERATURE_DISH,
        response_format={"type": "json_schema",
                         "json_schema": {"name": "dish", "strict": True, "schema": _DISH_SCHEMA}},
        model_kind=C.LLM_MODEL_ROUTING["dish"],
    )
    if resp is None:
        return None
    return {"dish": resp.dish.strip(), "reason": resp.reason.strip()}


# ---- совет по вкусу (мультиязычно) ----
def generate_tip(top_cuisine: str, frequent: list[str], lang: str = "en") -> dict:
    system = C.TIP_SYSTEM_TEMPLATE.format(lang=lang)
    user = f"Favourite cuisine: {top_cuisine}. Often buys: {', '.join(frequent) or 'n/a'}."
    out = _chat(system, user, temperature=C.TEMPERATURE_TIP,
                model_kind=C.LLM_MODEL_ROUTING["tip"])
    if out:
        return {"tip": out.strip(), "lang": lang, "source": "llm"}
    template = C.TIP_FALLBACKS.get(lang, C.TIP_FALLBACKS["en"])
    return {"tip": template.format(cuisine=top_cuisine), "lang": lang, "source": "fallback"}
