# -*- coding: utf-8 -*-
"""Pora AI — единый мультиязычный LLM-модуль (scope + категоризация + советы + рецепты).

Один код работает с локальной Ollama И с облаком — отличие только в env:
  Ollama:  LLM_BASE_URL=http://localhost:11434/v1  LLM_API_KEY=ollama   LLM_MODEL=qwen3
  Облако:  LLM_BASE_URL=https://api.openai.com/v1   LLM_API_KEY=sk-...    LLM_MODEL=gpt-4o-mini

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

import httpx
from pydantic import BaseModel, Field

import brain
import constants as C
from _cache import TTLCache

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
# In-process caches — categorize (per name+sections) and parse_recipe (per URL)
# --------------------------------------------------------------------------
def _cache_enabled_from_env() -> bool:
    raw = os.getenv(C.CACHE_ENABLED_ENV)
    if raw is None:
        return C.CACHE_ENABLED_DEFAULT
    return raw.strip().lower() not in C.ENV_FALSY


_CACHE_ENABLED = _cache_enabled_from_env()
_categorize_cache = TTLCache(C.CATEGORIZE_CACHE_SIZE, C.CATEGORIZE_CACHE_TTL_S)
_recipe_cache = TTLCache(C.RECIPE_CACHE_SIZE, C.RECIPE_CACHE_TTL_S)


def llm_enabled() -> bool:
    return bool(API_KEY)


def client():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(base_url=BASE_URL, api_key=API_KEY or "noop")
    return _client


# Re-exports so external callers keep the pora_llm.<name> surface stable
REFUSALS = C.REFUSALS
SCOPE_SYSTEM = C.SCOPE_SYSTEM
DEFAULT_USER_AGENT = C.DEFAULT_USER_AGENT


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

    for attempt in range(C.LLM_MAX_RETRIES + 1):
        try:
            resp = client().chat.completions.create(**kwargs)
            return resp.choices[0].message.content
        except Exception as e:
            is_transient = transient and isinstance(e, transient)
            if not is_transient or attempt == C.LLM_MAX_RETRIES:
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


class _RecipeResponse(BaseModel):
    title: Optional[str] = None
    ingredients: list[dict] = Field(default_factory=list)


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


# ---- рецепты: JSON-LD (бесплатно) → LLM-фолбэк, любой язык ----
class Ingredient(BaseModel):
    raw: str
    name: Optional[str] = None
    qty: Optional[float] = None
    unit: Optional[str] = None
    section: str = C.DEFAULT_FALLBACK_SECTION


class Recipe(BaseModel):
    title: Optional[str] = None
    ingredients: list[Ingredient] = Field(default_factory=list)
    source: str = "none"


_RECIPE_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["title", "ingredients"],
    "properties": {
        "title": {"type": ["string", "null"]},
        "ingredients": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["raw", "name", "qty", "unit"],
            "properties": {"raw": {"type": "string"}, "name": {"type": ["string", "null"]},
                           "qty": {"type": ["number", "null"]}, "unit": {"type": ["string", "null"]}},
        }},
    },
}


# --------------------------------------------------------------------------
# HTML utils + browser-like web fetch + anti-hallucination validation
# --------------------------------------------------------------------------
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_STYLE_RE = re.compile(r"<(script|style|noscript)[^>]*>.*?</\1>", re.S | re.I)
_BOILER_RE = re.compile(r"<(nav|header|footer|aside|form)\b[^>]*>.*?</\1>", re.S | re.I)
_MAIN_CONTENT_RE = re.compile(r"<(main|article)\b[^>]*>(.*?)</\1>", re.S | re.I)
_WS_RE = re.compile(r"\s+")


def html_to_text(html: str) -> str:
    """Strip <script>/<style>, tags, decode common entities, collapse whitespace."""
    s = _SCRIPT_STYLE_RE.sub(" ", html)
    s = _HTML_TAG_RE.sub(" ", s)
    for k, v in C.HTML_ENTITIES.items():
        s = s.replace(k, v)
    return _WS_RE.sub(" ", s).strip()


def _accept_language(lang: Optional[str]) -> str:
    if not lang:
        return C.ACCEPT_LANGUAGE_DEFAULT
    lang = lang.split("-")[0].lower()
    return f"{lang},{lang};q=0.9,en;q=0.5"


def extract_main_content(html: str) -> str:
    """Pick the main content of an HTML page: first <main>/<article>, else strip boilerplate."""
    m = _MAIN_CONTENT_RE.search(html)
    if m:
        return m.group(2)
    return _BOILER_RE.sub(" ", html)


def web_fetch(url: str, lang: Optional[str] = None,
              timeout: float = C.FETCH_TIMEOUT_S,
              max_bytes: int = C.FETCH_MAX_BYTES,
              retries: int = C.FETCH_RETRIES) -> dict:
    """Browser-like HTTP fetch with realistic headers, retries on 429/5xx, and content extraction.

    Returns ``{"url", "status", "html", "text"}``: the final URL after redirects, status code,
    the truncated raw HTML, and the readable plain text from the main content area.

    Raises ``httpx.HTTPError`` if every attempt fails.
    """
    headers = {
        "User-Agent": C.DEFAULT_USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": _accept_language(lang),
        "Accept-Encoding": "gzip, deflate",
        "DNT": "1",
        "Upgrade-Insecure-Requests": "1",
    }
    last_exc: Optional[Exception] = None
    with httpx.Client(timeout=timeout, follow_redirects=True, headers=headers) as cli:
        for attempt in range(retries + 1):
            try:
                r = cli.get(url)
                if r.status_code in C.FETCH_RETRY_STATUSES and attempt < retries:
                    continue
                r.raise_for_status()
                html = r.text[:max_bytes]
                return {"url": str(r.url), "status": r.status_code,
                        "html": html, "text": html_to_text(extract_main_content(html))}
            except httpx.HTTPError as e:
                last_exc = e
                if attempt == retries:
                    raise
    assert last_exc is not None
    raise last_exc


def _norm(s: str) -> str:
    return _WS_RE.sub(" ", s.lower()).strip()


def _build_synonym_lookup() -> dict:
    """word → tuple of alternatives, both directions, built once at import."""
    lookup: dict[str, tuple[str, ...]] = {}
    for a, b in C.INGREDIENT_SYNONYM_PAIRS:
        lookup[a] = lookup.get(a, ()) + (b,)
        lookup[b] = lookup.get(b, ()) + (a,)
    return lookup


_SYNONYMS = _build_synonym_lookup()


def _name_in_source(name: str, haystack: str) -> bool:
    """Match an ingredient name against the source with graceful degradation.

    1. verbatim substring;
    2. singular-strip: "eggs" matches a source that only has "egg";
    3. cross-lingual synonym bridge (constants.INGREDIENT_SYNONYM_PAIRS) —
       the LLM sometimes translates the name it extracts.
    """
    if len(name) < 3:
        return False
    if name in haystack:
        return True
    if name.endswith("s") and len(name) >= 4 and name[:-1] in haystack:
        return True
    for alt in _SYNONYMS.get(name, ()):
        if alt in haystack:
            return True
    return False


def validate_against_source(ingredients: list[dict], source_text: str) -> list[dict]:
    """Drop ingredients whose ``raw`` or ``name`` is not present in the source.

    Anti-hallucination guard: LLM may invent ingredients that aren't in the page.
    We require the full ``raw`` line OR the ``name`` to appear in the source —
    verbatim, singular-stripped, or through the RU↔EN synonym bridge
    (see ``_name_in_source``). Case-insensitive, whitespace-collapsed.
    """
    haystack = _norm(source_text)
    kept: list[dict] = []
    for ing in ingredients:
        raw = _norm(ing.get("raw") or "")
        name = _norm(ing.get("name") or "")
        if raw and raw in haystack:
            kept.append(ing)
        elif name and _name_in_source(name, haystack):
            kept.append(ing)
    return kept


def _iter_recipe_nodes(data):
    stack = [data]
    while stack:
        node = stack.pop()
        if isinstance(node, list):
            stack.extend(node)
        elif isinstance(node, dict):
            if "@graph" in node:
                g = node["@graph"]
                stack.extend(g if isinstance(g, list) else [g])
            t = node.get("@type")
            for x in (t if isinstance(t, list) else [t]):
                if x and str(x).lower() == "recipe":
                    yield node


def extract_jsonld(html: str) -> Optional[dict]:
    for b in re.findall(r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>', html, re.S | re.I):
        try:
            data = json.loads(b.strip())
        except Exception:
            continue
        for node in _iter_recipe_nodes(data):
            ings = node.get("recipeIngredient") or node.get("ingredients")
            if ings:
                ings = [ings] if isinstance(ings, str) else ings
                return {"title": node.get("name"),
                        "ingredients": [{"raw": str(i).strip(), "name": None, "qty": None, "unit": None} for i in ings],
                        "source": "jsonld"}
    return None


def extract_recipe_from_text(text: str) -> dict:
    """LLM-based extraction with anti-hallucination validation against source text."""
    resp = _chat_model(
        C.RECIPE_EXTRACT_SYSTEM, text[:C.LLM_TEXT_CAP], _RecipeResponse,
        examples=C.FEW_SHOT_EXAMPLES.get("recipe_extract"),
        response_format={"type": "json_schema",
                         "json_schema": {"name": "recipe", "strict": True, "schema": _RECIPE_SCHEMA}},
        model_kind=C.LLM_MODEL_ROUTING["recipe_extract"],
    )
    if resp is None:
        return {"title": None, "ingredients": [], "source": "none"}
    validated = validate_against_source(resp.ingredients, text)
    return {
        "title": resp.title,
        "ingredients": validated,
        "source": "llm" if validated else "none",
    }


def parse_recipe(url: str, categorizer: brain.Categorizer,
                 sections: Optional[list[str]] = None, lang: Optional[str] = None) -> Recipe:
    """Full URL → Recipe pipeline (cached by URL + sections + lang)."""
    key = ("recipe", url, tuple(sorted(sections or ())), lang or "")
    if _CACHE_ENABLED:
        cached = _recipe_cache.get(key)
        if cached is not None:
            return Recipe.model_validate(cached)

    fetched = web_fetch(url, lang=lang)
    html, text = fetched["html"], fetched["text"]

    data = extract_jsonld(html)
    if not data:
        data = extract_recipe_from_text(text)

    ings = data.get("ingredients") or []
    if ings:
        if sections:
            labels = [ing.get("name") or ing.get("raw") or "" for ing in ings]
            tagged = categorize_llm_batch(labels, sections)
            fallback = _fallback_section(sections)
            for ing, (sec, _conf) in zip(ings, tagged):
                ing["section"] = sec if (ing.get("name") or ing.get("raw")) else fallback
        else:
            for ing in ings:
                label = ing.get("name") or ing.get("raw") or ""
                ing["section"] = categorizer.predict(label)[0] if label else C.DEFAULT_FALLBACK_SECTION

    recipe = Recipe.model_validate(data)
    if _CACHE_ENABLED and data.get("source") in ("jsonld", "llm"):
        _recipe_cache.set(key, recipe.model_dump())
    return recipe
