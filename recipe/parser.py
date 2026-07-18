# -*- coding: utf-8 -*-
"""Tier 2 — pure-Python recipe extraction. Free, fast, no LLM.

Most recipe sites without JSON-LD still mark their ingredients up in a way a
parser can find: schema.org microdata, an ``ingredient``-ish CSS class, or a
plain list right after an "Ингредиенты"/"Ingredients" heading. Reaching for an
LLM on those pages is slow and costs money for nothing.

Three strategies are tried in descending order of trust; the first one that
produces a plausible ingredient block wins. Returning ``None`` means "I am not
confident" — the caller then falls back to the LLM.

Stdlib only (``html.parser``) — no new runtime dependency.
"""
from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Optional

from . import constants as RC

_WS_RE = re.compile(r"\s+")
_SKIP_TAGS = frozenset({"script", "style", "noscript", "template"})


def _norm(s: str) -> str:
    return _WS_RE.sub(" ", s).replace("\xa0", " ").strip()


# --------------------------------------------------------------------------
# Quantity splitting
# --------------------------------------------------------------------------
# Longest alternatives first — regex alternation is first-match ("кг" before "г").
# Russian recipe sites abbreviate heavily ("бан.", "уп.", "ст. л."), so short
# forms are listed alongside the full words.
_UNIT_ALTERNATION = (
    r"кг|гр|г|мл|л|шт|штук[иа]?|зубчик[иов]*|зуб|щепотк[аи]|щеп|"
    r"ст\.?\s*л\.?|ч\.?\s*л\.?|стакан[аов]*|стак|пучок|пучка|пуч|"
    r"банк[аи]|бан|упаковк[аи]|уп|ложк[аи]|долек|долька|кусок|куск[аи]|кус|"
    r"kg|g|gr|ml|l|oz|lb|lbs|cups?|tbsps?|tsps?|pcs?|cloves?|pinch|cans?|"
    r"tablespoons?|teaspoons?|pounds?|ounces?|slices?|packs?"
)
_NUM = r"\d+(?:[.,]\d+)?(?:\s*/\s*\d+)?"

# "2 cups flour" / "3 eggs"
_LEADING_RE = re.compile(
    rf"^({_NUM})\s*({_UNIT_ALTERNATION})?\.?\s+(.{{2,}})$", re.I)
# "Кабачок - 550 г" / "Sugar 100 g" / "Сметана 1,5 ст. л."
_TRAILING_RE = re.compile(
    rf"^(.+?)[\s\-–—:,]*({_NUM})\s*({_UNIT_ALTERNATION})?\.?$", re.I)


def _to_float(raw: str) -> Optional[float]:
    raw = raw.replace(",", ".").replace(" ", "")
    if "/" in raw:                                   # "1/2"
        try:
            a, b = raw.split("/", 1)
            return round(float(a) / float(b), 3) if float(b) else None
        except (ValueError, ZeroDivisionError):
            return None
    try:
        return float(raw)
    except ValueError:
        return None


_PARENTHETICAL_RE = re.compile(r"\s*\([^)]*\)")


def split_quantity(raw: str) -> tuple[str, Optional[float], Optional[str]]:
    """Split an ingredient line into (name, qty, unit).

    Handles the three shapes seen in the wild: quantity first ("2 cups flour"),
    quantity last ("Кабачок - 550 г"), and no quantity at all ("соль по вкусу").
    Parentheticals are stripped first — "Чеснок - 13 г (2 зубчика)" states the
    weight, and "(2 зубчика)" would otherwise be read as the quantity.
    Anything unparseable degrades to (whole_line, None, None) — never raises.
    """
    line = _PARENTHETICAL_RE.sub("", _norm(raw)).strip()
    if not line:
        return "", None, None

    m = _LEADING_RE.match(line)
    if m and (m.group(2) or not _TRAILING_RE.match(line)):
        qty, unit, name = _to_float(m.group(1)), m.group(2), m.group(3)
        name = name.strip(" -–—:,.")
        if name:
            return name, qty, (_norm(unit) if unit else None)

    m = _TRAILING_RE.match(line)
    if m and m.group(3):                             # require a real unit here
        name = m.group(1).strip(" -–—:,.")
        if name:
            return name, _to_float(m.group(2)), _norm(m.group(3))

    return line, None, None


# --------------------------------------------------------------------------
# HTML walking
# --------------------------------------------------------------------------
class _Collector(HTMLParser):
    """Flatten HTML into (tag, attrs, text, order) records, skipping scripts.

    ``convert_charrefs`` is on, so entities arrive already decoded. Text is
    attributed to the innermost open element, which is what makes the
    class/microdata strategies work on nested markup like
    ``<td class="ingr"><span>Кабачок - 550 г</span></td>``.
    """

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.records: list[dict] = []
        self._stack: list[dict] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        rec = {"tag": tag, "attrs": dict(attrs), "text": "",
               "order": len(self.records)}
        self.records.append(rec)
        self._stack.append(rec)

    def handle_endtag(self, tag):
        if tag in _SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth:
            return
        for i in range(len(self._stack) - 1, -1, -1):
            if self._stack[i]["tag"] == tag:
                del self._stack[i:]
                break

    def handle_data(self, data):
        if self._skip_depth or not data.strip():
            return
        for rec in self._stack:                      # bubble text to ancestors
            rec["text"] += " " + data


def _attr_blob(rec: dict) -> str:
    a = rec["attrs"]
    return " ".join(str(v) for v in (a.get("class"), a.get("id"),
                                     a.get("itemprop")) if v).lower()


# --------------------------------------------------------------------------
# Candidate scoring / validation
# --------------------------------------------------------------------------
_QTY_RE = re.compile(RC.QTY_UNIT_PATTERN, re.I)


def _is_heading_line(s: str) -> bool:
    """A block heading ("Продукты (на 13 порций)") is not an ingredient."""
    low = s.lower()
    return (len(s) <= RC.PARSER_HEADING_MAX_LEN
            and any(h in low for h in RC.PARSER_INGREDIENT_HEADINGS))


def _drop_wrappers(lines: list[str]) -> list[str]:
    """Remove container rows that merely repeat their children.

    Text bubbles up to ancestors while parsing, so a wrapper like
    ``<td class="ingr_title">`` surfaces as one long line containing several
    real ingredient lines. Any candidate that contains another (shorter)
    candidate verbatim is such a wrapper.
    """
    if len(lines) < 2:
        return lines
    keep = []
    for i, s in enumerate(lines):
        others = (o for j, o in enumerate(lines) if j != i and len(o) < len(s))
        if any(o in s for o in others):
            continue
        keep.append(s)
    return keep


def _clean_candidates(lines: list[str]) -> list[str]:
    """Normalize, drop junk/headings/wrappers, collapse duplicates in order."""
    out: list[str] = []
    seen: set[str] = set()
    for line in lines:
        s = _norm(line)
        if not (RC.PARSER_MIN_LINE_LEN <= len(s) <= RC.PARSER_MAX_LINE_LEN):
            continue
        if _is_heading_line(s):
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return _drop_wrappers(out)


def _looks_like_ingredients(lines: list[str], trusted: bool) -> bool:
    """Guard against returning a navigation menu as a recipe.

    A trusted source (schema.org microdata) only needs enough items; anything
    heuristic must also *look* like a shopping list — a decent share of the
    lines carrying a quantity+unit.
    """
    if len(lines) < RC.PARSER_MIN_INGREDIENTS:
        return False
    if trusted:
        return True
    with_qty = sum(1 for s in lines if _QTY_RE.search(s))
    return with_qty / len(lines) >= RC.PARSER_MIN_QTY_RATIO


# --------------------------------------------------------------------------
# Strategies
# --------------------------------------------------------------------------
def _by_microdata(records: list[dict]) -> list[str]:
    """schema.org microdata — itemprop="recipeIngredient" (or legacy plural)."""
    return [r["text"] for r in records
            if r["attrs"].get("itemprop", "").lower()
            in ("recipeingredient", "ingredients")]


def _by_class_name(records: list[dict]) -> list[str]:
    """Elements whose class/id looks ingredient-ish.

    Keeps only the deepest matches (a wrapper ``<div class="ingredients">``
    would otherwise swallow the whole block into one line) by preferring
    records whose text is short enough to be a single ingredient.
    """
    hits = [r for r in records
            if any(m in _attr_blob(r) for m in RC.PARSER_INGREDIENT_CLASS_MARKERS)]
    lines = [r["text"] for r in hits
             if len(_norm(r["text"])) <= RC.PARSER_MAX_LINE_LEN]
    return lines


def _by_heading(records: list[dict]) -> list[str]:
    """List items following an "Ингредиенты" / "Ingredients" heading.

    Stops at the next heading so the cooking steps do not leak in.
    """
    start = None
    for r in records:
        text = _norm(r["text"]).lower()
        if len(text) <= RC.PARSER_HEADING_MAX_LEN and any(
                h in text for h in RC.PARSER_INGREDIENT_HEADINGS):
            start = r["order"]
            break
    if start is None:
        return []

    stop = len(records)
    for r in records:
        if r["order"] <= start or r["tag"] not in ("h1", "h2", "h3", "h4"):
            continue
        text = _norm(r["text"]).lower()
        if not any(h in text for h in RC.PARSER_INGREDIENT_HEADINGS):
            stop = r["order"]
            break

    return [r["text"] for r in records
            if start < r["order"] < stop and r["tag"] in ("li", "td", "tr", "p")]


def _extract_title(records: list[dict]) -> Optional[str]:
    for tag in ("h1", "title"):
        for r in records:
            if r["tag"] == tag:
                t = _norm(r["text"])
                if t:
                    return t.split(" - ")[0].split(" | ")[0].strip() or t
    return None


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------
def parse_ingredients_html(html: str) -> Optional[dict]:
    """Extract a recipe from HTML without touching an LLM.

    Returns ``{"title", "ingredients": [{raw, name, qty, unit}], "source":
    "parser"}`` or ``None`` when no strategy yields a confident result — the
    caller should then fall back to the LLM.
    """
    if not html or not html.strip():
        return None

    collector = _Collector()
    try:
        collector.feed(html)
    except Exception:
        return None                                  # malformed markup — let the LLM try
    records = collector.records
    if not records:
        return None

    for strategy, trusted in ((_by_microdata, True),
                              (_by_class_name, False),
                              (_by_heading, False)):
        lines = _clean_candidates(strategy(records))
        if not _looks_like_ingredients(lines, trusted):
            continue
        ingredients = []
        for raw in lines[:RC.PARSER_MAX_INGREDIENTS]:
            name, qty, unit = split_quantity(raw)
            ingredients.append({"raw": raw, "name": name, "qty": qty, "unit": unit})
        return {"title": _extract_title(records),
                "ingredients": ingredients,
                "source": RC.SOURCE_PARSER}
    return None
