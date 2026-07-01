# -*- coding: utf-8 -*-
"""DEPRECATED: модуль объединён в pora_llm.py (мультиязычный).
Оставлен как тонкий шим для обратной совместимости — используйте pora_llm.

Экспорты обновлены под текущую поверхность API v2 (web_fetch, batched
categorize, anti-hallucination, dynamic sections).
"""
from pora_llm import (  # noqa: F401
    # core
    llm_enabled,
    detect_lang,
    refusal,
    REFUSALS,
    # chat / tip / dish
    chat,
    generate_tip,
    suggest_dish_llm,
    # categorization
    categorize_llm,
    categorize_llm_batch,
    # recipes
    parse_recipe,
    extract_jsonld,
    extract_recipe_from_text,
    validate_against_source,
    # HTTP / HTML utilities
    web_fetch,
    html_to_text,
    extract_main_content,
)
