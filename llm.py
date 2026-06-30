# -*- coding: utf-8 -*-
"""DEPRECATED: модуль объединён в pora_llm.py (мультиязычный).
Оставлен как тонкий шим для обратной совместимости — используйте pora_llm."""
from pora_llm import (  # noqa: F401
    llm_enabled, chat, categorize_llm, generate_tip,
    parse_recipe, extract_jsonld, extract_recipe_from_text, detect_lang,
)
