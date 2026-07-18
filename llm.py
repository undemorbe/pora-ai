# -*- coding: utf-8 -*-
"""DEPRECATED: используйте ``pora_llm`` (LLM) и ``recipe`` (разбор рецептов).

Тонкий шим для обратной совместимости — оставлен, чтобы старые импорты
(в том числе со стороны Go-интеграции) не сломались. Новый код должен брать
функции напрямую из соответствующего модуля.
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
)

# Разбор рецептов переехал в пакет ``recipe`` (три ступени: JSON-LD →
# Python-парсер → LLM). Имена сохранены прежними.
from recipe import (  # noqa: F401
    parse_recipe,
    extract_jsonld,
    extract_recipe_from_text,
    validate_against_source,
    web_fetch,
    html_to_text,
    extract_main_content,
    Recipe,
    Ingredient,
)
