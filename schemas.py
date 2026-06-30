# -*- coding: utf-8 -*-
"""Pydantic-схемы запросов. Это контракт для Go-бэкенда. Поле lang опционально:
если не передать — язык определяется автоматически (по тексту)."""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class Purchase(BaseModel):
    product: str
    date: str = Field(..., description="ISO дата, напр. 2026-06-01")


class ReplenishmentRequest(BaseModel):
    today: Optional[str] = None
    purchases: list[Purchase]


class CategorizeRequest(BaseModel):
    names: list[str]
    lang: Optional[str] = Field(None, description="ru/en/… — если None, определим сами")
    sections: Optional[list[str]] = Field(
        None,
        description="Custom store taxonomy (canonical KEYS, language-agnostic). "
                    "If provided, the fast classifier is bypassed and the LLM is used "
                    "with this enum. If None, brain.SECTIONS is used.",
    )
    section_labels: Optional[dict] = Field(
        None,
        description="Optional {section_key: localized_label} map for custom sections. "
                    "If absent, the section key itself is echoed as section_label.",
    )


class NotifyTimeRequest(BaseModel):
    opens: list[str] = Field(..., description="ISO datetime заходов в приложение")


class RecommendRequest(BaseModel):
    recipe_imports: list[str] = Field(default_factory=list)
    regular_products: list[str] = Field(default_factory=list)
    lang: Optional[str] = None


class ParseRecipeRequest(BaseModel):
    url: str
    lang: Optional[str] = None
    sections: Optional[list[str]] = Field(
        None,
        description="Custom section taxonomy applied to every ingredient. "
                    "If None, the fast classifier with brain.SECTIONS is used.",
    )


class TipRequest(BaseModel):
    top_cuisine: str = "Итальянская"
    frequent: list[str] = Field(default_factory=list)
    lang: Optional[str] = None


class ChatRequest(BaseModel):
    message: str
    lang: Optional[str] = None


class BriefRequest(BaseModel):
    today: Optional[str] = None
    purchases: list[Purchase] = Field(default_factory=list)
    opens: list[str] = Field(default_factory=list)
    recipe_imports: list[str] = Field(default_factory=list)
    regular_products: list[str] = Field(default_factory=list)
    lang: Optional[str] = None


class SuggestRequest(BaseModel):
    today: Optional[str] = None
    purchases: list[Purchase] = Field(default_factory=list)
    recipe_imports: list[str] = Field(default_factory=list)
    regular_products: list[str] = Field(default_factory=list)
    current_cart: list[str] = Field(default_factory=list,
                                    description="Items already in the user's cart right now")
    lang: Optional[str] = None
    limit: int = Field(5, ge=1, le=20)
