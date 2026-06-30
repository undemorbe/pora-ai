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


class NotifyTimeRequest(BaseModel):
    opens: list[str] = Field(..., description="ISO datetime заходов в приложение")


class RecommendRequest(BaseModel):
    recipe_imports: list[str] = Field(default_factory=list)
    regular_products: list[str] = Field(default_factory=list)
    lang: Optional[str] = None


class ParseRecipeRequest(BaseModel):
    url: str
    lang: Optional[str] = None


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
