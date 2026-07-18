# -*- coding: utf-8 -*-
"""Tests for _recipe_parse — the pure-Python (no LLM) recipe extractor."""
from __future__ import annotations

from _recipe_parse import parse_ingredients_html, split_quantity


class TestSplitQuantity:
    def test_ru_dash_separated(self):
        assert split_quantity("Кабачок - 550 г") == ("Кабачок", 550.0, "г")

    def test_ru_no_dash(self):
        assert split_quantity("Молоко 200 мл") == ("Молоко", 200.0, "мл")

    def test_en_leading_quantity(self):
        assert split_quantity("2 cups flour") == ("flour", 2.0, "cups")

    def test_en_trailing_quantity(self):
        assert split_quantity("Sugar 100 g") == ("Sugar", 100.0, "g")

    def test_fraction_comma_decimal(self):
        name, qty, unit = split_quantity("Сметана 1,5 ст. л.")
        assert name == "Сметана" and qty == 1.5

    def test_piece_unit(self):
        assert split_quantity("Яйцо - 3 шт") == ("Яйцо", 3.0, "шт")

    def test_no_quantity_keeps_whole_name(self):
        assert split_quantity("соль по вкусу") == ("соль по вкусу", None, None)

    def test_trailing_parenthetical_ignored(self):
        # "(2 зубчика)" is a clarification, not the quantity
        assert split_quantity("Чеснок - 13 г (2 зубчика)") == ("Чеснок", 13.0, "г")

    def test_parenthetical_brand_dropped_from_name(self):
        name, qty, unit = split_quantity("Кукуруза (Фрау Марта) — 0.5 бан.")
        assert name == "Кукуруза"
        assert qty == 0.5

    def test_empty(self):
        assert split_quantity("") == ("", None, None)


class TestMicrodataStrategy:
    def test_itemprop_recipe_ingredient(self):
        html = """<html><body>
          <h1>Оладьи</h1>
          <ul>
            <li><span itemprop="recipeIngredient">Мука - 200 г</span></li>
            <li><span itemprop="recipeIngredient">Молоко - 250 мл</span></li>
            <li><span itemprop="recipeIngredient">Яйцо - 2 шт</span></li>
          </ul></body></html>"""
        out = parse_ingredients_html(html)
        assert out is not None
        assert out["source"] == "parser"
        assert out["title"] == "Оладьи"
        assert [i["raw"] for i in out["ingredients"]] == [
            "Мука - 200 г", "Молоко - 250 мл", "Яйцо - 2 шт"]
        assert out["ingredients"][0]["qty"] == 200.0
        assert out["ingredients"][0]["unit"] == "г"
        assert out["ingredients"][0]["name"] == "Мука"

    def test_legacy_ingredients_itemprop(self):
        html = ('<div><span itemprop="ingredients">Соль - 5 г</span>'
                '<span itemprop="ingredients">Перец - 2 г</span></div>')
        out = parse_ingredients_html(html)
        assert out and len(out["ingredients"]) == 2


class TestClassNameStrategy:
    def test_russianfood_style_table_rows(self):
        # Real-world shape: no microdata, ingredients live in class="ingr_tr_N"
        html = """<html><head><title>Рулетики - RussianFood.com</title></head><body>
          <table><tr><td class="ingr_title"><span class="prod">Продукты</span></td></tr>
          <tr class="ingr_tr_0"><td><span>Кабачок - 550 г</span></td></tr>
          <tr class="ingr_tr_1"><td><span>Брынза - 190 г</span></td></tr>
          <tr class="ingr_tr_0"><td><span>Творог - 60 г</span></td></tr>
          </table></body></html>"""
        out = parse_ingredients_html(html)
        assert out is not None
        raws = [i["raw"] for i in out["ingredients"]]
        assert "Кабачок - 550 г" in raws
        assert "Брынза - 190 г" in raws
        assert "Творог - 60 г" in raws

    def test_english_ingredient_class(self):
        html = """<div>
          <li class="recipe-ingredient">2 cups flour</li>
          <li class="recipe-ingredient">1 tbsp sugar</li>
          <li class="recipe-ingredient">3 eggs</li></div>"""
        out = parse_ingredients_html(html)
        assert out and len(out["ingredients"]) == 3


class TestHeadingStrategy:
    def test_list_after_ingredients_heading(self):
        html = """<html><body>
          <nav><ul><li>Главная</li><li>Рецепты</li><li>Войти</li></ul></nav>
          <h2>Ингредиенты</h2>
          <ul><li>Картофель - 500 г</li><li>Лук - 1 шт</li><li>Масло - 30 мл</li></ul>
          <h2>Приготовление</h2><p>Нарезать и обжарить.</p></body></html>"""
        out = parse_ingredients_html(html)
        assert out is not None
        raws = [i["raw"] for i in out["ingredients"]]
        assert "Картофель - 500 г" in raws
        # navigation must not leak in
        assert not any("Главная" in r for r in raws)


class TestRejection:
    def test_page_without_recipe_returns_none(self):
        html = "<html><body><h1>О компании</h1><p>Мы продаём мебель с 1999 года.</p></body></html>"
        assert parse_ingredients_html(html) is None

    def test_too_few_ingredients_rejected(self):
        # one lonely quantity is not a recipe — let the LLM decide instead
        html = '<div><li class="ingredient">Соль - 5 г</li></div>'
        assert parse_ingredients_html(html) is None

    def test_navigation_only_page_rejected(self):
        html = ("<html><body><ul>" +
                "".join(f"<li>Пункт меню {i}</li>" for i in range(30)) +
                "</ul></body></html>")
        assert parse_ingredients_html(html) is None

    def test_empty_html(self):
        assert parse_ingredients_html("") is None


class TestNoise:
    def test_scripts_and_styles_ignored(self):
        html = """<html><body>
          <script>var ingredients = "Яд - 100 г";</script>
          <style>.ingredient { color: red }</style>
          <li class="ingredient">Мука - 200 г</li>
          <li class="ingredient">Соль - 5 г</li>
          <li class="ingredient">Вода - 100 мл</li></body></html>"""
        out = parse_ingredients_html(html)
        assert out is not None
        assert not any("Яд" in i["raw"] for i in out["ingredients"])

    def test_wrapper_rows_dropped(self):
        # The <td class="ingr_title"> wrapper repeats the whole block; only the
        # leaf rows are real ingredients.
        html = """<table>
          <tr class="ingr_tr_0"><td class="ingr_title">Продукты Кабачок - 550 г Брынза - 190 г</td></tr>
          <tr class="ingr_tr_1"><td>Кабачок - 550 г</td></tr>
          <tr class="ingr_tr_0"><td>Брынза - 190 г</td></tr>
          <tr class="ingr_tr_1"><td>Творог - 60 г</td></tr></table>"""
        out = parse_ingredients_html(html)
        raws = [i["raw"] for i in out["ingredients"]]
        assert "Кабачок - 550 г" in raws
        assert not any(r.startswith("Продукты") for r in raws), raws

    def test_heading_row_dropped(self):
        html = """<div>
          <li class="ingredient">Продукты (на 4 порции)</li>
          <li class="ingredient">Мука - 200 г</li>
          <li class="ingredient">Соль - 5 г</li>
          <li class="ingredient">Вода - 100 мл</li></div>"""
        out = parse_ingredients_html(html)
        raws = [i["raw"] for i in out["ingredients"]]
        assert not any("Продукты" in r for r in raws), raws

    def test_duplicates_collapsed(self):
        html = """<div>
          <li class="ingredient">Мука - 200 г</li>
          <li class="ingredient">Мука - 200 г</li>
          <li class="ingredient">Соль - 5 г</li>
          <li class="ingredient">Вода - 100 мл</li></div>"""
        out = parse_ingredients_html(html)
        raws = [i["raw"] for i in out["ingredients"]]
        assert raws.count("Мука - 200 г") == 1

    def test_absurdly_long_lines_dropped(self):
        long_line = "Мука " + "очень " * 100 + "- 200 г"
        html = f"""<div>
          <li class="ingredient">{long_line}</li>
          <li class="ingredient">Соль - 5 г</li>
          <li class="ingredient">Вода - 100 мл</li>
          <li class="ingredient">Сахар - 50 г</li></div>"""
        out = parse_ingredients_html(html)
        assert all(len(i["raw"]) < 200 for i in out["ingredients"])
