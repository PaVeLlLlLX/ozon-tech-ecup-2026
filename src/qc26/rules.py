"""Признаки-улики по опубликованным правилам категорий.

⚠ Это НЕ решающие правила. Истина — разметка, и она местами расходится с текстом
правил. Улики нужны для двух вещей: как признаки для модели и как каркас объяснения
(в комментарии надо назвать причину, а не просто вердикт).

Напоминание про смысл метки: `label=1` («не бан») значит «товар действительно
относится к заявленной категории». Поэтому для БАД уликой «за» служит маркировка
биологически активной добавки, а для легковоспламеняющихся — то, что товар сам
является источником огня или содержит горючее, а не то, что он рядом с огнём.
"""
from __future__ import annotations

import re

import pandas as pd

CATEGORY_SUPPLEMENT = "БАД"
CATEGORY_FLAMMABLE = "Легковоспламеняющиеся"

# --- БАД -------------------------------------------------------------------
SUPPLEMENT_MARK = re.compile(
    r"биологически\s+активн|дietary\s+supplement|dietary\s+supplement|бад\s+к\s+пище|"
    r"свидетельств\w*\s+о\s+государственной\s+регистрации|сгр\b", re.I)
SUPPLEMENT_WORD = re.compile(r"\bбад\w*\b|\bbad\b", re.I)
SPORT_NUTRITION = re.compile(
    r"спортивн\w+\s+питани|bcaa|всаа|л[-\s]?карнитин|l[-\s]?carnitine|протеин|"
    r"изолят\s+сыворот|гейнер|креатин|предтрениров|whey|аминокислотн\w+\s+комплекс", re.I)
NOT_SUPPLEMENT = re.compile(r"не\s+являет\w*\s+(?:бад|биологически)", re.I)

# ⚠ Маркировка НА УПАКОВКЕ разделена по языку: замер на 725 карточках, где в тексте нет
# ни одного маркера (фон «не бан» 16.0%), показал противоположные знаки.
#   кириллическое «БАД» на фото      — 21 товар,  «не бан» 4.8%   (подтверждает БАД)
#   «биологически активн…» на фото   — 17 товаров, «не бан» 11.8%
#   английское «dietary supplement»  — 36 товаров, «не бан» 52.8% (втрое ВЫШЕ фона!)
# Английская надпись обязательна на импортных банках по праву США, но по правилам Ozon
# такой товар часто проходит как спортпитание. Слитый признак supplement_mark_ocr
# смешивал оба смысла и давал бессмыслицу, поэтому ниже они разведены.
# ⚠ Разбор 35 срабатываний на распознанном тексте показал, что наивный шаблон ловит НЕ
# маркировку товара, а три посторонние вещи, и все они чаще встречаются у НЕ-БАДов:
#   1) дисклеймер-отрицание «НЕ ЯВЛЯЕТСЯ ЛЕКАРСТВЕННЫМ СРЕДСТВОМ И БАД»;
#   2) строку состава на банке спортпитания «Биологически активное ВЕЩЕСТВО, % от нормы»;
#   3) номер СГР — он есть и у обычной пищевой продукции.
# Из 35 совпадений содержательным было одно. Поэтому: только «биологически активная
# ДОБАВКА» (не «вещество»), отрицание отсекается отдельно, голый «сгр» убран.
SUPPLEMENT_MARK_RU = re.compile(
    r"биологически\s*активн\w*\s*добавк|бад\s*к\s*пище|"
    r"свидетельств\w*\s+о\s+государственной\s+регистрации", re.I)
# окно вокруг «бад», в котором слово стоит в отрицании, а не в маркировке
SUPPLEMENT_MARK_RU_NEG = re.compile(r"не\s+являе\w*[^.]{0,60}\bбад\b", re.I)
SUPPLEMENT_MARK_EN = re.compile(r"dietary\s*suppl|\bsuppl\w*", re.I)

# --- Легковоспламеняющиеся --------------------------------------------------
IGNITION_SOURCE = re.compile(
    r"\bспичк|\bспичек|зажигалк|розжиг|растопк|сухое\s+горюч|горюч\w+\s+таблет|"
    r"фитил|бенгальск\w+\s+огн|петард|фейерверк|дымов\w+\s+шашк|цветной\s+дым|хлопушк", re.I)
FUEL_CONTENT = re.compile(
    r"баллон\w*\s+(?:с\s+)?газ|газ\w*\s+в\s+баллон|мапп[-\s]?газ|бутан|пропан|"
    r"бензин|керосин|уайт[-\s]?спирит|жидкост\w+\s+для\s+розжига|спирт\w+\s+горюч|"
    r"топлив\w+\s+для|уголь\s+древесн|брикет\w*\s+древесн", re.I)
IN_KIT = re.compile(
    r"в\s+комплект\w*[^.]{0,80}(?:угол|уголь|газ|спичк|розжиг|горюч)|"
    r"комплектаци\w*[^.]{0,80}(?:угол|уголь|газ|спичк|розжиг|горюч)|"
    r"\+\s*\d*\s*(?:цанговы\w+\s+)?баллон", re.I)
# исключения из правил: устройство для огня, аксессуар, пустая тара
DEVICE_FOR_FIRE = re.compile(
    r"\bмангал|\bгриль|барбекю|\bжаровн|газов\w+\s+плит|горелк|паяльн\w+\s+ламп|"
    r"\bрезак\b|\bкамин|\bпечь|\bпечк|коптильн|таган", re.I)
ACCESSORY = re.compile(
    r"чехол|футляр|подставк|держател|опахало|веер\b|кофр|брелок|сувенир|прикол|игрушк|"
    r"спичечниц|пепельниц|решётк|решетк|шампур|щипц|перчатк|салфетк|наклейк|"
    r"без\s+спичек|без\s+газа|без\s+топлива|газ\s+не\s+вход|в\s+комплект\w*\s+не\s+вход", re.I)


def _text(row: pd.Series) -> str:
    return f"{row.get('name', '')} \n {row.get('description', '')}"


def evidence(row: pd.Series, ocr_text: str = "") -> dict:
    """Улики по одной карточке. ocr_text — распознанный текст с изображений.

    Улики про огонь снимаются дважды: по НАЗВАНИЮ (что это за товар) и по всему
    тексту (что вообще упомянуто). Разница принципиальна: описание мангала полно
    слов «уголь» и «розжиг», но сам мангал не горюч — по правилам это исключение.
    """
    name = str(row.get("name", ""))
    txt = _text(row)
    full = txt + " \n " + str(ocr_text or "")
    cat = str(row.get("category", ""))

    ev = {
        "supplement_mark": bool(SUPPLEMENT_MARK.search(txt)),
        "supplement_mark_ocr": bool(SUPPLEMENT_MARK.search(str(ocr_text or ""))),
        # разведённые по языку улики с фото (см. комментарий у шаблонов)
        "ocr_mark_ru": bool(SUPPLEMENT_MARK_RU.search(str(ocr_text or "")))
        and not SUPPLEMENT_MARK_RU_NEG.search(str(ocr_text or "")),
        "ocr_mark_en": bool(SUPPLEMENT_MARK_EN.search(str(ocr_text or ""))),
        "ocr_sport": bool(SPORT_NUTRITION.search(str(ocr_text or ""))),
        "supplement_word": bool(SUPPLEMENT_WORD.search(txt)),
        "sport_nutrition": bool(SPORT_NUTRITION.search(txt)),
        "explicit_not_supplement": bool(NOT_SUPPLEMENT.search(txt)),
        # что за товар — по названию
        "ignition_source_name": bool(IGNITION_SOURCE.search(name)),
        "fuel_content_name": bool(FUEL_CONTENT.search(name)),
        "device_for_fire": bool(DEVICE_FOR_FIRE.search(name)),
        "accessory": bool(ACCESSORY.search(name)),
        # что упомянуто вообще — по всему тексту вместе с распознанным с фото
        "ignition_source": bool(IGNITION_SOURCE.search(full)),
        "fuel_content": bool(FUEL_CONTENT.search(full)),
        "in_kit": bool(IN_KIT.search(full)),
    }
    ev["category"] = cat
    return ev


def evidence_frame(df: pd.DataFrame, ocr: pd.Series | None = None) -> pd.DataFrame:
    """Таблица улик по всем карточкам (булевы колонки, пригодны как признаки)."""
    ocr_map = ocr if ocr is not None else pd.Series("", index=df.index)
    rows = [evidence(r, ocr_map.get(i, "")) for i, r in df.iterrows()]
    out = pd.DataFrame(rows, index=df.index)
    return out.drop(columns=["category"])


def rule_guess(row: pd.Series, ocr_text: str = "") -> int:
    """Что сказали бы сами правила: 1 — товар соответствует заявленной категории.

    Используется только как ориентир и как опора для сравнения в отчётах EDA.
    """
    ev = evidence(row, ocr_text)
    if ev["category"] == CATEGORY_SUPPLEMENT:
        if ev["explicit_not_supplement"]:
            return 0
        if ev["supplement_mark"] or ev["supplement_mark_ocr"]:
            return 1
        if ev["sport_nutrition"]:
            return 0
        return 1 if ev["supplement_word"] else 0
    # Легковоспламеняющиеся. ⚠ Замерено в EDA-3: на ключевых словах правила здесь почти
    # не работают (лучший F1 ~0.20 при фоне 3.6%) — «товар САМ является горючим» словами
    # не выражается. Оставляем как слабый ориентир и каркас объяснения, не как решение.
    if ev["accessory"] or ev["device_for_fire"]:
        return 0
    return 1 if (ev["ignition_source_name"] or ev["fuel_content_name"]) else 0
