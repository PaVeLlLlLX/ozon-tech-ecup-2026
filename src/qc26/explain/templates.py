"""Объяснение вердикта для проверяющего сотрудника (50-300 символов).

Базовая версия — сборка из улик, а не свободная генерация: она гарантированно
укладывается в лимит и никогда не противоречит вердикту. Это опора, с которой будем
сравнивать генерацию моделью (её на этапе финалистов оценивает llm-as-a-judge).

Комментарий строим по схеме «что за товар → какая улика найдена → вывод о категории»,
потому что сотруднику нужна причина решения, а не его пересказ.
"""
from __future__ import annotations

import pandas as pd

from ..rules import CATEGORY_SUPPLEMENT, evidence

_SUP_OK = [
    ("supplement_mark_ocr", "на изображении упаковки видна маркировка биологически активной добавки"),
    ("supplement_mark", "в описании есть прямое указание «биологически активная добавка к пище»"),
    ("supplement_word", "карточка прямо обозначает товар как БАД"),
]
_SUP_BAN = [
    ("explicit_not_supplement", "в описании прямо сказано, что товар не является БАД"),
    ("sport_nutrition", "товар отнесён к спортивному питанию, а оно к БАД не относится"),
]
_FLAM_OK = [
    ("fuel_content_name", "товар содержит горючее вещество или газ"),
    ("ignition_source_name", "товар сам является источником открытого огня"),
    ("in_kit", "в комплект входит горючее содержимое"),
    ("fuel_content", "в составе товара указано горючее вещество"),
]
_FLAM_BAN = [
    # порядок важен: мангал с шампурами — устройство для огня, а не аксессуар
    ("device_for_fire", "это устройство для использования с огнём, а само по себе оно не горючее"),
    ("accessory", "это аксессуар или тара без горючего содержимого"),
]


def _pick(ev: dict, options: list[tuple[str, str]]) -> str | None:
    for key, phrase in options:
        if ev.get(key):
            return phrase
    return None


def explain(row: pd.Series, is_ok: int, ocr_text: str = "") -> str:
    """Комментарий к вердикту. is_ok=1 — товар соответствует заявленной категории."""
    ev = evidence(row, ocr_text)
    name = str(row.get("name", "")).strip()
    short = (name[:70].rsplit(" ", 1)[0] if len(name) > 70 else name) or "Товар"
    is_supplement = ev["category"] == CATEGORY_SUPPLEMENT

    if is_supplement:
        reason = _pick(ev, _SUP_OK if is_ok else _SUP_BAN)
        if is_ok:
            reason = reason or "состав и назначение соответствуют биологически активной добавке"
            tail = "Товар относится к категории БАД, карточка заполнена корректно."
        else:
            reason = reason or ("ни в тексте, ни на изображениях нет маркировки БАД "
                                "или dietary supplement")
            tail = "Отнести товар к категории БАД нельзя."
    else:
        reason = _pick(ev, _FLAM_OK if is_ok else _FLAM_BAN)
        if is_ok:
            reason = reason or "товар относится к горючим изделиям по составу и назначению"
            tail = "Товар относится к легковоспламеняющимся, категория указана верно."
        else:
            reason = reason or "источника воспламенения и горючего содержимого не выявлено"
            tail = "К легковоспламеняющимся товар не относится."

    return f"«{short}»: {reason}. {tail}"


def explain_frame(df: pd.DataFrame, preds, ocr: pd.Series | None = None) -> list[str]:
    ocr_map = ocr if ocr is not None else pd.Series("", index=df.index)
    return [explain(row, int(p), ocr_map.get(i, ""))
            for (i, row), p in zip(df.iterrows(), preds)]
