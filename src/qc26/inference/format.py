"""Формат ответа — часть контракта: нарушение обнуляет решение целиком.

Строка результата: `<комментарий>ТЕКСТ<вердикт>ВЕРДИКТ` без закрывающих тегов,
вердикт строго «бан» либо «не бан», длина комментария строго 50-300 символов.
Здесь и сборка строки, и валидатор — валидатор гоняем перед каждой отправкой.
"""
from __future__ import annotations

import re

import pandas as pd

COMMENT_TAG = "<комментарий>"
VERDICT_TAG = "<вердикт>"
VERDICT_BAN = "бан"
VERDICT_OK = "не бан"
MIN_LEN = 50
MAX_LEN = 300

# Запасной текст ровно нужной длины: если объяснение потерялось, строка всё равно валидна.
_FALLBACK = ("Автоматическая проверка карточки завершена, решение принято по названию, "
             "описанию и изображениям товара.")

_RESULT_RE = re.compile(
    rf"^{re.escape(COMMENT_TAG)}(?P<comment>.*){re.escape(VERDICT_TAG)}(?P<verdict>.*)$",
    re.DOTALL,
)


def clip_comment(text: str) -> str:
    """Приводит комментарий к 50-300 символам: обрезает по границе слова, добивает хвостом."""
    comment = re.sub(r"\s+", " ", str(text or "")).strip()
    # теги внутри комментария сломали бы разбор строки проверяющей системой
    comment = comment.replace(COMMENT_TAG, " ").replace(VERDICT_TAG, " ").strip()
    if not comment:
        comment = _FALLBACK

    if len(comment) > MAX_LEN:
        cut = comment.rfind(" ", 0, MAX_LEN + 1)
        comment = (comment[:cut] if cut > MIN_LEN else comment[:MAX_LEN]).rstrip(" ,;:-")
    if len(comment) < MIN_LEN:
        comment = (comment.rstrip(". ") + ". " + _FALLBACK).strip()
        comment = re.sub(r"\s+", " ", comment)[:MAX_LEN]
    if len(comment) < MIN_LEN:  # патологический случай — добиваем пробелами до минимума
        comment = comment + " " * (MIN_LEN - len(comment))
    return comment


def build_result(comment: str, is_ok: int | bool) -> str:
    """Собирает строку result. is_ok=1 → «не бан» (товар соответствует категории)."""
    verdict = VERDICT_OK if int(is_ok) == 1 else VERDICT_BAN
    return f"{COMMENT_TAG}{clip_comment(comment)}{VERDICT_TAG}{verdict}"


def validate_submission(df: pd.DataFrame, expected_ids=None) -> list[str]:
    """Возвращает список нарушений формата. Пустой список — файл валиден."""
    errors: list[str] = []
    if list(df.columns) != ["id", "result"]:
        errors.append(f"колонки должны быть ровно ['id','result'], а не {list(df.columns)}")
    if "result" not in df.columns or "id" not in df.columns:
        return errors

    if df["id"].duplicated().any():
        errors.append(f"повторяющиеся id: {int(df['id'].duplicated().sum())}")
    if df["result"].isna().any():
        errors.append(f"пустых result: {int(df['result'].isna().sum())}")

    if expected_ids is not None:
        exp = {str(i) for i in expected_ids}
        got = {str(i) for i in df["id"]}
        if exp - got:
            errors.append(f"нет ответа для {len(exp - got)} товаров")
        if got - exp:
            errors.append(f"лишних товаров в ответе: {len(got - exp)}")

    bad_shape = bad_verdict = bad_len = 0
    for value in df["result"].astype(str):
        m = _RESULT_RE.match(value)
        if not m:
            bad_shape += 1
            continue
        if m.group("verdict") not in (VERDICT_BAN, VERDICT_OK):
            bad_verdict += 1
        if not (MIN_LEN <= len(m.group("comment")) <= MAX_LEN):
            bad_len += 1
    if bad_shape:
        errors.append(f"строк не по шаблону <комментарий>...<вердикт>: {bad_shape}")
    if bad_verdict:
        errors.append(f"вердикт не «бан»/«не бан»: {bad_verdict}")
    if bad_len:
        errors.append(f"длина комментария вне 50-300: {bad_len}")
    return errors


def submission_stats(df: pd.DataFrame) -> dict:
    lens, verdicts = [], []
    for value in df["result"].astype(str):
        m = _RESULT_RE.match(value)
        if m:
            lens.append(len(m.group("comment")))
            verdicts.append(m.group("verdict"))
    s = pd.Series(lens)
    return {
        "n": len(df),
        "comment_len_min": int(s.min()) if len(s) else 0,
        "comment_len_median": float(s.median()) if len(s) else 0.0,
        "comment_len_max": int(s.max()) if len(s) else 0,
        "share_not_ban": float(pd.Series(verdicts).eq(VERDICT_OK).mean()) if verdicts else 0.0,
    }
