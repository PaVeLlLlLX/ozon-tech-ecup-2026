"""Поиск почти-дубликата обучающего товара во время прогона.

Наши сплиты групповые: почти-дубликаты не разрываются между фолдами, иначе метрика
завышается на 19.5 п.п. Это верно для честного замера МОДЕЛИ — и ровно поэтому мы почти
две недели не замечали, что на закрытых данных всё наоборот. Если организаторы делили
каталог по товарам, а не по группам, то у тестового товара его дубликат лежит в выданной
нам обучающей выборке **вместе с меткой**.

Замер моделированием (случайное разбиение 70/30 по товарам, пять повторов):

    только текст   покрытие редкой 67.3%, вердикт по соседу верен в 99.8%
    текст + фото   покрытие редкой 74.4%, верен в 99.6%

То есть текстовый поиск забирает почти всю выгоду и **не требует ни одной картинки** —
ни времени на прогоне, ни зависимости от easyocr, ни риска на файловой системе только
для чтения.

Подписи те же, что склеивают группы в `qc26.groups`: точное совпадение нормализованного
названия и описания, отдельно описание, отдельно название, плюс полосы мини-хэшей по
восьмисловным шинглам. Таблица подписей строится заранее и кладётся в архив.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from ..groups import _shingle_bands, normalize_text

# ⚠ Таблица хранится в csv.gz, а НЕ в parquet: в базовом образе проверяющей системы нет
# ни pyarrow, ни fastparquet, и `pd.read_parquet` там падает с ImportError. Отправка от
# 15.08 на этом обнулилась. Гзип разбирает стандартная библиотека, csv читает сам pandas.
TABLE_SUFFIX = ".csv.gz"


def _digest(key: str) -> str:
    """Ключ фиксированной длины вместо сырого текста.

    Название с описанием целиком — это сотни байт на строку, из-за чего таблица
    раздувалась до 22 МБ. Хэш даёт 16 символов, а заодно в архив не попадает текст
    обучающих карточек.
    """
    return hashlib.md5(key.encode("utf-8")).hexdigest()[:16]

# Ключи ищутся в этом порядке: чем выше, тем строже совпадение. Первый сработавший и
# даёт ответ — иначе слабый ключ мог бы перебить точное совпадение названия с описанием.
KEY_KINDS = ("name_desc", "desc", "name", "band")

# Полоса мини-хэшей, встречающаяся у слишком многих товаров, — это шаблон описания,
# а не дубликат. Тот же порог, что и при построении групп.
MAX_BAND_MEMBERS = 60


def signatures(df: pd.DataFrame, name_col: str, desc_col: str) -> dict[str, list[tuple]]:
    """Подписи каждого товара по видам ключей: {вид: [(позиция, ключ), ...]}."""
    norm_name = df[name_col].map(normalize_text)
    norm_desc = df[desc_col].map(normalize_text)
    pos = {idx: k for k, idx in enumerate(df.index)}

    out: dict[str, list[tuple]] = {k: [] for k in KEY_KINDS}
    for idx in df.index:
        p, nm, ds = pos[idx], norm_name[idx], norm_desc[idx]
        if nm and ds:
            out["name_desc"].append((p, nm + "||" + ds))
        if len(ds) >= 40:
            out["desc"].append((p, ds))
        if len(nm) >= 10:
            out["name"].append((p, nm))
        for b in _shingle_bands(ds) or []:
            out["band"].append((p, str(b)))
    return out


def build_table(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Таблица «ключ -> метка» по обучающей выборке. Кладётся в архив решения."""
    name_col, desc_col = cfg["data"]["name_col"], cfg["data"]["desc_col"]
    tgt, cat = cfg["data"]["target_col"], cfg["data"]["category_col"]
    sig = signatures(df, name_col, desc_col)
    labels = df[tgt].to_numpy()
    cats = df[cat].astype(str).to_numpy()

    rows = []
    for kind, pairs in sig.items():
        agg: dict[str, list] = {}
        for p, key in pairs:
            agg.setdefault(key, []).append(p)
        for key, members in agg.items():
            # Вырожденный шаблон описания склеил бы пол-категории — такие ключи выкидываем
            if kind == "band" and len(members) > MAX_BAND_MEMBERS:
                continue
            rows.append({"kind": kind, "key": _digest(key),
                         "label": round(float(np.mean(labels[members])), 4),
                         "n": len(members),
                         "category": pd.Series(cats[members]).mode().iat[0]})
    return pd.DataFrame(rows)


def load_table(path) -> pd.DataFrame:
    """Чтение таблицы подписей без зависимостей сверх pandas."""
    return pd.read_csv(path, dtype={"key": str, "kind": str})


def lookup(df: pd.DataFrame, table: pd.DataFrame, cfg: dict) -> pd.Series:
    """Метка найденного дубликата для каждой строки df; NaN — дубликат не найден.

    Ключи проверяются от строгого к слабому, первый сработавший даёт ответ.
    """
    name_col, desc_col = cfg["data"]["name_col"], cfg["data"]["desc_col"]
    sig = signatures(df, name_col, desc_col)
    out = np.full(len(df), np.nan, dtype=float)

    for kind in KEY_KINDS:
        part = table[table["kind"] == kind]
        if part.empty:
            continue
        lut = dict(zip(part["key"].astype(str), part["label"].astype(float)))
        for p, key in sig.get(kind, []):
            if not np.isnan(out[p]):
                continue  # уже нашли по более строгому ключу
            val = lut.get(_digest(key))
            if val is not None:
                out[p] = val
    return pd.Series(out, index=df.index)


def apply_lookup(preds: np.ndarray, hits: pd.Series, undecided: float = 0.5) -> np.ndarray:
    """Вердикт: где дубликат нашёлся — его метка, где нет — вердикт модели.

    Ничьи (в группе дубликатов метки разошлись ровно пополам) оставляем модели: такие
    группы составляют около 6% и доверять им нечему.
    """
    out = np.asarray(preds).astype(int).copy()
    h = hits.to_numpy()
    decided = ~np.isnan(h) & (np.abs(h - undecided) > 1e-9)
    out[decided] = (h[decided] > undecided).astype(int)
    return out
