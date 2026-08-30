"""Группировка почти-дубликатов — основа честной валидации.

Продавца в данных 2026 нет, а 40% товаров дословно повторяют друг друга названием и
описанием. Случайный сплит кладёт копии одного товара и в обучение, и в валидацию, и
оценка получается завышенной. Здесь строим группы: точный текст, шинглы описания
(почти-дубликаты) и совпадение изображений по хэшу. Всё сливаем в непересекающиеся
множества — одна группа целиком уходит в один фолд.
"""
from __future__ import annotations

import hashlib
import re
from collections import defaultdict

import numpy as np
import pandas as pd

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^0-9a-zа-яё ]+")


class DisjointSet:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, a: int) -> int:
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def normalize_text(s: str) -> str:
    s = str(s).lower().replace("ё", "е")
    s = _PUNCT.sub(" ", s)
    return _WS.sub(" ", s).strip()


N_HASHES = 16
BAND_SIZE = 4


def _shingle_bands(text: str, k: int = 8, n_hashes: int = N_HASHES,
                   band: int = BAND_SIZE) -> list[tuple] | None:
    """Полосы из подписи по k-словным шинглам (мини-хэш + разбиение на полосы).

    Подпись — n минимальных хэшей множества шинглов. Склеиваем только те тексты, у
    которых совпала целая полоса из `band` хэшей подряд: совпадение одного хэша даёт
    ложные склейки на общих шаблонных фразах («не является лекарственным средством»)
    и утягивает в одну группу пол-категории.
    """
    words = normalize_text(text).split()
    if len(words) < k + band:
        return None
    shingles = {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}
    if len(shingles) < band:
        return None
    hashes = sorted(int(hashlib.md5(s.encode()).hexdigest()[:12], 16) for s in shingles)
    sig = hashes[:n_hashes]
    return [(bi, *sig[bi:bi + band]) for bi in range(0, len(sig) - band + 1, band)]


def _cosine_pairs(texts: pd.Series, threshold: float, chunk: int = 512):
    """Пары строк с косинусной близостью не ниже threshold на мешке n-грамм.

    ⚠ Зачем это нужно сверх мини-хэшей. Замер 18.08: 1234 товара (9.5%) имели соседа
    с близостью >0.90 в ЧУЖОЙ группе, и метка соседа совпадала у 98.5%. Мини-хэши их
    не ловили — 16 подписей при полосе 4 дают всего четыре полосы, и пара с Жаккаром
    0.5 находится с вероятностью 23%; тексты короче 12 слов подписи не получают вовсе;
    шинглуется только описание, название — нет. Утечка стоила редкой категории 16 п.п.
    AUC (0.9969 против 0.8338), то есть офлайн-оценка держалась на копировании двойника.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(max_features=60000, ngram_range=(1, 2), min_df=2)
    X = vec.fit_transform(texts.tolist())
    norms = np.sqrt(X.multiply(X).sum(axis=1)) + 1e-9
    X = X.multiply(1.0 / norms).tocsr()
    for i in range(0, X.shape[0], chunk):
        sim = (X[i:i + chunk] @ X.T).toarray()
        for r, c in zip(*np.where(sim >= threshold)):
            a = i + int(r)
            b = int(c)
            if a < b:
                yield a, b


def build_groups(df: pd.DataFrame, name_col: str = "name", desc_col: str = "description",
                 image_hashes: dict[str, list[str]] | None = None,
                 id_col: str = "id", shingle_k: int = 8,
                 cosine_threshold: float | None = None) -> pd.Series:
    """Возвращает номер группы для каждой строки df (индекс сохраняется).

    cosine_threshold — если задан, дополнительно склеивает пары по косинусной близости
    названия с описанием. Это закрывает дыры в полноте мини-хэшей (см. `_cosine_pairs`).
    """
    n = len(df)
    ds = DisjointSet(n)
    pos = {idx: k for k, idx in enumerate(df.index)}

    def link_by_key(keys) -> None:
        buckets: dict = defaultdict(list)
        for idx, key in keys.items():
            if key is None or key == "":
                continue
            buckets[key].append(pos[idx])
        for members in buckets.values():
            for other in members[1:]:
                ds.union(members[0], other)

    # 1. точное совпадение названия и описания (после нормализации)
    norm_name = df[name_col].map(normalize_text)
    norm_desc = df[desc_col].map(normalize_text)
    link_by_key((norm_name + "||" + norm_desc).to_dict())
    # 2. одинаковое непустое описание — тот же товар от другого продавца
    link_by_key(norm_desc.where(norm_desc.str.len() >= 40).to_dict())
    # 3. одинаковое название
    link_by_key(norm_name.where(norm_name.str.len() >= 10).to_dict())

    # 4. почти-дубликаты описаний: совпадение целой полосы мини-хэшей
    sig_buckets: dict = defaultdict(list)
    for idx, text in norm_desc.items():
        bands = _shingle_bands(text, k=shingle_k)
        if bands is None:
            continue
        for b in bands:
            sig_buckets[b].append(pos[idx])
    for members in sig_buckets.values():
        if len(members) > 60:  # вырожденный шаблон описания — не склеивать пол-категории
            continue
        for other in members[1:]:
            ds.union(members[0], other)

    # 5. общие изображения (хэши приходят снаружи, чтобы не читать файлы дважды).
    # Точную копию файла (md5) считаем надёжной уликой, похожесть (перцептивный хэш) —
    # слабее: белый фон каталожных фото даёт совпадения у разных товаров.
    if image_hashes:
        img_buckets: dict = defaultdict(list)
        for idx in df.index:
            for h in image_hashes.get(str(df.at[idx, id_col]), []):
                img_buckets[h].append(pos[idx])
        for key, members in img_buckets.items():
            cap = 60 if str(key).startswith("m:") else 12
            if len(members) > cap:
                continue
            for other in members[1:]:
                ds.union(members[0], other)

    # 6. близкие по словам карточки, которых не поймали ни точное совпадение, ни полосы
    if cosine_threshold is not None:
        joined = (norm_name + " " + norm_desc).reset_index(drop=True)
        for a, b in _cosine_pairs(joined, float(cosine_threshold)):
            ds.union(a, b)

    roots = np.array([ds.find(k) for k in range(n)])
    _, group_ids = np.unique(roots, return_inverse=True)
    return pd.Series(group_ids, index=df.index, name="group")


def group_purity(df: pd.DataFrame, group_col: str = "group",
                 target_col: str = "label") -> dict:
    """Насколько метки согласованы внутри групп — оценка шума разметки и потолка."""
    agg = df.groupby(group_col)[target_col].agg(["size", "mean"])
    multi = agg[agg["size"] > 1]
    mixed = multi[(multi["mean"] > 0) & (multi["mean"] < 1)]
    return {
        "n_groups": int(len(agg)),
        "n_multi_groups": int(len(multi)),
        "n_items_in_multi": int(multi["size"].sum()),
        "n_mixed_groups": int(len(mixed)),
        "n_items_in_mixed": int(mixed["size"].sum()),
        "purity": float(1 - len(mixed) / len(multi)) if len(multi) else 1.0,
    }
