"""Смесь двух решений с подмешиванием ТОЛЬКО в редкой категории.

Метрика — среднее двух F1, и категория БАД у нас насыщена (около 0.94 при 74.5%
позитивов): даже идеальная работа там прибавит к метрике три пункта. Весь запас лежит в
категории «Легковоспламеняющиеся», где позитивов 3.6%. Поэтому смешивать по всей выборке
смысла нет — в БАД это только вносит шум, а лишняя степень свободы ничем не окупается.

Устройство: основа держит обе категории, а в редкой к её оценке подмешивается оценка
второго решения другой природы. Замер на предсказаниях вне обучения с поправкой на
баланс закрытой выборки: основа на тексте с распознанным дала в редкой категории 0.590,
модель на эмбеддингах — 0.642, их смесь — **0.660**.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class BlendModel:
    """Основа + подмешивание в одной категории. Читается из run.py как обычные веса."""

    def __init__(self, base, admix, weight: float, category: str,
                 thresholds: dict[str, float], cfg: dict):
        self.base = base
        self.admix = admix
        self.weight = float(weight)
        self.category = category
        self.thresholds = dict(thresholds)
        self.cfg = cfg
        # run.py отличает решения, которым нужны эмбеддинги, по наличию этого поля —
        # смеси они нужны так же, как и вложенной модели, поэтому поле выставляется явно.
        self.builders = {"смесь": True}

    def _scores(self, model, df: pd.DataFrame, ocr, emb) -> np.ndarray:
        """Оценки любой из наших моделей: у одних есть scores, у других predict_proba."""
        if hasattr(model, "scores"):
            return np.asarray(model.scores(df, emb=emb, ocr=ocr), dtype=np.float64)
        return np.asarray(model.predict_proba(df, ocr), dtype=np.float64)

    def predict(self, df: pd.DataFrame, emb=None, ocr=None) -> np.ndarray:
        base = self._scores(self.base, df, ocr, emb)
        mixed = base.copy()
        cat_col = self.cfg["data"]["category_col"]
        is_target = (df[cat_col].astype(str) == self.category).to_numpy()
        if is_target.any():
            other = self._scores(self.admix, df, ocr, emb)
            mixed[is_target] = ((1 - self.weight) * base[is_target]
                                + self.weight * other[is_target])
        thr = df[cat_col].astype(str).map(self.thresholds).fillna(0.5).to_numpy()
        return (mixed >= thr).astype(int)
