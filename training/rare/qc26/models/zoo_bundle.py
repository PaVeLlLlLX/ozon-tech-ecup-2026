"""Выгруженная модель из набора: цепочка признаков, модели по категориям и пороги.

⚠ Класс живёт в пакете, а не в скрипте выгрузки, и это принципиально: имя класса
записывается внутрь файла весов. Пока он объявлялся в `scripts/export_zoo_model.py`,
веса ссылались на `__main__.ZooBundle` и загружались только тем же скриптом — внутри
`run.py`, то есть на самой сдаче, такой файл не открылся бы вовсе.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .zoo import predict_scores


class ZooBundle:
    """Цепочка признаков и модели по категориям + пороги. Читается из run.py."""

    def __init__(self, cfg: dict, feature_set: str, use_ocr: bool = True):
        self.cfg = cfg
        self.feature_set = feature_set
        # Признак нужен именно здесь: наборы отличаются тем, входит ли распознанный
        # текст в обучение. Подать его модели, которая училась без него, — молча
        # испортить признаки, а не получить ошибку.
        self.use_ocr = use_ocr
        self.builders: dict = {}
        self.models: dict = {}
        self.thresholds: dict[str, float] = {}

    def _features(self, builder, part: pd.DataFrame, sub_emb, sub_ocr):
        """Признаки по обученному преобразованию.

        У плотных наборов преобразование — объект с методом transform, у разреженных —
        словарь обученных векторизаторов, и вызывать его надо иначе. Раньше класс умел
        только первое, поэтому любая выгруженная линейная модель падала при первом же
        предсказании.
        """
        if hasattr(builder, "transform"):
            return builder.transform(part, sub_emb, sub_ocr)
        from ..features import sparse_text_matrix

        X, _ = sparse_text_matrix(self.cfg, part, sub_ocr, vectorizers=builder)
        return X

    def scores(self, df: pd.DataFrame, emb=None, ocr=None) -> np.ndarray:
        """Оценки без порога — нужны отдельно, чтобы смешивать решения между собой."""
        cat = self.cfg["data"]["category_col"]
        out = np.zeros(len(df), dtype=np.float64)
        pos_of = {idx: k for k, idx in enumerate(df.index)}
        for c, part in df.groupby(cat):
            idx = [pos_of[i] for i in part.index]
            builder, model = self.builders.get(str(c)), self.models.get(str(c))
            if builder is None or model is None:
                continue
            sub_ocr = None if (ocr is None or not getattr(self, "use_ocr", True)) \
                else ocr.loc[part.index]
            sub_emb = None if emb is None else emb[idx]
            X = self._features(builder, part, sub_emb, sub_ocr)
            # Моделей может быть несколько — по одной на сид. Усреднение по сидам дало
            # крупнейший дешёвый прирост из измеренных, поэтому оно вшито в саму выгрузку.
            ests = model if isinstance(model, list) else [model]
            out[idx] = np.mean([predict_scores(e, X) for e in ests], axis=0)
        return out

    def predict(self, df: pd.DataFrame, emb=None, ocr=None) -> np.ndarray:
        cat = self.cfg["data"]["category_col"]
        score = self.scores(df, emb=emb, ocr=ocr)
        thr = df[cat].astype(str).map(self.thresholds).fillna(0.5).to_numpy()
        return (score >= thr).astype(int)
