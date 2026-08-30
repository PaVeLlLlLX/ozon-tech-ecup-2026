"""Решение на мультимодальных эмбеддингах: представление даёт VLM, границу — классификатор.

Прогон №3 показал, что эмбеддинги изображений сами по себе дают 0.7529 против 0.7264 у
текстового бейзлайна, а добавленные к тексту поднимают PR-AUC редкой категории на
8.8 п.п. При этом генеративная VLM на тех же пикселях проигрывала всему. Разница в том,
кто строит разделяющую границу: 198 позитивов мало для дообучения языковой модели, но
достаточно для классификатора поверх готового представления.

Модель везёт с собой ВСЕ обученные преобразования (tf-idf, усечённые разложения,
разложение эмбеддинга), потому что на инференсе их надо воспроизвести ровно теми же —
иначе признаки окажутся другими, а ошибка будет тихой.

По умолчанию классификатор — ExtraTrees из sklearn: он есть в базовом образе
организаторов, поэтому решение не требует своего образа и проверяет ровно одну вещь —
работают ли эмбеддинги.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..features import DenseFeatureBuilder


def _make_estimator(kind: str, seed: int, y_train=None):
    from ..models.zoo import build

    return build(kind, seed, y_train=y_train)


class EmbedModel:
    """Признаки (эмбеддинги + сжатый текст + явные) → классификатор, отдельно по категориям."""

    def __init__(self, cfg: dict, kind: str = "extra_trees", seed: int = 42):
        self.cfg = cfg
        self.kind = kind
        self.seed = seed
        self.builders: dict[str, DenseFeatureBuilder] = {}
        self.models: dict[str, object] = {}
        self.thresholds: dict[str, float] = {}
        self.prior: dict[str, float] = {}

    def fit(self, df: pd.DataFrame, emb: np.ndarray, ocr: pd.Series | None) -> "EmbedModel":
        cat, tgt = self.cfg["data"]["category_col"], self.cfg["data"]["target_col"]
        pos_of = {idx: k for k, idx in enumerate(df.index)}
        for c, part in df.groupby(cat):
            rows = [pos_of[i] for i in part.index]
            builder = DenseFeatureBuilder(self.cfg, use_text=True, use_ocr_text=ocr is not None)
            X = builder.fit_transform(part, emb[rows], None if ocr is None else ocr.loc[part.index])
            y = part[tgt].to_numpy()
            est = _make_estimator(self.kind, self.seed, y_train=y).fit(X, y)
            self.builders[str(c)] = builder
            self.models[str(c)] = est
            self.thresholds.setdefault(str(c), 0.5)
            self.prior[str(c)] = float(y.mean())
        return self

    def predict_proba(self, df: pd.DataFrame, emb: np.ndarray,
                      ocr: pd.Series | None) -> np.ndarray:
        from ..models.zoo import predict_scores

        cat = self.cfg["data"]["category_col"]
        out = np.zeros(len(df), dtype=np.float32)
        pos_of = {idx: k for k, idx in enumerate(df.index)}
        for c, part in df.groupby(cat):
            rows = [pos_of[i] for i in part.index]
            builder = self.builders.get(str(c))
            if builder is None:  # незнакомая категория — отдаём средний приор
                out[rows] = float(np.mean(list(self.prior.values()) or [0.5]))
                continue
            X = builder.transform(part, emb[rows],
                                  None if ocr is None else ocr.loc[part.index])
            out[rows] = predict_scores(self.models[str(c)], X)
        return out

    def predict(self, df: pd.DataFrame, emb: np.ndarray,
                ocr: pd.Series | None) -> np.ndarray:
        cat = self.cfg["data"]["category_col"]
        proba = self.predict_proba(df, emb, ocr)
        thr = df[cat].astype(str).map(self.thresholds).fillna(0.5).to_numpy()
        return (proba >= thr).astype(int)

    def save(self, path) -> None:
        import joblib

        joblib.dump(self, path, compress=3)

    @staticmethod
    def load(path) -> "EmbedModel":
        import joblib

        return joblib.load(path)
