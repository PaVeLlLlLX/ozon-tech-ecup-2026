"""Классификатор на эмбеддингах — решение, пригодное к отправке.

⚠ Класс живёт в пакете, а не в скрипте обучения: имя класса пишется внутрь файла весов,
и объявленный в скрипте он сослался бы на `__main__` и не открылся бы в `run.py`.

Устройство простое: своя модель на каждую категорию, обученная прямо на векторах, плюс
свой порог. Никаких построителей признаков внутри нет — вектор приходит готовым, его
строит `qc26.inference.embed_runtime` во время прогона.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class EmbeddingModel:
    """Модели по категориям поверх готовой матрицы эмбеддингов."""

    def __init__(self, cfg: dict, emb_set: str = "text"):
        self.cfg = cfg
        # какой набор векторов ждёт модель: «text», «image» или «multimodal».
        # Значение уезжает в архив вместе с весами, чтобы прогон не угадывал.
        self.emb_set = emb_set
        self.models: dict[str, object] = {}
        self.thresholds: dict[str, float] = {}
        self.n_features: int | None = None

    def fit(self, df: pd.DataFrame, emb: np.ndarray, model_name: str,
            seed: int = 42) -> "EmbeddingModel":
        from .zoo import build as build_estimator

        cat, tgt = self.cfg["data"]["category_col"], self.cfg["data"]["target_col"]
        X = np.asarray(emb)
        self.n_features = int(X.shape[1])
        y = df[tgt].to_numpy()
        for c in df[cat].astype(str).unique():
            m = (df[cat].astype(str) == c).to_numpy()
            if y[m].sum() < 2 or (y[m] == 0).sum() < 2:
                continue
            est = build_estimator(model_name, seed, y_train=y[m])
            est.fit(X[m], y[m])
            self.models[c] = est
            self.thresholds.setdefault(c, 0.5)
        return self

    def _scores(self, X: np.ndarray, est) -> np.ndarray:
        if hasattr(est, "predict_proba"):
            return est.predict_proba(X)[:, 1]
        return 1.0 / (1.0 + np.exp(-est.decision_function(X)))

    def predict_proba(self, df: pd.DataFrame, ocr=None, emb=None) -> np.ndarray:
        """Оценки по категориям. `ocr` не используется — принимается ради общего вызова."""
        if emb is None:
            raise RuntimeError("решению нужны эмбеддинги, а они не построены")
        # Набор может приехать словарём или парой, если прогон строил несколько.
        if isinstance(emb, dict):
            emb = emb.get(self.emb_set)
        elif isinstance(emb, (tuple, list)):
            emb = emb[0]
        if emb is None:
            raise RuntimeError(f"нужен набор векторов «{self.emb_set}», его нет")
        X = np.asarray(emb)
        if self.n_features is not None and X.shape[1] != self.n_features:
            # Молчаливое несоответствие размерности дало бы мусор вместо вердикта.
            raise RuntimeError(f"размерность векторов {X.shape[1]}, "
                               f"а модель обучена на {self.n_features}")
        cat = self.cfg["data"]["category_col"]
        out = np.full(len(df), 0.5, dtype=float)
        pos_of = {idx: k for k, idx in enumerate(df.index)}
        for c, part in df.groupby(cat):
            est = self.models.get(str(c))
            if est is None:
                continue
            idx = [pos_of[i] for i in part.index]
            out[idx] = self._scores(X[idx], est)
        return out

    def predict(self, df: pd.DataFrame, ocr=None, emb=None) -> np.ndarray:
        proba = self.predict_proba(df, ocr=ocr, emb=emb)
        thr = (df[self.cfg["data"]["category_col"]].astype(str)
               .map(self.thresholds).fillna(0.5).to_numpy())
        return (proba >= thr).astype(int)

    def save(self, path) -> None:
        import joblib

        joblib.dump(self, path)
