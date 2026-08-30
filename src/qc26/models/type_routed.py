"""Решение с маршрутизацией по типу товара: общая модель плюс свои обработчики.

⚠ Класс живёт в пакете, а не в скрипте обучения, и это принципиально: имя класса
записывается внутрь файла весов. Пока такой класс объявлялся в скрипте, веса ссылались
на `__main__` и открывались только тем же скриптом — внутри `run.py`, то есть на самой
сдаче, файл не открылся бы вовсе.

Устройство простое. Основа — обычная модель по категориям. Для тех типов редкой
категории, где замер показал выигрыш, оценку выдаёт свой обработчик; для остальных
работает основа. Тип определяется по названию и началу описания теми же правилами, что
и при обучении.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..features import assign_product_type
from .text_baseline import TextBaseline

RARE = "Легковоспламеняющиеся"


class TypeRoutedModel:
    """Общая модель + свои обработчики на выбранные типы товара."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.base: TextBaseline | None = None
        self.picks: dict[str, str] = {}
        self.handlers: dict[str, object] = {}
        self.builders: dict[str, object] = {}
        self.thresholds: dict[str, float] = {}

    def fit(self, df: pd.DataFrame, ocr, picks: dict[str, str],
            rare_mask: np.ndarray) -> "TypeRoutedModel":
        from ..models.zoo import build as build_estimator
        from train_zoo import FEATURE_SETS, build_fold_features

        cfg, tgt = self.cfg, self.cfg["data"]["target_col"]
        self.base = TextBaseline(cfg).fit(df, ocr)
        self.picks = {t: k for t, k in picks.items() if k != "global"}
        if not self.picks:
            return self

        types = assign_product_type(df, cfg["data"]["name_col"], cfg["data"]["desc_col"])
        y = df[tgt].to_numpy()
        # оценка основы на всех данных — она же признак для обработчиков типа
        base_score = self.base.predict_proba(df, ocr)
        spec = FEATURE_SETS["dense_text_ocr"]

        for t, kind in self.picks.items():
            tm = rare_mask & (types == t).to_numpy()
            tr = tm if kind in ("svm_type", "boost_type", "stack_type") else rare_mask
            if y[tr].sum() < 2:
                continue
            if kind == "svm_type":
                self.handlers[t] = TextBaseline(cfg).fit(df[tr], ocr)
                continue
            Xtr, _, builder = build_fold_features(
                spec, cfg, df[tr], df[tr].head(1),
                None if ocr is None else ocr[tr], None if ocr is None else ocr[tr].head(1),
                None, None)
            Xtr = np.asarray(Xtr)
            if kind == "stack_type":
                Xtr = np.column_stack([Xtr, base_score[tr]])
            est = build_estimator("lightgbm", 42, y_train=y[tr])
            est.fit(Xtr, y[tr])
            self.handlers[t] = est
            self.builders[t] = builder
        return self

    def predict_proba(self, df: pd.DataFrame, ocr=None) -> np.ndarray:
        cfg = self.cfg
        out = self.base.predict_proba(df, ocr)
        if not self.handlers:
            return out
        types = assign_product_type(df, cfg["data"]["name_col"], cfg["data"]["desc_col"])
        cat = df[cfg["data"]["category_col"]].astype(str).to_numpy()
        for t, model in self.handlers.items():
            m = (types.to_numpy() == t) & (cat == RARE)
            if not m.any():
                continue
            part = df[m]
            if isinstance(model, TextBaseline):
                out[m] = model.predict_proba(part, ocr)
                continue
            builder = self.builders[t]
            X = np.asarray(builder.transform(part, None,
                                             None if ocr is None else ocr[m]))
            if self.picks.get(t) == "stack_type":
                X = np.column_stack([X, out[m]])
            out[m] = model.predict_proba(X)[:, 1]
        return out

    def predict(self, df: pd.DataFrame, ocr=None) -> np.ndarray:
        proba = self.predict_proba(df, ocr)
        thr = (df[self.cfg["data"]["category_col"]].astype(str)
               .map(self.thresholds).fillna(0.5).to_numpy())
        return (proba >= thr).astype(int)

    def save(self, path) -> None:
        import joblib

        joblib.dump(self, path)
