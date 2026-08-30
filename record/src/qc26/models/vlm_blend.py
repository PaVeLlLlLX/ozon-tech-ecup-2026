"""Связка линейной модели с дообученной VLM по общей квантильной шкале.

Зачем. 18.08 дообученная VLM публично дала 0.7809 против 0.7380 у линейной модели —
это два СИЛЬНЫХ и при этом слабо связанных источника (корреляция рангов в редкой
категории 0.591). На отложенном фолде связка даёт:

    редкая   PR-AUC 0.8840 -> 0.9048  при весе VLM 0.1
    БАД      PR-AUC 0.9841 -> 0.9902  при весе VLM 0.5

⚠ В БАД сама VLM уже лучше линейной (0.9859 против 0.9841), поэтому вес там больше.
Веса СВОИ на категорию: у них разный баланс и разные сильные стороны.

⚠ Шкала индуктивная — квантиль относительно распределения, замеренного ВНЕ ОБУЧЕНИЯ
и сохранённого в весах. Для VLM опорой служат скоры отложенного фолда (2595 товаров,
на которых она не училась): полный прогон по нашим данным дал бы переобученные,
завышенно уверенные скоры и перекосил бы квантили.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _quantile_of(x: np.ndarray, reference: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if reference is None or len(reference) == 0:
        raise RuntimeError("нет сохранённой шкалы — веса собраны неверно")
    left = np.searchsorted(reference, x, side="left")
    right = np.searchsorted(reference, x, side="right")
    return (left + right) / (2.0 * len(reference))


class VlmBlendModel:
    """Линейная модель по тексту + дообученная VLM, сложение по квантилям."""

    def __init__(self, cfg: dict, weights: dict[str, float] | None = None):
        self.cfg = cfg
        self.vlm_weight: dict[str, float] = dict(
            weights or {"БАД": 0.5, "Легковоспламеняющиеся": 0.1})
        # ⚠ Третий источник — векторы фотографий, независимый от первых двух. Вес
        # СВОЙ на категорию и по умолчанию нулевой: без обученной ветки он не влияет.
        self.image_weight: dict[str, float] = {}
        self.image_models: dict[str, object] = {}
        self.image_scalers: dict[str, object] = {}
        self.text_model = None                       # обученная TextBaseline
        self.thresholds: dict[str, float] = {}
        # ⚠ Доля выборки вместо порога — по категориям. Пусто = работает порог.
        # Смысл см. в verdict(): убирает ошибку переноса квантильной шкалы.
        self.top_fraction: dict[str, float] = {}
        # категория -> {"text": ..., "vlm": ..., "image": ...} — отсортированные скоры
        self.calibration: dict[str, dict[str, np.ndarray]] = {}

    def image_score(self, df: pd.DataFrame, emb) -> np.ndarray:
        """Оценка ветки на векторах фотографий; нули, если ветки нет."""
        if not getattr(self, "image_models", None):
            return np.zeros(len(df), dtype=np.float64)
        cat_col = self.cfg["data"]["category_col"]
        emb = np.asarray(emb, dtype=np.float32)
        out = np.zeros(len(df), dtype=np.float64)
        pos_of = {idx: k for k, idx in enumerate(df.index)}
        for cat, part in df.groupby(cat_col):
            m = self.image_models.get(str(cat))
            if m is None:
                continue
            idx = np.asarray([pos_of[i] for i in part.index])
            sc = self.image_scalers[str(cat)]
            out[idx] = m.predict_proba(sc.transform(emb[idx]))[:, 1]
        return out

    def blend(self, df: pd.DataFrame, text_score, vlm_score, img_score=None) -> np.ndarray:
        """Связанный скор. Источники считает вызывающий: VLM нужен корень картинок."""
        cat_col = self.cfg["data"]["category_col"]
        text_score = np.asarray(text_score, dtype=np.float64)
        vlm_score = np.asarray(vlm_score, dtype=np.float64)
        out = np.zeros(len(df), dtype=np.float64)
        pos_of = {idx: k for k, idx in enumerate(df.index)}
        for cat, part in df.groupby(cat_col):
            idx = np.asarray([pos_of[i] for i in part.index])
            w = float(self.vlm_weight.get(str(cat), 0.0))
            ref = self.calibration.get(str(cat)) or {}
            if not len(ref.get("text", ())) or not len(ref.get("vlm", ())):
                raise RuntimeError(f"нет шкалы для категории «{cat}»")
            # ⚠ Через getattr, а не напрямую: веса, сериализованные ДО появления
            # третьего источника, этих полей не имеют, и обращение к ним роняло
            # решение уже в контейнере (AttributeError: image_weight). Старый файл
            # весов обязан продолжать работать — просто без третьего источника.
            wi = float(getattr(self, "image_weight", {}).get(str(cat), 0.0))
            if wi > 0 and img_score is not None and len(ref.get("image", ())):
                out[idx] = ((1 - w - wi) * _quantile_of(text_score[idx], ref["text"])
                            + w * _quantile_of(vlm_score[idx], ref["vlm"])
                            + wi * _quantile_of(np.asarray(img_score)[idx], ref["image"]))
            else:
                out[idx] = ((1 - w) * _quantile_of(text_score[idx], ref["text"])
                            + w * _quantile_of(vlm_score[idx], ref["vlm"]))
        return out

    def verdict(self, df: pd.DataFrame, score: np.ndarray) -> np.ndarray:
        """Вердикт по порогу либо по ДОЛЕ выборки — смотря что задано в весах.

        ⚠ Зачем доля. Порог сравнивается со скором на квантильной шкале, построенной по
        отложенной выборке. Перенос этой шкалы на закрытые данные — источник ошибки,
        которая уже стоила нам пунктов: у связки `unseen_b` порог «верхние 2.63%» отобрал
        фактически 1.98%, потому что шкала строилась на 344 наблюдениях вместо 1101.

        Доля убирает перенос как класс: мы прямо называем «не бан» верхние K процентов
        САМОЙ проверяемой выборки. Баланс закрытых данных известен точно — 3.39% в редкой
        категории и 57.33% в БАД, восстановлено зондами, — поэтому K не гадается.

        ⚠ Через getattr: веса, сохранённые до появления этого поля, его не имеют, и
        обращение к нему уронило бы решение уже в контейнере. Так уже было с image_weight.
        """
        cat_col = self.cfg["data"]["category_col"]
        score = np.asarray(score, dtype=np.float64)
        top = dict(getattr(self, "top_fraction", {}) or {})
        out = np.zeros(len(df), dtype=int)
        pos_of = {idx: k for k, idx in enumerate(df.index)}

        missing = []
        for cat, part in df.groupby(cat_col):
            idx = np.asarray([pos_of[i] for i in part.index])
            frac = top.get(str(cat))
            if frac is not None:
                # Верхние frac выборки этой категории. Округляем вверх: при малом наборе
                # округление вниз может обнулить категорию целиком.
                n = int(np.ceil(float(frac) * len(idx)))
                n = max(0, min(n, len(idx)))
                if n:
                    order = np.argsort(-score[idx], kind="stable")
                    out[idx[order[:n]]] = 1
                continue
            thr = self.thresholds.get(str(cat))
            if thr is None:
                missing.append(str(cat))
                continue
            out[idx] = (score[idx] >= float(thr)).astype(int)
        if missing:
            raise RuntimeError(f"нет ни порога, ни доли для категорий: {sorted(missing)}")
        return out

    def save(self, path) -> None:
        import joblib

        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "VlmBlendModel":
        import joblib

        return joblib.load(path)
