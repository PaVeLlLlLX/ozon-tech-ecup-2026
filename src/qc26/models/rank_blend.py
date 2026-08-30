"""Ранговая связка боевой текстовой модели с классификатором на эмбеддингах фотографий.

⚠ Класс живёт в пакете, а не в скрипте обучения: имя класса пишется внутрь файла весов,
и объявленный в скрипте он сослался бы на `__main__` и не открылся бы в `run.py`.

Зачем связка. Замер 18.08 на честных фолдах (`artifacts/splits/groups_v2.parquet`, где
почти-дубликаты разведены по построению) в редкой категории:

    боевая линейная модель      PR-AUC 0.6334
    эмбеддинги фотографий       PR-AUC 0.3181
    связка рангов, вес 0.2      PR-AUC 0.7068   (+7.3 п.п.)

Фотографии сами по себе СЛАБЕЕ текста, ценность в независимости: корреляция рангов
двух источников всего 0.287. Прирост устойчив — 600 повторов с возвратом дали +7.26 п.п.
при разбросе 1.80 и положительном знаке в 100% случаев.

⚠ Почему именно РАНГИ, а не вероятности. Связка вероятностей даёт всего +0.1 п.п.
(0.6344 против 0.6334): шкалы линейного SVM и логистической модели на эмбеддингах
несопоставимы, и сложение их «в лоб» просто тонет в более уверенном источнике.
Ранг делает источники сравнимыми.

⚠⚠ Ранг по партии зависит от её состава, и это не теория: на смоуке из трёх товаров
редкой категории ранги равны 0.333/0.667/1.0, поэтому ВЕРХНИЙ товар проходит любой порог
независимо от своего скора. На 707 товарах публичной выборки это безобидно, но
конструкция хрупкая. Поэтому связка сделана ИНДУКТИВНОЙ: вместо ранга по партии берётся
квантиль относительно распределения, замеренного ВНЕ ФОЛДА на обучающих данных и
сохранённого в весах. Результат для одного товара больше не зависит от того, с кем он
приехал. Шкалы считаются ВНУТРИ КАТЕГОРИИ: в БАД и в редкой они разные.

⚠ Распределение для калибровки берётся именно ВНЕ ФОЛДА. Скоры линейной модели на её
собственных обучающих строках заметно увереннее, и калибровка по ним сместила бы все
тестовые товары вниз по шкале.

⚠⚠ Связка включается ТОЛЬКО в редкой категории, и это не осторожность, а расчёт.
Порог по рангу — это квантиль, то есть он фиксирует ДОЛЮ предсказанных позитивов.
В редкой категории доли почти совпадают (у нас 3.60%, на публике 3.39%), и квантиль
переносится. В БАД они расходятся вдвое сильнее любой другой величины в задаче —
74.49% против 57.33%, — и квантильный порог там переразметил бы выборку грубо.
Вдобавок фотографии в БАД не помогают вовсе: F1 связки 0.8922 против 0.8973 у текста.
Поэтому в БАД отдаётся обычная вероятность текстовой модели со своим порогом.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _quantile_of(x: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Доля значений `reference`, которые не больше каждого из `x`.

    Это и есть индуктивная замена ранга: шкала задаётся сохранённым распределением,
    а не тем, с кем товар приехал в одной партии.
    """
    x = np.asarray(x, dtype=np.float64)
    if reference is None or len(reference) == 0:
        return np.full(len(x), 0.5)
    left = np.searchsorted(reference, x, side="left")
    right = np.searchsorted(reference, x, side="right")
    return (left + right) / (2.0 * len(reference))


class RankBlendModel:
    """Текстовая модель плюс модель на эмбеддингах фотографий, связка по рангам."""

    SET_IMAGE = "image"

    def __init__(self, cfg: dict, image_weight: dict[str, float] | None = None):
        self.cfg = cfg
        # вес фотографий СВОЙ на категорию; ноль означает «остаться на тексте целиком»
        self.image_weight: dict[str, float] = dict(
            image_weight or {"Легковоспламеняющиеся": 0.2, "БАД": 0.0})
        self.text_model = None                       # обученная TextBaseline
        self.image_models: dict[str, object] = {}    # категория -> классификатор
        self.scalers: dict[str, object] = {}
        self.thresholds: dict[str, float] = {}
        # категория -> {"text": отсортированные скоры, "image": отсортированные скоры},
        # замеренные вне фолда. По ним считается квантиль на прогоне.
        self.calibration: dict[str, dict[str, np.ndarray]] = {}

    # ------------------------------------------------------------------ обучение
    def fit(self, df: pd.DataFrame, emb: np.ndarray, ocr=None,
            text_model=None, seed: int = 42,
            calibration: dict[str, dict[str, np.ndarray]] | None = None
            ) -> "RankBlendModel":
        """calibration — скоры источников ВНЕ ФОЛДА по категориям, задают шкалу.

        Если не передана, шкала снимается со скоров на обучающих строках. Так делать
        можно только для черновых замеров: эти скоры переобучены, и калибровка по ним
        смещает тестовые товары вниз по шкале.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        from .text_baseline import TextBaseline

        cat_col, tgt = self.cfg["data"]["category_col"], self.cfg["data"]["target_col"]
        if text_model is None:
            text_model = TextBaseline(self.cfg).fit(df, ocr)
        self.text_model = text_model

        emb = np.asarray(emb, dtype=np.float32)
        y = df[tgt].to_numpy()
        pos_of = {idx: k for k, idx in enumerate(df.index)}
        for cat, part in df.groupby(cat_col):
            idx = [pos_of[i] for i in part.index]
            sc = StandardScaler().fit(emb[idx])
            m = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced",
                                   random_state=seed)
            m.fit(sc.transform(emb[idx]), y[idx])
            self.image_models[str(cat)] = m
            self.scalers[str(cat)] = sc
            self.thresholds.setdefault(str(cat), 0.5)

        if calibration is not None:
            self.calibration = {k: {n: np.sort(np.asarray(v, dtype=np.float64))
                                    for n, v in d.items()}
                                for k, d in calibration.items()}
        else:
            txt = np.asarray(text_model.predict_proba(df, ocr), dtype=np.float64)
            for cat, part in df.groupby(cat_col):
                idx = [pos_of[i] for i in part.index]
                img = self.image_models[str(cat)].predict_proba(
                    self.scalers[str(cat)].transform(emb[idx]))[:, 1]
                self.calibration[str(cat)] = {"text": np.sort(txt[idx]),
                                              "image": np.sort(img)}
        return self

    def source_scores(self, df: pd.DataFrame, ocr=None, emb=None):
        """Сырые скоры обоих источников — нужны, чтобы собрать калибровку вне фолда."""
        cat_col = self.cfg["data"]["category_col"]
        emb = np.asarray(self._unpack(emb), dtype=np.float32)
        text = np.asarray(self.text_model.predict_proba(df, ocr), dtype=np.float64)
        image = np.full(len(df), np.nan)
        pos_of = {idx: k for k, idx in enumerate(df.index)}
        for cat, part in df.groupby(cat_col):
            idx = np.asarray([pos_of[i] for i in part.index])
            model, sc = self.image_models.get(str(cat)), self.scalers.get(str(cat))
            if model is not None:
                image[idx] = model.predict_proba(sc.transform(emb[idx]))[:, 1]
        return text, image

    # ---------------------------------------------------------------- применение
    def _unpack(self, emb):
        """Прогон отдаёт словарь наборов, когда их несколько; нам нужен только «image».

        ⚠ Падаем, а не берём что попало: веса обучены на эмбеддингах ФОТОГРАФИЙ, и
        молча подставить сюда набор текста значит выдать мусорный вердикт без единой
        ошибки в журнале. Так уже терялись отправки.
        """
        if isinstance(emb, dict):
            got = emb.get(self.SET_IMAGE)
            if got is None:
                raise RuntimeError(
                    f"связке нужен набор векторов «{self.SET_IMAGE}», пришли: {sorted(emb)}")
            return got
        if emb is None:
            raise RuntimeError("связке нужны эмбеддинги фотографий, не пришло ничего")
        return emb

    def predict_proba(self, df: pd.DataFrame, ocr=None, emb=None) -> np.ndarray:
        """Скор по каждой категории в своей шкале.

        ⚠ Шкала РАЗНАЯ и это намеренно: где вес фотографий нулевой — вероятность
        текстовой модели, где ненулевой — квантиль связки. Порог тоже свой на категорию,
        поэтому смешения не происходит. Причина — в шапке модуля.
        """
        cat_col = self.cfg["data"]["category_col"]
        emb = np.asarray(self._unpack(emb), dtype=np.float32)
        if len(emb) != len(df):
            raise RuntimeError(f"эмбеддингов {len(emb)}, а товаров {len(df)}")

        text_score = np.asarray(self.text_model.predict_proba(df, ocr), dtype=np.float64)
        out = np.zeros(len(df), dtype=np.float64)
        pos_of = {idx: k for k, idx in enumerate(df.index)}
        for cat, part in df.groupby(cat_col):
            idx = np.asarray([pos_of[i] for i in part.index])
            w = float(self.image_weight.get(str(cat), 0.0))
            model, sc = self.image_models.get(str(cat)), self.scalers.get(str(cat))
            if w <= 0 or model is None:  # текст целиком, в его собственной шкале
                out[idx] = text_score[idx]
                continue
            img = model.predict_proba(sc.transform(emb[idx]))[:, 1]
            ref = self.calibration.get(str(cat)) or {}
            # ⚠ Падаем, а не считаем без шкалы: без сохранённого распределения квантиль
            # выродится в одну и ту же величину для всех товаров, порог отработает как
            # «всем один вердикт», и решение молча отдаст мусор с валидным форматом.
            if not len(ref.get("text", ())) or not len(ref.get("image", ())):
                raise RuntimeError(
                    f"нет сохранённой шкалы для категории «{cat}» — веса собраны неверно")
            out[idx] = ((1 - w) * _quantile_of(text_score[idx], ref["text"])
                        + w * _quantile_of(img, ref["image"]))
        return out

    def predict(self, df: pd.DataFrame, ocr=None, emb=None) -> np.ndarray:
        cat_col = self.cfg["data"]["category_col"]
        score = self.predict_proba(df, ocr=ocr, emb=emb)
        thr = df[cat_col].astype(str).map(self.thresholds).fillna(0.5).to_numpy()
        return (score >= thr).astype(int)

    def save(self, path) -> None:
        import joblib

        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "RankBlendModel":
        import joblib

        return joblib.load(path)
