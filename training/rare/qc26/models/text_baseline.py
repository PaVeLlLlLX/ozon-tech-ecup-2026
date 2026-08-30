"""Текстовый бейзлайн: tf-idf + логрег, обученные ОТДЕЛЬНО по каждой категории.

Категории противоположны по балансу (74.5% против 3.6% позитивов) и опираются на
разные слова, поэтому общая модель на них — заведомо компромисс. Отдельные модели
заодно дают отдельные пороги, а порог здесь — узкое место метрики.

Не требует GPU и весов моделей: годится как запасной путь в контейнере и как нижняя
граница, с которой сравнивается VLM.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from ..rules import evidence_frame

_WS = re.compile(r"\s+")


def build_text(df: pd.DataFrame, cfg: dict, ocr: pd.Series | None = None) -> pd.Series:
    """Вход модели: название + описание (+ распознанный с фото текст)."""
    bc = cfg["baseline"]
    limit = int(bc["desc_max_chars"])
    name = df[cfg["data"]["name_col"]].astype(str)
    desc = df[cfg["data"]["desc_col"]].astype(str).str.slice(0, limit)
    text = name + " \n " + desc
    if bc.get("use_ocr", False) and ocr is not None:
        text = text + " \n НА ФОТО: " + ocr.reindex(df.index).fillna("").astype(str)
    return text.map(lambda s: _WS.sub(" ", s.lower().replace("ё", "е")).strip())


def _class_weight(value):
    """Вес классов в виде, который понимает sklearn.

    ⚠ Через YAML и JSON словарь весов приезжает со СТРОКОВЫМИ ключами («1» вместо 1), а
    sklearn сверяет их с метками и молча не находит ни одной: вес просто не применяется,
    ошибки при этом нет. Приводим ключи к целым.
    """
    if isinstance(value, dict):
        return {int(k): float(v) for k, v in value.items()}
    return value


def category_settings(cfg: dict, category: str | None) -> dict:
    """Настройки бейзлайна с поправкой на категорию.

    Категории — разные задачи: в БАД 7469 товаров и 74.5% позитивов, в редкой 5502
    товара и 198 позитивов. Одни и те же регуляризация, вес классов и параметры текста
    им заведомо не подходят, а до сих пор навязывались обеим. Блок
    `baseline.per_category.<категория>` перекрывает верхний уровень точечно.
    """
    bc = dict(cfg["baseline"])
    extra = (bc.get("per_category") or {}).get(str(category)) if category else None
    if not extra:
        return bc
    for k, v in extra.items():
        bc[k] = {**bc[k], **v} if isinstance(v, dict) and isinstance(bc.get(k), dict) else v
    return bc


class CategoryTextModel:
    """Одна tf-idf + логрег связка на одну категорию."""

    def __init__(self, cfg: dict, category: str | None = None):
        self.cfg = cfg
        self.category = category
        bc = category_settings(cfg, category)
        self.word = TfidfVectorizer(
            analyzer="word", ngram_range=tuple(bc["word"]["ngram_range"]),
            max_features=bc["word"]["max_features"], min_df=bc["word"]["min_df"],
            sublinear_tf=bc["word"]["sublinear_tf"],
        )
        self.char = None
        if bc["char"]["enabled"]:
            self.char = TfidfVectorizer(
                analyzer="char_wb", ngram_range=tuple(bc["char"]["ngram_range"]),
                max_features=bc["char"]["max_features"], min_df=bc["char"]["min_df"],
                sublinear_tf=bc["char"]["sublinear_tf"],
            )
        # Форма линейной модели — параметр, а не данность: в прогоне №3 линейный SVM
        # на тех же признаках дал 0.7785 против 0.7448 у логистической регрессии.
        kind = bc.get("classifier", "logreg")
        weight = _class_weight(bc["logreg"]["class_weight"])
        if kind == "linear_svm":
            from sklearn.svm import LinearSVC

            self.clf = LinearSVC(C=bc.get("svm", {}).get("C", 1.0), class_weight=weight)
        else:
            self.clf = LogisticRegression(
                C=bc["logreg"]["C"], max_iter=bc["logreg"]["max_iter"],
                class_weight=weight,
            )
        self.use_rules = bc.get("use_rule_features", True)
        self.threshold = 0.5

    def _features(self, texts: pd.Series, rules: pd.DataFrame | None, fit: bool):
        blocks = [self.word.fit_transform(texts) if fit else self.word.transform(texts)]
        if self.char is not None:
            blocks.append(self.char.fit_transform(texts) if fit else self.char.transform(texts))
        if self.use_rules and rules is not None:
            blocks.append(sparse.csr_matrix(rules.to_numpy(dtype=np.float32)))
        return sparse.hstack(blocks).tocsr()

    def fit(self, texts: pd.Series, rules: pd.DataFrame | None, y) -> "CategoryTextModel":
        X = self._features(texts, rules, fit=True)
        self.clf.fit(X, np.asarray(y))
        return self

    def predict_proba(self, texts: pd.Series, rules: pd.DataFrame | None) -> np.ndarray:
        X = self._features(texts, rules, fit=False)
        if hasattr(self.clf, "predict_proba"):
            return self.clf.predict_proba(X)[:, 1]
        # у линейного SVM вероятности нет — приводим решающую функцию сигмоидой:
        # для порядка и подбора порога важна монотонность, а не калибровка
        return 1.0 / (1.0 + np.exp(-self.clf.decision_function(X)))


class TextBaseline:
    """Набор пер-категорийных моделей + пороги. Умеет сохраняться в один файл."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.models: dict[str, CategoryTextModel] = {}
        self.thresholds: dict[str, float] = {}
        self.prior: dict[str, float] = {}

    def _rules(self, df: pd.DataFrame, ocr):
        """Признаки-улики плюс, если включено, ответы языковой модели на анкету.

        Ответы приходят отдельной таблицей и подмешиваются к уликам: линейная модель
        сохраняет свой откалиброванный порог, а знание о товаре приходит извне.
        """
        bc = self.cfg["baseline"]
        rules = evidence_frame(df, ocr) if bc.get("use_rule_features") else None
        if not bc.get("use_qa_features"):
            return rules
        from ..config import resolve_path

        # На прогоне ответы считаются заново и кладутся сюда: офлайн-файл привязан к
        # идентификаторам НАШИХ товаров, на закрытой выборке их нет.
        qa = getattr(self, "qa_runtime", None)
        if qa is None:
            path = resolve_path(bc.get("qa_answers", "artifacts/qa/qa_answers.parquet"))
            if not path.exists():
                # Молчаливая работа без заявленных признаков — это другая модель под тем
                # же именем. Лучше упасть здесь, чем разбираться по баллу на лидерборде.
                raise RuntimeError(f"признаки-ответы включены, но файла нет: {path}")
            qa = pd.read_parquet(path)
        idc = self.cfg["data"]["id_col"]
        qa = qa.set_index(qa[idc].astype(str))
        cols = [c for c in qa.columns if c.startswith("qa_")]
        aligned = qa.reindex(df[idc].astype(str))[cols].astype(np.float32)
        aligned.index = df.index
        aligned = aligned.fillna(0.5)  # нет ответа — «не знаю», середина шкалы
        return aligned if rules is None else pd.concat([rules, aligned], axis=1)

    def fit(self, df: pd.DataFrame, ocr: pd.Series | None = None) -> "TextBaseline":
        cat_col, tgt = self.cfg["data"]["category_col"], self.cfg["data"]["target_col"]
        texts = build_text(df, self.cfg, ocr)
        rules = self._rules(df, ocr)
        for cat, part in df.groupby(cat_col):
            m = CategoryTextModel(self.cfg, category=str(cat))
            m.fit(texts.loc[part.index], None if rules is None else rules.loc[part.index],
                  part[tgt].to_numpy())
            self.models[str(cat)] = m
            self.thresholds.setdefault(str(cat), 0.5)
            self.prior[str(cat)] = float(part[tgt].mean())
        return self

    def predict_proba(self, df: pd.DataFrame, ocr: pd.Series | None = None) -> np.ndarray:
        cat_col = self.cfg["data"]["category_col"]
        texts = build_text(df, self.cfg, ocr)
        rules = self._rules(df, ocr)
        out = np.zeros(len(df), dtype=np.float32)
        pos_of = {idx: k for k, idx in enumerate(df.index)}
        for cat, part in df.groupby(cat_col):
            model = self.models.get(str(cat))
            idx = [pos_of[i] for i in part.index]
            if model is None:  # незнакомая категория — отдаём средний приор по обучению
                out[idx] = float(np.mean(list(self.prior.values()) or [0.5]))
                continue
            out[idx] = model.predict_proba(
                texts.loc[part.index], None if rules is None else rules.loc[part.index])
        return out

    def predict(self, df: pd.DataFrame, ocr: pd.Series | None = None) -> np.ndarray:
        cat_col = self.cfg["data"]["category_col"]
        proba = self.predict_proba(df, ocr)
        thr = df[cat_col].astype(str).map(self.thresholds).fillna(0.5).to_numpy()
        return (proba >= thr).astype(int)

    def save(self, path) -> None:
        import joblib

        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "TextBaseline":
        import joblib

        return joblib.load(path)
