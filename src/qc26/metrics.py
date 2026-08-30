"""Метрика соревнования: F1 считается по каждой категории отдельно и усредняется.

⚠ В условиях метрика названа «Macro Averaged F1», но пояснение и код baseline
организаторов (`f1_score(y, pred)` без `average`) говорят о ДВОИЧНОЙ F1 по классу 1
внутри категории. Формулировка допускает и второе прочтение — F1 macro по обоим
классам внутри категории. Считаем оба варианта и всегда показываем оба: они сильно
расходятся на почти вырожденной категории и по-разному двигают оптимальный порог.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def competition_score(y_true, y_pred, categories, average: str = "binary") -> dict:
    """F1 по каждой категории + среднее по категориям.

    average="binary" — F1 по классу 1 (как в baseline организаторов);
    average="macro"  — F1 macro по обоим классам внутри категории.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cats = pd.Series(np.asarray(categories))

    per_cat: dict[str, float] = {}
    for cat in sorted(cats.unique()):
        m = (cats == cat).values
        kw = {"average": "macro"} if average == "macro" else {"pos_label": 1}
        per_cat[str(cat)] = float(f1_score(y_true[m], y_pred[m], zero_division=0, **kw))
    return {"per_category": per_cat, "mean": float(np.mean(list(per_cat.values())))}


def ranking_scores(y_true, y_score, categories) -> dict:
    """AUC и PR-AUC по категориям — по ним отбираем конфигурации (устойчивее F1)."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    cats = pd.Series(np.asarray(categories))

    out: dict[str, float] = {}
    for cat in sorted(cats.unique()):
        m = (cats == cat).values
        yt, ys = y_true[m], y_score[m]
        if len(np.unique(yt)) < 2:
            out[f"auc::{cat}"] = float("nan")
            out[f"pr_auc::{cat}"] = float("nan")
            continue
        out[f"auc::{cat}"] = float(roc_auc_score(yt, ys))
        out[f"pr_auc::{cat}"] = float(average_precision_score(yt, ys))
    aucs = [v for k, v in out.items() if k.startswith("auc::")]
    prs = [v for k, v in out.items() if k.startswith("pr_auc::")]
    out["auc_mean"] = float(np.nanmean(aucs)) if aucs else float("nan")
    out["pr_auc_mean"] = float(np.nanmean(prs)) if prs else float("nan")
    return out


def evaluate_scores(y_true, y_score, categories, groups, seed: int = 42) -> dict:
    """Полная оценка непрерывного скора на отложенной выборке.

    Порог подбирается не на тех же данных, на которых считается итоговая F1: выборка
    делится пополам по ГРУППАМ (чтобы копии одного товара не оказались по разные
    стороны), на одной половине порог подбирается, на другой применяется, и наоборот.

    Вынесено в пакет, потому что этим пользуются и оценка модели, и пересборка журнала
    экспериментов: посчитай они по-разному, числа в отчётах и в журнале разойдутся.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    cats = np.asarray(categories)

    rng = np.random.default_rng(seed)
    uniq = pd.unique(np.asarray(groups))
    half = set(rng.choice(uniq, size=len(uniq) // 2, replace=False).tolist())
    side = pd.Series(groups).isin(half).to_numpy()

    out = {"ranking": ranking_scores(y_true, y_score, cats), "by_average": {}}
    for avg in ("binary", "macro"):
        pred = np.zeros(len(y_true), dtype=int)
        thr: dict[str, list[float]] = {}
        for c in pd.unique(cats):
            in_cat = cats == c
            for tune_side, apply_side in ((side, ~side), (~side, side)):
                tune, apply = in_cat & tune_side, in_cat & apply_side
                if tune.sum() == 0 or apply.sum() == 0 or len(np.unique(y_true[tune])) < 2:
                    continue
                t, _ = best_threshold(y_true[tune], y_score[tune], average=avg)
                pred[apply] = (y_score[apply] >= t).astype(int)
                thr.setdefault(str(c), []).append(t)
        sc = competition_score(y_true, pred, cats, average=avg)
        sc["thresholds"] = {k: float(np.mean(v)) for k, v in thr.items()}
        out["by_average"][avg] = sc
    return out


def best_threshold(y_true, y_score, average: str = "binary",
                   grid: np.ndarray | None = None) -> tuple[float, float]:
    """Порог, максимизирующий F1 внутри ОДНОЙ категории. Возвращает (порог, F1).

    Пороги подбираем на валидации, а не на тесте: иначе оценка завышена.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if grid is None:
        grid = np.unique(np.round(np.linspace(0.01, 0.99, 197), 4))
    kw = {"average": "macro"} if average == "macro" else {"pos_label": 1}
    best_f1, best_t = -1.0, 0.5
    for t in grid:
        f1 = f1_score(y_true, (y_score >= t).astype(int), zero_division=0, **kw)
        if f1 > best_f1:
            best_f1, best_t = float(f1), float(t)
    return best_t, best_f1
