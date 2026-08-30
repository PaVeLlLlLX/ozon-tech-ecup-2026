"""Сборка связки линейной модели с дообученной VLM.

Веса и пороги подбираются на ОТЛОЖЕННОМ ФОЛДЕ — единственных строках, где скоры VLM
получены вне её обучения. Полный прогон VLM по нашим данным для этого не нужен и даже
вреден: 80% скоров вышли бы переобученными и перекосили бы квантильную шкалу.

Метрика считается под баланс публичной выборки — в этом виде прибор ошибся на 0.10 п.п.
на прошлой отправке.

Запуск: python scripts/train_vlm_blend.py --name vlm_blend
"""
from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from qc26.config import load_config, resolve_path
from qc26.data import load_cards
from qc26.models.text_baseline import TextBaseline
from qc26.models.vlm_blend import VlmBlendModel, _quantile_of

PUBLIC_RATE = {"БАД": 528 / 921, "Легковоспламеняющиеся": 24 / 707}
RARE = "Легковоспламеняющиеся"


def weighted_f1(y, p, w) -> float:
    tp = w[(y == 1) & (p == 1)].sum()
    fp = w[(y == 0) & (p == 1)].sum()
    fn = w[(y == 1) & (p == 0)].sum()
    d = 2 * tp + fp + fn
    return float(2 * tp / d) if d > 0 else 0.0


def prior_weights(y, target: float) -> np.ndarray:
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    w = np.ones(len(y), dtype=float)
    if n_pos and n_neg:
        w[y == 0] = (n_pos * (1 - target)) / (target * n_neg)
    return w


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="vlm_blend")
    ap.add_argument("--vlm-holdout", default="artifacts/preds/vlm_full_e3_holdout_holdout.parquet")
    ap.add_argument("--linear-oof", default="artifacts/preds/svm_c4_bal_oof.parquet")
    ap.add_argument("--ocr-index", default="artifacts/ocr/ocr_index_fast.2frames.parquet")
    args = ap.parse_args()

    cfg = load_config("configs/data.yaml", "configs/baseline.yaml", overrides=[
        "baseline.use_ocr=true", "baseline.classifier=linear_svm",
        "baseline.svm.C=4.0", "baseline.logreg.class_weight=balanced",
        f"paths.ocr_index={args.ocr_index}"])
    idc, tgt, cat_col = (cfg["data"]["id_col"], cfg["data"]["target_col"],
                         cfg["data"]["category_col"])

    v = pd.read_parquet(resolve_path(args.vlm_holdout))
    lin = pd.read_parquet(resolve_path(args.linear_oof))
    v[idc] = v[idc].astype(str)
    lin[idc] = lin[idc].astype(str)
    d = v.merge(lin[[idc, "score_seed42"]].rename(columns={"score_seed42": "lin"}), on=idc)
    print(f"отложенный фолд: {len(d)} товаров", flush=True)

    # --- веса и пороги на отложенном фолде, под баланс публики
    grid_w = np.round(np.arange(0.0, 1.01, 0.05), 2)
    grid_t = np.unique(np.round(np.linspace(0.005, 0.995, 199), 4))
    weights, thresholds, per_cat, calib = {}, {}, {}, {}
    for cat, g in d.groupby(cat_col):
        y = g[tgt].to_numpy()
        wpri = prior_weights(y, PUBLIC_RATE[str(cat)])
        ref_t, ref_v = np.sort(g["lin"].to_numpy()), np.sort(g["score"].to_numpy())
        calib[str(cat)] = {"text": ref_t, "vlm": ref_v}
        best = None
        for w in grid_w:
            s = ((1 - w) * _quantile_of(g["lin"].to_numpy(), ref_t)
                 + w * _quantile_of(g["score"].to_numpy(), ref_v))
            for t in grid_t:
                f1 = weighted_f1(y, (s >= t).astype(int), wpri)
                if best is None or f1 > best[0]:
                    best = (f1, float(w), float(t))
        per_cat[str(cat)], weights[str(cat)], thresholds[str(cat)] = best[0], best[1], best[2]
        print(f"  {cat}: вес VLM {best[1]:.2f}, порог {best[2]:.4f}, F1 {best[0]:.4f}",
              flush=True)
    metric = float(np.mean(list(per_cat.values())))
    print(f"метрика (под баланс публики, на отложенном фолде): {metric:.4f}", flush=True)

    # --- линейная ветка обучается на ВСЕХ данных: на прогоне она увидит чужие карточки
    df = load_cards(cfg)
    from eda_04_separability import load_ocr_aligned
    ocr = load_ocr_aligned(cfg, df)
    assert ocr is None or ocr.map(len).mean() > 20, "распознанный текст подозрительно короткий"
    print("обучаю линейную ветку на всех данных...", flush=True)
    text_model = TextBaseline(cfg).fit(df, ocr)

    model = VlmBlendModel(cfg, weights=weights)
    model.text_model = text_model
    model.thresholds = thresholds
    model.calibration = calib
    dst = resolve_path(f"artifacts/models/{args.name}.joblib")
    model.save(dst)
    resolve_path(f"artifacts/models/{args.name}.metrics.json").write_text(
        json.dumps({"metric_binary": metric, "f1_rare": per_cat.get(RARE, 0.0),
                    "f1_supplement": per_cat.get("БАД", 0.0),
                    "vlm_weight": weights, "thresholds": thresholds,
                    "как получено": "отложенный фолд, под баланс публичной выборки"},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"веса: {dst}", flush=True)


if __name__ == "__main__":
    main()
