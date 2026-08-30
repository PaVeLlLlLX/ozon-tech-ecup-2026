"""Обучение текстового бейзлайна и подбор порогов на групповых сплитах.

Пороги — отдельная сущность от модели: главный урок подготовки в том, что усиление
обучения поднимает ранжирование и роняет F1 при пороге. Поэтому порог подбираем
честно, вне фолда, усредняем по фолдам и по нескольким seed, и только потом
переобучаем модель на всех данных.

Артефакт: artifacts/models/<имя>.joblib (модель + пороги, читается из run.py)
"""
import argparse

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from qc26.config import load_config, resolve_path
from qc26.data import load_cards
from qc26.metrics import best_threshold, competition_score, ranking_scores
from qc26.models.text_baseline import TextBaseline
from qc26.report import Report
from qc26.tracking import M_AUC, M_BINARY, M_MACRO, M_PR_AUC, log_run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--override", action="append", default=[])
    ap.add_argument("--name", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--note", default=None, help="краткая суть прогона для журнала")
    args = ap.parse_args()

    cfg = load_config("configs/data.yaml", "configs/baseline.yaml", overrides=args.override)
    name = args.name or cfg["baseline"]["name"]
    df = load_cards(cfg).reset_index(drop=True)
    idc, cat, tgt = (cfg["data"]["id_col"], cfg["data"]["category_col"],
                     cfg["data"]["target_col"])
    seeds = args.seeds or cfg["split"]["seeds"]

    groups = pd.read_parquet(resolve_path(cfg["paths"]["groups_file"]))
    groups[idc] = groups[idc].astype(str)
    fold_cols = [c for c in groups.columns if c.startswith("fold_")]
    df = df.merge(groups[[idc, "group"] + fold_cols], on=idc, how="left")

    from eda_04_separability import cross_validate, load_ocr_aligned

    ocr = load_ocr_aligned(cfg, df) if cfg["baseline"].get("use_ocr") else None
    print(f"распознанный с фото текст: {'используется' if ocr is not None else 'нет'}",
          flush=True)

    # --- подбор порогов и честная оценка ---
    thr_acc: dict[str, list[float]] = {}
    rank_rows, score_rows = [], []
    oof_store: dict[int, np.ndarray] = {}
    for seed in seeds:
        fold_col = f"fold_seed{seed}"
        if fold_col not in df.columns:
            continue
        folds = df[fold_col].to_numpy()
        oof, rank = cross_validate(cfg, df, folds, ocr)
        rank_rows.append({"seed": seed, **rank})
        oof_store[seed] = oof

        y = df[tgt].to_numpy()
        pred = np.zeros(len(df), dtype=int)
        for c in df[cat].unique():
            in_cat = (df[cat] == c).to_numpy()
            for f in np.unique(folds):
                tune, apply = in_cat & (folds != f), in_cat & (folds == f)
                if tune.sum() == 0 or apply.sum() == 0:
                    continue
                t, _ = best_threshold(y[tune], oof[tune], average=cfg["metric"]["average"])
                pred[apply] = (oof[apply] >= t).astype(int)
                thr_acc.setdefault(str(c), []).append(t)
        for avg in ("binary", "macro"):
            sc = competition_score(y, pred, df[cat].to_numpy(), average=avg)
            score_rows.append({"seed": seed, "average": avg, **sc["per_category"],
                               "метрика": sc["mean"]})
        print(f"seed {seed}: AUC {rank['auc_mean']:.4f} | "
              f"метрика {score_rows[-2]['метрика']:.4f}", flush=True)

    thresholds = {c: float(np.median(v)) for c, v in thr_acc.items()}
    rank_df, score_df = pd.DataFrame(rank_rows), pd.DataFrame(score_rows)

    # --- финальная модель на всех данных ---
    model = TextBaseline(cfg).fit(df, ocr)
    model.thresholds = thresholds
    out_dir = resolve_path(cfg["paths"]["artifacts_dir"]) / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.joblib"
    model.save(path)

    # Предсказания вне обучения сохраняем: по ним можно честно оценить ЛЮБОЙ порог,
    # не переобучая модель, — например когда порог подменяется в конфиге отправки.
    preds_dir = resolve_path(cfg["paths"]["artifacts_dir"]) / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)
    oof_df = df[[idc, cat, tgt, "group"]].copy()
    for seed, arr in oof_store.items():
        oof_df[f"score_seed{seed}"] = arr
    oof_df.to_parquet(preds_dir / f"{name}_oof.parquet", index=False)

    # Честная оценка рядом с артефактом: сборщик архива берёт её в журнал отправок,
    # чтобы локальные числа и публичный результат можно было сопоставлять.
    import json

    (out_dir / f"{name}.metrics.json").write_text(json.dumps({
        "name": name,
        "use_ocr": bool(ocr is not None),
        "seeds": seeds,
        "thresholds": thresholds,
        "metric_binary": float(score_df[score_df["average"] == "binary"]["метрика"].mean()),
        "metric_macro": float(score_df[score_df["average"] == "macro"]["метрика"].mean()),
        "auc_mean": float(rank_df["auc_mean"].mean()),
        "pr_auc_mean": float(rank_df["pr_auc_mean"].mean()),
        "validation": "групповой сплит, порог вне оцениваемого фолда",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    rep = Report(f"Бейзлайн «{name}»: обучение и пороги",
                 resolve_path(cfg["paths"]["eda_dir"]).parent / "reports" / f"{name}.md")
    rep.kv({"модель": cfg["baseline"]["name"],
            "распознанный с фото текст": bool(ocr is not None),
            "улики по правилам как признаки": cfg["baseline"].get("use_rule_features"),
            "seed": seeds, "пороги": thresholds, "артефакт": path})
    rep.h("Ранжирование по фолдам")
    rep.table(rank_df.round(4), floatfmt="{:.4f}")
    rep.h("Метрика соревнования (порог подобран вне оцениваемого фолда)")
    rep.table(score_df.round(4), floatfmt="{:.4f}")
    if len(seeds) > 1:
        sp = score_df[score_df["average"] == cfg["metric"]["average"]]["метрика"]
        rep.p(f"Разброс по seed: {sp.min():.4f}…{sp.max():.4f} "
              f"(размах {(sp.max() - sp.min()) * 100:.2f} п.п.) — порог значимости для "
              "сравнения с другими конфигурациями.")
    rep.save()

    main_avg = cfg["metric"]["average"]
    metric_mean = float(score_df[score_df["average"] == main_avg]["метрика"].mean())
    by_avg = {a: float(score_df[score_df["average"] == a]["метрика"].mean())
              for a in ("binary", "macro")}
    note = args.note or (
        f"tf-idf+логрег по категориям, {'с распознанным текстом с фото' if ocr is not None else 'только текст карточки'}, "
        f"порог под {main_avg}; групповой сплит, {len(seeds)} seed")
    log_run(cfg, f"baseline_{name}",
            params={"model": cfg["baseline"]["name"], "use_ocr": bool(ocr is not None),
                    "use_rule_features": cfg["baseline"].get("use_rule_features"),
                    "seeds": seeds, "thresholds": thresholds},
            metrics={M_BINARY: by_avg["binary"], M_MACRO: by_avg["macro"],
                     M_AUC: float(rank_df["auc_mean"].mean()),
                     M_PR_AUC: float(rank_df["pr_auc_mean"].mean())},
            tags={"stage": "baseline", "split": "grouped"}, note=note)
    print(f"модель сохранена: {path}")
    print(f"метрика (average={main_avg}, групповой сплит): {metric_mean:.4f}")


if __name__ == "__main__":
    main()
