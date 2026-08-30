"""Оценка VLM на отложенной выборке: ранжирование, порог, метрика соревнования.

Порог подбирается не на тех же данных, на которых считается итоговая F1: отложенную
выборку делим пополам по группам, на одной половине подбираем, на другой меряем, и
наоборот. Иначе результат систематически завышен, а порог у нас — узкое место.

Артефакты: artifacts/preds/<имя>_holdout.parquet  Отчёт: artifacts/reports/<имя>.md
"""
import argparse

import _bootstrap  # noqa: F401

import pandas as pd

from qc26.config import load_config, resolve_path
from qc26.data import load_cards
from qc26.metrics import evaluate_scores
from qc26.report import Report
from qc26.tracking import (M_AUC, M_BINARY, M_MACRO, M_PR_AUC, f1_metric_name,
                           log_run)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--override", action="append", default=[])
    ap.add_argument("--model", required=True, help="папка LoRA-адаптера или id базовой модели")
    ap.add_argument("--name", default=None)
    ap.add_argument("--holdout-fold", type=int, default=0)
    ap.add_argument("--holdout-seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--note", default=None, help="краткая суть эксперимента для журнала")
    # ⚠ Тот же профиль, что и при обучении. Оценивать модель на входе, отличном от
    # обучающего (другое разрешение, другое число кадров), — это мерить не то, что
    # поедет: ровно так уже терялись отправки.
    ap.add_argument("--config", action="append", default=[],
                    help="дополнительный YAML поверх configs/vlm_sft.yaml")
    # ⚠ Половина отложенного фолда. Нужна, когда модель прошла вторую ступень на другой
    # его половине: связке от этих строк требуются только СКОРЫ (шкала квантилей), а
    # метки для шкалы не нужны — поэтому даже небольшой нетронутый кусок годится.
    ap.add_argument("--half", choices=["a", "b"], default=None,
                    help="оценивать только половину фолда (a — обучающая, b — резерв)")
    args = ap.parse_args()

    cfg = load_config("configs/data.yaml", "configs/vlm_sft.yaml", *args.config,
                      overrides=args.override)
    name = args.name or f"eval_{args.model.replace('/', '_').split(chr(92))[-1]}"
    df = load_cards(cfg).reset_index(drop=True)
    idc, cat, tgt = (cfg["data"]["id_col"], cfg["data"]["category_col"],
                     cfg["data"]["target_col"])

    g = pd.read_parquet(resolve_path(cfg["paths"]["groups_file"]))
    g[idc] = g[idc].astype(str)
    fold_col = f"fold_seed{args.holdout_seed}"
    df = df.merge(g[[idc, "group", fold_col]], on=idc, how="left")
    ho = df[df[fold_col] == args.holdout_fold]
    if args.half:
        split_col = f"fold_seed{args.holdout_seed + 1}"
        if split_col not in g.columns:
            raise SystemExit(f"нет колонки {split_col} — половину не выделить")
        ho = ho.merge(g[[idc, split_col]], on=idc, how="left")
        ho = ho[(ho[split_col] % 2 == 1) if args.half == "a" else (ho[split_col] % 2 == 0)]
        print(f"половина «{args.half}» отложенного фолда: {len(ho)} товаров")
    ho = ho.reset_index(drop=True)
    if args.limit:
        ho = ho.head(args.limit).reset_index(drop=True)
    print(f"отложенная выборка: {len(ho)} товаров "
          f"(позитивов {int(ho[tgt].sum())})", flush=True)

    ocr = None
    if cfg["vlm"].get("use_ocr"):
        p = resolve_path(cfg["paths"]["ocr_index"])
        if p.exists():
            idx = pd.read_parquet(p)
            mapping = dict(zip(idx[idc].astype(str), idx["ocr_text"].astype(str)))
            ocr = ho[idc].astype(str).map(mapping).fillna("")

    from qc26.models.vlm import VlmScorer, make_texts

    texts = make_texts(ho, cfg, ocr)
    scorer = VlmScorer(cfg, model_id=args.model)
    scores = scorer.score(ho, texts)

    preds_dir = resolve_path(cfg["paths"]["artifacts_dir"]) / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)
    out = ho[[idc, cat, tgt, "group"]].copy()
    out["score"] = scores
    out.to_parquet(preds_dir / f"{name}_holdout.parquet", index=False)

    evaluated = evaluate_scores(ho[tgt].to_numpy(), scores, ho[cat].to_numpy(),
                                ho["group"].to_numpy(), seed=args.holdout_seed)
    rank, results = evaluated["ranking"], evaluated["by_average"]

    rep = Report(f"Оценка VLM «{name}»",
                 resolve_path(cfg["paths"]["artifacts_dir"]) / "reports" / f"{name}.md")
    rep.kv({"модель": args.model, "товаров в отложенной выборке": len(ho),
            "позитивов": int(ho[tgt].sum()), "кадров на товар": cfg["vlm"]["max_images"],
            "распознанный текст": bool(ocr is not None)})
    rep.h("Ранжирование")
    rep.table(pd.DataFrame([rank]).round(4), floatfmt="{:.4f}")
    rep.h("Метрика соревнования (порог подобран на другой половине выборки)")
    rows = [{"average": a, **s["per_category"], "метрика": s["mean"],
             "пороги": str(s["thresholds"])} for a, s in results.items()]
    rep.table(pd.DataFrame(rows), floatfmt="{:.4f}")
    rep.p("Сравнивать с текстовым бейзлайном из `artifacts/reports/tfidf_logreg.md` — "
          "и только по одной и той же отложенной выборке.")
    rep.save()

    main_avg = cfg["metric"]["average"]
    note = args.note or (
        f"VLM {cfg['vlm']['model_id'].split('/')[-1]}: {cfg['vlm']['max_images']} кадра, "
        f"max_pixels {cfg['vlm']['max_pixels']}"
        f"{', распознанный текст в промпте' if ocr is not None else ''}; "
        f"отложенный фолд {args.holdout_fold}")
    log_run(cfg, name,
            params={"model": args.model, "max_images": cfg["vlm"]["max_images"],
                    "max_pixels": cfg["vlm"]["max_pixels"],
                    "use_ocr": bool(ocr is not None)},
            metrics={M_BINARY: results["binary"]["mean"], M_MACRO: results["macro"]["mean"],
                     M_AUC: rank["auc_mean"], M_PR_AUC: rank["pr_auc_mean"],
                     **{f1_metric_name(k, "binary"): v
                        for k, v in results["binary"]["per_category"].items()}},
            tags={"stage": "vlm", "split": "holdout"}, note=note)
    print(f"AUC {rank['auc_mean']:.4f} | метрика ({main_avg}) "
          f"{results[main_avg]['mean']:.4f}")


if __name__ == "__main__":
    main()
