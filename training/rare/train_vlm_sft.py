"""Обучение LoRA-адаптера VLM: «Да»/«Нет» на вопрос о принадлежности категории.

Обучается на всех фолдах, кроме отложенного, — отложенный нужен для честной оценки и
подбора порога. Смоук: `--smoke` берёт 6 примеров и одну эпоху, чтобы проверить, что
цепочка процессор → батч → шаг оптимизатора работает, не тратя ночь.

    python scripts/train_vlm_sft.py --smoke
    python scripts/train_vlm_sft.py --name vlm2b_v1
"""
import argparse
import time

import _bootstrap  # noqa: F401
import pandas as pd

from qc26.config import load_config, resolve_path
from qc26.data import load_cards
from qc26.models.vlm import build_sft_examples, run_sft


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--override", action="append", default=[])
    ap.add_argument("--name", default="vlm_sft")
    ap.add_argument("--holdout-fold", type=int, default=0)
    ap.add_argument("--holdout-seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    # ⚠ Профиль-надстройка поверх configs/vlm_sft.yaml. Локальные запуски его не
    # передают и работают ровно как раньше; аренда подкладывает свой файл с bf16,
    # большим батчем и сохранениями по эпохам, не трогая локальный конфиг.
    ap.add_argument("--config", action="append", default=[],
                    help="дополнительный YAML поверх configs/vlm_sft.yaml")
    # ⚠ Вторая ступень на НЕВИДАННЫХ данных. Дообучение на тех же строках бессмысленно:
    # замер 23.08 дал потери около одной миллионной — модель их запомнила, градиента нет,
    # и прогон только встряхивает веса (F1 редкой 0.6957 -> 0.6774 без всякого обучения).
    ap.add_argument("--train-fold", type=int, default=None,
                    help="учить ТОЛЬКО на этом фолде (по умолчанию — на всех, кроме него)")
    ap.add_argument("--half", choices=["a", "b"], default=None,
                    help="половина фолда по второму групповому разбиению: a — 755 строк, "
                         "b — 1840. Учить на большей, меньшую оставить связке под шкалу")
    ap.add_argument("--resume", action="store_true",
                    help="продолжить с последней сохранённой точки, если она есть")
    args = ap.parse_args()

    overrides = list(args.override)
    if args.smoke:
        overrides += ["vlm.max_samples=6", "vlm.epochs=1", "vlm.grad_accum=2",
                      "vlm.save_strategy=no"]

    cfg = load_config("configs/data.yaml", "configs/vlm_sft.yaml", *args.config,
                      overrides=overrides)
    df = load_cards(cfg).reset_index(drop=True)
    idc = cfg["data"]["id_col"]

    groups_path = resolve_path(cfg["paths"]["groups_file"])
    fold_col = f"fold_seed{args.holdout_seed}"
    split_col = f"fold_seed{args.holdout_seed + 1}"      # второе групповое разбиение
    if groups_path.exists():
        g = pd.read_parquet(groups_path)
        g[idc] = g[idc].astype(str)
        cols = [idc, "group", fold_col] + ([split_col] if split_col in g.columns else [])
        df = df.merge(g[cols], on=idc, how="left")
        if args.train_fold is None:
            train_df = df[df[fold_col] != args.holdout_fold].reset_index(drop=True)
        else:
            # ⚠ Вторая ступень на НЕВИДАННЫХ данных. Обычное дообучение на тех же
            # строках бессмысленно: замер 23.08 показал потери около одной миллионной —
            # модель их запомнила, градиента нет, прогон только встряхивает веса.
            # Отложенный фолд — единственные строки, которых модель не видела.
            train_df = df[df[fold_col] == args.train_fold]
            if args.half:
                # Половина фолда остаётся нетронутой: связке нужна ШКАЛА скоров, а для
                # неё метки не нужны — достаточно скоров модели на строках, которых она
                # не касалась. Делим вторым ГРУППОВЫМ разбиением, поэтому почти-дубликаты
                # не расходятся между половинами.
                if split_col not in train_df.columns:
                    raise SystemExit(f"нет колонки {split_col} — половину не выделить")
                odd = train_df[split_col] % 2 == 1
                train_df = train_df[odd if args.half == "a" else ~odd]
            train_df = train_df.reset_index(drop=True)
            print(f"⚠ ВТОРАЯ СТУПЕНЬ: учимся ТОЛЬКО на фолде {args.train_fold}"
                  f"{f', половина {args.half}' if args.half else ''} — "
                  f"{len(train_df)} строк, которых модель не видела")
    else:
        print("⚠ файла со сплитами нет — учимся на всех данных (оценка будет нечестной)")
        train_df = df

    ocr = None
    if cfg["vlm"].get("use_ocr"):
        p = resolve_path(cfg["paths"]["ocr_index"])
        if p.exists():
            idx = pd.read_parquet(p)
            mapping = dict(zip(idx[idc].astype(str), idx["ocr_text"].astype(str)))
            ocr = train_df[idc].astype(str).map(mapping).fillna("")
        else:
            print("⚠ распознанного текста нет — обучаемся без него")

    examples = build_sft_examples(train_df, cfg, ocr)
    print(f"обучающих примеров: {len(examples)} "
          f"(позитивов {int(examples[cfg['data']['target_col']].sum())})", flush=True)
    print(examples.groupby([cfg["data"]["category_col"], "answer"]).size().to_string(),
          flush=True)
    print("--- пример промпта ---\n" + examples.iloc[0]["text"][:1200], flush=True)

    out_dir = resolve_path(cfg["paths"]["artifacts_dir"]) / "models" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    run_sft(cfg, examples, str(out_dir), resume=args.resume)
    print(f"адаптер сохранён: {out_dir} (обучение {(time.time() - t0) / 60:.1f} мин)")


if __name__ == "__main__":
    main()
