"""Состоятелен ли перенос порога редкой половины в код второй половины.

    python check_calib.py            # готовит выборку (нужен pyarrow, значит вне образа)
    python check_calib.py --score    # считает оценки и сверяет (запускать В ОБРАЗЕ)

⚠⚠ Что проверяется. Редкая половина ставил «не бан» в редкой категории по
АБСОЛЮТНОМУ срезу: квантиль оценки относительно эталона из 1101 значения выше 0.9699.
Такой срез переносится только вместе со всей машинкой, которая эти оценки считала, а она
отличается: набор написаний ответа (починено), сторона выравнивания пачки (невоспроизво-
димо в принципе — там оценка зависит от соседей по пачке из тридцати двух) и один тензор
зрительной части, у нас хранящийся точнее.

⭐ Эталон построен ровно на фолде 0 редкой категории — 1101 значение, столько же, сколько
там товаров. Значит наши оценки на том же фолде должны лечь на то же распределение. Это
и есть прямая проверка: сравниваем не догадки, а два распределения.

Что смотрим:
    1. долю, которую отбирает порог — в редкой половине это около 3%;
    2. насколько наше распределение совпало с эталонным по квантилям;
    3. F1 по разметке при отборе порогом — фолд 0 для этого адаптера отложен.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BUILD = HERE / "build"
SAMPLE = HERE / "calib_fold0.csv"
CAL = BUILD / "artifacts/models/rare2_calibration.json"
ADAPTER = BUILD / "artifacts/sft_vlm/vlm_rare2_a"
BASE = BUILD / "artifacts/models/qwen3vl4b_int8"


def prepare(repo: Path) -> None:
    df = pd.read_csv(repo / "data" / "data.csv")
    df = df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed")])
    df["id"] = df["id"].astype(str)
    g = pd.read_parquet(repo / "artifacts" / "splits" / "groups.parquet")
    g["id"] = g["id"].astype(str)
    df = df.merge(g[["id", "fold_seed42"]], on="id", how="left")
    part = df[(df["category"] != "БАД") & (df["fold_seed42"] == 0)].reset_index(drop=True)
    part.to_csv(SAMPLE, index=False, encoding="utf-8")
    print(f"выборка готова: {len(part)} товаров редкой категории с фолда 0, "
          f"позитивов {int(part.label.sum())} -> {SAMPLE}")


def score() -> None:
    sys.path.insert(0, str(BUILD / "src"))
    from qc26.config import load_config
    from qc26.data import images_root
    from qc26.inference.vlm import score_frame

    cfg = load_config(str(BUILD / "configs" / "data.yaml"),
                      str(BUILD / "configs" / "baseline.yaml"),
                      str(BUILD / "configs" / "submission.yaml"))
    df = pd.read_csv(SAMPLE)
    df["id"] = df["id"].astype(str)
    cfg["_test_path"] = str(SAMPLE)
    cfg["paths"]["data_csv"] = str(SAMPLE)
    root = images_root(cfg, str(SAMPLE))
    print(f"считаю {len(df)} товаров адаптером vlm_rare2_a", flush=True)
    z = score_frame(df, cfg, root, ADAPTER, batch=32, base_override=str(BASE))
    if z is None:
        raise SystemExit("оценки не получены")
    z = np.asarray(z, dtype=np.float32)
    prob = (1.0 / (1.0 + np.exp(-z))).astype(np.float64)
    out = df[["id", "label"]].copy()
    out["z"] = z
    out["prob"] = prob
    out.to_csv(HERE / "calib_scored.csv", index=False, encoding="utf-8")

    cal = json.loads(CAL.read_text(encoding="utf-8"))
    ref = np.asarray(cal["эталон"], dtype=np.float64)
    thr = float(cal["порог"])
    left = np.searchsorted(ref, prob, side="left")
    right = np.searchsorted(ref, prob, side="right")
    q = (left + right) / (2.0 * len(ref))
    sel = q >= thr
    y = df["label"].to_numpy().astype(int)
    tp = int((sel & (y == 1)).sum())
    f1 = 2 * tp / (int(sel.sum()) + int((y == 1).sum())) if tp else 0.0

    print()
    print(f"=== 1. Сколько отбирает порог {thr} ===")
    print(f"   отобрано {int(sel.sum())} из {len(df)} = {sel.mean() * 100:.2f}%")
    print(f"   в редкой половине на публичной выборке было 22 из 707 = 3.11%")
    print()
    print(f"=== 2. Насколько наше распределение совпало с эталонным ===")
    print(f"{'квантиль':>10}{'эталон':>14}{'наше':>14}")
    for p in (0.50, 0.90, 0.95, 0.97, 0.9699, 0.99):
        print(f"{p:>10.4f}{np.quantile(ref, p):>14.6f}{np.quantile(prob, p):>14.6f}")
    print(f"   доля нулей: эталон {float((ref < 1e-9).mean()):.4f}, "
          f"наше {float((prob < 1e-9).mean()):.4f}")
    print()
    print(f"=== 3. Качество на отложенном фолде ===")
    print(f"   позитивов {int(y.sum())}, отобрано {int(sel.sum())}, верных {tp}, "
          f"F1 {f1:.4f}")
    # для сравнения — что дала бы доля 3.11% вместо порога
    k = max(1, int(round(0.0311 * len(df))))
    idx = np.argsort(-prob, kind="stable")[:k]
    tp_r = int(y[idx].sum())
    print(f"   для сравнения, доля 3.11%: отобрано {k}, верных {tp_r}, "
          f"F1 {2 * tp_r / (k + int(y.sum())):.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--repo", default=str(HERE.parent.parent))
    a = ap.parse_args()
    if a.score:
        score()
    else:
        prepare(Path(a.repo))


if __name__ == "__main__":
    sys.exit(main())
