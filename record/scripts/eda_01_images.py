"""EDA-1: инвентаризация всех изображений + хэши для склейки дубликатов.

Считает по каждому файлу: размер, геометрию, md5 содержимого и перцептивный хэш.
md5 ловит точные копии, перцептивный — пережатые и слегка изменённые. Оба нужны:
одинаковые фото у разных товаров означают один и тот же товар от разных продавцов,
а такие пары обязаны попадать в один фолд валидации.

Артефакт: artifacts/eda/image_index.parquet   Отчёт: artifacts/eda/01_images.md
"""
import argparse
import hashlib

import _bootstrap  # noqa: F401
import pandas as pd
from PIL import Image
from tqdm import tqdm

from qc26.config import load_config, resolve_path
from qc26.data import image_paths, images_root, load_cards
from qc26.report import Report

PHASH_SIZE = 8


def _phash(img: Image.Image) -> str:
    """Перцептивный хэш через imagehash; при сбое — пустая строка."""
    import imagehash

    return str(imagehash.phash(img, hash_size=PHASH_SIZE))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="ограничить число товаров (для смоука)")
    args = ap.parse_args()

    cfg = load_config("configs/data.yaml")
    df = load_cards(cfg)
    if args.limit:
        df = df.head(args.limit)
    root = images_root(cfg)
    idc = cfg["data"]["id_col"]

    out_path = resolve_path(cfg["paths"]["eda_dir"]) / "image_index.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    missing = []
    for item_id in tqdm(df[idc].tolist(), desc="изображения", unit="товар"):
        paths = image_paths(root, item_id)
        if not paths:
            missing.append(item_id)
            continue
        for order, p in enumerate(paths):
            rec = {idc: item_id, "order": order, "file": p.name, "bytes": p.stat().st_size,
                   "width": 0, "height": 0, "format": "", "md5": "", "phash": "", "ok": False}
            try:
                raw = p.read_bytes()
                rec["md5"] = hashlib.md5(raw).hexdigest()
                with Image.open(p) as img:
                    rec["format"] = img.format or ""
                    rec["width"], rec["height"] = img.size
                    # draft ускоряет декодирование JPEG в разы — точный размер тут не нужен
                    img.draft("RGB", (256, 256))
                    rec["phash"] = _phash(img.convert("RGB"))
                    rec["ok"] = True
            except Exception as e:
                rec["format"] = f"ОШИБКА: {type(e).__name__}"
            rows.append(rec)

    idx = pd.DataFrame(rows)
    idx.to_parquet(out_path, index=False)

    rep = Report("EDA-1. Изображения: инвентаризация и хэши",
                 resolve_path(cfg["paths"]["eda_dir"]) / "01_images.md")
    rep.kv({
        "корень изображений": root,
        "товаров в выборке": len(df),
        "товаров без изображений": len(missing),
        "всего файлов": len(idx),
        "битых файлов": int((~idx["ok"]).sum()),
        "изображений на товар (среднее)": f"{len(idx) / max(1, len(df) - len(missing)):.2f}",
    })

    rep.h("Число изображений на товар")
    per_item = idx.groupby(idc).size().rename("n_images")
    cnt = per_item.value_counts().sort_index().rename("товаров").reset_index()
    cnt.columns = ["изображений", "товаров"]
    cnt["доля, %"] = (cnt["товаров"] / cnt["товаров"].sum() * 100).round(1)
    rep.table(cnt, floatfmt="{:.1f}")

    rep.h("Геометрия и вес")
    stats = idx.loc[idx["ok"], ["width", "height", "bytes"]].describe(
        percentiles=[0.05, 0.5, 0.95]).round(0).reset_index()
    rep.table(stats, floatfmt="{:.0f}")
    rep.p("Форматы: " + ", ".join(f"{k} — {v}" for k, v in
                                  idx["format"].value_counts().head(5).items()))

    rep.h("Дубликаты изображений")
    ok = idx[idx["ok"]]
    md5_groups = ok.groupby("md5")[idc].nunique()
    shared_md5 = md5_groups[md5_groups > 1]
    items_shared = ok[ok["md5"].isin(shared_md5.index)][idc].nunique()
    ph_groups = ok.groupby("phash")[idc].nunique()
    shared_ph = ph_groups[ph_groups > 1]
    items_shared_ph = ok[ok["phash"].isin(shared_ph.index)][idc].nunique()
    rep.kv({
        "точных копий файлов (групп md5 > 1 товара)": len(shared_md5),
        "товаров, делящих файл с другим товаром": f"{items_shared} "
                                                  f"({items_shared / max(1, len(df)) * 100:.1f}%)",
        "групп по перцептивному хэшу (> 1 товара)": len(shared_ph),
        "товаров с похожим фото у другого товара": f"{items_shared_ph} "
                                                   f"({items_shared_ph / max(1, len(df)) * 100:.1f}%)",
    })
    rep.p("Это прямая утечка при случайном сплите: один и тот же товар от разных продавцов "
          "попадёт и в обучение, и в валидацию. Группировка — в блоке EDA-2.")
    rep.save()
    print(f"индекс изображений: {out_path} ({len(idx)} файлов)")


if __name__ == "__main__":
    main()
