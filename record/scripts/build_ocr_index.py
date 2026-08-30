"""Распознавание текста на изображениях товаров.

Зачем: 725 разрешённых БАДов не содержат в тексте вообще никакой маркировки —
надпись «биологически активная добавка» напечатана на упаковке. Такие карточки по
тексту неотличимы от нарушителей, и это прямой замер пользы от зрения.

Индекс дописывается: повторный запуск досчитывает только недостающие товары, поэтому
прогон переживает перезапуск и его можно ставить на ночь.

⚠ cv2 внутри easyocr не открывает пути с кириллицей — подаём массив из PIL, а не путь.

Артефакт: artifacts/ocr/ocr_index.parquet (id, ocr_text, ocr_per_image)
"""
import argparse
import time

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from tqdm import tqdm

from qc26.config import load_config, resolve_path
from qc26.data import image_paths, images_root, load_cards

NO_TEXT = "(текст не распознан)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="ограничить число товаров (смоук)")
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--profile", default="ocr",
                    help="блок настроек: ocr (полный) либо ocr_shippable (в лимит контейнера)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config("configs/data.yaml", "configs/baseline.yaml", "configs/zoo.yaml")
    oc = cfg[args.profile]
    df = load_cards(cfg)
    if args.limit:
        df = df.head(args.limit)
    idc = cfg["data"]["id_col"]
    root = images_root(cfg)

    out = resolve_path(args.out or oc.get("out_path") or cfg["paths"]["ocr_index"])
    print(f"профиль «{args.profile}»: {oc['max_images']} кадра, сторона {oc['max_side']}",
          flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)

    done: dict[str, dict] = {}
    if out.exists():
        prev = pd.read_parquet(out)
        done = {str(r[idc]): dict(r) for _, r in prev.iterrows()}
    todo = [i for i in df[idc].astype(str) if i not in done]
    print(f"товаров всего {len(df)}, уже распознано {len(done)}, осталось {len(todo)}",
          flush=True)
    if not todo:
        print("всё готово")
        return

    import easyocr

    reader = easyocr.Reader(oc["languages"], gpu=bool(args.gpu), verbose=False)

    from qc26.data import load_image

    rows: list[dict] = []
    t0 = time.time()
    for k, item_id in enumerate(tqdm(todo, desc="распознавание", unit="товар")):
        paths = image_paths(root, item_id)[: int(oc["max_images"])]
        per_image: list[str] = []
        for p in paths:
            try:
                arr = np.array(load_image(p, max_side=int(oc["max_side"])))
                parts = reader.readtext(arr, detail=0, paragraph=True)
                text = " ".join(str(x).strip() for x in parts if str(x).strip())
                per_image.append(text[: int(oc["max_chars_per_image"])])
            except Exception:
                per_image.append("")
        joined = " | ".join(t for t in per_image if t)
        rows.append({idc: item_id, "ocr_text": joined or NO_TEXT,
                     "ocr_per_image": " ||| ".join(per_image), "n_images": len(paths)})
        if (k + 1) % int(oc["batch_flush"]) == 0:
            pd.DataFrame(list(done.values()) + rows).to_parquet(out, index=False)
            speed = (k + 1) / max(1e-9, time.time() - t0)
            print(f"  сброс на диск: {len(done) + len(rows)} товаров, "
                  f"{speed:.1f} товар/с, осталось ~{(len(todo) - k - 1) / speed / 60:.0f} мин",
                  flush=True)

    res = pd.DataFrame(list(done.values()) + rows).drop_duplicates(subset=[idc])
    res.to_parquet(out, index=False)
    empty = int((res["ocr_text"] == NO_TEXT).sum())
    print(f"готово: {len(res)} товаров, без текста {empty} ({empty / len(res) * 100:.0f}%) → {out}")


if __name__ == "__main__":
    main()
