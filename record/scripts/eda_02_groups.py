"""EDA-2: группы почти-дубликатов и схема валидации.

Продавца в данных 2026 нет, а повторов много: без группировки случайный сплит течёт.
Здесь строим группы (точный текст, шинглы описаний, общие изображения), меряем
согласованность меток внутри групп (это оценка шума разметки и потолка качества) и
сохраняем разметку по фолдам для всех дальнейших экспериментов.

Артефакты: artifacts/splits/groups.parquet   Отчёт: artifacts/eda/02_groups.md
"""
import _bootstrap  # noqa: F401
import pandas as pd

from qc26.config import load_config, resolve_path
from qc26.groups import build_groups, group_purity
from qc26.report import Report
from qc26.splits import stratified_group_folds


def _image_hashes(cfg: dict, idc: str) -> dict[str, list[str]]:
    """Хэши фото по товарам из индекса EDA-1 (если он уже построен)."""
    p = resolve_path(cfg["paths"]["eda_dir"]) / "image_index.parquet"
    if not p.exists() or not cfg["groups"].get("use_image_hashes", True):
        return {}
    idx = pd.read_parquet(p)
    idx = idx[idx["ok"]] if "ok" in idx.columns else idx
    out: dict[str, list[str]] = {}
    for item_id, part in idx.groupby(idc):
        # md5 — точная копия, перцептивный — пережатая; префикс, чтобы не смешивать
        out[str(item_id)] = (["m:" + h for h in part["md5"].dropna().unique() if h]
                             + ["p:" + h for h in part["phash"].dropna().unique() if h])
    return out


def main() -> None:
    cfg = load_config("configs/data.yaml")
    from qc26.data import load_cards

    df = load_cards(cfg)
    idc, tgt, cat = (cfg["data"]["id_col"], cfg["data"]["target_col"],
                     cfg["data"]["category_col"])

    hashes = _image_hashes(cfg, idc)
    print(f"хэши изображений загружены для {len(hashes)} товаров", flush=True)

    rep = Report("EDA-2. Дубликаты, группы и валидация",
                 resolve_path(cfg["paths"]["eda_dir"]) / "02_groups.md")

    # последовательно накапливаем источники склейки, чтобы видеть вклад каждого
    variants = {
        "только точный текст (название+описание)": dict(image_hashes=None, shingle_k=10**9),
        "текст + шинглы описаний": dict(image_hashes=None,
                                        shingle_k=cfg["groups"]["shingle_k"]),
        "текст + шинглы + изображения": dict(image_hashes=hashes or None,
                                             shingle_k=cfg["groups"]["shingle_k"]),
    }
    summary = []
    groups = None
    for label, kw in variants.items():
        g = build_groups(df, name_col=cfg["data"]["name_col"], desc_col=cfg["data"]["desc_col"],
                         id_col=idc, **kw)
        tmp = df.assign(group=g)
        pur = group_purity(tmp, "group", tgt)
        summary.append({
            "источник склейки": label,
            "групп": pur["n_groups"],
            "товаров в группах >1": pur["n_items_in_multi"],
            "доля таких товаров, %": round(pur["n_items_in_multi"] / len(df) * 100, 1),
            "смешанных по метке групп": pur["n_mixed_groups"],
            "чистота групп, %": round(pur["purity"] * 100, 1),
        })
        groups = g

    rep.h("Сколько данных склеивается")
    rep.table(pd.DataFrame(summary), floatfmt="{:.1f}")
    rep.p("Чистота групп — оценка сверху на достижимое качество: внутри группы карточки "
          "неотличимы по тексту, значит расхождение меток там воспроизвести нечем.")

    df = df.assign(group=groups)

    rep.h("Чистота групп по категориям")
    rows = []
    for c, part in df.groupby(cat):
        pur = group_purity(part, "group", tgt)
        rows.append({"категория": c, "товаров": len(part), "групп": pur["n_groups"],
                     "смешанных групп": pur["n_mixed_groups"],
                     "товаров в смешанных": pur["n_items_in_mixed"],
                     "чистота, %": round(pur["purity"] * 100, 1)})
    rep.table(pd.DataFrame(rows), floatfmt="{:.1f}")

    rep.h("Редкий класс: сколько его на самом деле")
    rows = []
    for c, part in df.groupby(cat):
        pos = part[part[tgt] == 1]
        rows.append({
            "категория": c,
            "позитивов (label=1)": len(pos),
            "уникальных групп среди них": pos["group"].nunique(),
            "позитивов с дублем-собратом": int((pos["group"].map(
                df["group"].value_counts()) > 1).sum()),
        })
    rep.table(pd.DataFrame(rows), floatfmt="{:.0f}")
    rep.p("Эффективный размер редкого класса — число ГРУПП, а не строк. При случайном "
          "сплите копии позитивов попадают в обучение и в валидацию одновременно, и "
          "оценка получается завышенной; в блоке EDA-4 эта разница измерена напрямую.")

    # раскладка по фолдам для всех дальнейших экспериментов
    out = df[[idc, cat, tgt, "group"]].copy()
    for seed in cfg["split"]["seeds"]:
        out[f"fold_seed{seed}"] = stratified_group_folds(
            df, cfg["split"]["n_folds"], seed, "group", (cat, tgt))
    path = resolve_path(cfg["paths"]["groups_file"])
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)

    rep.h("Раскладка по фолдам (seed 42)")
    fold_col = f"fold_seed{cfg['split']['seeds'][0]}"
    pivot = out.pivot_table(index=fold_col, columns=cat, values=tgt,
                            aggfunc=["size", "sum"]).fillna(0).astype(int)
    pivot.columns = [f"{a}: {b}" for a, b in pivot.columns]
    rep.table(pivot.reset_index(), floatfmt="{:.0f}")
    rep.p(f"Файл со сплитами: `{path.relative_to(resolve_path('.'))}` "
          f"(группы неделимы, страта — пара категория+метка).")
    rep.save()
    print(f"группы и фолды: {path}")


if __name__ == "__main__":
    main()
