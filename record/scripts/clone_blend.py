"""Клон готовой связки с изменёнными порогами и весами — без переобучения.

Зачем отдельный скрипт. Диагностические и пробные отправки отличаются от боевой ровно
одним числом внутри весов: порогом категории или весом источника. Раньше такие клоны
делались разовыми вставками в командной строке, и в журнале оставалось только словесное
«клон связки-рекордсмена» — восстановить, что именно поменяли, было нельзя. Здесь
изменение записывается в поле `как получено` рядом с весами.

⚠ Пороги связки лежат ВНУТРИ файла весов, а не в конфиге отправки: `run.py` для ветки
`vlmblend:` берёт `loaded.thresholds` и ключ `thresholds` из описания варианта не читает.
Поэтому диагностику «в редкой категории всем бан» нельзя собрать одной строкой в
описании варианта — нужен именно клон весов.

Примеры:

    # диагностика: в редкой категории всем «бан», публичный балл = F1 по БАД / 2
    python scripts/clone_blend.py --src artifacts/models/vlm_blend_big.joblib \
        --dst artifacts/models/vlm_blend_big_supplement_only.joblib \
        --threshold Легковоспламеняющиеся=1.01 \
        --why "диагностика: вклад редкой категории обнулён"

    # смягчённый порог редкой категории
    python scripts/clone_blend.py --src artifacts/models/vlm_blend_big.joblib \
        --dst artifacts/models/vlm_blend_big_recall.joblib \
        --threshold Легковоспламеняющиеся=0.96
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import joblib

from qc26.config import resolve_path


def _pairs(items: list[str], what: str) -> dict[str, float]:
    """Разбирает «Категория=число» с проверкой — опечатка в названии молчать не должна."""
    out: dict[str, float] = {}
    for raw in items or []:
        key, sep, value = raw.partition("=")
        if not sep:
            raise SystemExit(f"{what}: ожидалось «Категория=число», получено «{raw}»")
        try:
            out[key.strip()] = float(value)
        except ValueError:
            raise SystemExit(f"{what}: «{value}» не число (в «{raw}»)") from None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="файл весов готовой связки")
    ap.add_argument("--dst", required=True, help="куда положить клон")
    ap.add_argument("--threshold", action="append", default=[],
                    help="Категория=порог, можно несколько раз")
    ap.add_argument("--vlm-weight", action="append", default=[],
                    help="Категория=вес языковой модели, можно несколько раз")
    ap.add_argument("--image-weight", action="append", default=[],
                    help="Категория=вес векторов фотографий, можно несколько раз")
    # ⚠ Доля выборки вместо порога: «называть не бан верхние K процентов этой
    # категории». Убирает ошибку переноса квантильной шкалы на закрытые данные.
    ap.add_argument("--top-fraction", action="append", default=[],
                    help="Категория=доля (0..1); задаёт долю ВМЕСТО порога")
    ap.add_argument("--why", default="", help="строка в поле «как получено» рядом с весами")
    args = ap.parse_args()

    src, dst = resolve_path(args.src), resolve_path(args.dst)
    if not src.exists():
        raise SystemExit(f"нет файла весов: {src}")
    model = joblib.load(src)

    thresholds = _pairs(args.threshold, "--threshold")
    top_fraction = _pairs(args.top_fraction, "--top-fraction")
    vlm_weights = _pairs(args.vlm_weight, "--vlm-weight")
    image_weights = _pairs(args.image_weight, "--image-weight")
    if not (thresholds or vlm_weights or image_weights or top_fraction):
        raise SystemExit("нечего менять: задайте хотя бы один порог или вес")

    # ⚠ Категории проверяем по уже сохранённым ключам: молча добавленный ключ с
    # опечаткой не применился бы, а связка продолжила бы работать со старым числом.
    known = set(map(str, getattr(model, "thresholds", {}) or {}))
    for cat in (list(thresholds) + list(vlm_weights) + list(image_weights)
                + list(top_fraction)):
        if known and cat not in known:
            raise SystemExit(f"категории «{cat}» нет в весах; известны: {sorted(known)}")

    before = {"top_fraction": dict(getattr(model, "top_fraction", {}) or {}),
              "thresholds": dict(getattr(model, "thresholds", {}) or {}),
              "vlm_weight": dict(getattr(model, "vlm_weight", {}) or {}),
              "image_weight": dict(getattr(model, "image_weight", {}) or {})}

    model.thresholds = {**before["thresholds"], **thresholds}
    if vlm_weights:
        model.vlm_weight = {**before["vlm_weight"], **vlm_weights}
    if image_weights:
        model.image_weight = {**before["image_weight"], **image_weights}
    if top_fraction:
        model.top_fraction = {**before["top_fraction"], **top_fraction}

    dst.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, dst)

    changes = {k: {"было": before[k], "стало": v} for k, v in (
        ("thresholds", model.thresholds),
        ("vlm_weight", getattr(model, "vlm_weight", {})),
        ("image_weight", getattr(model, "image_weight", {})),
        ("top_fraction", getattr(model, "top_fraction", {}))) if before[k] != v}
    meta = {"источник": str(Path(args.src).as_posix()),
            "как получено": args.why or f"клон {Path(args.src).stem} с правкой весов",
            "изменено": changes,
            "thresholds": model.thresholds,
            "vlm_weight": getattr(model, "vlm_weight", {}),
            "image_weight": getattr(model, "image_weight", {})}
    dst.with_suffix(".metrics.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"клон сохранён: {dst}")
    for key, diff in changes.items():
        print(f"  {key}: {diff['было']} -> {diff['стало']}")


if __name__ == "__main__":
    main()
