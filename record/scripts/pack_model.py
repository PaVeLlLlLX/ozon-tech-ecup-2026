"""Упаковка весов модели в архив решения и проверка потери точности.

    python scripts/pack_model.py --src <каталог модели> --dst artifacts/models/<имя>_packed

Модель, которой нет в каталоге проверяющей системы, приходится везти в архиве. В bf16
Qwen3-VL-4B занимает 8.28 ГБ при лимите архива 5 ГБ; здесь веса пакуются в int8 с
масштабом на строку и разворачиваются обратно при загрузке.

Отчёт: artifacts/reports/pack_<имя>.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from qc26.config import resolve_path
from qc26.models.packing import pack_model


def _dir_size_gb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1024 ** 3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="каталог с safetensors и конфигами")
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()

    src, dst = Path(args.src), resolve_path(args.dst)
    if not src.is_dir():
        raise SystemExit(f"нет каталога {src}")
    before = _dir_size_gb(src)
    info = pack_model(src, dst)
    after = _dir_size_gb(dst)

    lines = [f"# Упаковка весов: {dst.name}", "",
             f"- источник: `{src}`", f"- результат: `{dst}`", "",
             "| величина | значение |", "|---|---|",
             f"| каталог до | {before:.2f} ГБ |",
             f"| каталог после | {after:.2f} ГБ |",
             f"| упаковано тензоров | {info['packed_tensors']} |",
             f"| доля от исходного размера | {info['ratio']:.3f} |",
             f"| средняя относительная ошибка веса | **{info['err_mean_rel']:.5f}** |",
             f"| наибольшая относительная ошибка | **{info['err_max_rel']:.5f}** |",
             "",
             "Ошибка считается по каждому упакованному тензору как средний модуль "
             "отклонения, делённый на средний модуль самого веса.", "",
             f"Лимит архива — 5 ГБ. Запас после упаковки: **{5.0 - after:.2f} ГБ** "
             "(в него должны уложиться адаптер, код и прочие файлы решения).", ""]
    rep = resolve_path(f"artifacts/reports/pack_{dst.name}.md")
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"отчёт: {rep}")


if __name__ == "__main__":
    main()
