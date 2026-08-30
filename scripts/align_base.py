"""Приводит базу к весам редкой половины: сжимает единственный расходящийся тензор.

    python scripts/align_base.py --check     # только показать, ничего не менять
    python scripts/align_base.py --apply     # переписать веса

⚠⚠ Зачем. Порог 0.9699 в редкой категории — срез АБСОЛЮТНЫЙ, откалиброванный на той
стопке, которой считала редкая половина. Обе упаковки базы сжимают одинаково — int8 с масштабом на
строку, — и списки сжатых тензоров совпадают на 357 из 358. Расходится ровно один:

    model.visual.pos_embed.weight   в редкой половине СЖАТ, у нас хранится точно

«Точнее» здесь не значит «правильнее»: порог калиброван на сжатом варианте. Разложение
публичных баллов показывает цену расхождения — в редкой половине в верхних 22 местах 21 верный
товар, у нас в верхних 24 только 20, то есть один подходящий провалился ниже отбора.

Правка повторяет сжатие редкой половины и кладёт РАЗВЁРНУТЫЙ результат обратно в bf16: ровно то
число, которое получает загрузчик редкой половины, разворачивая свой int8. Переобучения не
требуется, размер архива не меняется.

⚠ Формула сжатия взята из кода редкой половины (models/packing.py) дословно:
    scale = max|w| по строке / 127;  q = clamp(round(w / scale), -127, 127);  back = q * scale
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "artifacts/models/qwen3vl4b_int8"
NAME = "model.visual.pos_embed.weight"


def main() -> None:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    packed = set(json.loads((BASE / "packing.json").read_text(encoding="utf-8"))["packed"])
    if NAME in packed:
        raise SystemExit(f"{NAME} уже сжат — правка не нужна")

    shard = None
    for p in sorted(BASE.glob("*.safetensors")):
        with safe_open(p, framework="pt") as f:
            if NAME in f.keys():
                shard = p
                break
    if shard is None:
        raise SystemExit(f"не нашёл {NAME} ни в одном файле весов")

    with safe_open(shard, framework="pt") as f:
        tensors = {k: f.get_tensor(k) for k in f.keys()}
        meta = f.metadata() or {}
    w = tensors[NAME]
    print(f"файл:   {shard.name}, тензоров {len(tensors)}")
    print(f"тензор: {NAME}, форма {tuple(w.shape)}, тип {w.dtype}")

    # ⚠ Дословно как в packing.py редкой половины: масштаб на СТРОКУ, деление в float32.
    w32 = w.to(torch.float32)
    scale = w32.abs().amax(dim=1, keepdim=True) / 127.0
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = torch.clamp(torch.round(w32 / scale), -127, 127).to(torch.int8)
    back = (q.to(torch.float32) * scale).to(w.dtype)

    diff = (back.to(torch.float32) - w32).abs()
    denom = w32.abs().mean().clamp_min(1e-12)
    print(f"расхождение после сжатия: среднее {float(diff.mean() / denom):.5f} "
          f"относительных, максимум {float(diff.max() / denom):.5f}")
    print(f"изменилось значений: {int((back != w).sum())} из {w.numel()}")

    if not a.apply:
        print("\n(это просмотр; чтобы записать, запусти с --apply)")
        return

    backup = shard.with_suffix(shard.suffix + ".orig")
    if not backup.exists():
        shutil.copy2(shard, backup)
        print(f"исходный файл сохранён: {backup.name}")
    tensors[NAME] = back
    save_file(tensors, str(shard), metadata=meta or None)
    print(f"записано: {shard.name}")

    # ⚠ Проверяем ЗАПИСАННОЕ, а не то, что собирались записать.
    with safe_open(shard, framework="pt") as f:
        got = f.get_tensor(NAME)
    if not torch.equal(got, back):
        raise SystemExit("после записи тензор не совпал — файл повреждён")
    print("сверка после записи: совпадает")


if __name__ == "__main__":
    sys.exit(main())
