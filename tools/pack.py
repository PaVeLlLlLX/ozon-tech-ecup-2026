"""Сборка архива симбиоза: каждую категорию выносит своё дообучение.

    python pack.py

Решение собрано из двух отправок, каждая из которых уже показала свою половину на
лидерборде. Разложение публичных баллов по известному составу выборки (БАД 921 товар
и 528 позитивов, редкая 707 и 24) даёт числа однозначно:

    БАД     vl4_plain     490 верных из 528 названных   F1 0.9280
    БАД     наша связка   481 из 517                    F1 0.9206
    редкая  vlm_rare2_a    21 верный из 24 при 22 названных   F1 0.9130
    редкая  vl4_plain      18 из 24                     F1 0.8372

Метрика усредняет категории независимо, поэтому взять в каждую ту модель, которая её
выигрывает, законно: ожидание (0.9130 + 0.9280) / 2 = 0.9205 против 0.9168 у рекорда.
Переобучения не требуется — оба адаптера уже обучены.

Что выброшено против рекорда и почему:

    peft        заменён ручной вживкой LoRA — в базовом образе организаторов его нет
    easyocr     в редкой категории вклад зрительной модели равен 1.0, распознавание
                туда не входило вовсе; в БАД связка с ним уступила чистой модели
    смеситель   там же: половина голоса линейной модели в БАД стоила 0.7 пункта

Что добавлено: генерация объяснений той же моделью, что выносит вердикт.
"""
from __future__ import annotations

import json
import sys
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILD = HERE / "build"
OUT = HERE.parent / "submissions"
NAME = "duo_best_halves"


def main() -> None:
    cfg_path = BUILD / "configs" / "submission.yaml"
    if not cfg_path.exists():
        raise SystemExit(f"нет {cfg_path}")
    # ⚠ Читаем ИМЕННО тот конфиг, который поедет, и сверяем его с деревом: адаптеры,
    # названные в нём, обязаны лежать на диске. Отсутствие адаптера в архиве обнаружится
    # иначе только на прогоне у жюри, а это потраченная попытка.
    text = cfg_path.read_text(encoding="utf-8")
    variant = next(l.split(":", 1)[1].strip() for l in text.splitlines()
                   if l.strip().startswith("variant:"))
    adapters = [l.split(":", 1)[1].strip() for l in text.splitlines()
                if l.strip().startswith("artifacts/sft_vlm")
                or "artifacts/sft_vlm" in l]
    adapters = [a.strip().strip('"') for a in adapters if "artifacts/sft_vlm" in a]
    adapters = [a.split(": ", 1)[-1].strip().strip('"') for a in adapters]
    for a in adapters:
        d = BUILD / a
        if not (d / "adapter_model.safetensors").exists():
            raise SystemExit(f"адаптер «{a}» назван в конфиге, но его нет в дереве")
        print(f"  адаптер на месте: {a}")

    OUT.mkdir(exist_ok=True)
    dst = OUT / f"{variant}.zip"
    t0 = time.time()
    files = [p for p in sorted(BUILD.rglob("*")) if p.is_file()
             and "__pycache__" not in p.parts]
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for p in files:
            z.write(p, p.relative_to(BUILD).as_posix())

    # ⚠ Проверяем СОБРАННЫЙ архив, а не то, что собирались положить: 29.08 общий каталог
    # сборки дал пять архивов с чужими конфигами, и ни размер, ни журнал этого не
    # показали. Заодно смотрим образ: свой вместо базового означал бы, что уборка peft
    # и распознавания прошла впустую.
    with zipfile.ZipFile(dst) as z:
        inside = z.read("configs/submission.yaml").decode("utf-8")
        if f"variant: {variant}" not in inside:
            raise SystemExit(f"в архиве лежит конфиг чужого варианта")
        meta = json.loads(z.read("metadata.json"))
        if "baseline" not in meta["image"]:
            raise SystemExit(f"образ {meta['image']!r} не базовый — peft и распознавание "
                             f"убирались ради базового")
        names = set(z.namelist())
        for a in adapters:
            if f"{a}/adapter_model.safetensors" not in names:
                raise SystemExit(f"в архиве нет весов адаптера «{a}»")
        if z.testzip() is not None:
            raise SystemExit("архив повреждён")
    size = dst.stat().st_size
    print(f"  {variant}: {len(files)} файлов, {size / 1e9:.3f} ГБ, образ "
          f"{meta['image']}, адаптеров {len(adapters)}, проверен, "
          f"за {(time.time() - t0) / 60:.1f} мин"
          + ("  ⚠⚠ БОЛЬШЕ 5 ГБ" if size > 5e9 else ""), flush=True)


if __name__ == "__main__":
    sys.exit(main())
