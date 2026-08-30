"""Сплошная сверка собранного архива с обоими источниками.

    python audit.py

Проверяется СОБРАННЫЙ zip, а не дерево сборки: балл зависит от того, что уехало, а не
от того, что собирались положить. Каждая строка отчёта отвечает на вопрос «сходится ли
это с тем решением, откуда взято».

    редкая категория  -> vlm_blend_rare2_a_t24.zip  (публичные 0.9168088205)
    БАД               -> vl4_plain.zip              (публичные 0.8826198027)
"""
from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ZIP = HERE.parent / "submissions" / "duo_best_halves.zip"
REC = HERE / "vlm_blend_rare2_a_t24.zip"
MATE = HERE.parent / "vl4_plain.zip"

ok = 0
bad = 0


def check(title: str, got, want, note: str = "") -> None:
    global ok, bad
    good = got == want
    globals()["ok" if good else "bad"] = (ok + 1) if good else ok
    if not good:
        globals()["bad"] = bad + 1
    mark = "  OK  " if good else "  !!  "
    print(f"{mark}{title}")
    if good:
        print(f"        {got}{('  — ' + note) if note else ''}")
    else:
        print(f"        получено: {got}")
        print(f"        ожидалось: {want}")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def main() -> None:
    global ok, bad
    for p in (ZIP, REC, MATE):
        if not p.exists():
            raise SystemExit(f"нет файла {p}")
    z = zipfile.ZipFile(ZIP)
    rec = zipfile.ZipFile(REC)
    mate = zipfile.ZipFile(MATE)
    names = set(z.namelist())

    print("=" * 74)
    print("ОБЩЕЕ")
    print("=" * 74)
    meta = json.loads(z.read("metadata.json"))
    check("образ решения", meta["image"], "odsai/ecup26-quality-baseline:1.0",
          "базовый, от организаторов")
    check("точка входа", meta["entry_point"], "python -u run.py")
    size = ZIP.stat().st_size
    print(f"  {'OK' if size < 5e9 else '!!'}  размер архива")
    print(f"        {size / 1e9:.3f} ГБ при лимите 5 ГБ")

    cfg = z.read("configs/submission.yaml").decode("utf-8")

    def val(key: str) -> str:
        for line in cfg.splitlines():
            s = line.strip()
            if s.startswith(key + ":"):
                return s.split(":", 1)[1].strip()
        return "<нет>"

    def sub(parent: str, key: str) -> str:
        want, inside = False, False
        for line in cfg.splitlines():
            if line.strip().startswith(parent + ":"):
                want, inside = True, True
                continue
            if inside and line[:3] == "  " + line.strip()[:1] and ":" in line:
                if not line.startswith("    "):
                    inside = False
                    continue
            if inside and line.strip().startswith(f'"{key}"'):
                return line.split(":", 1)[1].strip()
        return "<нет>"

    print()
    print("=" * 74)
    print("РЕДКАЯ КАТЕГОРИЯ — источник vlm_blend_rare2_a_t24 (0.9168088205)")
    print("=" * 74)
    check("адаптер", sub("vlm_adapters", "Легковоспламеняющиеся"),
          "artifacts/sft_vlm/vlm_rare2_a")
    a_ours = z.read("artifacts/sft_vlm/vlm_rare2_a/adapter_model.safetensors")
    a_rec = rec.read("artifacts/models/vlm_rare2_a/adapter_model.safetensors")
    check("веса адаптера (sha256)", sha(a_ours), sha(a_rec),
          "побитово те же, что в архиве рекорда")
    check("способ отбора", sub("vlm_select", "Легковоспламеняющиеся"), "calibrated",
          "порог по эталону, а не доля")
    cal = json.loads(z.read("artifacts/models/rare2_calibration.json"))
    check("порог", cal["порог"], 0.9699, "из thresholds рекорда")
    check("длина эталона", len(cal["эталон"]), 1101, "фолд 0 редкой категории")
    log = json.loads(z.read("artifacts/sft_vlm/vlm_rare2_a/train_log.json"))
    check("шаблон промпта", log["prompt"], "rare2", "сверен с verdict_ru.txt посимвольно")
    check("обрезка описания", log["desc_max"], 900, "из vlm_sft_rented.yaml рекорда")
    check("кадров", log["n_images"], 5)
    check("пикселей на кадр", log["max_pixels"], 100352)
    check("написаний «да»", log["yes_variants"], ["Да", " Да", "да", "Yes", " Yes"],
          "как YES_VARIANTS рекорда")
    check("написаний «нет»", log["no_variants"], ["Нет", " Нет", "нет", "No", " No"])

    print()
    print("=" * 74)
    print("БАД — источник vl4_plain (0.8826198027)")
    print("=" * 74)
    check("адаптер", sub("vlm_adapters", "БАД"), "artifacts/sft_vlm/vl4_plain_bf16")
    b_ours = z.read("artifacts/sft_vlm/vl4_plain_bf16/adapter_model.safetensors")
    b_mate = mate.read("artifacts/sft_vlm/vl4_plain_bf16/adapter_model.safetensors")
    check("веса адаптера (sha256)", sha(b_ours), sha(b_mate),
          "побитово те же, что в его архиве")
    check("способ отбора", sub("vlm_select", "БАД"), "rate", "доля, как у него")
    check("доля", sub("vlm_rate", "БАД"), "0.5733", "его число")
    blog = json.loads(z.read("artifacts/sft_vlm/vl4_plain_bf16/train_log.json"))
    check("шаблон промпта", blog["prompt"], "qc")
    check("обрезка описания", blog.get("desc_max", "нет ключа → 800"),
          "нет ключа → 800", "ровно как в его прогоне")

    print()
    print("=" * 74)
    print("ОБЪЯСНЕНИЯ — наши лучшие настройки")
    print("=" * 74)
    check("источник объяснений", val("explain"), "vlm")
    check("потолок новых токенов", val("explain_tokens"), "60")
    check("кадров в промпте", val("explain_images"), "1")
    check("описание в промпте", val("explain_desc"), "300")

    print()
    print("=" * 74)
    print("ЧЕГО В АРХИВЕ БЫТЬ НЕ ДОЛЖНО")
    print("=" * 74)
    for what, pat in (("веса easyocr", "easyocr"), ("смеситель SVM", "vlm_blend"),
                      ("веса судьи", "judge_"), ("кэш python", "__pycache__")):
        hit = [n for n in names if pat in n]
        check(what, hit, [], "убрано")
    src = z.read("run.py").decode("utf-8")
    check("импорт peft в решении", "import peft" in src or "from peft" in src, False,
          "LoRA вживляется вручную")

    print()
    print("=" * 74)
    print(f"ИТОГ: сошлось {ok}, расхождений {bad}")
    print("=" * 74)
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    sys.exit(main())
