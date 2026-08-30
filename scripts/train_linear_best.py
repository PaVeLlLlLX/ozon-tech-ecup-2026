"""Линейные модели в настройках, отобранных подбором прогона №5.

Подбор (`artifacts/reports/tune_linear.md`, среднее по трём сидам) поставил на первое
место линейный SVM с C=1.0 **без балансировки классов и без калибровки**: F1 редкой
категории 0.6238 против 0.6008 у той же модели с балансировкой и 0.5652 у лучшей
логистической регрессии.

Наш публичный рекорд `svm_ocr_easyocr` (0.7233) обучен той же связкой, но с
балансировкой — то есть отличается от лучшей настройки ровно одним параметром. Это
редкий случай, когда улучшение проверяется одной переменной, поэтому обучение идёт на
ТОМ ЖЕ двухкадровом индексе распознавания, что и рекорд: пятикадровый индекс вводит
вторую переменную, а прогон №5 показал, что кадры 3-5 ничего не добавляют.

Модели:
    svm_c1_nobal       C=1.0, без балансировки, двухкадровый индекс — главный кандидат
    svm_c4_bal         C=4.0, с балансировкой, двухкадровый индекс — вторая строка подбора
    svm_c1_nobal_ocr5  C=1.0, без балансировки, ПЯТИкадровый индекс — цена рассогласования

Запуск (детач): scripts\train_linear_best.cmd   Журнал: artifacts/logs/linear_best.log
"""
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import _bootstrap  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
OCR2 = "artifacts/ocr/ocr_index_fast.2frames.parquet"
OCR5 = "artifacts/ocr/ocr_index_fast.parquet"

# Общая часть: распознанный текст во входе, улики по правилам как признаки, линейный SVM.
COMMON = ["-o", "baseline.use_ocr=true", "-o", "baseline.classifier=linear_svm"]

JOBS = [
    ("svm_c1_nobal", [*COMMON, "-o", f"paths.ocr_index={OCR2}",
                      "-o", "baseline.svm.C=1.0", "-o", "baseline.logreg.class_weight=null"],
     "Линейный SVM C=1.0 без балансировки классов, распознавание с двух кадров. "
     "Лучшая строка подбора прогона №5; от публичного рекорда отличается ровно "
     "снятой балансировкой."),
    ("svm_c4_bal", [*COMMON, "-o", f"paths.ocr_index={OCR2}",
                    "-o", "baseline.svm.C=4.0", "-o", "baseline.logreg.class_weight=balanced"],
     "Линейный SVM C=4.0 с балансировкой классов, распознавание с двух кадров. "
     "Вторая строка подбора; вместе с рекордом и главным кандидатом замыкает квадрат "
     "по двум осям — сила регуляризации и вес классов."),
    ("svm_c1_nobal_ocr5", [*COMMON, "-o", f"paths.ocr_index={OCR5}",
                           "-o", "baseline.svm.C=1.0",
                           "-o", "baseline.logreg.class_weight=null"],
     "Та же лучшая настройка, но обученная на пятикадровом индексе — том самом, что "
     "контейнер считает на инференсе. Замеряет цену рассогласования обучения и прогона."),
]


def say(msg: str) -> None:
    print(f"[{datetime.now():%m-%d %H:%M:%S}] {msg}", flush=True)


def main() -> None:
    say("=== обучение линейных моделей в отобранных настройках ===")
    for name, opts, note in JOBS:
        say(f"--- НАЧАЛО: {name}")
        t0 = time.time()
        rc = subprocess.run([PY, "-u", str(ROOT / "scripts" / "train_baseline.py"),
                             "--name", name, *opts, "--note", note],
                            cwd=ROOT).returncode
        say(f"--- {'ГОТОВО' if rc == 0 else f'СБОЙ ({rc})'}: {name} "
            f"за {(time.time() - t0) / 60:.1f} мин")
    say("=== обучение завершено ===")


if __name__ == "__main__":
    main()
