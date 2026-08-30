"""Сборка архива решения ровно в том виде, в каком его ждёт проверяющая система.

Что кладётся в архив (всё пути от его корня):
    metadata.json     — образ и строка запуска, без него решение не примут
    run.py            — точка входа, запускается как `python -u run.py`
    src/qc26/**       — код решения
    configs/**        — конфиги (+ копии в JSON на случай образа без PyYAML)
    artifacts/models/ — веса выбранного решения

Архив не просто пакуется, а ПРОВЕРЯЕТСЯ: во временной папке решение запускается так же,
как это сделает проверяющая система — на десяти товарах (стадия check) и на полном
объёме (замер времени), после чего результат прогоняется через валидатор формата.
Проверять надо именно распакованный архив, а не рабочую копию: в архиве другой набор
файлов, и любая забытая зависимость всплывёт только здесь.

    python scripts/build_submission.py --variant tfidf_binary
    python scripts/build_submission.py --all      # все заготовленные варианты
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd

from qc26.config import load_config, resolve_path
from qc26.data import load_cards
from qc26.inference.format import submission_stats, validate_submission
from qc26.metrics import competition_score, ranking_scores

ROOT = Path(__file__).resolve().parents[1]


class BuildLock:
    """Не даёт двум сборкам идти одновременно.

    Обе используют одну папку artifacts/_stage и один журнал отправок. При наложении
    одна стирает промежуточные файлы другой: в лучшем случае это PermissionError, в
    худшем — молча собранный пустой архив, который выглядит готовым. Такое уже случалось.
    """

    def __init__(self, path: Path):
        self.path = path

    def __enter__(self):
        if self.path.exists():
            try:
                who = self.path.read_text(encoding="utf-8").strip()
            except OSError:
                who = "неизвестно кем"
            age_min = (time.time() - self.path.stat().st_mtime) / 60
            if age_min < 180:
                raise SystemExit(
                    f"Сборка уже идёт ({who}, {age_min:.0f} мин назад).\n"
                    f"Дождитесь её окончания. Если она точно оборвалась, удалите файл:\n"
                    f"  {self.path}")
            print(f"⚠ замок от {who} старше трёх часов — считаю его брошенным", flush=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(f"запущена {datetime.now():%d.%m %H:%M}", encoding="utf-8")
        return self

    def __exit__(self, *exc):
        self.path.unlink(missing_ok=True)
        return False


def _child_env() -> dict:
    """Окружение дочернего прогона с гарантированной utf-8 на выводе.

    Без этого под Windows дочерний питон берёт кодировку консоли (cp1251), и первый же
    символ вне неё роняет решение уже после подсчёта ответа. В контейнере кодировка
    другая, поэтому дефект виден только локально — тем важнее ловить его здесь.
    """
    import os as _os

    # ⚠ HF_HUB_OFFLINE обязателен и для ЛОКАЛЬНЫХ проверок, не только для контейнера.
    # Без него загрузчик пытается сходить в сеть за базовой моделью, на которую
    # ссылается adapter_config.json, и проверка виснет в таймаутах: сборка решения на
    # языковой модели простояла так сорок минут, не дойдя даже до первой стадии.
    # У проверяющей системы сети нет вовсе, так что это ещё и честный режим.

    return {**_os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1",
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}


ARCHIVE_LIMIT_GB = 5.0


def expects_vision(spec: dict) -> bool:
    """Ждать ли в журнале маркер о состоявшейся работе с фотографиями.

    Решение на языковой модели может кадры не читать вовсе (max_images=0). Маркер
    печатается ТОЛЬКО при реально прочитанных кадрах, поэтому такому варианту его
    ждать нельзя — иначе здоровое решение помечается как «ЗРЕНИЕ НЕ СРАБОТАЛО».
    """
    return works_with_photos(spec) and not spec.get("no_vision")


def works_with_photos(spec: dict) -> bool:
    """Читает ли решение фотографии во время прогона.

    ⚠ Одно определение на все три места: выбор стадий проверки, замер времени на
    приватном размере и боевой прогон в контейнере. Пока их было три разных, решение
    на языковой модели проскочило мимо ДВУХ последних и получило статус «ГОТОВ», ни
    разу не подтвердив, что вердикт выносит оно само, а не откат на правила.
    """
    # ⚠ Это про «решение тяжёлое», а НЕ про «ждём маркер зрения». Смешивать нельзя:
    # когда я вернул False для варианта без кадров, сборка погнала его на полный объём
    # в 12 971 товар и зависла. За ожидание маркера отвечает expects_vision ниже.
    return bool(spec.get("use_ocr") or spec.get("use_embeddings")
                or spec.get("use_qa")
                or str(spec.get("model", "")).startswith(("vlm:", "vlmblend:")))
# Сколько товаров подавать в боевой проверке: выше отладочного порога в
# сто товаров, иначе зрение пропускается и проверять нечего.
# ⚠ Переопределяется QC_HOSTILE_ITEMS: полная сборка с распознаванием идёт около
# получаса на товар-набор 300+3800, и при пяти попытках в день это съедает день.
# Ниже 150 не опускать — на 100 товарах зрение пропускается, и проверять нечего.
HOSTILE_ITEMS = int(os.environ.get("QC_HOSTILE_ITEMS", 300))

# ⚠ Таймаут дочерних прогонов. Раньше его не было вовсе, и зависший этап висел
# бесконечно: проверка из десяти товаров у решения с анкетой ушла в бесконечную
# рекурсию и держала сборку восемь минут, съев всю оперативную память. Сборка обязана
# падать быстро и внятно, а не превращать пятиминутную задачу в часовую.
CHILD_TIMEOUT_SEC = int(os.environ.get("QC_CHILD_TIMEOUT", 900))

# Порог, ниже которого решение НАМЕРЕННО пропускает работу с фотографиями (отладочная
# стадия жюри — десять товаров и три минуты). Держим то же значение, что в run.py.
CHECK_STAGE_MAX_ITEMS = 100

# Имена образов: базовый от организаторов и наш. Вариант объявляет, какой ему нужен,
# а не пишет имя строкой — иначе при смене тега придётся править десяток мест.
IMAGES = json.loads((ROOT / "docker" / "images.json").read_text(encoding="utf-8"))

# Заготовленные варианты отправки. В сутки доступно 5 попыток, поэтому набор
# подобран так, чтобы каждая попытка что-то сообщала, а не просто повторяла соседнюю.
# note — комментарий, который прикрепляется к отправке на платформе и становится
#        описанием прогона в MLflow. Он должен читаться САМОСТОЯТЕЛЬНО: и сторонний
#        человек, и мы сами через месяц должны по нему понять, что это за решение —
#        какая модель, на каком входе, как обучена, чем отличается по настройкам.
#        Ссылок вида «то же решение» быть не должно: рядом с ним соседней строки нет.
# why  — внутреннее обоснование, зачем эта попытка потрачена. В отправку не идёт.
VARIANTS = {
    'vlm_blend_rare2_a_t24': {   'model': 'vlmblend:artifacts/models/vlm_blend_rare2_a_t24.joblib:artifacts/models/vlm_rare2_a',
    'fallback_model': 'artifacts/models/tfidf_logreg.joblib',
    'include_dirs': [   'artifacts/models/vlm_rare2_a',
                        'artifacts/models/qwen4b_packed',
                        'artifacts/easyocr'],
    'packed_base_path': 'artifacts/models/qwen4b_packed',
    'use_ocr': True,
    'ocr_engine': 'easyocr',
    'ocr_max_images': 2,
    'image': 'custom',
    'note': 'Решение, давшее наш лучший результат, с единственной правкой: в категории '
            '«Легковоспламеняющиеся» порог смягчён так, чтобы вердикт «не бан» получала доля '
            'товаров около трёх с половиной процентов вместо трёх.',
    'why': 'Разбор лучшего результата показал, что там названо двадцать одно товарное '
           'предложение и двадцать из них верны. Точность 0.95 при полноте 0.83: ошибок '
           'почти нет, а ненайденных четыре. Отправка проверяет, добавит ли более широкая '
           'воронка верных больше, чем ложных.'},
}


def _copy_tree(src: Path, dst: Path) -> None:
    """Копирует дерево без служебного мусора: скомпилированный кэш в архиве не нужен."""
    shutil.copytree(src, dst, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo",
                                                  ".ipynb_checkpoints"))
    for junk in list(dst.rglob("__pycache__")):
        shutil.rmtree(junk, ignore_errors=True)


def _write_json_copies(configs_dir: Path) -> None:
    """Копия каждого YAML в JSON: базовый образ может быть без PyYAML."""
    import yaml

    for y in configs_dir.rglob("*.yaml"):
        data = yaml.safe_load(y.read_text(encoding="utf-8")) or {}
        y.with_suffix(".json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def make_foreign_layout(cfg: dict, root: Path, n: int = 10) -> Path:
    """Собирает маленький набор данных ВНЕ репозитория, как его подложит жюри.

    Наши данные лежат в `data/`, и прогон по ним ничего не доказывает: путь к картинкам
    решение выводит из пути к csv, а на сервере тот путь будет чужим. Здесь создаётся
    отдельная папка с `test.csv` и `images/<id>/`, чтобы поймать любое предположение о
    раскладке до отправки, а не после обнуления попытки.
    """
    from qc26.data import image_paths, images_root

    if root.exists():
        shutil.rmtree(root)
    (root / "images").mkdir(parents=True)
    df = load_cards(cfg)
    src_root = images_root(cfg)
    sample = df.head(n)
    for item_id in sample[cfg["data"]["id_col"]]:
        dst = root / "images" / str(item_id)
        dst.mkdir(parents=True, exist_ok=True)
        for p in image_paths(src_root, item_id):
            shutil.copy2(p, dst / p.name)
    csv_path = root / "test.csv"
    # целевой колонки в тестовых данных нет — воспроизводим это же
    sample.drop(columns=[cfg["data"]["target_col"]], errors="ignore").to_csv(
        csv_path, index=False, encoding="utf-8")
    return csv_path


def _metric_at_thresholds(cfg: dict, model_rel: str, thresholds: dict) -> dict | None:
    """Честная метрика при ЗАДАННОМ пороге — по предсказаниям вне обучения.

    Порог можно менять как угодно, не переобучая модель: сами предсказания от него не
    зависят. Так вариант со сдвинутым порогом сравним с основным, а не выглядит лучше
    только потому, что его померили на обучающих данных.
    """
    name = Path(model_rel).stem
    p = resolve_path(cfg["paths"]["artifacts_dir"]) / "preds" / f"{name}_oof.parquet"
    if not p.exists():
        return None
    oof = pd.read_parquet(p)
    tgt, cat = cfg["data"]["target_col"], cfg["data"]["category_col"]
    score_cols = [c for c in oof.columns if c.startswith("score_seed")]
    if not score_cols:
        return None
    out: dict[str, float] = {}
    for avg in ("binary", "macro"):
        vals = []
        for col in score_cols:
            thr = oof[cat].astype(str).map(thresholds).fillna(0.5).to_numpy()
            pred = (oof[col].to_numpy() >= thr).astype(int)
            vals.append(competition_score(oof[tgt].to_numpy(), pred,
                                          oof[cat].to_numpy(), average=avg)["mean"])
        out[f"локально {avg}"] = round(float(sum(vals) / len(vals)), 4)
    # AUC от порога не зависит — считаем по тем же предсказаниям, чтобы в журнале
    # у вариантов со сдвинутым порогом не было пустой клетки
    aucs = [ranking_scores(oof[tgt].to_numpy(), oof[col].to_numpy(),
                           oof[cat].to_numpy())["auc_mean"] for col in score_cols]
    out["AUC"] = round(float(sum(aucs) / len(aucs)), 4)
    return out


def build(variant: str, spec: dict, cfg: dict, stage: Path, out_dir: Path,
          check_csv: Path, full_csv: Path, docker_image: str | None = None) -> dict:
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    # --- состав архива ---
    shutil.copy2(ROOT / "run.py", stage / "run.py")
    # Образ можно задать отдельно для каждой отправки: варианту с распознаванием,
    # LoRA-адаптером или бустингом нужен наш образ, простому текстовому — базовый.
    meta = json.loads((ROOT / "docker" / "metadata.json").read_text(encoding="utf-8"))
    meta["image"] = IMAGES.get(spec.get("image", "base"), IMAGES["base"])
    (stage / "metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    _copy_tree(ROOT / "src", stage / "src")
    _copy_tree(ROOT / "configs", stage / "configs")

    model = spec["model"]
    # ⚠ probe_model обязан попасть в архив наравне с основной моделью: зонд-замер
    # считает вердикты именно им, а без файла молча уходит в откат на правила и
    # передаёт наружу замер несуществующей модели.
    # ⚠ У связки путь составной: «vlmblend:веса:адаптер». Без разбора копировалась бы
    # строка целиком, расширения .joblib у неё нет, файл молча не попадал в архив —
    # и решение падало уже на выборке приватного размера с FileNotFoundError.
    model_files = [model]
    if str(model).startswith("vlmblend:"):
        model_files = list(str(model).split(":")[1:])
    for rel in (model_files + [spec.get("fallback_model"), spec.get("probe_model")]
                + list(spec.get("include", []))):
        # .csv.gz — таблица подписей для поиска дубликата: такой же обязательный файл
        # решения, как веса. ⚠ Список расширений — грабли: пока в нём не было нужного,
        # файл молча не копировался, а решение падало уже в контейнере. Поэтому не
        # «файл с известным расширением», а «всё, что вариант перечислил явно».
        if not rel or not str(rel).endswith((".joblib", ".pth", ".parquet", ".csv.gz",
                                             ".csv", ".json", ".npy", ".npz")):
            continue
        src_model = resolve_path(rel)
        if not src_model.exists():
            return {"variant": variant, "статус": f"нет файла {rel}"}
        dst = stage / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_model, dst)
    # папки целиком (веса распознавателя: в контейнере их не скачать)
    for rel in spec.get("include_dirs", []):
        src_dir = resolve_path(rel)
        if not src_dir.is_dir():
            return {"variant": variant, "статус": f"нет папки {rel}"}
        _copy_tree(src_dir, stage / rel)

    # какое решение отправляем — читает run.py
    lines = ["submission:", f"  model: {model}", f"  variant: {variant}",
             f"  comment: \"{spec['note']}\"",
             f"  use_ocr: {'true' if spec.get('use_ocr') else 'false'}",
             f"  use_embeddings: {'true' if spec.get('use_embeddings') else 'false'}",
             f"  ocr_engine: {spec.get('ocr_engine', 'vlm')}",
             f"  on_vision_failure: {spec.get('on_vision_failure', 'fail')}",
             ] + ([f"  ocr_budget_sec: {spec['ocr_budget_sec']}"]
                  if spec.get("ocr_budget_sec") else []) \
        + ([f"  ocr_max_images: {spec['ocr_max_images']}"]
           if spec.get("ocr_max_images") else [])
    if spec.get("fallback_model"):
        lines.append(f"  fallback_model: {spec['fallback_model']}")
    if spec.get("probe_model"):
        lines.append(f"  probe_model: {spec['probe_model']}")
    if spec.get("use_duplicates"):
        lines.append("  use_duplicates: true")
    if spec.get("smooth_duplicates"):
        # ⚠ Сглаживание скоров по почти-дубликатам ВНУТРИ проверяемой выборки. Пишем
        # СПИСОК категорий, а не «да»: замер на отложенном фолде дал в редкой категории
        # AUC 0.9113 -> 0.9729, а в БАД 0.9577 -> 0.9524, то есть чуть хуже.
        value = spec["smooth_duplicates"]
        if value is True:
            lines.append("  smooth_duplicates: true")
        else:
            lines.append("  smooth_duplicates:")
            lines += [f'    - "{c}"' for c in value]
    if spec.get("use_qa"):
        # ответы языковой модели на вопросы о товаре считаются НА ПРОГОНЕ
        lines.append("  use_qa: true")
    if spec.get("packed_base_path"):
        # базовая модель едет в архиве упакованной: её нет в каталоге жюри
        lines.append(f"  packed_base_path: {spec['packed_base_path']}")
    if spec.get("embedding_sets"):
        # список наборов векторов, которые решению нужно построить на прогоне
        lines.append("  embedding_sets:")
        lines += [f"    - {name}" for name in spec["embedding_sets"]]
    if spec.get("thresholds"):
        lines.append("  thresholds:")
        lines += [f"    \"{k}\": {v}" for k, v in spec["thresholds"].items()]
    # ⚠ Настройки ВХОДА, с которыми обучалась модель этого варианта. Прогон читает
    # configs/vlm_sft.yaml, а там значения рабочей модели; решение, обученное на другом
    # разрешении или другом числе кадров, поехало бы со ЧУЖИМ входом и молча выдало
    # мусор. Блок пишется последним и перекрывает vlm_sft.yaml: run.py сливает
    # submission.yaml после него.
    if spec.get("vlm_overrides"):
        lines.append("")
        lines.append("vlm:")
        lines += [f"  {k}: {v}" for k, v in spec["vlm_overrides"].items()]
    (stage / "configs" / "submission.yaml").write_text("\n".join(lines) + "\n",
                                                       encoding="utf-8")
    # в контейнере файловая система только для чтения — трекинг обязан молчать
    data_yaml = stage / "configs" / "data.yaml"
    data_yaml.write_text(data_yaml.read_text(encoding="utf-8")
                         .replace("enabled: true", "enabled: false"), encoding="utf-8")
    _write_json_copies(stage / "configs")

    # --- проверка распакованного архива тем же вызовом, что и у жюри ---
    res: dict = {"variant": variant, "модель": model}
    # Полный объём (12 971 товар) прогоняем только у решений без зрения: у остальных он
    # в 3.4 раза больше приватного набора, чтение честно не укладывается в бюджет, и
    # строгий режим роняет прогон. Время у них меряется отдельно, на выборке приватного
    # размера — это и есть настоящий сценарий.
    stages = [("чужая раскладка (10 товаров)", check_csv, 3.0)]
    # Полный объём гоняем только у ЛЁГКИХ решений. У всех, кто работает с
    # фотографиями — распознавание, векторы или сама зрительно-языковая модель —
    # 12 971 товар втрое больше приватного набора, и проверка растянулась бы на часы.
    # Их время меряется отдельно, на выборке приватного размера.
    if not works_with_photos(spec):
        stages.append(("полный объём", full_csv, None))
    for tag, csv_path, limit_min in stages:
        out_csv = stage / f"_probe_{tag.split()[0]}.csv"
        t0 = time.time()
        try:
            proc = subprocess.run(
                [sys.executable, "-u", "run.py", "--test_data_path", str(csv_path),
                 "--output_path", str(out_csv)],
                cwd=stage, capture_output=True, text=True, encoding="utf-8",
                errors="replace", env=_child_env(), timeout=CHILD_TIMEOUT_SEC)
        except subprocess.TimeoutExpired:
            return {**res, "статус": f"ЗАВИС на «{tag}»",
                    "лог": f"не уложился в {CHILD_TIMEOUT_SEC} с"}
        dt = time.time() - t0
        if proc.returncode != 0:
            return {**res, "статус": f"ПАДЕНИЕ на «{tag}»",
                    "лог": (proc.stderr or proc.stdout)[-800:]}
        sub = pd.read_csv(out_csv)
        expected = load_cards(cfg, path=csv_path)[cfg["data"]["id_col"]]
        errors = validate_submission(sub, expected_ids=expected)
        if errors:
            return {**res, "статус": f"ФОРМАТ НАРУШЕН на «{tag}»: {'; '.join(errors)}"}
        if tag.startswith("чужая"):
            res["время check, с"] = round(dt, 1)
        else:
            res["время полного прогона, с"] = round(dt, 1)
            res["товаров"] = len(sub)
            # пересчёт на приватный набор: 3800 товаров при лимите 40 минут
            res["оценка private, мин"] = round(dt / max(1, len(sub)) * 3800 / 60, 1)
            st = submission_stats(sub)
            res["доля «не бан»"] = round(st["share_not_ban"], 4)
            res["длина комментария"] = f"{st['comment_len_min']}–{st['comment_len_max']}"
        out_csv.unlink(missing_ok=True)

    # --- отдельная проверка времени для вариантов с чтением фото ---
    # Полный прогон по нашим 12 971 товарам ничего не доказывает: это в 3.4 раза больше
    # приватного набора, и чтение фото там честно отменяется по бюджету. Поэтому время
    # меряем на выборке ровно приватного размера, положенной рядом с изображениями.
    if works_with_photos(spec):
        # ⚠ Замер НАМЕРЕННО не пользуется нашими кэшами распознавания и эмбеддингов:
        # он меряет, сколько займёт настоящий прогон у жюри, где всё считается с нуля.
        # Но мерить это на полных 3800 товарах незачем — темп линеен, и на шестой части
        # выборки он оценивается так же, а времени уходит вшестеро меньше. Проверку
        # того, что зрение реально состоялось, делает боевой прогон в контейнере.
        # ⚠ По умолчанию НЕ 3800: у решений со зрением полный замер идёт часами и
        # упирается в таймаут, роняя всю партию сборок. Темп линеен, шестая часть
        # выборки оценивает его так же. Ночь №10 потеряла на этом три архива.
        n_timing = int(os.environ.get("QC_TIMING_ITEMS", 300))
        sized = full_csv.parent / "_timing_private.csv"
        try:
            load_cards(cfg).head(n_timing).to_csv(sized, index=False, encoding="utf-8")
            out_csv = stage / "_probe_timing.csv"
            t0 = time.time()
            try:
                proc = subprocess.run(
                    [sys.executable, "-u", "run.py", "--test_data_path", str(sized),
                     "--output_path", str(out_csv)],
                    cwd=stage, capture_output=True, text=True, encoding="utf-8",
                    errors="replace", env=_child_env(), timeout=CHILD_TIMEOUT_SEC)
            except subprocess.TimeoutExpired:
                # ⚠ Таймаут замера времени НЕ должен ронять сборку: архив уже собран,
                # а не уложившийся замер — это факт для журнала, а не отказ. Прежде
                # исключение уходило наверх и убивало всю партию вариантов.
                return {**res, "статус": "ЗАВИС на замере времени",
                        "лог": f"замер не уложился в {CHILD_TIMEOUT_SEC} с"}
            dt = time.time() - t0
            log = (proc.stdout or "") + (proc.stderr or "")
            if n_timing >= 3800:
                res["время на 3800 товарах, мин"] = round(dt / 60, 1)
            else:
                # линейный пересчёт: загрузка модели в него тоже попадает, поэтому
                # оценка получается с запасом в большую сторону — это безопасно
                res[f"замер на {n_timing} товарах, мин"] = round(dt / 60, 1)
                res["время на 3800 товарах, мин"] = round(dt / 60 * 3800 / n_timing, 1)
            # ищем именно сообщение движка чтения, а не финальную строку решения:
            # «готово за» встречается и там, и ложное срабатывание скрыло бы отказ
            from qc26.inference import VISION_DONE_MARKERS

            done = any(m in log for m in VISION_DONE_MARKERS)
            # ⚠ Ниже отладочного порога зрение пропускается ПО ЗАМЫСЛУ, маркера там не
            # будет никогда. Судить по такому замеру о зрении нельзя: так рождались
            # ложные «ЗРЕНИЕ НЕ СРАБОТАЛО» у здоровых архивов.
            judged = expects_vision(spec) and n_timing > CHECK_STAGE_MAX_ITEMS
            if not expects_vision(spec):
                res["работа с фото состоялась"] = "не требуется (решение без кадров)"
            elif not judged:
                res["работа с фото состоялась"] = (
                    f"не проверено ({n_timing} товаров — ниже порога зрения)")
            else:
                res["работа с фото состоялась"] = "да" if done else "нет (откат)"
            if judged and not done:
                # Вариант заявил работу с изображениями, а она не состоялась — решение
                # тихо выродилось в текстовое. Отправлять такое нельзя: на лидерборде
                # оно повторит базовый результат, а попытка будет потрачена. Именно так
                # мы потеряли три попытки подряд.
                res["статус"] = "ЗРЕНИЕ НЕ СРАБОТАЛО"
            if proc.returncode != 0:
                return {**res, "статус": "ПАДЕНИЕ на выборке приватного размера",
                        "лог": log[-800:]}
            out_csv.unlink(missing_ok=True)
        finally:
            sized.unlink(missing_ok=True)

    # --- локальная метрика на обучающих данных (для журнала) ---
    # Честная оценка есть только у варианта с теми же порогами, с какими модель училась.
    # Если порог переопределён конфигом отправки, число из metrics.json к нему не относится.
    if model.startswith(("vlm:", "vlmblend:")):
        # У решения на языковой модели пороги РОДНЫЕ — они получены той же оценкой на
        # отложенном фолде, что и метрика, поэтому число из файла к ним относится.
        # у связки путь к весам — ВТОРОЕ поле, у одиночной модели — первое
        _parts = model.split(":")
        metrics_file = ROOT / (_parts[1].removesuffix(".joblib") + ".metrics.json")
    else:
        metrics_file = (ROOT / f"{spec['model'].removesuffix('.joblib')}.metrics.json"
                        if model.endswith(".joblib") and not spec.get("thresholds")
                        else None)
    if metrics_file is not None and metrics_file.exists():
        m = json.loads(metrics_file.read_text(encoding="utf-8"))
        # ⚠ Читаем по НАЛИЧИЮ, а не по обязательности. Не каждый наш скрипт обучения
        # считает все три числа: слияние и модели на эмбеддингах меряют F1 по категориям
        # под баланс закрытой выборки и не считают ни F1 macro, ни средний AUC. Прежде
        # обращение по ключу роняло всю сборку целиком на KeyError, и партия из пяти
        # архивов не собиралась из-за отсутствующей строки в журнале.
        for field, key in (("локально binary", "metric_binary"),
                           ("локально macro", "metric_macro"),
                           ("AUC", "auc_mean"),
                           ("F1 редкой", "f1_rare"),
                           ("F1 БАД", "f1_supplement")):
            value = m.get(key)
            if isinstance(value, (int, float)):
                res[field] = round(float(value), 4)
        res["как считано"] = "групповой сплит"
    elif model.endswith(".joblib") and spec.get("thresholds"):
        # Порог подменён — считаем метрику по сохранённым предсказаниям вне обучения
        # при этом самом пороге. Замер на обучающих данных был бы завышен вдвое.
        honest = _metric_at_thresholds(cfg, spec["model"], spec["thresholds"])
        res.update(honest or {})
        res["как считано"] = ("групповой сплит, порог задан" if honest
                              else "нет предсказаний вне обучения — метрика не посчитана")
    elif works_with_photos(spec):
        # ⚠ Ветку со зрением нельзя гонять по всем 12 971 товару ради метрики: это в
        # 3.4 раза больше приватного набора, чтение честно не укладывается в бюджет, и
        # прогон падает. Метрику такой ветки берём из файла рядом с весами; если его
        # нет — оставляем неизвестной, но архив собираем. Раньше здесь была ошибка,
        # ронявшая сборку целиком.
        res["как считано"] = ("нет файла метрик рядом с весами — метрика не посчитана "
                              "(полный объём для ветки со зрением не гоняем)")
    else:
        # Правила и постоянный вердикт ничему не учились — замер на всех данных честный.
        res["как считано"] = "на всех данных, честно"
        df = load_cards(cfg, path=full_csv)
        out_csv = stage / "_probe_metric.csv"
        proc = subprocess.run(
            [sys.executable, "-u", "run.py", "--test_data_path", str(full_csv),
             "--output_path", str(out_csv)], cwd=stage, capture_output=True,
            text=True, encoding="utf-8", errors="replace", env=_child_env(),
            timeout=CHILD_TIMEOUT_SEC)
        if proc.returncode != 0 or not out_csv.exists():
            res["как считано"] = "прогон для метрики не удался"
            return {**res, "статус": "ПАДЕНИЕ на замере метрики",
                    "лог": (proc.stderr or proc.stdout or "")[-800:]}
        sub = pd.read_csv(out_csv)
        pred = sub["result"].str.endswith("<вердикт>не бан").astype(int).to_numpy()
        for avg in ("binary", "macro"):
            sc = competition_score(df[cfg["data"]["target_col"]].to_numpy(), pred,
                                   df[cfg["data"]["category_col"]].to_numpy(), average=avg)
            res[f"локально {avg}"] = round(sc["mean"], 4)
        out_csv.unlink(missing_ok=True)

    # --- упаковка ---
    out_dir.mkdir(parents=True, exist_ok=True)
    archive = out_dir / f"{variant}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in sorted(stage.rglob("*")):
            if not f.is_file() or f.name.startswith("_probe"):
                continue
            if f.suffix in (".pyc", ".pyo") or "__pycache__" in f.parts:
                continue
            z.write(f, f.relative_to(stage))
    size_gb = archive.stat().st_size / 1024 ** 3
    res["архив"] = str(archive.relative_to(ROOT))
    res["размер, МБ"] = round(size_gb * 1024, 1)
    res["образ"] = meta["image"]
    if size_gb > ARCHIVE_LIMIT_GB:
        res["статус"] = "АРХИВ БОЛЬШЕ 5 ГБ"
    elif res.get("статус") in (None, "ГОТОВ"):
        res["статус"] = "ГОТОВ"

    # Проверять надо в ТОМ образе, который вариант объявил своим: решение на
    # эмбеддингах заявлено под базовый образ именно потому, что не требует наших
    # библиотек, и прогон в нашем образе этого бы не доказал. Флаг --docker
    # подменяет только наш образ, базовый берётся из docker/images.json.
    #
    # ⚠ Прогон в контейнере обязателен, а не опционален. Отправка `dup_text_only`
    # (15.08) собиралась без флага --docker, проверка молча не выполнилась, статус всё
    # равно встал в «ГОТОВ» — и решение обнулилось на платформе: в базовом образе нет
    # pyarrow, а таблица подписей лежала в parquet. Локальный прогон этого не видит,
    # потому что у нас pyarrow есть. Теперь непроверенный архив так и называется.
    wanted = spec.get("image", "base")
    image = docker_image if wanted == "custom" else IMAGES.get(wanted, wanted)
    if wanted == "custom" and not docker_image:
        res["в контейнере"] = "не выполнена: нашему образу нужен флаг --docker"
        res["статус"] = "НЕ ПРОВЕРЕН В КОНТЕЙНЕРЕ"
    else:
        res["проверено в образе"] = image
        res["в контейнере"] = verify_in_docker(stage, check_csv, image)
        if res["в контейнере"] != "ОК":
            res["статус"] = "СБОЙ В КОНТЕЙНЕРЕ"
    shutil.rmtree(stage, ignore_errors=True)

    # ⚠ Проверка выше идёт на ДЕСЯТИ товарах, где зрение пропускается намеренно, и в
    # мягких условиях. Она не отвечает на единственный важный вопрос: вынесет ли вердикт
    # заявленная модель. Отправка на эмбеддингах прошла эту проверку и вернула на
    # лидерборде балл решения-заглушки, потому что вердикт вынесли правила.
    # Поэтому решение со зрением не имеет права называться готовым, пока не отработало
    # на выборке выше отладочного порога в условиях проверяющей системы.
    if res.get("статус") == "ГОТОВ" and works_with_photos(spec):
        from verify_vision_in_docker import check as hostile_check

        # У зонда-измерителя вердикт по определению выносит не модель, а сам зонд:
        # он кодирует замер долей категории БАД. Ждём от него собственного источника —
        # строгость проверки при этом не теряется, подменённый источник всё так же
        # виден. Для обычных отправок ожидание прежнее: только модель.
        model_name = str(spec["model"])
        expect = (f"зонд-замер {model_name.split(':', 2)[2]}"
                  if model_name.startswith("probe:measure:") else "модель")
        probe = hostile_check(variant, items=HOSTILE_ITEMS, tag="сборка",
                              expect_source=expect,
                              expect_vision=expects_vision(spec))
        res["боевая проверка"] = probe.get("итог", "не выполнена")
        res["вердикт от"] = probe.get("источник вердикта", "—")
        if probe.get("итог") != "ОК":
            res["статус"] = f"НЕ ПРОШЁЛ БОЕВУЮ ПРОВЕРКУ: {probe.get('итог')}"
    return res


def verify_in_docker(stage: Path, check_csv: Path, image: str) -> str:
    """Прогон решения ВНУТРИ образа — единственная честная проверка перед отправкой.

    Локальное окружение отличается от контейнера версиями библиотек и набором пакетов,
    и именно там всплывают неприятности вроде отсутствующего easyocr. Проверка идёт на
    той же выборке с чужой раскладкой, что и локальная.
    """
    out_dir = stage.parent / "_docker_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    args = ["docker", "run", "--rm",
            "-v", f"{stage}:/app", "-v", f"{check_csv.parent}:/data",
            "-v", f"{out_dir}:/out", "-w", "/app"]
    # Модели: у проверяющей системы они смонтированы в /shared_models, у нас лежат в
    # кэше huggingface. Подкладываем кэш по тому же пути — иначе ветки, которым нужна
    # модель, в контейнере молча отваливаются, и локально это не видно. Именно так
    # провал решения на эмбеддингах остался незамеченным до отправки.
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    if hub.is_dir():
        args += ["-v", f"{hub}:/shared_models:ro",
                 "-e", "SHARED_MODELS_PATH=/shared_models",
                 "-e", "HF_HUB_OFFLINE=1"]
    for gpu in (["--gpus", "all"], []):
        proc = subprocess.run(args[:3] + gpu + args[3:] + [
            image, "python", "-u", "run.py",
            "--test_data_path", "/data/" + check_csv.name,
            "--output_path", "/out/submit.csv"],
            capture_output=True, text=True, encoding="utf-8", errors="replace")
        if proc.returncode == 0:
            break
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
        return "ПАДЕНИЕ: " + " | ".join(tail)
    produced = out_dir / "submit.csv"
    if not produced.exists():
        return "файл ответа не создан"
    sub = pd.read_csv(produced)
    errors = validate_submission(sub)
    produced.unlink(missing_ok=True)
    return "ОК" if not errors else "ФОРМАТ: " + "; ".join(errors)


# Колонки, которые заполняются РУКАМИ и которые пересборка обязана сохранить.
MANUAL_COLS = ("public_f1", "private_f1", "дата отправки")
# Порядок колонок в журнале: сначала что это, потом чем оно оказалось, потом техника.
COLUMN_ORDER = [
    "variant", "статус", "дата сборки",
    "локально binary", "локально macro", "AUC", "как считано",
    "public_f1", "private_f1", "дата отправки",
    "модель", "доля «не бан»", "оценка private, мин", "размер, МБ",
    "время check, с", "время полного прогона, с", "товаров", "длина комментария",
    "архив", "комментарий к сабмиту", "обоснование",
]
# Старые названия колонок, оставшиеся от прежних сборок.
RENAMES = {"отправлено": "дата отправки"}
DROP_COLS = ("комментарий",)  # дубликат «комментария к сабмиту» из первой версии


def update_registry(rows: list[dict], path: Path) -> pd.DataFrame:
    """Журнал отправок. public_f1, private_f1 и дату отправки заполняешь руками.

    Пересборка перезаписывает только вычисляемые колонки: ручные значения переносятся
    из прежнего файла по имени варианта, иначе один запуск сборщика стирал бы историю
    результатов с лидерборда.
    """
    new = pd.DataFrame(rows)
    new["дата сборки"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    for col in MANUAL_COLS:
        new[col] = ""

    if path.exists():
        old = pd.read_csv(path, dtype=str).fillna("")
        old = old.rename(columns=RENAMES).drop(columns=list(DROP_COLS), errors="ignore")
        old = old.loc[:, ~old.columns.duplicated()]
        keep = {r["variant"]: r for _, r in old.iterrows()}
        for i, v in new["variant"].items():
            for col in MANUAL_COLS:
                prev = str(keep.get(v, {}).get(col, "")).strip() if v in keep else ""
                if prev:
                    new.at[i, col] = prev
        rest = old[~old["variant"].isin(new["variant"])]
        cols = list(dict.fromkeys(list(new.columns) + list(rest.columns)))
        new = pd.concat([rest.reindex(columns=cols).fillna("").astype(str),
                         new.reindex(columns=cols).fillna("").astype(str)],
                        ignore_index=True)

    ordered = [c for c in COLUMN_ORDER if c in new.columns]
    new = new[ordered + [c for c in new.columns if c not in ordered]]
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(path, index=False, encoding="utf-8-sig")
    return new


def _log_builds(cfg: dict, rows: list[dict]) -> None:
    """Каждая собранная отправка сразу заводит прогон в журнале экспериментов.

    Публичный результат вписывается в registry.csv руками и доезжает позже через
    scripts/sync_public_scores.py — прогон при этом дополняется, а не дублируется.
    """
    mc = cfg.get("mlflow", {})
    if not mc.get("enabled", False):
        return
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        from qc26.tracking import M_AUC, M_BINARY, M_MACRO, metric_name

        uri = str(mc["tracking_uri"])
        if uri.startswith("sqlite:///"):
            uri = "sqlite:///" + str(
                resolve_path(uri.removeprefix("sqlite:///"))).replace("\\", "/")
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(mc.get("experiment", "default"))
        client = MlflowClient()
        exp = client.get_experiment_by_name(mc.get("experiment", "default"))
    except Exception as e:
        print(f"⚠ трекинг недоступен: {e}")
        return

    for r in rows:
        if r.get("статус") != "ГОТОВ":
            continue
        # Имена метрик — те же, что у обучения и оценки, иначе в таблице MLflow
        # получаются отдельные столбцы с дырами и прогоны не сравнить взглядом.
        metrics = {k: v for k, v in {
            M_BINARY: r.get("локально binary"),
            M_MACRO: r.get("локально macro"),
            M_AUC: r.get("AUC"),
            "runtime_private_min": r.get("оценка private, мин"),
            "archive_mb": r.get("размер, МБ"),
            "share_not_ban": r.get("доля «не бан»"),
        }.items() if isinstance(v, (int, float))}
        # Прогон отправки один на вариант: пересборка обновляет его, а не плодит копии,
        # иначе публичный результат потом не к чему будет привязать.
        found = client.search_runs(
            [exp.experiment_id],
            filter_string=f"tags.submission_variant = '{r['variant']}'", max_results=1)
        run_id = (found[0].info.run_id if found else client.create_run(
            exp.experiment_id, run_name=f"submit_{r['variant']}",
            tags={"stage": "submission", "submission_variant": r["variant"]}).info.run_id)
        for k, v in metrics.items():
            client.log_metric(run_id, metric_name(k), float(v))
        # model/archive — ПАРАМЕТРЫ, как и у прогонов обучения: в MLflow параметры и
        # теги живут в разных столбцах, и одно и то же поле не должно попадать в оба.
        for key, val in (("model", r.get("модель")), ("archive", r.get("архив")),
                         ("metric_source", r.get("как считано")),
                         ("variant", r.get("variant"))):
            if val:
                try:
                    client.log_param(run_id, key, str(val)[:490])
                except Exception:
                    pass  # параметр уже записан прежней сборкой и неизменяем
        note = r.get("комментарий к сабмиту")
        if note:
            client.set_tag(run_id, "mlflow.note.content", note)
            client.set_tag(run_id, "comment", note[:490])
        client.set_terminated(run_id, "FINISHED")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", action="append", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--docker", nargs="?", const="ecup26-quality:local", default=None,
                    help="дополнительно прогнать архив внутри указанного образа")
    args = ap.parse_args()

    cfg = load_config("configs/data.yaml", "configs/baseline.yaml")
    names = list(VARIANTS) if args.all or not args.variant else args.variant
    lock = BuildLock(resolve_path(cfg["paths"]["artifacts_dir"]) / "build.lock")
    with lock:

        art = resolve_path(cfg["paths"]["artifacts_dir"])
        out_dir = art / "submissions"
        out_dir.mkdir(parents=True, exist_ok=True)

        full_csv = resolve_path(cfg["paths"]["data_csv"])
        check_csv = make_foreign_layout(cfg, art / "_checkset", n=10)

        rows = []
        for name in names:
            if name not in VARIANTS:
                print(f"⚠ неизвестный вариант {name}")
                continue
            print(f"\n=== сборка «{name}» ===", flush=True)
            res = build(name, VARIANTS[name], cfg, art / "_stage" / name, out_dir,
                        check_csv, full_csv, docker_image=args.docker)
            res["комментарий к сабмиту"] = VARIANTS[name]["note"]
            res["обоснование"] = VARIANTS[name]["why"]
            rows.append(res)
            print("   " + " | ".join(f"{k}: {v}" for k, v in res.items()
                                     if k not in ("комментарий к сабмиту", "обоснование", "лог")))
            print(f"   комментарий к сабмиту: {VARIANTS[name]['note']}")
            if "лог" in res:
                print("   лог:\n" + str(res["лог"]))

        reg = update_registry(rows, out_dir / "registry.csv")
        _log_builds(cfg, rows)
        print(f"\nжурнал отправок: {out_dir / 'registry.csv'}")
        print("Колонки public_f1 / private_f1 / отправлено заполняешь руками — пересборка "
              "их не затирает.")
        cols = [c for c in ("variant", "статус", "локально binary", "локально macro",
                            "оценка private, мин", "размер, МБ", "public_f1") if c in reg.columns]
        print(reg[cols].to_string(index=False))
        print("\nКомментарии к отправкам (их же вставлять в поле комментария на платформе):")
        for r in rows:
            if r.get("статус") == "ГОТОВ":
                print(f"  [{r['variant']}] {r['комментарий к сабмиту']}")


if __name__ == "__main__":
    main()
