"""Точка входа решения — так его запускает проверяющая система.

    python run.py --test_data_path test.csv --output_path submit.csv

Путь к изображениям выводится из пути к csv (`<папка csv>/images`), как в примере
организаторов; лишний уровень вложенности определяется автоматически.

В контейнере файловая система только для чтения и нет интернета: ничего не скачиваем
и не пишем никуда, кроме --output_path. Формат ответа проверяем перед записью — его
нарушение обнуляет решение целиком, поэтому запасной путь обязателен: если модель не
загрузилась или упала, ответ всё равно формируется (по правилам-уликам), лишь бы файл
был валиден и содержал строку для каждого товара.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

# Печать НИКОГДА не должна ронять решение. Кодировка вывода зависит от окружения: под
# Windows это может оказаться cp1251, в которой нет ни стрелки, ни длинного тире, и
# обычный print уронит прогон уже после того, как всё посчитано. Ставим utf-8 и замену
# непредставимых символов — потеря значка в логе несопоставима с потерей попытки.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # поток без reconfigure — не страшно
        pass

from qc26.config import load_config, resolve_path  # noqa: E402
from qc26.data import load_cards  # noqa: E402
from qc26.explain.templates import explain_frame  # noqa: E402
from qc26.inference.format import build_result, submission_stats, validate_submission  # noqa: E402
from qc26.rules import rule_guess  # noqa: E402

DEFAULT_MODEL = "artifacts/models/tfidf_logreg.joblib"
DATA_PATH: list[str] = []  # путь к тестовому csv: от него ищутся изображения
T0: list[float] = []  # момент старта прогона: от него считается остаток бюджета


# Лимиты проверки: check 10 товаров / 3 мин, public ~1600 / 20 мин, private ~3800 / 40 мин.
# Какой именно набор подан, решение не знает — определяем по числу строк.
TIME_LIMITS = ((100, 180.0), (2200, 1200.0), (10 ** 9, 2400.0))
# Доля лимита, отдаваемая работе с фотографиями; остальное — загрузка модели,
# предсказание и запись ответа.
# ⚠ Поднято с 0.72 после замеров НА ПЛАТФОРМЕ (18.08, отправка со связкой):
# распознавание с двух кадров 0.200 с/товар, построение векторов 0.298 с/товар, вместе
# 31.4 минуты на приватных 3789. При доле 0.72 бюджет составлял 28.8 минуты, и решение
# останавливалось, хотя до лимита в 40 минут оставался запас: накладные расходы по
# замеру всего около двух минут (модель грузится за 4 с, вердикт линейной мгновенный).
BUDGET_SHARE = 0.82


def _ocr_budget(n_items: int, sub: dict) -> float:
    """Сколько секунд можно потратить на чтение фото при данном объёме входа.

    Считать бюджет фиксированным числом нельзя: локально мы прогоняем 12 971 товар, а
    на проверке будет 1600 или 3800, и прогноз времени отличается в разы. С фиксированным
    бюджетом чтение отменялось бы там, где на самом деле укладывается.
    """
    limit = next(lim for bound, lim in TIME_LIMITS if n_items <= bound)
    budget = limit * BUDGET_SHARE
    cap = sub.get("ocr_budget_sec")
    return min(budget, float(cap)) if cap else budget


CHECK_STAGE_MAX_ITEMS = 100


def _skip_vision_on_check(n_items: int) -> bool:
    """Отладочная стадия проверки — десять товаров и три минуты на всё.

    Загрузка модели на два миллиарда параметров сама по себе может не уложиться в этот
    лимит, а смысла в зрении там нет: метрика на отладочных примерах не считается, важно
    лишь отдать корректный файл. Поэтому на крошечном входе зрение пропускается молча —
    это не отказ, а экономия. Иначе решение падало бы на первой же стадии, и проверка
    прекращалась бы, не дойдя до оцениваемых наборов.
    """
    return n_items <= CHECK_STAGE_MAX_ITEMS


def _smooth_by_duplicates(cfg: dict, df, score):
    """Скор каждого товара = среднее по группе его почти-дубликатов.

    Группы строятся ТОЛЬКО по тексту карточки: изображения тут не нужны, а лишние
    минуты в контейнере нужны ещё меньше — приватная стадия теснее публичной на 10%.

    ⚠ При любой неудаче возвращаем исходные скоры, а не падаем: сглаживание — это
    улучшение поверх работающего решения, и терять из-за него всю отправку нельзя.
    Сообщение печатается всегда, чтобы по журналу было видно, сработало оно или нет.
    """
    import numpy as np

    score = np.asarray(score, dtype=np.float64)
    # ⚠ Только те категории, где сглаживание доказано полезным. Замер на отложенном
    # фолде: в редкой категории AUC 0.9113 -> 0.9729 и в верхних 37 верных 24 -> 32,
    # а в БАД 0.9577 -> 0.9524, то есть чуть хуже. Применять ко всему подряд значило бы
    # покупать шесть пунктов в одной категории ценой полупункта в другой без нужды.
    want = cfg.get("submission", {}).get("smooth_duplicates")
    only = None if want is True else {str(c) for c in want} if isinstance(want, list) else None
    try:
        from qc26.groups import build_groups

        groups = build_groups(df, id_col=cfg["data"]["id_col"]).to_numpy()
        if only is not None:
            cat = df[cfg["data"]["category_col"]].astype(str).to_numpy()
            # Товары вне выбранных категорий делаем каждый своей группой — тогда их
            # среднее равно им самим, и общий код остаётся одним на всех.
            groups = np.where(np.isin(cat, list(only)), groups, -np.arange(1, len(df) + 1))
        smoothed = score.copy()
        order = np.argsort(groups, kind="stable")
        g_sorted = groups[order]
        # Границы групп в отсортированном порядке — без словарей и циклов по товарам.
        starts = np.flatnonzero(np.r_[True, g_sorted[1:] != g_sorted[:-1]])
        sums = np.add.reduceat(score[order], starts)
        sizes = np.diff(np.r_[starts, len(order)])
        means = np.repeat(sums / sizes, sizes)
        smoothed[order] = means
        changed = int((sizes > 1).sum())
        print(f"сглаживание по дубликатам: групп с повторами {changed} из "
              f"{len(sizes)}, затронуто товаров {int(sizes[sizes > 1].sum())}", flush=True)
        return smoothed
    except Exception as exc:                        # noqa: BLE001
        print(f"⚠ сглаживание по дубликатам НЕ выполнено ({type(exc).__name__}: {exc}) — "
              f"вердикт по исходным скорам", flush=True)
        return score


def _vision_failed(cfg: dict, what: str) -> None:
    """Как вести себя, если ветка со зрением не запустилась.

    Тихий откат на текстовую модель кажется безопасным, но обходится дороже падения:
    решение отдаёт валидный ответ, попытка засчитывается, а результат оказывается
    точным повтором уже известного числа — и мы ничего не узнаём. Упавшее решение
    попытку не тратит. Поэтому у вариантов, ради зрения и затеянных, поведение по
    умолчанию — падать громко; мягкий откат оставлен для финальных отправок, где важнее
    отдать хоть что-то.
    """
    mode = str(cfg.get("submission", {}).get("on_vision_failure", "fail"))
    if mode == "fail":
        raise SystemExit(
            f"ОСТАНОВ: {what} не состоялось, а решение построено вокруг него.\n"
            "Отдавать вместо этого текстовую модель бессмысленно: результат повторит "
            "уже известный, а попытка будет потрачена.\n"
            "Чтобы вместо остановки использовался запасной путь, поставьте в конфиге "
            "отправки on_vision_failure: fallback")
    print(f"⚠ {what} не состоялось — работаю запасным путём", flush=True)


def _resolve_ocr(cfg: dict, df: pd.DataFrame, elapsed: float):
    """Распознанный с фото текст, если оно по карману; иначе None.

    Бюджет считается от оставшегося времени приватного прогона с запасом: лучше отдать
    решение без распознавания, чем не отдать вовсе.
    """
    sub = cfg.get("submission", {})
    if not sub.get("use_ocr"):
        return None
    if _skip_vision_on_check(len(df)):
        print(f"отладочная стадия ({len(df)} товаров) — чтение фото пропущено", flush=True)
        return None
    from qc26.data import images_root
    from qc26.inference.ocr_runtime import (default_weights_dir, run_ocr,
                                            run_vlm_transcript)

    root = Path(__file__).resolve().parent
    budget = _ocr_budget(len(df), sub) - elapsed
    if budget <= 60:
        return None
    engine = str(sub.get("ocr_engine", "vlm"))
    if engine == "vlm":
        # В базовом образе организаторов нет easyocr — читаем упаковку моделью из
        # /shared_models, она доступна решению без сборки своего образа.
        return run_vlm_transcript(cfg, df, budget_sec=budget)
    return run_ocr(cfg, df, images_root(cfg, DATA_PATH[0]), budget_sec=budget,
                   weights_dir=default_weights_dir(root))


def _predict(cfg: dict, df: pd.DataFrame, model: str, ocr=None, emb=None):
    """Вердикт по указанному решению; при любой ошибке — откат на правила-улики.

    Значения --model:
      путь к .joblib — обученная модель (основной путь);
      rules         — только правила-улики, без sklearn (страховка на случай
                      несовместимости версий библиотек в образе);
      const:0 / const:1 — постоянный вердикт (зонд для калибровки лидерборда:
                      по его публичному значению видно, какое из двух прочтений
                      метрики считает проверяющая система).
    """
    if model == "rules":
        return df.apply(lambda r: rule_guess(r), axis=1).to_numpy(), "правила"
    if model == "probe:layout":
        # Логов с платформы мы не видим, поэтому находки кодируются в самом ответе.
        # В БАД «не бан» ставится там, где найдены изображения товара; в редкой
        # категории — там, где нашлась модель в /shared_models. Публичный балл
        # раскладывается однозначно: 0.3972 — нашли всё, 0.3644 — только картинки,
        # 0.0328 — только модели, 0.0000 — ничего.
        from qc26.data import image_paths, images_root
        from qc26.models.vlm_backends import resolve_model_id

        root = images_root(cfg, DATA_PATH[0]) if DATA_PATH else None
        wanted = cfg["embeddings"]["model_id"]
        model_found = int(resolve_model_id(wanted) != wanted)
        print(f"зонд раскладки: корень изображений {root}, "
              f"модель {wanted} {'найдена' if model_found else 'НЕ найдена'}", flush=True)

        cat_col = cfg["data"]["category_col"]
        idc = cfg["data"]["id_col"]
        preds = np.zeros(len(df), dtype=int)
        for k, (_, row) in enumerate(df.iterrows()):
            if str(row[cat_col]) == "БАД":
                preds[k] = int(bool(image_paths(root, str(row[idc]))))
            else:
                preds[k] = model_found
        found_img = int(preds[(df[cat_col] == "БАД").to_numpy()].sum())
        print(f"зонд раскладки: изображения найдены у {found_img} товаров БАД", flush=True)
        return preds, "зонд раскладки"
    if model == "const:supplement":
        # «не бан» только категории БАД: обнуляет вклад редкой категории и делает
        # публичное значение прямой мерой доли позитивов в БАД
        preds = (df[cfg["data"]["category_col"]].astype(str) == "БАД").astype(int).to_numpy()
        return preds, "постоянный вердикт по категории"
    if model.startswith("probe:measure:"):
        # Замер по закрытым данным, закодированный в вердиктах: доля категории БАД,
        # помеченная «не бан», задаётся уровнем замера, редкая категория целиком уходит
        # в «бан». Публичный балл читается по таблице уровней (qc26/inference/probe.py).
        from qc26.data import images_root
        from qc26.inference import probe as probe_mod

        what = model.split(":", 2)[2]
        # ⚠ Отладочная стадия жюри — десять товаров, и редкой категории среди них может
        # не оказаться вовсе. Замер там бессмыслен (стадия не оценивается), а падение
        # останавливает проверку целиком. Отдаём валидный ответ нулевого уровня.
        if _skip_vision_on_check(len(df)):
            print(f"отладочная стадия ({len(df)} товаров) — замер пропущен", flush=True)
            return probe_mod.encode(cfg, df, 0), f"зонд-замер {what} (отладочная стадия)"
        scores = None
        if what == "rare_predpos":
            # Замеру нужны вердикты самой модели — считаем их той же дорогой, что и
            # обычная отправка, включая распознавание с фотографий.
            src = cfg.get("submission", {}).get("probe_model")
            if not src:
                raise RuntimeError("зонду не указана модель в probe_model")
            scores, source = _predict(cfg, df, src, ocr=ocr, emb=emb)
            # ⚠ _predict при любой ошибке молча откатывается на правила. Для обычной
            # отправки это страховка, а для зонда — ложь: наружу ушёл бы замер правил
            # под видом замера модели, и прочитали бы мы его как факт про модель.
            if source != "модель":
                raise RuntimeError(
                    f"зонду нужны вердикты модели, а их вынес источник «{source}» — "
                    f"замер отменён, чтобы не передать наружу неверное число")
        root = images_root(cfg, DATA_PATH[0]) if DATA_PATH else None
        dup_table = None
        if what == "rare_dupcov":
            table_path = resolve_path(cfg.get("submission", {}).get(
                "dup_index", "artifacts/dup/dup_index.csv.gz"))
            if not table_path.exists():
                raise RuntimeError(f"таблица подписей не найдена: {table_path}")
            from qc26.inference.duplicates import load_table
            dup_table = load_table(table_path)
        value, level = probe_mod.measure(what, cfg, df, scores=scores, images_root=root,
                                         dup_table=dup_table)
        print(f"зонд «{what}»: замер {value:.4f} -> уровень {level} "
              f"(ожидаемый публичный балл {probe_mod.expected_score(level):.4f})", flush=True)
        return probe_mod.encode(cfg, df, level), f"зонд-замер {what}"
    if model.startswith("const:"):
        value = int(model.split(":", 1)[1])
        return np.full(len(df), value, dtype=int), f"постоянный вердикт {value}"
    if model.startswith("vlmblend:"):
        # Связка линейной модели с дообученной VLM. Оба источника считаются здесь:
        # модели нужен корень картинок, а он известен только прогону.
        import joblib

        from qc26.data import images_root
        from qc26.models.vlm import VlmScorer, make_texts
        from qc26.models.vlm_blend import VlmBlendModel  # noqa: F401

        weights_path, adapter_path = model.split(":", 2)[1:]
        loaded = joblib.load(resolve_path(weights_path))
        adapter = resolve_path(adapter_path)
        if not adapter.exists():
            raise RuntimeError(f"адаптер не найден: {adapter}")
        root = images_root(cfg, DATA_PATH[0]) if DATA_PATH else None
        need_images = not _skip_vision_on_check(len(df))
        text_score = np.asarray(loaded.text_model.predict_proba(df, ocr), dtype=np.float64)
        packed = cfg.get("submission", {}).get("packed_base_path")
        if packed:
            packed_dir = resolve_path(packed)
            if not packed_dir.is_dir():
                raise RuntimeError(f"упакованная база не найдена: {packed_dir}")
            cfg = {**cfg, "vlm": {**cfg["vlm"], "packed_base_path": str(packed_dir)}}
            print(f"базовая модель из архива: {packed_dir}", flush=True)
        scorer = VlmScorer(cfg, model_id=str(adapter), images_root_path=root,
                           require_images=need_images)
        vlm_score = scorer.score(df, make_texts(df, cfg, ocr), show_progress=False)
        # ⚠ Третий источник — векторы фотографий, если у связки есть такая ветка.
        # Их строит прогон и передаёт сюда; без ветки аргумент просто игнорируется.
        img_score = None
        if getattr(loaded, "image_models", None):
            if emb is None:
                raise RuntimeError("связке нужны векторы фотографий, а они не построены")
            got = emb.get("image") if isinstance(emb, dict) else emb
            if got is None:
                raise RuntimeError(f"нужен набор «image», пришли: {sorted(emb)}")
            img_score = loaded.image_score(df, got)
            print(f"третий источник: векторы фотографий, вес {loaded.image_weight}",
                  flush=True)
        score = loaded.blend(df, text_score, vlm_score, img_score)
        # ⚠ Сглаживание по почти-дубликатам ВНУТРИ проверяемой выборки. В наших данных
        # метки совпадают в 96.7% групп почти-одинаковых карточек, значит и скоры у них
        # должны быть близки, а расхождение — это шум модели. Заменяем скор каждого
        # товара средним по его группе: усреднение снижает разброс и поднимает наверх
        # тех, кого одиночный прогон недооценил. Полноту в редкой категории мы теряем
        # именно на таких.
        #
        # ⚠ Это НЕ поиск дубликатов обучающей выборки в тестовой — та гипотеза была
        # опровергнута отправкой 15.08. Здесь группы строятся только внутри теста, по
        # тексту карточки, и меткам взяться неоткуда.
        if cfg.get("submission", {}).get("smooth_duplicates"):
            score = _smooth_by_duplicates(cfg, df, score)
        preds = loaded.verdict(df, score)
        print(f"связка: веса {loaded.vlm_weight}, пороги {loaded.thresholds}, "
              f"доля «не бан» {preds.mean():.3f}", flush=True)
        return preds, "модель"
    if model.startswith("vlm:"):
        # Дообученная VLM как решение: адаптер LoRA лежит в архиве, базовая модель —
        # в общем каталоге проверяющей системы. Вердикт по логитам первого токена
        # ответа «Да»/«Нет».
        from qc26.data import images_root
        from qc26.models.vlm import VlmScorer, make_texts

        adapter = resolve_path(model.split(":", 1)[1])
        if not adapter.exists():
            raise RuntimeError(f"адаптер не найден: {adapter}")
        sub = cfg.get("submission", {})
        thresholds = sub.get("thresholds") or {}
        if not thresholds:
            raise RuntimeError("решению на VLM нужны пороги в конфиге отправки")
        # ⚠ Корень изображений — от пути к данным ПРОГОНА, а не из конфига. Скорер
        # подставляет белый кадр вместо ненайденного, поэтому при неверном корне
        # вердикт вынесся бы по пустым картинкам без единой ошибки в журнале.
        root = images_root(cfg, DATA_PATH[0]) if DATA_PATH else None
        # На отладочной стадии зрение пропускается штатно, требовать кадры там нельзя.
        need_images = not _skip_vision_on_check(len(df))
        # ⚠ Базовая модель может ехать в архиве упакованной: адаптер помнит её имя,
        # но в каталоге проверяющей системы её нет, а сети в контейнере нет тоже.
        packed = sub.get("packed_base_path")
        if packed:
            packed_dir = resolve_path(packed)
            if not packed_dir.is_dir():
                raise RuntimeError(f"упакованная база не найдена: {packed_dir}")
            cfg = {**cfg, "vlm": {**cfg["vlm"], "packed_base_path": str(packed_dir)}}
            print(f"базовая модель из архива: {packed_dir}", flush=True)
        scorer = VlmScorer(cfg, model_id=str(adapter), images_root_path=root,
                           require_images=need_images)
        texts = make_texts(df, cfg, ocr)
        scores = scorer.score(df, texts, show_progress=False)
        cat_col = cfg["data"]["category_col"]
        thr = df[cat_col].astype(str).map(thresholds)
        if thr.isna().any():
            missing = sorted(set(df.loc[thr.isna(), cat_col].astype(str)))
            raise RuntimeError(f"нет порога для категорий: {missing}")
        preds = (scores >= thr.to_numpy()).astype(int)
        print(f"VLM {adapter.name}: пороги {thresholds}, "
              f"доля «не бан» {preds.mean():.3f}", flush=True)
        return preds, "модель"
    try:
        import joblib

        from qc26.models.text_baseline import TextBaseline  # noqa: F401

        path = resolve_path(model)
        loaded = joblib.load(path)
        override = cfg.get("submission", {}).get("thresholds") or {}
        if override:
            loaded.thresholds = {**loaded.thresholds, **override}
            print(f"порог переопределён конфигом отправки: {override}", flush=True)
        # Умеет ли модель принимать матрицу эмбеддингов — видно по её же сигнатуре.
        # Прежде признаком служило поле builders, но оно есть у ВСЕХ выгруженных весов,
        # включая чисто текстовые: линейная модель спотыкалась о проверку «ждёт
        # эмбеддинги», хотя они ей не нужны, и вердикт выносили правила.
        import inspect

        # Признаки-ответы языковой модели: считаются на прогоне и кладутся в модель.
        # Офлайн-файл привязан к идентификаторам наших товаров и на закрытой выборке
        # бесполезен — без этого шага решение падало на выборке приватного размера.
        # ⚠ Запасной модели анкета не нужна: она обучена без неё. Проверять это надо
        # ПЕРЕД входом в ветку — откат по распознаванию мог переключить модель раньше,
        # и тогда решение либо уходило в бесконечную рекурсию (каждый виток грузил
        # веса, память росла до потолка, проверка из десяти товаров висела вечно),
        # либо падало с требованием запасной модели, будучи уже в ней.
        _fb = str(cfg.get("submission", {}).get("fallback_model") or "")
        if cfg.get("submission", {}).get("use_qa") and str(model) != _fb:
            from qc26.inference.qa_runtime import build_qa_features

            fb = cfg.get("submission", {}).get("fallback_model")
            # ⚠ Откат зовёт _predict заново, а use_qa в конфиге остаётся — без сравнения
            # с текущей моделью запасная модель снова попадала бы сюда и откатывалась
            # на себя же. Это давало БЕСКОНЕЧНУЮ РЕКУРСИЮ: каждый виток грузил веса,
            # память росла до потолка, и проверка из десяти товаров висела вечно.
            if _skip_vision_on_check(len(df)):
                if fb and str(fb) != str(model):
                    print("отладочная стадия — беру запасную модель без анкеты",
                          flush=True)
                    return _predict(cfg, df, fb, ocr=ocr, emb=emb)
                raise RuntimeError(
                    "решению нужны ответы языковой модели, на отладочной стадии они не "
                    "считаются, а запасная модель без них не задана — укажите "
                    "fallback_model в описании варианта")
            left = _ocr_budget(len(df), cfg.get("submission", {})) - (time.time() - T0[0])
            qa = build_qa_features(cfg, df, budget_sec=max(0.0, left))
            if qa is None:
                _vision_failed(cfg, "ответы языковой модели на вопросы о товаре")
                if fb and str(fb) != str(model):
                    print(f"переключаюсь на запасную модель без анкеты: {fb}", flush=True)
                    return _predict(cfg, df, fb, ocr=ocr, emb=emb)
            else:
                qa = qa.set_index(qa[cfg["data"]["id_col"]].astype(str))
                loaded.qa_runtime = qa

        takes_emb = "emb" in inspect.signature(loaded.predict).parameters
        # Нужны ли эмбеддинги ЭТОМУ решению, знает конфиг отправки, а не файл весов.
        wants_emb = bool(cfg.get("submission", {}).get("use_embeddings"))
        if takes_emb:
            if wants_emb and emb is None:
                raise RuntimeError("модель ждёт эмбеддинги, а они не построены")
            # ⚠ Только по именам: у разных наших моделей порядок аргументов
            # различался, и матрица эмбеддингов уходила в аргумент
            # распознанного текста. Ошибки при этом нет — модель бросает
            # исключение, вердикт молча выносят правила, а на лидерборде
            # это выглядит как балл решения-заглушки.
            preds = loaded.predict(df, emb=emb, ocr=ocr)
        else:
            preds = loaded.predict(df, ocr)
        print(f"модель: {path.name}, пороги {loaded.thresholds}", flush=True)
        return preds, "модель"
    except Exception:
        traceback.print_exc()
        # ⚠ Молчаливый откат на правила — самая дорогая ошибка этого проекта. Он отдаёт
        # валидный ответ, попытка засчитывается, а балл оказывается баллом решения-
        # заглушки (0.4308), который мы потом принимаем за результат модели. Упавшая
        # отправка попытку НЕ тратит. Поэтому по умолчанию падаем, а откат включается
        # явным ключом и только для финальной отправки, где важнее отдать хоть что-то.
        if str(cfg.get("submission", {}).get("on_model_failure", "fail")) == "fail":
            raise SystemExit(
                "ОСТАНОВ: модель не отработала (трассировка выше).\n"
                "Отдавать вместо неё вердикт по правилам бессмысленно: балл будет "
                "баллом решения-заглушки, а попытка потрачена.\n"
                "Для финальной отправки поставьте в конфиге on_model_failure: fallback")
        print("ОТКАТ: модель недоступна, вердикт по правилам-уликам", flush=True)
        return df.apply(lambda r: rule_guess(r), axis=1).to_numpy(), "правила"


def main() -> None:
    ap = argparse.ArgumentParser(description="Контроль качества карточек товаров")
    # В условиях аргументы названы то через подчёркивание (пример baseline и раздел
    # «Формат выходного файла»), то через дефис («--output-path» в описании входных
    # аргументов). Принимаем оба написания: неверно разобранный аргумент обнуляет
    # решение целиком ещё до подсчёта метрики.
    ap.add_argument("--test_data_path", "--test-data-path", "-i", dest="test_data_path",
                    required=True)
    ap.add_argument("--output_path", "--output-path", "-o", dest="output_path",
                    required=True)
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--model", default=None,
                    help="путь к .joblib, либо rules, либо const:0 / const:1")
    args = ap.parse_args()

    t0 = time.time()
    T0.append(t0)
    DATA_PATH.append(args.test_data_path)
    cfg = load_config(args.config, "configs/baseline.yaml", "configs/zoo.yaml",
                      "configs/vlm_sft.yaml", "configs/submission.yaml")
    # Проверяющая система запускает ровно `python -u run.py` без своих аргументов,
    # поэтому выбор решения для конкретной отправки живёт в configs/submission.yaml,
    # который кладёт сборщик архива. Явный --model (локальные прогоны) главнее.
    model = (args.model or os.environ.get("QC26_MODEL")
             or cfg.get("submission", {}).get("model") or DEFAULT_MODEL)
    df = load_cards(cfg, path=args.test_data_path)
    idc = cfg["data"]["id_col"]
    print(f"товаров на входе: {len(df)}", flush=True)

    # Распознавание — единственный этап, способный не уложиться в лимит на чужом
    # железе, поэтому оно опционально и умеет отказаться. При отказе берётся запасная
    # модель, обученная без него: подавать модели вход не того вида, на котором её
    # учили, хуже, чем взять более слабую, но согласованную.
    ocr = _resolve_ocr(cfg, df, elapsed=time.time() - t0)
    if cfg.get("submission", {}).get("use_ocr") and ocr is None:
        if not _skip_vision_on_check(len(df)):
            _vision_failed(cfg, "чтение текста с изображений")
        fallback = cfg.get("submission", {}).get("fallback_model")
        if fallback:
            print(f"переключаюсь на запасную модель без распознавания: {fallback}",
                  flush=True)
            model = fallback

    # Эмбеддинги: тоже опционально и тоже с отказом вместо риска. Бюджет делим с
    # чтением текста — если оба этапа включены, второй получает остаток времени.
    emb = None
    if cfg.get("submission", {}).get("use_embeddings") and _skip_vision_on_check(len(df)):
        # На отладочной стадии эмбеддинги не строятся намеренно, а модель без них
        # работать не умеет — без подмены она бросит исключение и вердикт вынесут
        # правила. Ответ выйдет валидный, но собранный не тем, чем заявлено. Берём
        # запасную текстовую модель: она обучена без эмбеддингов и ответит честно.
        fb = cfg.get("submission", {}).get("fallback_model")
        if fb:
            print("отладочная стадия — беру запасную текстовую модель", flush=True)
            model = fb
    elif cfg.get("submission", {}).get("use_embeddings"):
        from qc26.inference.embed_runtime import build_embedding_sets, build_embeddings

        left = _ocr_budget(len(df), cfg.get("submission", {})) - (time.time() - t0)
        # Какие наборы векторов нужны решению, знает конфиг отправки. Позднему слиянию
        # нужны два — текстовый и картиночный по отдельности; модели на эмбеддингах
        # текста — один. По умолчанию строится один совмещённый, как во всех прежних
        # отправках, иначе они изменили бы поведение молча.
        sets = list(cfg.get("submission", {}).get("embedding_sets") or [])
        # ⚠ Корень изображений берётся от пути к данным ПРОГОНА, а не из конфига.
        # Построитель искал его по paths.data_csv, а внутри архива папки data/ нет:
        # картинки не находились и модель кодировала пустоту молча.
        from qc26.data import images_root as _images_root

        img_root = _images_root(cfg, DATA_PATH[0]) if DATA_PATH else None
        if left <= 60:
            emb = None
        elif sets:
            built = build_embedding_sets(cfg, df, budget_sec=max(0.0, left), sets=sets,
                                         images_root=img_root)
            # Один набор отдаём матрицей, несколько — словарём: так модель сама берёт
            # то, что ей нужно, по имени, и порядок аргументов ничего не решает.
            emb = (built[sets[0]] if built is not None and len(sets) == 1 else built)
        else:
            emb = build_embeddings(cfg, df, budget_sec=max(0.0, left),
                                   images_root=img_root)
        if emb is None:
            _vision_failed(cfg, "построение эмбеддингов изображений")
            fb = cfg.get("submission", {}).get("fallback_model")
            if fb:
                print(f"переключаюсь на запасную модель без эмбеддингов: {fb}", flush=True)
                model = fb

    preds, source = _predict(cfg, df, model, ocr, emb)

    # Поиск почти-дубликата обучающего товара. Наши сплиты групповые — дубликаты не
    # разрываются между фолдами, иначе оценка модели завышается на 19.5 п.п. Но на
    # закрытых данных всё наоборот: если каталог делили по товарам, дубликат тестового
    # товара лежит в выданной нам обучающей выборке ВМЕСТЕ С МЕТКОЙ. Замер по своим
    # данным: покрытие редкой категории 67.3%, вердикт по соседу верен в 99.7%.
    if cfg.get("submission", {}).get("use_duplicates"):
        from qc26.inference.duplicates import apply_lookup, load_table, lookup

        table_path = resolve_path(cfg["submission"].get(
            "dup_index", "artifacts/dup/dup_index.csv.gz"))
        if not table_path.exists():
            # Молчаливое вырождение уже стоило нам трёх попыток: без таблицы решение
            # выглядело бы обычным и мы бы неверно прочли результат.
            raise RuntimeError(f"таблица подписей не найдена: {table_path}")
        table = load_table(table_path)
        hits = lookup(df, table, cfg)
        before = preds.copy()
        preds = apply_lookup(preds, hits)
        cat_col = cfg["data"]["category_col"]
        for c in df[cat_col].astype(str).unique():
            m = (df[cat_col].astype(str) == c).to_numpy()
            got = (~np.isnan(hits.to_numpy())) & m
            print(f"дубликаты «{c}»: найдены у {got.sum()} из {m.sum()} "
                  f"({got.sum() / max(1, m.sum()):.1%}), вердиктов изменено "
                  f"{int((before[m] != preds[m]).sum())}", flush=True)
        source = f"{source} + поиск дубликата"

    comments = explain_frame(df, preds, ocr)
    out = pd.DataFrame({
        idc: df[idc].to_numpy(),
        "result": [build_result(c, p) for c, p in zip(comments, preds)],
    })
    out.columns = ["id", "result"]

    errors = validate_submission(out, expected_ids=df[idc])
    if errors:
        # Такого быть не должно: build_result сам приводит длину и вердикт к формату.
        print("НАРУШЕНИЯ ФОРМАТА: " + "; ".join(errors), flush=True)
        raise SystemExit(2)

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_path, index=False, encoding="utf-8")
    stats = submission_stats(out)
    print(f"готово за {time.time() - t0:.1f} с | источник вердикта: {source} | "
          f"«не бан»: {stats['share_not_ban'] * 100:.1f}% | "
          f"длина комментария {stats['comment_len_min']}–{stats['comment_len_max']} "
          f"(медиана {stats['comment_len_median']:.0f}) → {args.output_path}", flush=True)


if __name__ == "__main__":
    main()
