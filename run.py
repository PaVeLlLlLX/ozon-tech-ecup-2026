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

# ⚠ Печать не имеет права ронять решение. На Windows stdout отдаётся в cp1251, и любой
# символ вне неё (у нас была стрелка в итоговой строке) валит прогон с UnicodeEncodeError
# уже ПОСЛЕ записи ответа. В контейнере кодировка другая, но полагаться на это нельзя:
# сбой на печати обнуляет попытку так же, как сбой модели.
# Применяем несколько стратегий для надёжности:
import io
for _stream_var, _stream in [("stdout", sys.stdout), ("stderr", sys.stderr)]:
    try:
        # Стратегия 1: reconfigure для Python 3.7+
        if hasattr(_stream, 'reconfigure'):
            _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, TypeError):
        pass
    try:
        # Стратегия 2: замена на TextIOWrapper с buffer (надёжнее для Windows)
        if hasattr(_stream, 'buffer') and not isinstance(_stream, io.TextIOWrapper):
            new_stream = io.TextIOWrapper(_stream.buffer, encoding="utf-8", errors="replace")
            setattr(sys, _stream_var, new_stream)
    except (AttributeError, ValueError, TypeError):
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from qc26.config import load_config, resolve_path  # noqa: E402
from qc26.data import load_cards  # noqa: E402
from qc26.explain.templates import explain_frame  # noqa: E402
from qc26.inference.format import build_result, submission_stats, validate_submission  # noqa: E402
from qc26.rules import rule_guess  # noqa: E402

DEFAULT_MODEL = "artifacts/models/tfidf_logreg.joblib"


def _predict(cfg: dict, df: pd.DataFrame, model: str):
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
    if model == "const:supplement":
        # «не бан» только категории БАД: обнуляет вклад редкой категории и делает
        # публичное значение прямой мерой доли позитивов в БАД
        preds = (df[cfg["data"]["category_col"]].astype(str) == "БАД").astype(int).to_numpy()
        return preds, "постоянный вердикт по категории"
    if model.startswith("const:"):
        value = int(model.split(":", 1)[1])
        return np.full(len(df), value, dtype=int), f"постоянный вердикт {value}"
    try:
        from qc26.models.text_baseline import TextBaseline

        path = resolve_path(model)
        loaded = TextBaseline.load(path)
        override = cfg.get("submission", {}).get("thresholds") or {}
        if override:
            loaded.thresholds = {**loaded.thresholds, **override}
            print(f"порог переопределён конфигом отправки: {override}", flush=True)
        proba = loaded.predict_proba(df)
        head_path = cfg.get("submission", {}).get("emb_head")
        source = "модель"
        if head_path:
            proba, source = _blend_embeddings(cfg, df, proba, head_path, loaded)
        sub = cfg.get("submission", {})
        # ⚠⚠ Ворота пропускали ТОЛЬКО ключ vlm_adapter. С пер-категорийным ключом
        # vlm_adapters зрительная модель молча выпадала, вердикт выносила текстовая, и
        # решение спокойно дописывало «источник вердикта: модель» — ни ошибки, ни
        # предупреждения. Смоук поймал это на десяти карточках; на отправке мы приняли
        # бы балл слабой модели за результат сильной. Ровно та беда, ради которой в
        # этом решении вообще нет откатов.
        vlm_path = sub.get("vlm_adapter")
        per_cat_cfg = sub.get("vlm_adapters")
        if isinstance(per_cat_cfg, dict) and per_cat_cfg:
            vlm_path = next(iter(per_cat_cfg.values()))   # адаптер по умолчанию
        if vlm_path:
            proba, source = _blend_vlm(cfg, df, proba, vlm_path, loaded, source)
        elif sub.get("vlm_categories"):
            raise SystemExit(
                "заданы vlm_categories, но ни vlm_adapter, ни vlm_adapters — "
                "вердикт вынесла бы текстовая модель, а мы приняли бы её балл за "
                "результат зрительной")
        cat = df[cfg["data"]["category_col"]].astype(str)
        thr = cat.map(loaded.thresholds).fillna(0.5).to_numpy()
        print(f"модель: {path.name}, пороги {loaded.thresholds}", flush=True)
        return (proba >= thr).astype(int), source
    except Exception:
        # ⚠ ОТКАТА НЕТ НАМЕРЕННО. Упавшая отправка попытку не тратит, а тихо
        # выродившаяся тратит её и возвращает число, которое мы примем за результат.
        traceback.print_exc()
        raise SystemExit("модель недоступна — решение падает, а не откатывается")


def _blend_embeddings(cfg: dict, df: pd.DataFrame, proba: np.ndarray, head_path: str,
                      loaded) -> tuple[np.ndarray, str]:
    """Подмешивает оценку головы над эмбеддингами — по категориям, с откатом.

    Замерено (docs/emb_head_results.local.md): «Легковоспламеняющиеся» +7.5…+16.2 п.п.
    F1 при весе 0.45, БАД — прироста нет, вес 0. Веса и пороги лежат в артефакте
    головы, а не подбираются здесь.

    ⚠ Любой сбой на этом пути обязан оставить вердикт текстовой модели. Эмбеддинги
    дают несколько пунктов, падение решения стоит всей попытки.
    """
    try:
        from qc26.data import images_root
        from qc26.inference.embed import embed_frame
        from qc26.models.emb_head import EmbHeadBundle

        bundle = EmbHeadBundle.load(resolve_path(head_path))
        active = {c: w for c, w in bundle.weights.items() if w > 0 and c in bundle.heads}
        if not active:
            return proba, "модель (голова отключена весами)"

        cat = df[cfg["data"]["category_col"]].astype(str)
        need = cat.isin(active).to_numpy()
        if not need.any():
            return proba, "модель (нет товаров нужных категорий)"

        # считаем эмбеддинги ТОЛЬКО там, где они влияют на вердикт: на БАДах вес 0,
        # а это больше половины выборки — экономия времени без потери качества
        sub = df[need]
        emb = embed_frame(sub, cfg, images_root(cfg, cfg.get("_test_path")),
                          batch=int(cfg.get("submission", {}).get("embed_batch", 32)))
        if emb is None:
            return proba, "модель (эмбеддинги недоступны)"

        blended = proba.copy()
        pos = np.where(need)[0]
        for c, w in active.items():
            m = (cat.to_numpy()[need] == c)
            if not m.any():
                continue
            p_head = bundle.heads[c].predict_proba(emb[m])
            blended[pos[m]] = (1 - w) * proba[pos[m]] + w * p_head
            print(f"смесь по категории {c}: вес {w}, товаров {int(m.sum())}", flush=True)
        for c, t in bundle.thresholds.items():
            loaded.thresholds[c] = float(t)
        return blended, "модель + эмбеддинги"
    except Exception:
        # ⚠ ОТКАТА НЕТ НАМЕРЕННО — см. комментарий в _predict.
        traceback.print_exc()
        raise SystemExit("голова над эмбеддингами недоступна — решение падает")


def _blend_vlm(cfg: dict, df: pd.DataFrame, proba: np.ndarray, adapter: str,
               loaded, source: str) -> tuple[np.ndarray, str]:
    """Подменяет оценку дообученной VLM в указанных категориях. С откатом.

    Базовые веса берутся из /shared_models (модель есть в списке организаторов),
    в архиве едет только адаптер LoRA.

    ⚠ Шкала оценок VLM не имеет ничего общего со шкалой tf-idf, поэтому вместе с
    оценкой ОБЯЗАТЕЛЬНО подменяется и порог — из thresholds.json рядом с адаптером,
    подобранный на внутренней проверке. Забыть про порог здесь значит применить к
    новому распределению чужую отсечку и получить вердикт «всем бан» или «всем не бан».
    """
    try:
        from qc26.data import images_root
        from qc26.inference.vlm import score_frame

        sub_cfg = cfg.get("submission", {})
        cats = sub_cfg.get("vlm_categories") or ["Легковоспламеняющиеся"]
        # ⚠ Вес СВОЙ у каждой категории, как и доля отбора. У сокомандника в
        # рекордной связке вес зрительной модели в БАД равен 0.5, а в редкой 1.0:
        # там, где текстовая модель сильна, ей отдаётся половина голоса, а где
        # слаба — она выключается. Общий вес на обе категории такого не выражает.
        # Принимаем и число (тогда оно общее), и словарь по категориям.
        _w_raw = sub_cfg.get("vlm_weight", 1.0)
        if isinstance(_w_raw, dict):
            weights = {str(c): float(v) for c, v in _w_raw.items()}
        else:
            weights = {}
        weight_default = 1.0 if isinstance(_w_raw, dict) else float(_w_raw)
        adapter_p = resolve_path(adapter)

        # ⚠ thresholds.json нужен ТОЛЬКО при отборе по порогу. При отборе по доле
        # порог вычисляется из самих оценок, и требовать файл — значит молча
        # выключить VLM у адаптера, обученного без отложенной выборки. Ровно такой
        # молчаливый пропуск уже трижды давал публичные значения, совпавшие до
        # десятого знака с решением без VLM.
        import json

        # ⚠ vlm_select может быть словарём по категориям: в «Легковоспламеняющихся»
        # отбор идёт по эталону рекорда, в «БАД» — по доле. Ни то, ни другое не требует
        # thresholds.json рядом с адаптером, и str(словарь) != "rate" отправил бы
        # решение в ветку ниже, где VLM МОЛЧА пропускается.
        _sel_raw = sub_cfg.get("vlm_select", "rate")
        select_by = "rate" if isinstance(_sel_raw, dict) else str(_sel_raw)
        thr_file = adapter_p / "thresholds.json"
        vlm_thr: dict = {}
        if thr_file.exists():
            vlm_thr = json.loads(thr_file.read_text(encoding="utf-8")).get(
                "thresholds", {}) or {}
        elif select_by != "rate":
            # ⚠ ОТКАТА НЕТ. Раньше здесь стоял тихий пропуск, и решение возвращало
            # вердикт текстовой модели как ни в чём не бывало.
            raise SystemExit(
                f"нет {thr_file}, а отбор задан по порогу «{select_by}» — вердикт "
                f"вынесла бы текстовая модель, и мы приняли бы её балл за результат")

        cat = df[cfg["data"]["category_col"]].astype(str)
        need = cat.isin(cats).to_numpy()
        if not need.any():
            return proba, source + " (нет товаров для VLM)"

        # ⚠ vlm_base — веса, привезённые с решением. Qwen3-VL-4B нет в каталоге
        # жюри, она едет в архиве упакованной в int8. Путь относительный, от корня
        # распакованного архива: абсолютный на проверяющей машине не существует.
        base_dir = sub_cfg.get("vlm_base")
        base_ov = str(resolve_path(base_dir)) if base_dir else None
        root_imgs = images_root(cfg, cfg.get("_test_path"))
        n_batch = int(sub_cfg.get("vlm_batch", 8))
        mode = str(sub_cfg.get("exotic_mode") or "none")
        sub_df = df[need]

        per_cat = sub_cfg.get("vlm_adapters")
        if isinstance(per_cat, dict) and mode == "none":
            # ⚠ СВОЙ адаптер на каждую категорию. Обе половины метрики выигрываются
            # разными дообучениями, и разложение публичных баллов по известному составу
            # выборки (БАД 921 товар / 528 позитивов, редкая 707 / 24) показывает это
            # прямо:
            #     БАД     vl4_plain    490 верных из 528 названных, F1 0.9280
            #     БАД     наша связка  481 из 517,                  F1 0.9206
            #     редкая  vlm_rare2_a   21 верных из 24,            F1 0.9130
            #     редкая  vl4_plain     18 из 24,                   F1 0.8372
            # Метрика усредняет категории независимо, поэтому брать в каждую ту модель,
            # которая её выигрывает, законно и ничего не смешивает.
            #
            # ⚠ Шкалы двух адаптеров НЕСОПОСТАВИМЫ между собой: логит-разность у каждого
            # своя. Сравнивать их и нельзя — отбор идёт по доле ВНУТРИ категории, а в
            # каждой категории работает ровно один адаптер. Ни одна операция ниже не
            # сводит числа разных адаптеров вместе.
            scores = np.zeros(len(sub_df), dtype=np.float32)
            cat_sub = cat.to_numpy()[need]
            for c in cats:
                m = cat_sub == c
                if not m.any():
                    continue
                if c not in per_cat:
                    raise SystemExit(f"для категории «{c}» адаптер не задан")
                ap_c = resolve_path(per_cat[c])
                s_c = score_frame(sub_df[m], cfg, root_imgs, ap_c, batch=n_batch,
                                  n_images=sub_cfg.get("vlm_images"),
                                  base_override=base_ov)
                if s_c is None:
                    raise SystemExit(
                        f"адаптер «{ap_c.name}» категории «{c}» не дал оценок — "
                        f"решение падает, а не откатывается")
                scores[m] = s_c
                print(f"категория «{c}»: {int(m.sum())} товаров адаптером {ap_c.name}, "
                      f"оценки от {float(s_c.min()):+.2f} до {float(s_c.max()):+.2f}",
                      flush=True)
                # ⚠ Освобождаем карту перед следующим адаптером. Без этого смоук упал
                # с 22.35 ГБ занятых: score_frame держит модель до сборки мусора, и
                # три загрузки подряд жили одновременно.
                import gc

                gc.collect()
                try:
                    import torch

                    torch.cuda.empty_cache()
                except Exception:
                    pass
        elif mode == "duo":
            # ⚠ Объединение ДВУХ адаптеров максимумом ранга. Разложение публичных
            # баллов показало, что они находят РАЗНЫЕ товары: первый 18 верных из 19
            # названных, второй 19 из 22. Среднее рангов утопило бы находку, которую
            # сделал только один; максимум сохраняет обе.
            second = sub_cfg.get("exotic_adapter2")
            if not second:
                raise SystemExit("режим duo требует exotic_adapter2")
            a = score_frame(sub_df, cfg, root_imgs, adapter_p, batch=n_batch,
                            n_images=sub_cfg.get("vlm_images"), base_override=base_ov)
            b = score_frame(sub_df, cfg, root_imgs, resolve_path(second),
                            batch=n_batch, n_images=sub_cfg.get("vlm_images"),
                            base_override=base_ov)
            if a is None or b is None:
                raise SystemExit("один из двух адаптеров не дал оценок — решение падает")
            ra = pd.Series(a).rank(pct=True).to_numpy()
            rb = pd.Series(b).rank(pct=True).to_numpy()
            scores = np.maximum(ra, rb).astype(np.float32)
            print(f"duo: два адаптера объединены максимумом ранга, {len(scores)} оценок",
                  flush=True)
        elif mode != "none":
            from qc26.inference.exotic import exotic_scores

            scores = exotic_scores(mode, sub_df, cfg, root_imgs, adapter_p,
                                   base_override=base_ov, batch=n_batch)
        else:
            scores = score_frame(sub_df, cfg, root_imgs, adapter_p, batch=n_batch,
                                 n_images=sub_cfg.get("vlm_images"),
                                 base_override=base_ov)
        if scores is None:
            raise SystemExit("оценка VLM не получена — решение падает")

        blended = proba.copy()
        pos = np.where(need)[0]
        for c in cats:
            m = (cat.to_numpy()[need] == c)
            if not m.any():
                continue
            # ⚠ Оценка VLM приходит в ЛОГИТ-разности, а proba — вероятность tf-idf.
            # Складывать их напрямую бессмысленно: шкалы несопоставимы. Переводим
            # оценку VLM в перцентильный ранг ВНУТРИ КАТЕГОРИИ — после этого и смесь,
            # и отбор по доле работают независимо от масштаба.
            vlm_rank = pd.Series(scores[m]).rank(pct=True).to_numpy()
            how_blend = str(sub_cfg.get("vlm_blend", "replace"))
            if how_blend == "rank_max":
                # ⚠ Объединение, а не усреднение. Замерено на фолде 0: модели находят
                # РАЗНЫЕ позитивы (общих 6, только VLM 2, только текст 4, объединение
                # 12 из 14 против 10 у текста). Среднее рангов топит то, что нашла
                # одна модель; максимум ранга сохраняет находки обеих.
                a = pd.Series(proba[pos[m]]).rank(pct=True).to_numpy()
                blended[pos[m]] = np.maximum(a, vlm_rank)
            else:
                # обе части в ранговой шкале — иначе вес не имеет смысла
                a = pd.Series(proba[pos[m]]).rank(pct=True).to_numpy()
                w_c = weights.get(c, weight_default)
                blended[pos[m]] = (1 - w_c) * a + w_c * vlm_rank
            # ⚠ Порог из thresholds.json подобран в ВЕРОЯТНОСТНОЙ шкале обучения,
            # а здесь шкала ранговая. Применять его нельзя — только отбор по доле.
            if select_by != "rate":
                print(f"⚠ шкала VLM ранговая, порог из обучения неприменим — "
                      f"перехожу на отбор по доле для «{c}»", flush=True)

            # ⚠ Способ отбора важнее самой модели. Порог с внутренней проверки
            # подобран на 39 позитивах и по траектории гулял от 0.11 до 0.985; на
            # фолде 0 он отбирает 1.73%, то есть ~12 товаров из 707 при 24 истинных
            # позитивах — это ограничивает F1 сверху величиной 0.667 даже при
            # идеальной точности. Доля же 0.034 замерена зондами по закрытой выборке
            # (это внешнее знание о приоре, а не подгонка под отправляемые данные).
            # ⚠ Доля отбора СВОЯ у каждой категории. В закрытой выборке БАД имеет
            # 528 позитивов из 921 (доля 0.5733), редкая — 24 из 707 (0.0339).
            # Единая доля 0.0269 отобрала бы в БАД 25 товаров вместо ~528 и
            # уничтожила бы категорию.
            # ⚠ Правило пиротехники. Единственный признак, который в этой категории
            # даёт крупный подъём на БОЛЬШОЙ выборке, а не на десятке наблюдений:
            # замер по всем 5502 товарам редкой категории (198 позитивов) —
            #   хлопушки, дымовые шашки, цветной дым          236 шт, «не бан» 0.212
            #   они же БЕЗ упоминания сжатого воздуха         144 шт, «не бан» 0.340
            #   они же С упоминанием сжатого воздуха           92 шт, «не бан» 0.011
            # при фоне категории 0.036. Сжатый воздух разделяет класс в тридцать раз:
            # пневматическая хлопушка не содержит горючего, пиротехническая содержит.
            #
            # ⚠ Подъём МАЛЫЙ и намеренно. Он не добавляет предсказаний, а меняет их
            # состав: на отложенной выборке при девятнадцати названных верных стало 19
            # вместо 17 — вошли две дымовые шашки, вышли газовая насадка и роллы для
            # розжига, все четыре по разметке верно. Эффект держится на всём диапазоне
            # от 0.01 до 0.12; при 0.2 и выше правило начинает затаскивать негативы и
            # роняет результат до 12 верных.
            bonus = float(sub_cfg.get("pyro_bonus", 0.0))
            if bonus and c != "БАД":
                import re as _re

                # ⚠ Ищем ТОЛЬКО в названии. Описание тянет за собой сопутствующие
                # товары: краски Холи, язычки-дуделки, конфетти-пушки — там хлопушки
                # лишь упомянуты. Замер: 41 такой товар, и среди них «не бан» РОВНО
                # НОЛЬ. Без описания доля поднятых растёт с 0.340 до 0.400 (подъём
                # 11.1x против 9.5x), а группа опущенных становится чистой: 0.000.
                d_ = cfg["data"]
                txt = df.loc[df.index[need][m], d_["name_col"]].fillna("").str.lower()
                pyro = txt.str.contains(
                    r"хлопушк|дымов\w* шашк|цветн\w* дым|дымов\w* фонтан|шашк\w* дымов",
                    regex=True)
                air = txt.str.contains(r"пневматич|сжат\w* воздух|без пиротехн", regex=True)
                up, down = (pyro & ~air).to_numpy(), (pyro & air).to_numpy()
                blended[pos[m]] = blended[pos[m]] + bonus * up - bonus * down
                print(f"правило пиротехники: поднято {int(up.sum())}, "
                      f"опущено {int(down.sum())} из {int(m.sum())}", flush=True)

            # ⚠ Судья: линейная модель поверх слов КАРТОЧКИ и слов ОБЪЯСНЕНИЯ.
            # Объяснение порождается, НЕ ЗНАЯ вердикта: в промпт он не подаётся, а
            # подсказки для «бан» и «не бан» выровнены посимвольно.
            #
            # ⭐ ЧЕМ ЭТО ОТЛИЧАЕТСЯ ОТ ПРЕДЫДУЩЕЙ ВЕРСИИ, ОБНУЛИВШЕЙ ОТПРАВКУ. Тогда
            # судья получал оценку VLM ОТДЕЛЬНЫМ ПРИЗНАКОМ. Учился он на вероятностях
            # из [0,1], а на прогоне ему приходила логит-разность от -13.8 до +8.5;
            # при весе +4.7 этот признак подавлял все словесные, и судья выродился в
            # монотонную функцию оценки модели — вердикты совпали побитово, а балл
            # повторил рекорд до последнего знака. Здесь оценка ему НЕ подаётся вовсе,
            # а смешивание идёт в пространстве РАНГОВ, где шкалу подменить нечем:
            #
            #     итог = (1 - w) * ранг(оценка VLM) + w * ранг(судьи)
            #
            # ⚠⚠ И ГЛАВНОЕ: ниже стоит проверка на вырождение. Если верхушка после
            # поправки совпала с верхушкой модели, решение ПАДАЕТ. Отправка, молча
            # повторяющая рекорд, тратит попытку и возвращает число, которое мы
            # принимаем за результат; упавшая — попытку возвращает.
            judge_path = sub_cfg.get("judge_rare")
            if judge_path and c != "БАД":
                import joblib
                from scipy.sparse import hstack
                from scipy.stats import rankdata

                from qc26.explain.vlm_explain import generate_comments
                from qc26.inference.exotic import _Runner

                jp = resolve_path(judge_path)
                if not jp.exists():
                    raise SystemExit(f"нет весов судьи: {jp}")
                J = joblib.load(jp)
                if J.get("kind") != "rank-text-v2":
                    raise SystemExit(
                        f"судья версии {J.get('kind')!r}: у прежней оценка VLM входила "
                        f"признаком и подменялась шкалой на прогоне — не используем")
                w_j = float(sub_cfg.get("judge_weight", 0.15))
                if not 0.0 < w_j <= 0.5:
                    raise SystemExit(f"вес судьи {w_j} вне разумного диапазона (0, 0.5]")
                idx = df.index[need][m]
                sub = df.loc[idx]
                runner = _Runner(cfg, adapter_p, base_ov, n_batch)
                zero = np.zeros(len(sub), dtype=int)     # вердикта ещё нет и не нужно
                tpl = explain_frame(sub, zero)
                texts = generate_comments(
                    sub, cfg, root_imgs, runner, zero, tpl,
                    max_new_tokens=int(sub_cfg.get("explain_tokens", 60)),
                    n_images=int(sub_cfg.get("explain_images", 1)),
                    max_desc=int(sub_cfg.get("explain_desc", 300)),
                    batch=sub_cfg.get("explain_batch"))
                d_ = cfg["data"]
                card = (sub[d_["name_col"]].fillna("") + " "
                        + sub[d_["desc_col"]].fillna("")).str.lower()
                expl = pd.Series(texts, index=card.index).fillna("").str.lower()
                X = hstack([J["vec_card_w"].transform(card),
                            J["vec_card_c"].transform(card),
                            J["vec_expl_w"].transform(expl),
                            J["vec_expl_c"].transform(expl)]).tocsr()
                p_j = J["model"].predict_proba(X)[:, 1]
                cur = blended[pos[m]]
                n_j = len(p_j)
                r_model = rankdata(cur) / n_j
                r_judge = rankdata(p_j) / n_j
                mixed = (1.0 - w_j) * r_model + w_j * r_judge
                # ⚠ Диапазоны рядом: это ровно та сверка, которой вчера не было.
                print(f"судья «{J['kind']}»: обучен на {J['n_train']} товарах "
                      f"({J['n_pos']} позитивов), вес {w_j}", flush=True)
                print(f"   оценка VLM  от {float(cur.min()):+.3f} до {float(cur.max()):+.3f}"
                      f"   судья от {float(p_j.min()):.3f} до {float(p_j.max()):.3f}"
                      f"   (в обучении судья давал от {J['p_lo']:.3f} до {J['p_hi']:.3f})",
                      flush=True)
                raw_r = sub_cfg.get("vlm_rate", 0.034)
                rate_c = float(raw_r[c]) if isinstance(raw_r, dict) else float(raw_r)
                k_j = max(1, int(round(rate_c * n_j)))
                top_before = set(np.argsort(-cur, kind="stable")[:k_j].tolist())
                top_after = set(np.argsort(-mixed, kind="stable")[:k_j].tolist())
                moved = len(top_after - top_before)
                print(f"   верхушка {k_j}: заменено {moved} товаров", flush=True)
                if moved == 0:
                    # ⚠ Порог по размеру: на смоуке из десяти карточек в редкой
                    # категории верхушка это один товар, и несдвинутый один товар —
                    # не вырождение, а арифметика. Ворота ставим там, где они значат
                    # то, что должны: на боевом объёме.
                    if n_j >= 50:
                        raise SystemExit(
                            "судья не изменил ни одного товара в верхушке — отправка "
                            "повторила бы решение без судьи и впустую сожгла попытку")
                    print(f"   ⚠ на {n_j} товарах верхушка не сдвинулась; для ворот "
                          f"это слишком мало, продолжаю", flush=True)
                blended[pos[m]] = mixed
                # Объяснения уже посчитаны — переиспользуем их в ответе, а не считаем вновь.
                cfg.setdefault("_judge_comments", {}).update(
                    dict(zip(sub[d_["id_col"]].astype(str), texts)))

            # ⚠⚠ ОТБОР ПРАВИЛОМ РЕКОРДА. В «Легковоспламеняющихся» решение, давшее
            # 0.9168088205, брало не долю, а порог 0.9699 на КВАНТИЛЕ относительно
            # эталонного распределения из 1101 значения, сохранённого при обучении.
            # Критерий абсолютный: «оценка выше 96.99% эталона». Доля — относительный,
            # «верхние 3% этой выборки»; на публичной они совпали случайно, на приватной
            # разойдутся. Эталон и порог едут в архиве, шкала приводится к той же.
            sel_raw = sub_cfg.get("vlm_select", "rate")
            sel_c = str(sel_raw.get(c, "rate") if isinstance(sel_raw, dict) else sel_raw)
            if sel_c == "calibrated":
                import json as _json

                cal_raw = sub_cfg.get("vlm_calibration")
                cal_p = cal_raw.get(c) if isinstance(cal_raw, dict) else cal_raw
                if not cal_p:
                    raise SystemExit(f"для «{c}» задан отбор по эталону, но эталона нет")
                cal = _json.loads(resolve_path(cal_p).read_text(encoding="utf-8"))
                ref = np.asarray(cal["эталон"], dtype=np.float64)
                thr_c = float(cal["порог"])
                if not np.all(np.diff(ref) >= 0):
                    raise SystemExit("эталон не отсортирован — квантиль посчитается неверно")
                # Шкала рекорда: sigmoid от логит-разности. Новый код отдаёт саму
                # логит-разность, поэтому sigmoid возвращаем здесь, во float32 — ровно
                # той точности, в которой эталон и строился.
                z = np.asarray(scores, dtype=np.float32)[m]
                prob = (1.0 / (1.0 + np.exp(-z.astype(np.float32)))).astype(np.float64)
                left = np.searchsorted(ref, prob, side="left")
                right = np.searchsorted(ref, prob, side="right")
                q = (left + right) / (2.0 * len(ref))
                blended[pos[m]] = q
                loaded.thresholds[c] = thr_c
                n_sel = int((q >= thr_c).sum())
                print(f"отбор по эталону рекорда: порог {thr_c}, эталон {len(ref)} "
                      f"значений, отобрано {n_sel} из {int(m.sum())}", flush=True)
                print(f"   оценка: логит-разность от {float(z.min()):+.2f} до "
                      f"{float(z.max()):+.2f} → вероятность от {prob.min():.6f} до "
                      f"{prob.max():.6f}; эталон от {ref[0]:.6f} до {ref[-1]:.6f}",
                      flush=True)
                if n_sel == 0:
                    raise SystemExit(
                        f"по эталону в «{c}» не отобрано ни одного товара — "
                        f"шкала оценки не совпала с эталоном")
                continue

            raw_rate = sub_cfg.get("vlm_rate", 0.034)
            if isinstance(raw_rate, dict):
                if c not in raw_rate:
                    print(f"⚠ для категории {c} доля не задана — VLM в ней пропущена",
                          flush=True)
                    blended[pos[m]] = proba[pos[m]]
                    continue
                rate = float(raw_rate[c])
            else:
                rate = float(raw_rate)
            k = max(1, int(round(rate * int(m.sum()))))
            vals = np.sort(blended[pos[m]])[::-1]
            loaded.thresholds[c] = float(vals[k - 1])
            n_sel = int((blended[pos[m]] >= loaded.thresholds[c]).sum())
            print(f"отбор по замеренной доле {rate}: хотели {k} из {int(m.sum())}, "
                  f"отобрано {n_sel}", flush=True)
            if n_sel > k * 1.5:
                # ⚠ Ничьи. Ровно из-за них три отправки дали одинаковые 11 из 19:
                # вероятность насыщалась в 1.0 у трети выборки. С ранговой шкалой
                # такого быть не должно, но проверку оставляем — она дешёвая.
                print(f"⚠ ничьи в шкале: отобрано {n_sel} вместо {k}. "
                      f"Ранжирование вырождено, вердикт ненадёжен.", flush=True)
            print(f"VLM по категории {c}: вес {weights.get(c, weight_default)}, "
                  f"товаров {int(m.sum())}, "
                  f"порог {loaded.thresholds[c]:.4f}", flush=True)
        return blended, source + " + VLM"
    except Exception:
        # ⚠ ОТКАТА НЕТ НАМЕРЕННО. Именно здесь три отправки подряд молча выродились в
        # текстовое решение и вернули один и тот же балл, потратив три попытки.
        traceback.print_exc()
        raise SystemExit("зрительно-языковая модель недоступна — решение падает")


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
    cfg = load_config(args.config, "configs/baseline.yaml", "configs/submission.yaml")
    # Проверяющая система запускает ровно `python -u run.py` без своих аргументов,
    # поэтому выбор решения для конкретной отправки живёт в configs/submission.yaml,
    # который кладёт сборщик архива. Явный --model (локальные прогоны) главнее.
    model = (args.model or os.environ.get("QC26_MODEL")
             or cfg.get("submission", {}).get("model") or DEFAULT_MODEL)
    # путь к картинкам выводится из пути к csv, а не из конфига: в контейнере данные
    # лежат не там, где на нашей машине
    cfg["_test_path"] = args.test_data_path
    df = load_cards(cfg, path=args.test_data_path)
    idc = cfg["data"]["id_col"]
    print(f"товаров на входе: {len(df)}", flush=True)

    preds, source = _predict(cfg, df, model)

    # Диагностическая отправка: обнулить вердикт в указанных категориях. F1 по классу
    # «не бан» там становится ровно 0, поэтому публичное значение равно половине F1
    # оставшейся категории — так метрика раскладывается по категориям без догадок.
    force_ban = cfg.get("submission", {}).get("force_ban_categories") or []
    if force_ban:
        mask = df[cfg["data"]["category_col"]].astype(str).isin(force_ban).to_numpy()
        preds = np.where(mask, 0, preds)
        source += f" (вердикт «бан» принудительно: {', '.join(force_ban)})"
        print(f"обнулено категорий: {force_ban}, товаров {int(mask.sum())}", flush=True)

    # Шаблонное объяснение считаем всегда: оно и опора, и запасной путь.
    comments = explain_frame(df, preds)
    if str(cfg.get("submission", {}).get("explain") or "template") == "vlm":
        from qc26.data import images_root
        from qc26.explain.vlm_explain import generate_comments
        from qc26.inference.exotic import _Runner

        sub_cfg = cfg.get("submission", {})
        base_dir = sub_cfg.get("vlm_base")
        # ⚠ Настройки прохода объяснений подобраны замером на ста товарах и обязаны
        # ехать в архив целиком, а не по одному ключу. Один кадр вместо пяти, потолок
        # 60 новых токенов, описание 300 символов и свой батч дали 4.4 секунды на товар
        # против 27.5 у первой версии — в 6.2 раза быстрее при чистом тексте.
        # ⚠ Объяснения, уже посчитанные проходом судьи, ЗАНОВО НЕ СЧИТАЕМ. Судья
        # генерирует тексты по всей редкой категории, и повторный проход по ним —
        # чистая потеря времени: около 40% выборки вхолостую. Прежний прогон на сайте
        # шёл 39 минут при лимите 40 на приватной стадии, то есть запаса не было вовсе.
        ready = cfg.get("_judge_comments") or {}
        ids_all = df[idc].astype(str).to_numpy()
        todo = np.array([i not in ready for i in ids_all])
        if ready:
            print(f"объяснения редкой категории уже есть у {len(ids_all) - int(todo.sum())} "
                  f"товаров — основной проход считает {int(todo.sum())}", flush=True)
        # ⚠ Объяснение пишет ТОТ ЖЕ адаптер, который вынес вердикт в этой категории.
        # Иначе решение и его обоснование исходят от разных моделей, и текст объясняет
        # не то, что произошло. Когда адаптер один на всё, план вырождается в один шаг.
        base_ov_ex = str(resolve_path(base_dir)) if base_dir else None
        n_batch_ex = int(sub_cfg.get("vlm_batch", 8))
        root_ex = images_root(cfg, cfg.get("_test_path"))
        per_cat_ex = sub_cfg.get("vlm_adapters")
        cat_all = df[cfg["data"]["category_col"]].astype(str).to_numpy()
        if isinstance(per_cat_ex, dict):
            missed = todo & ~np.isin(cat_all, list(per_cat_ex))
            if missed.any():
                raise SystemExit(
                    f"у {int(missed.sum())} товаров категория без адаптера: "
                    f"{sorted(set(cat_all[missed]))}")
            plan = [(c, resolve_path(a), todo & (cat_all == c))
                    for c, a in per_cat_ex.items()]
        else:
            plan = [("все", resolve_path(sub_cfg["vlm_adapter"]), todo)]
        for cat_name, ap_ex, sel in plan:
            if not sel.any():
                continue
            runner = _Runner(cfg, ap_ex, base_ov_ex, n_batch_ex)
            got = generate_comments(
                df[sel], cfg, root_ex, runner, preds[sel],
                [c for c, t in zip(comments, sel) if t],
                max_new_tokens=int(sub_cfg.get("explain_tokens", 60)),
                n_images=int(sub_cfg.get("explain_images", 1)),
                max_desc=int(sub_cfg.get("explain_desc", 300)),
                batch=sub_cfg.get("explain_batch"))
            if len(got) != int(sel.sum()):
                raise SystemExit(
                    f"генерация вернула {len(got)} текстов на {int(sel.sum())} товаров")
            it = iter(got)
            comments = [next(it) if t else c for c, t in zip(comments, sel)]
            print(f"объяснения категории «{cat_name}»: {int(sel.sum())} товаров "
                  f"адаптером {ap_ex.name}", flush=True)
            # ⚠ Освобождаем карту перед загрузкой следующего адаптера: две модели по
            # 8.3 ГБ разом не нужны никогда, а на нашей карте просто не поместятся.
            del runner
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass
        # ⚠ Подставляем объяснения, на которых работал судья: в ответе обязан стоять
        # РОВНО тот текст, по которому принят вердикт. Основной проход этих строк уже
        # не считал, поэтому подстановка здесь не роскошь, а единственный источник.
        if ready:
            n = 0
            for k, i in enumerate(ids_all):
                if i in ready:
                    comments[k] = ready[i]
                    n += 1
            if n != len(ready):
                raise SystemExit(
                    f"судья дал {len(ready)} объяснений, подставилось {n} — "
                    f"в ответе оказался бы шаблон вместо текста, решившего вердикт")
            print(f"объяснения редкой категории взяты из прохода судьи: {n}", flush=True)
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
