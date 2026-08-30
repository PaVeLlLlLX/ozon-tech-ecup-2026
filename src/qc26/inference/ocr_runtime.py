"""Распознавание текста на фото во время прогона решения — с защитой по времени.

Распознавание даёт лучший прирост из всего, что мы нашли, но это единственный этап,
который может не уложиться в лимит на чужом железе. Поэтому здесь три страховки:

1. **Отказ вместо падения.** Нет easyocr в образе, нет весов, нет интернета — функция
   возвращает None, и решение работает по запасной модели без распознавания.
2. **Ранняя оценка скорости.** После первых товаров считается фактический темп и
   прогноз на весь объём. Если прогноз выходит за бюджет, распознавание бросается
   сразу, а не за минуту до конца лимита.
3. **Жёсткий рубеж.** Даже при хорошем прогнозе работа прекращается по истечении
   бюджета: частичный результат хуже целостного, поэтому в этом случае тоже отказ.

Веса easyocr кладутся в архив (около 94 МБ) — в контейнере интернета нет, скачать их
не получится.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

NO_TEXT = "(текст не распознан)"
# Сколько товаров просмотреть в поисках тех, у кого есть фотографии, чтобы замерить темп.
# Первые сорок могут оказаться без фото — тогда замер вырождается и защита слепнет.
PROBE_SCAN_LIMIT = 600
# Как часто пересчитывать темп по ходу чтения и подгонять число кадров под остаток.
RECHECK_EVERY = 100


def ocr_available(weights_dir: str | Path | None = None) -> bool:
    try:
        import easyocr  # noqa: F401
    except Exception:
        return False
    if weights_dir is None:
        return True
    d = Path(weights_dir)
    return d.is_dir() and any(d.rglob("*.pth"))


def run_ocr(cfg: dict, df: pd.DataFrame, images_root, budget_sec: float,
            weights_dir: str | Path | None = None,
            probe_items: int = 40) -> pd.Series | None:
    """Распознанный текст по товарам или None, если это не по карману."""
    oc = cfg.get("ocr_shippable") or cfg.get("ocr") or {}
    # Потолок кадров может задать сама отправка. Прогон №5 показал, что кадры 3-5
    # сигнала не добавляют (F1 редкой категории −1.3 п.п. при +0.7 п.п. PR-AUC, всё
    # внутри разброса по сидам), а времени стоят прямо пропорционально. Поэтому число
    # кадров — параметр варианта отправки, а не константа профиля распознавания.
    sub_cap = cfg.get("submission", {}).get("ocr_max_images")
    max_images = int(sub_cap) if sub_cap else int(oc.get("max_images", 2))
    max_side = int(oc.get("max_side", 768))
    max_chars = int(oc.get("max_chars_per_image", 400))
    langs = list(oc.get("languages", ["ru", "en"]))

    if images_root is None:
        print("папка с изображениями не найдена — распознавание пропущено", flush=True)
        return None

    try:
        import easyocr

        kwargs = {"gpu": True, "verbose": False}
        if weights_dir is not None and Path(weights_dir).is_dir():
            kwargs.update(model_storage_directory=str(weights_dir),
                          user_network_directory=str(weights_dir),
                          download_enabled=False)
        reader = easyocr.Reader(langs, **kwargs)
    except Exception as e:
        print(f"распознавание недоступно ({type(e).__name__}: {e}) — идём без него",
              flush=True)
        return None

    from ..data import image_paths, load_image

    idc = cfg["data"]["id_col"]
    ids = df[idc].astype(str).tolist()

    # ⚠ Поштучное чтение оставляет видеокарту простаивать: на H100 замерено 0.131 с на
    # кадр, тогда как модель на два миллиарда параметров по тем же фотографиям тратит
    # 0.12 с на ТОВАР. Время уходит не на вычисления, а на запуск ядер, декодирование и
    # изменение размера по одной картинке. Поэтому кадры собираются пачкой и подаются
    # разом; размер пачки — из конфига, чтобы под чужое железо его можно было поднять,
    # не трогая код.
    batch_size = max(1, int(oc.get("batch_size", 1)))
    # Значение в конфиге задано под 80 ГБ проверяющей системы. На нашей карте столько
    # кадров разом не поместится, а проверять код надо именно локально — поэтому здесь
    # батч ужимается по фактической видеопамяти, и один конфиг годится для обеих машин.
    try:
        import torch

        if torch.cuda.is_available():
            gb = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
            if gb < 40:
                small = max(4, int(batch_size * gb / 80))
                if small < batch_size:
                    print(f"видеопамяти {gb:.0f} ГБ — пачка кадров {batch_size} → {small}",
                          flush=True)
                    batch_size = small
    except Exception:
        pass

    def _clean(found) -> str:
        return " ".join(str(x).strip() for x in found if str(x).strip())[:max_chars]

    def read_batch(item_ids: list[str], n_frames: int) -> list[str]:
        """Текст по каждому товару пачки. Кадры всех товаров распознаются одним вызовом."""
        frames, owner = [], []
        for k, item_id in enumerate(item_ids):
            for p in image_paths(images_root, item_id)[:n_frames]:
                try:
                    frames.append(np.array(load_image(p, max_side=max_side)))
                    owner.append(k)
                except Exception:
                    pass
        parts: list[list[str]] = [[] for _ in item_ids]
        if frames:
            try:
                if batch_size > 1 and hasattr(reader, "readtext_batched"):
                    # Пачка требует одинакового размера кадров — его задаёт сам easyocr,
                    # если передать ширину и высоту.
                    side = max_side // 32 * 32
                    results = reader.readtext_batched(
                        frames, n_width=side, n_height=side, detail=0, batch_size=batch_size)
                else:
                    results = [reader.readtext(f, detail=0, paragraph=True) for f in frames]
            except Exception as e:
                # Пачечный путь может не поддерживаться сборкой easyocr в чужом образе.
                # Это не повод терять этап целиком — возвращаемся к поштучному чтению.
                print(f"пачечное распознавание недоступно ({type(e).__name__}: {e}) — "
                      "читаю по одному кадру", flush=True)
                results = [reader.readtext(f, detail=0, paragraph=True) for f in frames]
            for k, found in zip(owner, results):
                parts[k].append(_clean(found))
        return [" | ".join(t for t in p if t) or NO_TEXT for p in parts]

    def read_one(item_id: str, n_frames: int) -> str:
        return read_batch([item_id], n_frames)[0]

    # Сколько кадров мы можем себе позволить. Отказываться целиком, когда не влезают два
    # кадра, — расточительно: один кадр всё равно находит маркировку у части товаров, и
    # это лучше, чем ничего. Поэтому темп меряем НА КАДР и подбираем их число под бюджет.
    # ⚠ Мерить темп на первых попавшихся товарах нельзя: если у них нет фотографий,
    # замер вырождается в ноль кадров, прогноз выходит нулевым, защита считает чтение
    # бесплатным и не сокращает кадры. Дальше по выборке фотографии находятся, чтение
    # идёт на полной скорости и съедает весь бюджет — ровно так была потеряна отправка
    # 13.08 (в логе «0.0 кадра на товар, прогноз 0.0 мин», затем «бюджет исчерпан»).
    # Поэтому для замера берём товары, У КОТОРЫХ ФОТОГРАФИИ ЕСТЬ.
    probe: list[str] = []
    scanned = 0
    for item_id in ids:
        scanned += 1
        if image_paths(images_root, item_id):
            probe.append(item_id)
            if len(probe) >= probe_items:
                break
        if scanned >= max(PROBE_SCAN_LIMIT, probe_items):
            break
    if not probe:
        print(f"ни у одного из {scanned} проверенных товаров нет фотографий — "
              "распознавание пропущено", flush=True)
        return None
    if scanned > len(probe):
        print(f"для замера темпа просмотрено {scanned} товаров, с фотографиями "
              f"{len(probe)}", flush=True)

    t_probe = time.time()
    frames_seen = 0
    for item_id in probe:
        frames_seen += len(image_paths(images_root, item_id)[:max_images])
        read_one(item_id, max_images)
    per_frame = (time.time() - t_probe) / max(1, frames_seen)
    # Доля товаров с фотографиями по просмотренному куску — прогноз без неё завышен
    # ровно во столько раз, во сколько в выборке товаров без фото.
    with_photo_share = len(probe) / max(1, scanned)
    avg_frames = frames_seen / max(1, len(probe)) * with_photo_share
    projected = per_frame * avg_frames * len(ids)
    print(f"темп {per_frame:.3f} с/кадр, {avg_frames:.1f} кадра на товар, прогноз "
          f"{projected / 60:.1f} мин при бюджете {budget_sec / 60:.1f} мин", flush=True)

    if projected > budget_sec:
        affordable = budget_sec / max(1e-9, per_frame * len(ids))
        new_max = max(1, min(max_images, int(affordable)))
        if affordable < 1.0:
            print("даже один кадр не укладывается — распознавание отменено", flush=True)
            return None
        print(f"кадров на товар уменьшено {max_images} → {new_max}, новый прогноз "
              f"{per_frame * new_max * len(ids) / 60:.1f} мин", flush=True)
        max_images = new_max

    # ⚠ Прежняя защита сторожила бюджет и, исчерпав его, отказывалась от ВСЕЙ работы —
    # худший исход из возможных: время потрачено, результата нет, прогон остановлен.
    # Теперь темп пересчитывается по ходу дела, и при отставании сокращается число
    # кадров, а не выбрасывается сделанное. Дойти до конца обязаны в любом случае.
    texts: list[str] = []
    t0 = time.time()
    k = 0
    while k < len(ids):
        if k and max_images > 0:
            spent = time.time() - t0
            need = spent / k * (len(ids) - k)
            while need > budget_sec - spent and max_images > 0:
                max_images -= 1
                need *= max_images / max(1, max_images + 1)
                print(f"отстаём от бюджета на товаре {k}: кадров на товар "
                      f"{max_images + 1} → {max_images}", flush=True)
            if max_images == 0:
                print("оставшиеся товары идут без чтения фотографий — "
                      "иначе не уложиться в отведённое время", flush=True)
        # Товаров в пачке берём столько, чтобы кадров в одном вызове было около
        # заданного размера пачки: считает видеокарта именно кадрами.
        step = max(1, min(RECHECK_EVERY, batch_size // max(1, max_images)))
        chunk = ids[k:k + step]
        texts.extend(read_batch(chunk, max_images))
        k += len(chunk)

    out = pd.Series(texts, index=df.index, dtype=object)
    # Пустой результат по всей выборке означает, что читать было нечего: подавать такой
    # вход модели, обученной на распознанном тексте, — та же ошибка, что и отсутствие
    # признака, только незаметная. Лучше честно отказаться.
    empty_share = float((out == NO_TEXT).mean())
    if empty_share > 0.9:
        print(f"текст не найден у {empty_share * 100:.0f}% товаров — "
              "распознавание считаем неудавшимся", flush=True)
        return None
    print(f"распознавание готово за {(time.time() - t0) / 60:.1f} мин, "
          f"без текста {empty_share * 100:.0f}%", flush=True)
    return out


def run_vlm_transcript(cfg: dict, df: pd.DataFrame, budget_sec: float,
                       probe_items: int = 32) -> pd.Series | None:
    """Чтение упаковки моделью из /shared_models — путь без своего образа.

    В базовом образе организаторов нет ни easyocr, ни cv2, зато есть transformers и
    смонтированные модели из списка. Поэтому надписи читает та же VLM, что доступна
    решению бесплатно по бюджету архива. Защита по времени та же, что у easyocr.
    """
    try:
        from ..models.vlm import VlmTranscriber
    except Exception as e:
        print(f"чтение упаковки недоступно ({type(e).__name__}: {e})", flush=True)
        return None
    try:
        t0 = time.time()
        tr = VlmTranscriber(cfg)
        loaded_at = time.time()
        # оценка темпа на первых товарах, ПОСЛЕ загрузки модели: её разовая стоимость
        # не должна размазываться по паре десятков товаров и завышать прогноз
        probe = df.head(min(probe_items, len(df)))
        head_texts = tr.transcribe(probe)
        rate = (time.time() - loaded_at) / max(1, len(probe))
        projected = rate * len(df) + (loaded_at - t0)
        print(f"темп чтения упаковки {rate:.3f} с/товар, прогноз {projected / 60:.1f} мин "
              f"при бюджете {budget_sec / 60:.1f} мин", flush=True)
        if projected > budget_sec:
            print("прогноз выходит за бюджет — чтение отменено", flush=True)
            return None
        rest = df.iloc[len(probe):]
        texts = head_texts + (tr.transcribe(rest) if len(rest) else [])
        out = pd.Series(texts, index=df.index, dtype=object)
        out = out.where(out.str.strip().str.lower() != "нет", NO_TEXT)
        print(f"чтение упаковки готово за {(time.time() - t0) / 60:.1f} мин", flush=True)
        return out
    except Exception as e:
        print(f"чтение упаковки не удалось ({type(e).__name__}: {e})", flush=True)
        return None


def default_weights_dir(root: Path) -> Path | None:
    """Веса, положенные в архив рядом с кодом."""
    for candidate in (root / "artifacts" / "easyocr", root / "easyocr_models"):
        if candidate.is_dir():
            os.environ.setdefault("EASYOCR_MODULE_PATH", str(candidate))
            return candidate
    return None
