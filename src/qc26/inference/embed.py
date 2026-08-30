"""Мультимодальные эмбеддинги на инференсе: то же, что при обучении, но в контейнере.

Веса энкодера в архив НЕ кладём — организаторы монтируют модели в /shared_models
(переменная SHARED_MODELS_PATH). Интернета в контейнере нет, поэтому загрузка
строго локальная.

Раскладка обязана совпадать с обучением до мелочей, иначе голова получит вектор
из другого пространства и вердикт будет случайным:
  - текст: «Категория / Название / Описание», описание обрезано до 800 символов;
  - до 2 фото, площадь кадра ≤ 100352 пикселей, стороны кратны шагу сетки;
  - плейсхолдеры дописываются ПОСЛЕ текста;
  - пулинг — среднее по токенам с маской (не последний токен: замерено, что для
    обучаемой головы среднее лучше).

Любой сбой обязан возвращать None, а не бросать исключение: решение с откатом на
tf-idf хуже, чем решение с эмбеддингами, но несравнимо лучше, чем упавшее.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

MAX_PIXELS = 100352          # пресет S — замерено, что L не даёт прироста и нестабилен
MAX_IMAGES = 2
MAX_DESC = 800
DEFAULT_MODEL = "Qwen/Qwen3-VL-Embedding-2B"


def shared_model_path(name: str = DEFAULT_MODEL) -> Path:
    root = os.environ.get("SHARED_MODELS_PATH", "/shared_models")
    return Path(root) / name


def has_weights(path: Path) -> bool:
    """Есть ли в каталоге настоящие веса, а не только конфиги.

    ⚠ Проверять существование ПАПКИ недостаточно: пустой или недокачанный каталог
    её проходит, а загрузчик падает уже внутри с «no file named model.safetensors».
    Именно так прогон на второй машине упал после успешной проверки пути.
    """
    if not path.is_dir():
        return False

    # Разрешить symlink'и — модели могут быть смонтированы
    try:
        if path.is_symlink():
            path = path.resolve()
    except (OSError, RuntimeError):
        return False

    # Стандартные форматы весов Hugging Face моделей:
    # - model.safetensors (один файл), или
    # - model-00001-of-00003.safetensors и т.д. (шардированный)
    # - pytorch_model.bin (один файл), или
    # - pytorch_model-00001.bin и т.д. (шардированный)
    # - model.index.json или pytorch_model.bin.index.json (указатель на шарды)
    for pattern in (
        "model.safetensors",
        "model-*.safetensors",
        "pytorch_model.bin",
        "pytorch_model-*.bin",
        "*.index.json"
    ):
        try:
            if next(path.glob(pattern), None) is not None:
                return True
        except (OSError, PermissionError):
            pass

    return False


def resolve_local_model(name: str = DEFAULT_MODEL) -> str:
    """Путь к весам, которые ГАРАНТИРОВАННО лежат на диске. Иначе — понятный отказ.

    Порядок поиска: общий каталог проверяющей системы, затем локальный кэш
    HuggingFace (нужен для наших прогонов на своей машине). Сети не касаемся:
    в контейнере её нет, а попытка обращения превращается в таймаут.
    """
    import os

    tried = []
    path = shared_model_path(name)
    tried.append(path)
    if has_weights(path):
        return str(path)

    cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    hub = cache / "hub" / ("models--" + name.replace("/", "--"))
    snaps = sorted((hub / "snapshots").glob("*")) if (hub / "snapshots").is_dir() else []
    for snap in reversed(snaps):
        tried.append(snap)
        if has_weights(snap):
            print(f"весов нет в {path}, беру кэш: {snap}", flush=True)
            return str(snap)

    detail = []
    for p in tried:
        if p.is_dir():
            inside = sorted(x.name for x in p.iterdir())[:8]
            detail.append(f"    {p} — есть, но без весов, внутри: {inside}")
        else:
            detail.append(f"    {p} — нет такого пути")
    raise FileNotFoundError(
        f"весов {name} не найдено.\n" + "\n".join(detail) +
        f"\n  Скачать: python -c \"from huggingface_hub import snapshot_download; "
        f"snapshot_download('{name}')\"\n"
        f"  Либо указать каталог: export SHARED_MODELS_PATH=/путь/к/моделям")


def _grid_step(proc) -> int:
    """Шаг сетки кадра = размер патча × коэффициент слияния.

    ⚠ Жёсткое 28 (соглашение Qwen2-VL) роняет раскладку картиночных признаков
    с «index out of bounds» — у этой модели патч 16 и слияние 2, то есть 32.
    """
    ip = getattr(proc, "image_processor", proc)
    patch = int(getattr(ip, "patch_size", 16) or 16)
    merge = int(getattr(ip, "merge_size", 2) or 2)
    return patch * merge


def _resize(img, max_pixels: int, step: int):
    from PIL import Image

    w, h = img.size
    if w * h <= max_pixels and w % step == 0 and h % step == 0:
        return img
    scale = (max_pixels / (w * h)) ** 0.5 if w * h > max_pixels else 1.0
    floor = step * 2
    nw = max(floor, (int(w * scale) // step) * step)
    nh = max(floor, (int(h * scale) // step) * step)
    return img.resize((nw, nh), Image.LANCZOS)


def build_text(row, cfg: dict) -> str:
    d = cfg["data"]
    return (f"Категория: {row[d['category_col']]}\n"
            f"Название: {str(row[d['name_col']])[:300]}\n"
            f"Описание: {str(row.get(d['desc_col']) or '')[:MAX_DESC]}")


def embed_frame(df: pd.DataFrame, cfg: dict, images_root, *, batch: int = 32,
                model_name: str = DEFAULT_MODEL, verbose: bool = True):
    """Вектор на товар. Возвращает (N, d) или None, если энкодер недоступен."""
    try:
        import torch
        from PIL import Image
        from transformers import AutoModel, AutoProcessor

        from ..data import image_paths

        # ⚠ Только локальная загрузка, БЕЗ попытки скачивания. В команде это уже стоило
        # трёх попыток: построитель эмбеддингов передавал имя модели прямо в загрузчик,
        # на машине разработчика срабатывал кэш HuggingFace, а в контейнере интернета
        # нет — этап падал, и решение отдавало запасной текстовый вердикт. Три отправки
        # дали одинаковые до десятого знака 0.7177160719, и это заметили не сразу.
        # Здесь: путь ищется в общем каталоге, при отсутствии — громкий отказ сразу,
        # без сетевых таймаутов, которые ещё и съедали бы бюджет времени.
        src = resolve_local_model(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        proc = AutoProcessor.from_pretrained(src, trust_remote_code=True,
                                            local_files_only=True)
        model = AutoModel.from_pretrained(src, dtype=dtype, trust_remote_code=True,
                                          local_files_only=True).to(device).eval()
        # ⚠ батч подбирается под карту, а не берётся из конфига вслепую. На Windows
        # нехватка видеопамяти не даёт честной ошибки: драйвер начинает возить
        # тензоры через системную память, и прогон не падает, а встаёт намертво
        # (батч 32 на 6 ГБ не осилил 700 товаров за девять минут). В контейнере
        # H100 на 80 ГБ, там ограничение не срабатывает.
        if device == "cuda":
            total = torch.cuda.get_device_properties(0).total_memory
            if total < 16e9:
                capped = max(2, min(batch, 4))
                if capped != batch:
                    print(f"видеопамяти {total / 1e9:.0f} ГБ — батч снижен "
                          f"{batch} → {capped}", flush=True)
                batch = capped

        step = _grid_step(proc)
        start = getattr(proc, "vision_start_token", "<|vision_start|>")
        end = getattr(proc, "vision_end_token", "<|vision_end|>")
        image_tok = getattr(proc, "image_token", "<|image|>")
        placeholder = f"{start}{image_tok}{end}"
        if verbose:
            print(f"энкодер: {src}, устройство {device}, шаг сетки {step}", flush=True)
    except Exception as e:
        print(f"энкодер недоступен ({type(e).__name__}: {str(e)[:160]}) — "
              f"работаем без эмбеддингов", flush=True)
        return None

    idc = cfg["data"]["id_col"]
    out: list[np.ndarray] = []
    fails = 0
    t0 = time.time()
    for start_i in range(0, len(df), batch):
        chunk = df.iloc[start_i:start_i + batch]
        texts, images = [], []
        for _, row in chunk.iterrows():
            texts.append(build_text(row, cfg))
            imgs = []
            for p in image_paths(images_root, str(row[idc]))[:MAX_IMAGES]:
                try:
                    imgs.append(_resize(Image.open(p).convert("RGB"), MAX_PIXELS, step))
                except Exception:
                    pass
            images.append(imgs)
        try:
            import torch

            if any(len(im) > 0 for im in images):
                prompts = [t + placeholder * len(im) for t, im in zip(texts, images)]
                enc = proc(text=prompts, images=images, padding=True, truncation=True,
                           return_tensors="pt")
            else:
                enc = proc(text=texts, images=None, padding=True, truncation=True,
                           return_tensors="pt")
            enc = enc.to(model.device)
            with torch.no_grad():
                hidden = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).expand(hidden.size()).to(hidden.dtype)
            vec = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            out.append(vec.float().cpu().numpy().astype(np.float32))
        except Exception as e:
            # ⚠ товар без вектора терять нельзя: строка ответа нужна для каждого.
            # Нули после нормировки дадут «признаков нет», и сработает tf-idf.
            fails += len(chunk)
            if fails <= len(chunk) * 2:
                print(f"сбой батча ({type(e).__name__}): {str(e)[:140]}", flush=True)
            out.append(np.zeros((len(chunk), out[0].shape[1] if out else 2048),
                                dtype=np.float32))
        for im in images:
            for x in im:
                try:
                    x.close()
                except Exception:
                    pass

    emb = np.concatenate(out, axis=0) if out else None
    if emb is None:
        return None
    bad = ~np.isfinite(emb).all(axis=1)
    if bad.any():
        good = emb[~bad]
        emb[bad] = good.mean(axis=0) if len(good) else 0.0
    if verbose:
        print(f"эмбеддинги готовы: {emb.shape}, {(time.time() - t0) / max(1, len(df)) * 1000:.0f} мс/товар"
              + (f", сбоев {fails}" if fails else ""), flush=True)
    return emb
