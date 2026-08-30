"""Мультимодальные эмбеддинги карточки: текст и до пяти изображений одним проходом.

Единственная ветка зрения, которую мы ещё не пробовали. Генеративная VLM со скорингом
«Да/Нет» проиграла тексту во всех четырёх конфигурациях, но это не приговор зрению:
эмбеддинги решают другую задачу — они отдают представление, а разделяющую границу
строит уже обычный классификатор, которому 198 позитивов хватает.

Схема сборки входа повторяет baseline организаторов (текст + плейсхолдеры кадров в одном
проходе, усреднение по маске внимания) — она проверена именно с этой моделью, а
самодеятельность здесь ничего не улучшит.

На инференсе это один прямой проход на товар: на H100 быстро и заведомо в лимите, в
отличие от распознавания текста.
"""
from __future__ import annotations

import gc

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ..data import image_paths, images_root, load_image


def _pool(hidden: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    """Усреднение по токенам с учётом маски внимания (padding не должен влиять)."""
    m = mask.unsqueeze(-1).expand(hidden.size()).to(hidden.dtype)
    summed = (hidden * m).sum(dim=1)
    counts = m.sum(dim=1).clamp(min=1e-9)
    return (summed / counts).squeeze(1).float().cpu().numpy().astype(np.float32)


class MultimodalEmbedder:
    def __init__(self, cfg: dict):
        from transformers import AutoModel, AutoProcessor

        from .vlm_backends import resolve_model_id

        ec = cfg["embeddings"]
        self.cfg = cfg
        self.ec = ec
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_images = int(ec["max_images"])
        self.max_side = int(ec["image_max_side"])

        # Идентификатор обязан пройти через поиск в общем каталоге весов: у проверяющей
        # системы моделей нет ни в кэше, ни в интернете, они смонтированы в /shared_models.
        # Без этой строки решение на эмбеддингах молча уходило в сеть и вырождалось в
        # текстовое — так и потерялась попытка.
        model_id = resolve_model_id(ec["model_id"])
        kwargs = {"max_pixels": int(ec["max_pixels"])} if ec.get("max_pixels") else {}
        self.processor = AutoProcessor.from_pretrained(model_id, **kwargs)
        self.model = AutoModel.from_pretrained(
            model_id, dtype=torch.float16, trust_remote_code=True).to(self.device).eval()

        p = self.processor
        self.placeholder = (getattr(p, "vision_start_token", "<|vision_start|>")
                            + getattr(p, "image_token", "<|image|>")
                            + getattr(p, "vision_end_token", "<|vision_end|>"))

    def _text_of(self, row: pd.Series) -> str:
        limit = int(self.ec["desc_max_chars"])
        return (f"Название: {row.get('name', '')}\n"
                f"Категория: {row.get('category', '')}\n"
                f"Описание: {str(row.get('description', ''))[:limit]}")

    @torch.no_grad()
    def _encode(self, texts: list[str], images: list[list]) -> np.ndarray:
        prompts = [t + self.placeholder * len(im) for t, im in zip(texts, images)]
        flat = [im for group in images for im in group]
        inputs = self.processor(text=prompts, images=flat or None, padding=True,
                                truncation=True, return_tensors="pt").to(self.device)
        out = self.model(**inputs)
        return _pool(out.last_hidden_state, inputs["attention_mask"])

    @torch.no_grad()
    def encode_frame(self, df: pd.DataFrame, with_images: bool = True,
                     with_text: bool = True, root=None) -> np.ndarray:
        """Эмбеддинги для всех строк df в исходном порядке.

        ⚠ `root` — корень изображений ПРОГОНА. Без него путь брался из конфига
        (`paths.data_csv`), а на сдаче данные лежат в чужом месте: внутри архива папки
        `data/` нет вовсе. Картинки не находились, и модель кодировала пустоту — молча,
        без единой ошибки. Именно так, судя по всему, отработали обе прежние отправки
        на эмбеддингах (0.6663 и 0.5300): их веса учились на векторах С картинками, а
        на проверке получали векторы БЕЗ них.
        """
        root = root if root is not None else images_root(self.cfg)
        if with_images and root is None:
            raise RuntimeError(
                "корень изображений не найден, а набор векторов требует картинок. "
                "Кодировать пустоту нельзя: веса учились на других векторах")
        idc = self.cfg["data"]["id_col"]
        # ⚠ Батч подбирается под H100 80 ГБ проверяющей системы, а не под нашу карту:
        # при батче 4 видеокарта простаивает, ожидая, пока процессор раскодирует
        # следующие два десятка кадров. Локально он сам ужимается по нехватке памяти
        # (ветка OutOfMemoryError ниже), поэтому большое значение проверке не мешает.
        from .vlm_backends import auto_batch

        batch = auto_batch(int(self.ec.get("inference_batch_size", 64)),
                           min_batch=int(self.ec["batch_size"]))
        # Декодирование кадров — процессорная работа, и раньше она шла последовательно
        # ВНУТРИ цикла батчей: пока грузились картинки, видеокарта ждала. Теперь кадры
        # следующего батча читаются параллельно счёту текущего.
        workers = int(self.ec.get("loader_workers", 8))
        chunks: list[np.ndarray] = []

        rows = list(df.iterrows())

        def load_group(row) -> list:
            if not with_images:
                return []
            out = []
            for p in image_paths(root, str(row[idc]))[:self.max_images]:
                try:
                    out.append(load_image(p, max_side=self.max_side))
                except Exception:
                    pass
            return out

        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for start in tqdm(range(0, len(rows), batch), desc="эмбеддинги", unit="батч"):
                part = rows[start:start + batch]
                # Без текста остаётся только картинка: нужно, чтобы измерить вклад
                # изображений ОТДЕЛЬНО. Все прежние заходы кодировали карточку вместе с
                # фотографиями одним вектором, и разделить их вклад было нечем.
                texts = [self._text_of(r) if with_text else "Товар:" for _, r in part]
                images = list(pool.map(load_group, [r for _, r in part]))
                chunks.append(self._encode_adaptive(texts, images))
                for group in images:
                    for im in group:
                        try:
                            im.close()
                        except Exception:
                            pass
        return np.vstack(chunks) if chunks else np.zeros((0, 0), dtype=np.float32)

    def _encode_adaptive(self, texts: list, images: list) -> np.ndarray:
        """Счёт батча с постепенным дроблением при нехватке видеопамяти.

        Большой батч ставится ради H100; на нашей карте он не влезет, поэтому здесь он
        делится пополам, пока не пройдёт. Прежняя версия сразу падала до поштучной
        обработки — это спасало от падения, но было втрое медленнее необходимого.
        """
        try:
            return self._encode(texts, images)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(texts) == 1:
                # один товар не влез даже целиком — считаем его без картинок
                return self._encode(texts, [[]])
            half = len(texts) // 2
            left = self._encode_adaptive(texts[:half], images[:half])
            right = self._encode_adaptive(texts[half:], images[half:])
            return np.vstack([left, right])

    def close(self) -> None:
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def save_embeddings(path, ids: pd.Series, emb: np.ndarray) -> None:
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path.with_suffix(".npy"), emb.astype(np.float32))
    pd.DataFrame({"id": ids.astype(str).to_numpy()}).to_parquet(
        path.with_suffix(".ids.parquet"), index=False)


def load_embeddings(path) -> tuple[pd.Series, np.ndarray]:
    from pathlib import Path

    path = Path(path)
    emb = np.load(path.with_suffix(".npy"))
    ids = pd.read_parquet(path.with_suffix(".ids.parquet"))["id"].astype(str)
    return ids, emb


def align_embeddings(path, ids: pd.Series) -> np.ndarray | None:
    """Матрица эмбеддингов, выстроенная под порядок переданных идентификаторов."""
    from pathlib import Path

    if not Path(path).with_suffix(".npy").exists():
        return None
    src_ids, emb = load_embeddings(path)
    pos = {v: k for k, v in enumerate(src_ids)}
    idx = [pos.get(str(i), -1) for i in ids]
    out = np.zeros((len(ids), emb.shape[1]), dtype=np.float32)
    known = [k for k, j in enumerate(idx) if j >= 0]
    out[known] = emb[[idx[k] for k in known]]
    return out
