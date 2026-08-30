"""Построение эмбеддингов во время прогона решения — с защитой по времени.

Отличие от чтения текста с фото: здесь один прямой проход на товар без генерации,
поэтому этап предсказуемо быстрый. Но полагаться на это вслепую нельзя — железо чужое,
и та же схема страховок применяется и тут: замер темпа на первых товарах, прогноз на
весь объём, отказ вместо риска сорвать лимит.

Модель берётся из /shared_models: она в списке организаторов, значит смонтирована
заранее и бюджет архива не тратит.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd


# Какие наборы векторов умеет строить решение. «multimodal» — карточка вместе с
# фотографиями одним вектором (так делали все прежние отправки); «text» и «image» —
# по отдельности, для позднего слияния.
SET_MODES = {"multimodal": (True, True), "text": (False, True), "image": (True, False)}
#                           картинки, текст


def build_embedding_sets(cfg: dict, df: pd.DataFrame, budget_sec: float,
                         sets: list[str], probe_items: int = 24, images_root=None):
    """Несколько наборов векторов за ОДНУ загрузку модели.

    Позднему слиянию нужны два набора — текстовый и картиночный, — а загрузка модели
    занимает заметное время, поэтому она делается один раз. Возвращается словарь
    {имя набора: матрица} либо None, если хотя бы один набор не по карману: слиянию
    нужны оба, и отдавать половину нельзя.
    """
    unknown = [s for s in sets if s not in SET_MODES]
    if unknown:
        raise RuntimeError(f"неизвестные наборы векторов: {unknown}")
    try:
        from ..models.embeddings import MultimodalEmbedder
    except Exception as e:
        print(f"эмбеддинги недоступны ({type(e).__name__}: {e})", flush=True)
        return None

    embedder = None
    try:
        t0 = time.time()
        embedder = MultimodalEmbedder(cfg)
        loaded_at = time.time()
        print(f"модель загружена за {loaded_at - t0:.0f} с, наборов к построению "
              f"{len(sets)}: {', '.join(sets)}", flush=True)

        out: dict[str, np.ndarray] = {}
        spent_load = loaded_at - t0
        for name in sets:
            with_images, with_text = SET_MODES[name]
            t_set = time.time()
            probe = df.head(min(probe_items, len(df)))
            head = embedder.encode_frame(probe, with_images=with_images,
                                         with_text=with_text, root=images_root)
            rate = (time.time() - t_set) / max(1, len(probe))
            projected = rate * len(df)
            left = budget_sec - (time.time() - t0)
            print(f"  набор «{name}»: темп {rate:.3f} с/товар, прогноз "
                  f"{projected / 60:.1f} мин при остатке {left / 60:.1f} мин", flush=True)
            if projected > left:
                print(f"  набор «{name}» не укладывается в остаток времени", flush=True)
                return None
            rest = df.iloc[len(probe):]
            tail = (embedder.encode_frame(rest, with_images=with_images,
                                          with_text=with_text, root=images_root)
                    if len(rest) else np.zeros((0, head.shape[1]), dtype=np.float32))
            out[name] = np.vstack([head, tail]) if len(tail) else head
            print(f"  набор «{name}» готов за {(time.time() - t_set) / 60:.1f} мин, "
                  f"форма {out[name].shape}", flush=True)
        print(f"все наборы готовы за {(time.time() - t0) / 60:.1f} мин "
              f"(из них загрузка {spent_load / 60:.1f} мин)", flush=True)
        return out
    except Exception as e:
        print(f"эмбеддинги не удались ({type(e).__name__}: {e})", flush=True)
        return None
    finally:
        if embedder is not None:
            try:
                embedder.close()
            except Exception:
                pass


def build_embeddings(cfg: dict, df: pd.DataFrame, budget_sec: float,
                     probe_items: int = 24, images_root=None) -> np.ndarray | None:
    """Матрица эмбеддингов по всем строкам df или None, если это не по карману."""
    try:
        from ..models.embeddings import MultimodalEmbedder
    except Exception as e:
        print(f"эмбеддинги недоступны ({type(e).__name__}: {e})", flush=True)
        return None

    try:
        t0 = time.time()
        embedder = MultimodalEmbedder(cfg)
        loaded_at = time.time()
        print(f"модель загружена за {loaded_at - t0:.0f} с", flush=True)

        probe = df.head(min(probe_items, len(df)))
        head = embedder.encode_frame(probe, root=images_root)
        # Темп меряем ПОСЛЕ загрузки модели: иначе её разовая стоимость размазывается
        # по двум десяткам товаров, темп завышается в разы, и этап отменяется зря.
        rate = (time.time() - loaded_at) / max(1, len(probe))
        projected = rate * len(df) + (loaded_at - t0)
        print(f"темп эмбеддингов {rate:.3f} с/товар, прогноз {projected / 60:.1f} мин "
              f"при бюджете {budget_sec / 60:.1f} мин", flush=True)
        if projected > budget_sec:
            print("прогноз выходит за бюджет — эмбеддинги отменены", flush=True)
            embedder.close()
            return None

        rest = df.iloc[len(probe):]
        tail = embedder.encode_frame(rest, root=images_root) if len(rest) else np.zeros((0, head.shape[1]),
                                                                     dtype=np.float32)
        embedder.close()
        out = np.vstack([head, tail]) if len(tail) else head
        print(f"эмбеддинги готовы за {(time.time() - t0) / 60:.1f} мин, форма {out.shape}",
              flush=True)
        return out
    except Exception as e:
        print(f"эмбеддинги не удались ({type(e).__name__}: {e})", flush=True)
        return None
