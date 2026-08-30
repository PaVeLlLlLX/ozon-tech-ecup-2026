"""Ответы языковой модели на вопросы о товаре, посчитанные ВО ВРЕМЯ ПРОГОНА.

Офлайн эти ответы строит `scripts/build_probe_answers.py` и кладёт в parquet, но тот
файл привязан к идентификаторам НАШИХ товаров: на закрытой выборке их нет, и решение
падало бы с «признаки-ответы включены, но файла нет». Поэтому на прогоне ответы
считаются заново, тем же промптом и той же моделью.

⚠ Картинки анкете не нужны и не читаются: ответ про свойства товара берётся из текста,
а кадры утроили бы время. Это же сделано в офлайн-построителе (`max_images: 0`), и
рассогласование входа здесь стоило бы дороже экономии.

Замер офлайн-построителя: 12 971 товар на восьми вопросах — 161 минута, то есть
0.093 с на вопрос-товар. На 3789 товарах это около 47 минут на RTX 3060 Ti и заметно
меньше на карте проверяющей системы.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

PROMPT_HEAD = (
    "Ты помогаешь модератору маркетплейса. Ниже карточка товара.\n\n"
    "{card}\n\nВопрос: {question}\nОтветь одним словом: Да или Нет."
)


def load_questions(path: str | Path) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "|" not in line:
            continue
        key, question = line.split("|", 1)
        out.append((key.strip(), question.strip()))
    return out


def build_qa_features(cfg: dict, df: pd.DataFrame, budget_sec: float,
                      questions_path: str | Path | None = None,
                      probe_items: int = 16) -> pd.DataFrame | None:
    """Кадр признаков qa_* по товарам df. None — если не уложились в бюджет.

    Возвращает None ТОЛЬКО по времени; любая другая беда поднимается наверх, чтобы
    решение упало, а не отдало вердикт без заявленных признаков.
    """
    from ..config import resolve_path
    from ..models.vlm import VlmScorer

    qpath = resolve_path(questions_path or cfg["baseline"].get(
        "qa_questions", "configs/prompts/qa_probes.txt"))
    questions = load_questions(qpath)
    if not questions:
        raise RuntimeError(f"список вопросов пуст: {qpath}")

    idc = cfg["data"]["id_col"]
    name_col, desc_col = cfg["data"]["name_col"], cfg["data"]["desc_col"]
    limit = int(cfg["vlm"].get("desc_max_chars", 900))
    cards = (df[name_col].astype(str) + "\nОписание: "
             + df[desc_col].astype(str).str.slice(0, limit)).tolist()

    t0 = time.time()
    # Картинки не нужны — max_images=0 отдаёт скорингу пустой список кадров.
    scorer = VlmScorer({**cfg, "vlm": {**cfg["vlm"], "max_images": 0}})
    loaded_at = time.time()

    # Прикидка темпа на горстке товаров: лучше отказаться заранее, чем оборваться на
    # середине и отдать половину признаков.
    probe = df.head(min(probe_items, len(df)))
    probe_cards = cards[:len(probe)]
    t_probe = time.time()
    _ = scorer.score(probe, [PROMPT_HEAD.format(card=c, question=questions[0][1])
                             for c in probe_cards], show_progress=False)
    rate = (time.time() - t_probe) / max(1, len(probe))
    projected = rate * len(df) * len(questions)
    left = budget_sec - (time.time() - t0)
    print(f"анкета: {len(questions)} вопросов, темп {rate:.3f} с/вопрос-товар, "
          f"прогноз {projected / 60:.1f} мин при остатке {left / 60:.1f} мин "
          f"(загрузка модели {loaded_at - t0:.0f} с)", flush=True)
    if projected > left:
        print("анкета не укладывается в остаток времени", flush=True)
        return None

    block: dict[str, np.ndarray] = {}
    for key, question in questions:
        texts = [PROMPT_HEAD.format(card=c, question=question) for c in cards]
        block[f"qa_{key}"] = scorer.score(df, texts, show_progress=False)
        print(f"  {key}: среднее {np.nanmean(block[f'qa_{key}']):.3f}", flush=True)
    out = pd.DataFrame(block)
    out.insert(0, idc, df[idc].astype(str).to_numpy())
    print(f"анкета готова за {(time.time() - t0) / 60:.1f} мин, "
          f"признаков {len(block)}", flush=True)
    return out
