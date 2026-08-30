"""Приёмы поверх уже обученного адаптера: переобучения не требуется, только вход.

Каждый режим — это дополнительные проходы той же модели с изменённым входом и
арифметика над логит-разностями. Веса не трогаются.

    contrast   разность «заявленная категория» против «чужая категория»
    symmetry   полусумма прямого и отрицательного вопроса
    selfctx    модель сперва называет предмет, потом решает, видя своё описание

⚠ Чего здесь СОЗНАТЕЛЬНО нет и почему. Опросник из вопросов-зондов проверялся ранее:
восемь ответов как признаки линейной модели дали PR-AUC редкой 0.6721/0.6431/0.6658
против 0.665 у опоры — ветка закрыта. Покадровая агрегация и аугментации на прогоне
тоже закрыты замерами. Повторять их незачем.

⚠ Шкала везде ЛОГИТ-РАЗНОСТЬ log P(Да) - log P(Нет), как и в основном скоринге:
вероятность в float32 слипается в 1.0 у трети выборки, и отбор по доле тогда
выбирает по ошибке последнего бита.

⚠ Модуль НЕ имеет запасного пути. Если приём не отработал, решение обязано упасть:
упавшая отправка попытку не тратит, а тихо выродившаяся тратит её и возвращает
число, которое мы примем за результат.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .vlm import (MAX_DESC, MAX_NAME, NO, RULES, YES, _grid_step, _load_from_packed,
                  load_with_adapter,
                  _load_image, _paths, _resolve_base, apply_lora, auto_batch,
                  base_used_in_training, facts_used_in_training, find_adapter_dir,
                  first_token_ids, images_used_in_training, pixels_used_in_training,
                  prompt_used_in_training, quiet_known_noise)

MODES = ("contrast", "symmetry", "selfctx", "selfctx_inside")

# Вопрос «чем товар ЯВЛЯЕТСЯ». Ответ модели подставляется обратно в промпт вердикта.
# Правила категории прямо говорят: решает не связь товара с темой, а то, что он собой
# представляет. Модель это различение умеет, но в общем вопросе оно тонет — приходится
# одновременно узнать предмет и применить к нему правило. Разделяем на два шага.
WHAT_QUESTION = (
    "Посмотри на фотографии и карточку товара.\n\n"
    "Название: {name}\n"
    "Описание: {desc}\n\n"
    "Назови, ЧЕМ ЯВЛЯЕТСЯ сам продаваемый предмет — двумя-четырьмя словами, без "
    "оценок и без упоминания категорий. Если это набор, назови в нём главное.\n"
    "Предмет:"
)
SELF_CONTEXT = "Осмотр показал, что это: {what}.\n\n"

# ⚠ Второй вопрос осмотра, и он лучше обоснован, чем первый. Замер на пятидесяти
# товарах редкой категории: признак, вытащенный из ответов модели на вопрос «что
# ВНУТРИ товара», отделяет классы с F1 0.653 против 0.529 у самого вердикта, и
# одиннадцать позитивов, которых вердикт не нашёл, в тексте описаны верно
# («внутри пиротехнический состав», «содержит собственное топливо»).
# ⚠ Вопрос «чем ЯВЛЯЕТСЯ предмет» такого замера не проходил — он взят по рассуждению.
# Поэтому режима два, и сравнивает их лидерборд, а не я.
INSIDE_QUESTION = (
    "Посмотри на фотографии и карточку товара.\n\n"
    "Название: {name}\n"
    "Описание: {desc}\n\n"
    "Что находится ВНУТРИ этого товара и от чего он работает? Назови коротко: "
    "собственное топливо, газ, пиротехнический состав, спички или горючее в "
    "комплекте — либо внешний источник, либо ничего из этого.\n"
    "Внутри:"
)

# Отрицательная форма основного вопроса. Всё, кроме последних строк, совпадает с прямой
# формой — иначе разность двух проходов мерила бы не перекос ответа, а разницу текстов.
NEGATIVE_TAIL = (
    "Товар НЕ относится к категории «{cat}»? Ответь одним словом: Да или Нет.\n"
    "Ответ:"
)


def _other_category(cat: str) -> str:
    """Вторая категория задачи. Их ровно две, поэтому «чужая» определена однозначно."""
    for k in RULES:
        if k != cat:
            return k
    raise RuntimeError("в правилах меньше двух категорий — контраст не построить")


class _Runner:
    """Загруженная модель плюс проход по выборке с произвольным текстом промпта."""

    def __init__(self, cfg: dict, adapter_path, base_override: str | None, batch: int):
        from pathlib import Path

        import torch
        from transformers import AutoProcessor

        self.torch = torch
        self.cfg = cfg
        adapter = find_adapter_dir(adapter_path)
        if adapter is None:
            raise RuntimeError(f"адаптера нет в {adapter_path}")
        self.adapter = adapter
        quiet_known_noise()
        src = str(base_override) if base_override else _resolve_base(
            base_used_in_training(adapter))
        if not Path(src).is_dir():
            raise RuntimeError(f"нет каталога весов {src}")
        if torch.cuda.is_available():
            device, dtype = "cuda", torch.bfloat16
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device, dtype = "mps", torch.bfloat16
        else:
            device, dtype = "cpu", torch.float32
        self.proc = AutoProcessor.from_pretrained(src, local_files_only=True)
        if not (Path(src) / "packing.json").exists():
            raise RuntimeError("ожидались упакованные веса с packing.json")
        # ⚠ Та же сборка, что и на проходе вердикта, и тот же кэш: если этот адаптер
        # уже оценивал товары, распаковка не повторится.
        self.model = load_with_adapter(src, adapter, dtype, device)

        tok = self.proc.tokenizer
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        self.tok = tok
        self.yes_ids, self.no_ids = first_token_ids(tok, YES), first_token_ids(tok, NO)
        if set(self.yes_ids) & set(self.no_ids):
            raise RuntimeError("«Да» и «Нет» делят первый токен — шкала неопределена")
        self.step = _grid_step(self.proc)
        self.px = pixels_used_in_training(adapter)
        self.n_img = int(images_used_in_training(adapter) or 5)
        self.prompt_name, self.template = prompt_used_in_training(adapter)
        self.with_facts = facts_used_in_training(adapter)
        self.batch = auto_batch(self.model.device, self.n_img, asked=batch,
                                max_pixels=self.px)
        print(f"экзотика: устройство {device}, кадров {self.n_img}, пикселей {self.px}, "
              f"батч {self.batch}, промпт «{self.prompt_name}»", flush=True)

    def _fields(self, row) -> dict:
        d = self.cfg["data"]
        return {"name": str(row[d["name_col"]])[:MAX_NAME],
                "desc": str(row.get(d["desc_col"]) or "")[:MAX_DESC],
                "cat": str(row[d["category_col"]])}

    def _wrap(self, text: str, n_imgs: int) -> str:
        content = ([{"type": "image"} for _ in range(n_imgs)]
                   + [{"type": "text", "text": text}])
        return self.proc.apply_chat_template([{"role": "user", "content": content}],
                                             tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)

    def text_for(self, row, *, category=None, negative=False, what=None) -> str:
        """Текст промпта. Шаблон обучения не подменяется — только дополняется."""
        f = self._fields(row)
        if what == "__ask__":
            # Какой из двух вопросов осмотра задавать, решает режим — см. INSIDE_QUESTION.
            tpl_ask = (INSIDE_QUESTION if getattr(self, "ask_inside", False)
                       else WHAT_QUESTION)
            return tpl_ask.format(**f)
        cat = category or f["cat"]
        tpl = self.template
        fields = {"cat": cat, "rules": RULES.get(cat, ""),
                  "name": f["name"], "desc": f["desc"]}
        if "{facts}" in tpl:
            from ..facts import facts_block

            d = self.cfg["data"]
            fields["facts"] = (facts_block(row[d["name_col"]], row.get(d["desc_col"]), cat)
                               if self.with_facts else "")
        text = tpl.format(**fields)
        if what:
            # ⚠ Вставляем ПЕРЕД вопросом, а не в начало: описание должно стоять
            # непосредственно перед решением, а шапка с правилами остаётся на месте.
            marker = "Решает не то" if "Решает не то" in text else "Товар действительно"
            if marker in text:
                head, _, tail = text.partition(marker)
                text = head + SELF_CONTEXT.format(what=what) + marker + tail
            else:
                raise RuntimeError("не нашёл, куда вставить самоописание в шаблон")
        if negative:
            marker = ("Товар действительно относится" if "Товар действительно" in text
                      else "Относится ли товар")
            if marker not in text:
                raise RuntimeError("не нашёл прямой вопрос в шаблоне — симметрия невозможна")
            text = text.split(marker)[0] + NEGATIVE_TAIL.format(cat=cat)
        return text

    def score(self, df: pd.DataFrame, images_root, *, category_of=None,
              negative: bool = False, whats: list | None = None) -> np.ndarray:
        """Один проход по выборке. whats — самоописания предмета на каждую строку."""
        idc = self.cfg["data"]["id_col"]
        out: list[float] = []
        for start in range(0, len(df), self.batch):
            chunk = df.iloc[start:start + self.batch]
            items = []
            for k, (_, row) in enumerate(chunk.iterrows()):
                imgs = []
                for p in _paths(images_root, str(row[idc]), self.n_img):
                    try:
                        imgs.append(_load_image(p, self.step, self.px))
                    except Exception:
                        pass
                cat = category_of(row) if category_of else None
                w = whats[start + k] if whats else None
                text = self.text_for(row, category=cat, negative=negative, what=w)
                items.append((self._wrap(text, len(imgs)), imgs))
            out.extend(self._forward(items))
            for _, im in items:
                for x in im:
                    try:
                        x.close()
                    except Exception:
                        pass
        arr = np.asarray(out, dtype=np.float32)
        if len(arr) != len(df):
            raise RuntimeError(f"проход вернул {len(arr)} оценок на {len(df)} строк")
        return arr

    def describe(self, df: pd.DataFrame, images_root, max_new_tokens: int = 16) -> list:
        """Самоописание предмета: одна короткая генерация на товар."""
        torch = self.torch
        idc = self.cfg["data"]["id_col"]
        self.model.config.use_cache = True
        out: list = []
        b = max(1, self.batch // 2)
        for start in range(0, len(df), b):
            chunk = df.iloc[start:start + b]
            prompts, images = [], []
            for _, row in chunk.iterrows():
                imgs = []
                for p in _paths(images_root, str(row[idc]), self.n_img):
                    try:
                        imgs.append(_load_image(p, self.step, self.px))
                    except Exception:
                        pass
                prompts.append(self._wrap(self.text_for(row, what="__ask__"), len(imgs)))
                images.append(imgs)
            flat = [im for im in images if im]
            enc = self.proc(text=prompts, images=flat or None, padding=True,
                            return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                gen = self.model.generate(**enc, max_new_tokens=max_new_tokens,
                                          do_sample=False, num_beams=1,
                                          pad_token_id=self.tok.pad_token_id)
            new = gen[:, enc["input_ids"].shape[1]:]
            for t in self.tok.batch_decode(new, skip_special_tokens=True):
                t = " ".join(str(t).split())[:80].strip(" .,;:\n")
                out.append(t or None)
            for im in images:
                for x in im:
                    try:
                        x.close()
                    except Exception:
                        pass
        self.model.config.use_cache = False
        return out

    def _forward(self, items):
        """Логит-разность батчем, с дроблением при нехватке памяти."""
        torch = self.torch

        def run(part):
            pr = [t for t, _ in part]
            im = [i for _, i in part if i]
            try:
                enc = self.proc(text=pr, images=im or None, padding=True,
                                return_tensors="pt").to(self.model.device)
                with torch.no_grad():
                    lg = self.model(**enc).logits[:, -1, :].float()
            except Exception as e:
                oom = (type(e).__name__ == "OutOfMemoryError"
                       or "out of memory" in str(e).lower())
                if not oom or len(part) == 1:
                    raise
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                half = len(part) // 2
                return run(part[:half]) + run(part[half:])
            if not torch.isfinite(lg).all():
                raise RuntimeError("нечисловые логиты")
            yes = torch.logsumexp(lg[:, self.yes_ids], dim=-1)
            no = torch.logsumexp(lg[:, self.no_ids], dim=-1)
            return (yes - no).cpu().numpy().tolist()

        return run(items)


def exotic_scores(mode: str, df: pd.DataFrame, cfg: dict, images_root, adapter_path,
                  *, base_override: str | None = None, batch: int = 32) -> np.ndarray:
    """Оценка выбранным приёмом. Падает при любой беде — отката здесь нет намеренно."""
    if mode not in MODES:
        raise RuntimeError(f"неизвестный приём «{mode}», есть только {MODES}")
    t0 = time.time()
    r = _Runner(cfg, adapter_path, base_override, batch)
    cat_col = cfg["data"]["category_col"]

    if mode == "contrast":
        # Своя категория против чужой. Модель, которая просто склонна отвечать «Да»,
        # получает одинаковую надбавку в обоих проходах, и разность её убирает.
        own = r.score(df, images_root)
        alien = r.score(df, images_root,
                        category_of=lambda row: _other_category(str(row[cat_col])))
        out = own - alien
        print(f"контраст: своя {own.mean():+.3f}, чужая {alien.mean():+.3f}, "
              f"разность {out.mean():+.3f}", flush=True)

    elif mode == "symmetry":
        # Прямой вопрос и отрицательный. У согласованной модели знаки противоположны;
        # общий перекос «Да/Нет» входит в оба с одним знаком и в полуразности гаснет.
        direct = r.score(df, images_root)
        inverse = r.score(df, images_root, negative=True)
        out = (direct - inverse) / 2.0
        print(f"симметрия: прямой {direct.mean():+.3f}, отрицательный "
              f"{inverse.mean():+.3f}, итог {out.mean():+.3f}", flush=True)

    else:  # selfctx и selfctx_inside
        # Два шага вместо одного. Сначала модель называет, ЧЕМ является предмет, потом
        # отвечает на вопрос о категории, видя своё же описание перед вопросом.
        r.ask_inside = (mode == "selfctx_inside")
        # Ответ про содержимое длиннее, чем «чем является предмет», — даём запас токенов.
        whats = r.describe(df, images_root, max_new_tokens=24 if r.ask_inside else 16)
        print(f"вопрос осмотра: {'что ВНУТРИ товара' if r.ask_inside else 'чем ЯВЛЯЕТСЯ предмет'}",
              flush=True)
        named = sum(1 for w in whats if w)
        if named < len(df) * 0.5:
            raise RuntimeError(f"самоописание получено только у {named} из {len(df)}")
        print(f"самоописание у {named} из {len(df)}, примеры: "
              f"{[w for w in whats if w][:3]}", flush=True)
        out = r.score(df, images_root, whats=whats)
        print(f"с самоописанием: среднее {out.mean():+.3f}", flush=True)

    print(f"приём «{mode}» готов за {(time.time() - t0) / 60:.1f} мин, "
          f"{len(out)} оценок", flush=True)
    return np.asarray(out, dtype=np.float32)
