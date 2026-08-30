"""LoRA SFT и скоринг VLM под задачу 2026.

Модель отвечает одним словом «Да»/«Нет» на вопрос «относится ли товар к заявленной
категории». Вероятность берём из логитов первого токена ответа, а не из сгенерированного
текста: так получается непрерывный скор, по которому можно двигать порог — а порог у нас
и есть узкое место метрики.

Обучение — QLoRA в 4 битах на 8 ГБ; инференс в контейнере — bf16 на H100.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from ..data import image_paths, images_root, load_image

ANSWER_YES = "Да"
ANSWER_NO = "Нет"
YES_VARIANTS = ["Да", " Да", "да", "Yes", " Yes"]
NO_VARIANTS = ["Нет", " Нет", "нет", "No", " No"]


def _first_token_ids(tok, variants: list[str]) -> list[int]:
    ids = []
    for v in variants:
        enc = tok.encode(v, add_special_tokens=False)
        if enc:
            ids.append(enc[0])
    return sorted(set(ids))


class SafeDict(dict):
    def __missing__(self, key: str) -> str:
        return ""


def build_prompt(row: pd.Series, cfg: dict, template: str, rules: dict[str, str],
                 ocr_text: str = "") -> str:
    """Подставляет карточку в шаблон промпта. Правила подставляются по категории."""
    vc = cfg["vlm"]
    category = str(row.get("category", ""))
    desc = str(row.get("description", ""))[: int(vc["desc_max_chars"])]
    values = SafeDict(
        category=category,
        rules=rules.get(category, ""),
        name=str(row.get("name", ""))[:300],
        description=desc,
        ocr=(f"\nТекст, распознанный на изображениях: {ocr_text[:400]}" if ocr_text else ""),
    )
    return template.format_map(values)


def load_rules(cfg: dict) -> dict[str, str]:
    from ..config import resolve_path

    return {cat: resolve_path(path).read_text(encoding="utf-8").strip()
            for cat, path in cfg["vlm"]["rules_files"].items()}


def load_template(cfg: dict) -> str:
    from ..config import resolve_path

    return resolve_path(cfg["vlm"]["prompt_file"]).read_text(encoding="utf-8")


def make_texts(df: pd.DataFrame, cfg: dict, ocr: pd.Series | None = None) -> list[str]:
    """Промпты для всех строк df (порядок сохраняется)."""
    template, rules = load_template(cfg), load_rules(cfg)
    use_ocr = bool(cfg["vlm"].get("use_ocr")) and ocr is not None
    return [build_prompt(row, cfg, template, rules,
                         ocr_text=(str(ocr.iloc[k]) if use_ocr else ""))
            for k, (_, row) in enumerate(df.iterrows())]


def build_sft_examples(df: pd.DataFrame, cfg: dict, ocr: pd.Series | None = None,
                       seed: int | None = None) -> pd.DataFrame:
    """Обучающая выборка с балансом, заданным ОТДЕЛЬНО для каждой категории.

    Категории противоположны по балансу: в БАД позитивов 74.5%, в легковоспламеняющихся
    3.6%. Один общий баланс здесь бессмыслен — редкая категория утонет. Берём все
    позитивы редкой категории и ограниченное число негативов к ним.
    """
    vc = cfg["vlm"]
    seed = vc["seed"] if seed is None else seed
    idc, cat, tgt = (cfg["data"]["id_col"], cfg["data"]["category_col"],
                     cfg["data"]["target_col"])
    rare = "Легковоспламеняющиеся"

    # Обучение только на одной категории: редкую можно не разбавлять семью тысячами
    # БАД, а отдать модели целиком — узкое место метрики именно там.
    if vc.get("only_category"):
        df = df[df[cat] == vc["only_category"]]

    # Весь обучающий набор без балансировки. Нынешняя схема отбрасывает 60% доступных
    # строк, и это ни разу не проверялось как отдельная ось.
    if vc.get("use_all_data"):
        out = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        if vc.get("max_samples"):
            out = out.head(int(vc["max_samples"])).reset_index(drop=True)
        return _finalize_examples(out, cfg, df, ocr, idc, cat, tgt)

    parts = []
    for c, part in df.groupby(cat):
        pos, neg = part[part[tgt] == 1], part[part[tgt] == 0]
        if c == rare:
            n_neg = min(len(neg), int(len(pos) * float(vc["neg_per_pos_rare"])))
            parts.append(pd.concat([pos, neg.sample(n=n_neg, random_state=seed)]))
        else:
            n_neg = min(len(neg), int(len(pos) * float(vc["balance_supplement"])))
            n_pos = min(len(pos), max(n_neg, 1))
            parts.append(pd.concat([pos.sample(n=n_pos, random_state=seed),
                                    neg.sample(n=n_neg, random_state=seed)]))

    out = pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if vc.get("max_samples"):
        out = out.head(int(vc["max_samples"])).reset_index(drop=True)
    return _finalize_examples(out, cfg, df, ocr, idc, cat, tgt)


def _finalize_examples(out, cfg, df, ocr, idc, cat, tgt):
    """Добавляет промпт и ответ «Да»/«Нет» к отобранным строкам."""
    ocr_sel = None
    if ocr is not None:
        ocr_map = dict(zip(df[idc].astype(str), ocr.astype(str)))
        ocr_sel = out[idc].astype(str).map(ocr_map)
    out = out.copy()
    out["text"] = make_texts(out, cfg, ocr_sel)
    out["answer"] = np.where(out[tgt].to_numpy() == 1, ANSWER_YES, ANSWER_NO)
    return out[[idc, cat, tgt, "text", "answer"]].rename(columns={idc: "id"})


def select_images(cfg: dict, item_id: str, root, max_images: int, max_side: int) -> list:
    paths = image_paths(root, str(item_id))[:max_images]
    return [load_image(p, max_side=max_side) for p in paths]


class VlmSftDataset(Dataset):
    """Готовые примеры: колонки id, text, answer. Картинки грузятся лениво."""

    def __init__(self, examples: pd.DataFrame, cfg: dict):
        self.cfg = cfg
        self.vc = cfg["vlm"]
        self.df = examples.reset_index(drop=True)
        self.root = images_root(cfg)
        self.max_images = int(self.vc["max_images"])
        self.max_side = int(self.vc["image_max_side"])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> dict:
        row = self.df.iloc[i]
        images = select_images(self.cfg, row["id"], self.root, self.max_images, self.max_side)
        if not images:  # товаров без фото в данных нет, но контейнер должен пережить и это
            from PIL import Image

            images = [Image.new("RGB", (56, 56), (255, 255, 255))]
        return {"images": images, "text": row["text"], "answer": row["answer"]}


class SftCollator:
    """Сборка батча. Сторона добивки решает, работает ли приём «потери на хвосте».

    ⚠ При добивке СПРАВА настоящие токены каждого примера стоят в начале, поэтому ответы
    оказываются на РАЗНЫХ позициях, и хвост приходится тянуть до ответа самого короткого
    примера в батче. У нас длины от 676 до 1664 токенов, то есть хвост — почти вся
    последовательность, а логиты на словарь в 152 тысячи приводятся к float32. Замер
    22.08 на RTX 3090: не влезал НИ ОДИН вариант с батчем больше единицы, даже с
    пересчётом активаций (он экономит активации, а не логиты).

    При добивке СЛЕВА все последовательности кончаются в одной позиции, ответы стоят в
    последних двух-трёх, и хвост становится крошечным при любом батче. Потери от этого
    не меняются: выброшенные позиции и так шли с меткой −100.

    По умолчанию «right» — прежнее поведение, чтобы локальные прогоны при батче 1
    (где добивки нет вовсе) остались побайтно теми же.
    """

    def __init__(self, processor, model_id: str, padding_side: str = "right"):
        self.processor = processor
        self.model_id = model_id
        self.padding_side = padding_side
        self.tok = processor.tokenizer

    def __call__(self, batch: list[dict]) -> dict:
        from .vlm_backends import build_inputs, forward_input_keys, render_prompt

        texts, images, ans_lens = [], [], []
        for ex in batch:
            prompt = render_prompt(self.processor, ex["text"], self.model_id,
                                   n_images=len(ex["images"]))
            texts.append(prompt + ex["answer"] + self.tok.eos_token)
            images.append(ex["images"])
            ans_lens.append(len(self.tok.encode(ex["answer"] + self.tok.eos_token,
                                                add_special_tokens=False)))

        self.tok.padding_side = self.padding_side
        enc = build_inputs(self.processor, texts, images, self.model_id)
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        total = labels.shape[1]
        for k, alen in enumerate(ans_lens):  # лосс только на токенах ответа
            if self.padding_side == "left":
                # Слева: настоящие токены прижаты к концу, ответ — последние alen штук.
                labels[k, : total - alen] = -100
            else:
                # Справа: настоящие токены в начале, ответ перед добивкой.
                seq_len = int(enc["attention_mask"][k].sum())
                labels[k, : seq_len - alen] = -100
        enc["labels"] = labels
        keep = forward_input_keys(self.model_id)
        if keep is not None:
            enc = type(enc)({k: v for k, v in enc.items() if k in keep})
        return enc


class TailLossTrainer:
    """Потери считаются ТОЛЬКО на хвосте последовательности, где стоит ответ.

    ⚠ Зачем. Метки у нас −100 везде, кроме одного-двух токенов ответа в конце, но
    модель всё равно считает логиты для ВСЕХ позиций. При словаре в 262 тысячи токенов
    и полутора тысячах позиций это тензор на 1.57 ГБ в float32, плюс столько же на
    обратный проход — из-за него gemma-4-E4B не поднималась на 8 ГБ (замеры 13.08 и
    18.08: 20.5 ГБ при батче 1 и любом ранге LoRA).

    Здесь модель зовётся с `logits_to_keep`, то есть считает логиты только последних
    позиций. Экономия примерно в seq_len/K раз, у нас это сотни раз. На результат не
    влияет: отброшенные позиции всё равно шли в потери с меткой −100.
    """

    @staticmethod
    def build(base_cls):
        import torch
        from torch.nn import CrossEntropyLoss

        class _Trainer(base_cls):
            def compute_loss(self, model, inputs, return_outputs=False, **kw):
                labels = inputs.pop("labels")
                # сколько последних позиций реально участвуют в потерях
                used = (labels != -100)
                if not bool(used.any()):
                    out = model(**inputs)
                    zero = out.logits.sum() * 0.0
                    return (zero, out) if return_outputs else zero
                last_used = int(used.nonzero()[:, 1].max().item())
                first_used = int(used.nonzero()[:, 1].min().item())
                keep = labels.shape[1] - first_used + 1
                try:
                    out = model(**inputs, logits_to_keep=keep)
                    logits = out.logits
                    tail = labels[:, -logits.shape[1]:]
                except TypeError:
                    # модель не понимает logits_to_keep — считаем как раньше
                    out = model(**inputs)
                    logits, tail = out.logits, labels
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = tail[:, 1:].contiguous()
                loss = CrossEntropyLoss()(
                    shift_logits.view(-1, shift_logits.size(-1)).float(),
                    shift_labels.view(-1).to(shift_logits.device))
                return (loss, out) if return_outputs else loss

        return _Trainer


def run_sft(cfg: dict, examples: pd.DataFrame, out_dir: str,
            resume: bool | str | None = None) -> None:
    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainingArguments

    from .vlm_backends import load_model, load_processor, lora_targets, lora_task_type

    vc = cfg["vlm"]
    processor = load_processor(vc["model_id"], vc.get("max_pixels"))
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model = load_model(vc["model_id"], processor, vc.get("load_in_4bit", True), for_training=True)
    model.config.use_cache = False
    if vc.get("load_in_4bit", True):
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model)
    elif vc.get("gradient_checkpointing", True):
        model.enable_input_require_grads()

    # ⚠ Вторая ступень: продолжаем обучение ГОТОВОГО адаптера, а не начинаем новый.
    # Нужно, чтобы дообучить победителя одной эпохой только на редкой категории —
    # там половина метрики и весь дефицит ранжирования. Ключа нет — прежнее поведение,
    # то есть свежий адаптер с нуля.
    init_adapter = vc.get("init_adapter")
    if init_adapter:
        from peft import PeftModel

        from ..config import resolve_path

        path = resolve_path(init_adapter)
        if not path.is_dir():
            raise RuntimeError(f"адаптера для второй ступени нет: {path}")
        # is_trainable=True обязателен: без него peft грузит адаптер только для вывода,
        # градиенты не идут, и «обучение» проходит вхолостую, ничего не меняя.
        model = PeftModel.from_pretrained(model, str(path), is_trainable=True)
        print(f"вторая ступень: продолжаем адаптер {path}", flush=True)
    else:
        lora = LoraConfig(
            r=vc["lora_r"], lora_alpha=vc["lora_alpha"], lora_dropout=vc["lora_dropout"],
            target_modules=lora_targets(vc["model_id"], vc["lora_target_modules"]),
            task_type=lora_task_type(vc["model_id"]),
        )
        model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    ds = VlmSftDataset(examples, cfg)
    print(f"примеров для обучения: {len(ds)}", flush=True)

    # YAML 1.1 читает «no» как False — приводим обратно к строке, которую ждёт Trainer
    save_strategy = vc.get("save_strategy", "no")
    save_strategy = "no" if save_strategy in (False, None) else str(save_strategy)

    args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=vc["batch_size"],
        gradient_accumulation_steps=vc["grad_accum"],
        num_train_epochs=vc["epochs"],
        learning_rate=vc["lr"],
        warmup_steps=vc.get("warmup_steps", 10),
        logging_steps=vc.get("logging_steps", 20),
        save_strategy=save_strategy,
        save_steps=vc.get("save_steps", 500),
        # ⚠ По умолчанию единица — как было: локальный прогон хранит одну точку и не
        # занимает диск. Аренда ставит больше: промежуточные точки дают ось «сколько
        # эпох» бесплатно, одним прогоном, а единица затирала бы все ранние.
        # Ноль или null означает «хранить все» — так это понимает и сам Trainer.
        save_total_limit=_save_limit(vc.get("save_total_limit", 1)),
        bf16=True,
        gradient_checkpointing=vc.get("gradient_checkpointing", True),
        remove_unused_columns=False,
        dataloader_num_workers=vc.get("num_workers", 2),
        report_to=[],
        seed=vc["seed"],
    )
    # ⚠ Потери только на хвосте: без этого модели с огромным словарём не поднимаются
    trainer_cls = TailLossTrainer.build(Trainer)
    trainer = trainer_cls(model=model, args=args, train_dataset=ds,
                          data_collator=SftCollator(processor, vc["model_id"],
                                                    vc.get("padding_side", "right")))
    trainer.train(resume_from_checkpoint=_resume_point(out_dir, resume))
    model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)


def _save_limit(value) -> int | None:
    """Ноль и null означают «хранить все контрольные точки»."""
    return None if value in (None, 0, "0", "all") else int(value)


def _resume_point(out_dir: str, resume: bool | str | None):
    """Откуда продолжать обучение: None — с нуля (прежнее поведение).

    ⚠ Зачем. На арендованной карте прогон стоит денег: упавший на третьем часу и
    начатый с нуля — это выброшенные рубли. Здесь `resume=True` означает «продолжи с
    последней сохранённой точки, если она есть», а не «упади, если её нет»: сам
    Trainer при `resume_from_checkpoint=True` и пустой папке бросает исключение, и
    повторный запуск ДО первого сохранения ронял бы прогон на ровном месте.
    """
    if not resume:
        return None
    if isinstance(resume, str):
        return resume
    from transformers.trainer_utils import get_last_checkpoint

    last = get_last_checkpoint(out_dir) if os.path.isdir(out_dir) else None
    if last:
        print(f"продолжаем с сохранённой точки: {last}", flush=True)
        return last
    print("сохранённых точек нет — обучение начинается с нуля", flush=True)
    return None


TRANSCRIBE_PROMPT = (
    "Перепиши весь текст, который видно на изображениях товара: надписи на упаковке, "
    "этикетке и инфографике. Ничего не добавляй от себя и не переводи. "
    "Если текста нет, ответь одним словом: нет."
)


class VlmTranscriber:
    """Чтение надписей с упаковки базовой VLM — без LoRA и без внешних библиотек.

    Зачем отдельно от easyocr: в базовом образе организаторов нет ни easyocr, ни peft,
    ни cv2, зато есть transformers и модели из /shared_models. Значит единственный
    способ прочитать упаковку в контейнере, не собирая свой образ, — попросить об этом
    ту же VLM. Задача чтения текста не равна задаче классификации, в которой наша VLM
    проиграла: распознавание — то, что подобные модели умеют хорошо.
    """

    def __init__(self, cfg: dict, model_id: str | None = None):
        from .vlm_backends import load_model, load_processor

        vc = cfg["vlm"]
        self.cfg = cfg
        self.model_id = model_id or vc["model_id"]
        self.batch_size = int(vc.get("transcribe_batch_size", 4))
        self.max_images = int(vc.get("transcribe_max_images", 2))
        self.max_side = int(vc.get("transcribe_image_max_side", 1024))
        self.max_new_tokens = int(vc.get("transcribe_max_new_tokens", 96))

        self.processor = load_processor(self.model_id, vc.get("transcribe_max_pixels"))
        tok = self.processor.tokenizer
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        # ⚠ Квантование НЕ наследуется от обучения. Ключ load_in_4bit включён ради
        # нашей карты на 8 ГБ, а на инференсе он тянет за собой bitsandbytes, которого
        # нет ни в базовом образе, ни в нашем: ветка падала с ImportError уже в
        # контейнере, хотя локально проходила. У проверяющей системы 80 ГБ, модель на
        # два миллиарда параметров идёт в bf16 без всякого сжатия — и быстрее.
        # Базовая модель может ехать в архиве упакованной — тогда путь к ней задан
        # конфигом отправки, а не разрешается по имени через каталог моделей.
        self.model = load_model(self.model_id, self.processor,
                                vc.get("inference_load_in_4bit", False),
                                for_training=False,
                                base_override=vc.get("packed_base_path"))
        self.model.eval()
        self.root = images_root(cfg)

    @torch.no_grad()
    def transcribe(self, df: pd.DataFrame) -> list[str]:
        from .vlm_backends import build_inputs, render_prompt

        idc = self.cfg["data"]["id_col"]
        ids = df[idc].astype(str).tolist()
        out: list[str] = []
        for start in tqdm(range(0, len(ids), self.batch_size), desc="чтение упаковки",
                          unit="батч"):
            chunk = ids[start:start + self.batch_size]
            images, prompts = [], []
            for item_id in chunk:
                imgs = select_images(self.cfg, item_id, self.root, self.max_images,
                                     self.max_side)
                if not imgs:
                    from PIL import Image

                    imgs = [Image.new("RGB", (56, 56), (255, 255, 255))]
                images.append(imgs)
                prompts.append(render_prompt(self.processor, TRANSCRIBE_PROMPT,
                                             self.model_id, n_images=len(imgs)))
            enc = build_inputs(self.processor, prompts, images, self.model_id)
            enc = enc.to(self.model.device)
            gen = self.model.generate(**enc, max_new_tokens=self.max_new_tokens,
                                      do_sample=False)
            gen = gen[:, enc["input_ids"].shape[1]:]
            out += [t.strip() for t in
                    self.processor.batch_decode(gen, skip_special_tokens=True)]
        return out


class VlmScorer:
    """Вероятность ответа «Да» по логитам первого токена."""

    def __init__(self, cfg: dict, model_id: str | None = None,
                 images_root_path=None, require_images: bool = False):
        from .vlm_backends import load_model, load_processor

        vc = cfg["vlm"]
        self.cfg = cfg
        self.vc = vc
        self.model_id = model_id or vc["model_id"]
        # ⚠ Батч под H100 80 ГБ проверяющей системы, а не под нашу карту: при батче 4
        # видеокарта простаивает, ожидая раскодированных кадров. Локально он сам
        # ужимается по нехватке памяти (см. _forward_adaptive).
        from .vlm_backends import auto_batch

        # min_batch — прежнее локально проверенное значение: хуже, чем было, стать нельзя
        # ⚠ Батч анкеты СВОЙ и заметно больше: она идёт без картинок, чистым текстом,
        # и на H100 батч 32 занимал около шестнадцати тысяч токенов — карта недогружена
        # вчетверо. Замер на платформе: 0.061 с/вопрос-товар, из-за чего восемь вопросов
        # не укладывались в остаток бюджета.
        key = "qa_batch_size" if int(vc.get("max_images", 3)) == 0 else "inference_batch_size"
        self.batch_size = auto_batch(int(vc.get(key, 32)),
                                     min_batch=int(vc.get("eval_batch_size", 4)))
        self.loader_workers = int(vc.get("loader_workers", 8))
        self.max_images = int(vc["max_images"])
        self.max_side = int(vc["image_max_side"])

        self.processor = load_processor(self.model_id, vc.get("max_pixels"))
        tok = self.processor.tokenizer
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        # ⚠ Квантование НЕ наследуется от обучения. Ключ load_in_4bit включён ради
        # нашей карты на 8 ГБ, а на инференсе он тянет за собой bitsandbytes, которого
        # нет ни в базовом образе, ни в нашем: ветка падала с ImportError уже в
        # контейнере, хотя локально проходила. У проверяющей системы 80 ГБ, модель на
        # два миллиарда параметров идёт в bf16 без всякого сжатия — и быстрее.
        # Базовая модель может ехать в архиве упакованной — тогда путь к ней задан
        # конфигом отправки, а не разрешается по имени через каталог моделей.
        self.model = load_model(self.model_id, self.processor,
                                vc.get("inference_load_in_4bit", False),
                                for_training=False,
                                base_override=vc.get("packed_base_path"))
        self.model.eval()
        self.yes_ids = _first_token_ids(tok, YES_VARIANTS)
        self.no_ids = _first_token_ids(tok, NO_VARIANTS)
        # ⚠ Корень изображений приходит от пути к данным ПРОГОНА, а не из конфига.
        # Ровно на этом 18.08 попались эмбеддинги: внутри архива решения папки data/
        # нет, картинки не находились, и модель кодировала пустоту молча. Здесь было
        # бы хуже — score подставляет белый квадрат вместо ненайденных кадров, то есть
        # вердикт вынесся бы по пустым изображениям без единой ошибки в журнале.
        self.root = images_root(cfg) if images_root_path is None else images_root_path
        self.require_images = bool(require_images)
        if self.require_images and self.root is None:
            raise RuntimeError(
                "скорингу нужны изображения, а корень с ними не найден — "
                "веса учились на кадрах, вердикт по пустым картинкам недопустим")

    @torch.no_grad()
    def _forward_adaptive(self, prompts: list, images: list) -> np.ndarray:
        """Вероятность «Да» для батча; при нехватке видеопамяти батч делится пополам.

        ⚠ Без этого большой батч, поставленный ради H100, роняет локальную проверку —
        а проверять надо ровно тот архив, который поедет. Дробление пополам, а не сразу
        до поштучной обработки: поштучный откат втрое медленнее необходимого.
        """
        from .vlm_backends import build_inputs, forward_input_keys

        try:
            enc = build_inputs(self.processor, prompts, images, self.model_id)
            enc = enc.to(self.model.device)
            keep = forward_input_keys(self.model_id)
            model_in = enc if keep is None else {k: v for k, v in enc.items() if k in keep}
            logits = self.model(**model_in).logits[:, -1, :].float()
            yes = torch.logsumexp(logits[:, self.yes_ids], dim=-1)
            no = torch.logsumexp(logits[:, self.no_ids], dim=-1)
            return torch.sigmoid(yes - no).cpu().numpy()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(prompts) == 1:
                raise
            half = len(prompts) // 2
            left = self._forward_adaptive(prompts[:half], images[:half])
            right = self._forward_adaptive(prompts[half:], images[half:])
            return np.concatenate([left, right])

    @torch.no_grad()
    def score(self, df: pd.DataFrame, texts: list[str], show_progress: bool = True) -> np.ndarray:
        from .vlm_backends import build_inputs, forward_input_keys, render_prompt

        scores = np.full(len(df), np.nan, dtype=np.float32)
        order = list(range(len(df)))
        iterator = range(0, len(order), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="скоринг", unit="батч")
        ids = df[self.cfg["data"]["id_col"]].astype(str).tolist()

        n_blank = 0
        t_start = time.time()
        from concurrent.futures import ThreadPoolExecutor

        pool = ThreadPoolExecutor(max_workers=max(1, self.loader_workers))
        for start in iterator:
            chunk = order[start:start + self.batch_size]
            images, prompts = [], []
            # Кадры читаются параллельно: декодирование — процессорная работа, и
            # последовательное чтение внутри цикла батчей держало видеокарту в простое.
            groups = ([[] for _ in chunk] if self.max_images == 0 else list(pool.map(
                lambda k: select_images(self.cfg, ids[k], self.root,
                                        self.max_images, self.max_side), chunk)))
            for k, imgs in zip(chunk, groups):
                # ⚠ При max_images=0 картинки не нужны ПО ЗАМЫСЛУ (так работает анкета),
                # и пустой кадр здесь не экономия, а расход: он гонит зрительную часть
                # модели вхолостую. Замер: 0.0930 против 0.0551 с/вопрос-товар, то есть
                # 1.7x. Сами ответы при этом почти не меняются — корреляция 0.93-1.00,
                # средняя разница 0.017 на шкале от нуля до единицы.
                if not imgs and self.max_images > 0:
                    from PIL import Image

                    # ⚠ Подстановка пустого кадра нужна ради товаров без фотографий, но
                    # она же прячет ненайденный корень. Проверка — по всей выборке,
                    # в конце: у первых ~600 карточек приватного набора кадров нет
                    # по данным, и проверка по первым товарам роняла здоровый прогон.
                    n_blank += 1
                    imgs = [Image.new("RGB", (56, 56), (255, 255, 255))]
                images.append(imgs)
                prompts.append(render_prompt(self.processor, texts[k], self.model_id,
                                             n_images=len(imgs)))

            p = self._forward_adaptive(prompts, images)
            for k, v in zip(chunk, p):
                scores[k] = v
        pool.shutdown(wait=True)
        # При max_images=0 кадров нет НАМЕРЕННО (так работает анкета), и предупреждать
        # не о чем: ложная тревога в журнале дороже молчания, её потом ищут как дефект.
        if n_blank and self.max_images > 0:
            print(f"⚠ товаров без единого кадра: {n_blank} из {len(df)} "
                  f"({n_blank / max(len(df), 1):.1%})", flush=True)
        # ⚠ Маркер печатается ТОЛЬКО когда кадры действительно прочитаны: по нему
        # сборка отличает состоявшуюся работу с фотографиями от тихого вырождения.
        # Печатать его безусловно значило бы врать самим себе — так уже терялись
        # отправки, у которых зрение молча не сработало.
        # ⚠ В текстовом режиме (max_images=0) кадров не читалось вовсе, и маркер
        # «скоринг по фотографиям готов» печатать НЕЛЬЗЯ: по нему сборка решает, что
        # зрение состоялось. Ложный маркер — ровно та подмена, из-за которой мы уже
        # принимали балл решения-заглушки за результат модели.
        with_frames = 0 if self.max_images == 0 else len(df) - n_blank
        # ⚠ Падаем ТОЛЬКО когда кадров нет вообще ни у одного товара: это однозначно
        # неверный корень. Проверять первые товары нельзя — у первых ~600 карточек
        # приватного набора фотографий нет ПО ДАННЫМ (замерено: 5 из 600), и такая
        # проверка роняла приватную стадию на здоровом решении.
        if self.require_images and with_frames == 0:
            raise RuntimeError(
                f"ни у одного из {len(df)} товаров не найдено кадров (корень "
                f"{self.root}) — вердикт по пустым изображениям недопустим")
        if with_frames > 0:
            print(f"скоринг по фотографиям готов за {time.time() - t_start:.1f} с "
                  f"({with_frames} товаров с кадрами из {len(df)})", flush=True)
        return scores
