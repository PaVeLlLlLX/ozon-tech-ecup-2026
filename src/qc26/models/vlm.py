"""LoRA SFT и скоринг VLM под задачу 2026.

Модель отвечает одним словом «Да»/«Нет» на вопрос «относится ли товар к заявленной
категории». Вероятность берём из логитов первого токена ответа, а не из сгенерированного
текста: так получается непрерывный скор, по которому можно двигать порог — а порог у нас
и есть узкое место метрики.

Обучение — QLoRA в 4 битах на 8 ГБ; инференс в контейнере — bf16 на H100.
"""
from __future__ import annotations

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

    ocr_sel = None
    if ocr is not None:
        ocr_map = dict(zip(df[idc].astype(str), ocr.astype(str)))
        ocr_sel = out[idc].astype(str).map(ocr_map)
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
    def __init__(self, processor, model_id: str):
        self.processor = processor
        self.model_id = model_id
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

        self.tok.padding_side = "right"
        enc = build_inputs(self.processor, texts, images, self.model_id)
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        for k, alen in enumerate(ans_lens):  # лосс только на токенах ответа
            seq_len = int(enc["attention_mask"][k].sum())
            labels[k, : seq_len - alen] = -100
        enc["labels"] = labels
        keep = forward_input_keys(self.model_id)
        if keep is not None:
            enc = type(enc)({k: v for k, v in enc.items() if k in keep})
        return enc


def run_sft(cfg: dict, examples: pd.DataFrame, out_dir: str) -> None:
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
        save_total_limit=1,
        bf16=True,
        gradient_checkpointing=vc.get("gradient_checkpointing", True),
        remove_unused_columns=False,
        dataloader_num_workers=vc.get("num_workers", 2),
        report_to=[],
        seed=vc["seed"],
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds,
                      data_collator=SftCollator(processor, vc["model_id"]))
    trainer.train()
    model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)


class VlmScorer:
    """Вероятность ответа «Да» по логитам первого токена."""

    def __init__(self, cfg: dict, model_id: str | None = None):
        from .vlm_backends import load_model, load_processor

        vc = cfg["vlm"]
        self.cfg = cfg
        self.vc = vc
        self.model_id = model_id or vc["model_id"]
        self.batch_size = int(vc.get("eval_batch_size", vc["batch_size"]))
        self.max_images = int(vc["max_images"])
        self.max_side = int(vc["image_max_side"])

        self.processor = load_processor(self.model_id, vc.get("max_pixels"))
        tok = self.processor.tokenizer
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        self.model = load_model(self.model_id, self.processor,
                                vc.get("load_in_4bit", False), for_training=False)
        self.model.eval()
        self.yes_ids = _first_token_ids(tok, YES_VARIANTS)
        self.no_ids = _first_token_ids(tok, NO_VARIANTS)
        self.root = images_root(cfg)

    @torch.no_grad()
    def score(self, df: pd.DataFrame, texts: list[str], show_progress: bool = True) -> np.ndarray:
        from .vlm_backends import build_inputs, forward_input_keys, render_prompt

        scores = np.full(len(df), np.nan, dtype=np.float32)
        order = list(range(len(df)))
        iterator = range(0, len(order), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="скоринг", unit="батч")
        ids = df[self.cfg["data"]["id_col"]].astype(str).tolist()

        for start in iterator:
            chunk = order[start:start + self.batch_size]
            images, prompts = [], []
            for k in chunk:
                imgs = select_images(self.cfg, ids[k], self.root, self.max_images, self.max_side)
                if not imgs:
                    from PIL import Image

                    imgs = [Image.new("RGB", (56, 56), (255, 255, 255))]
                images.append(imgs)
                prompts.append(render_prompt(self.processor, texts[k], self.model_id,
                                             n_images=len(imgs)))
            enc = build_inputs(self.processor, prompts, images, self.model_id)
            enc = enc.to(self.model.device)
            keep = forward_input_keys(self.model_id)
            model_in = enc if keep is None else {k: v for k, v in enc.items() if k in keep}
            logits = self.model(**model_in).logits[:, -1, :].float()
            yes = torch.logsumexp(logits[:, self.yes_ids], dim=-1)
            no = torch.logsumexp(logits[:, self.no_ids], dim=-1)
            p = torch.sigmoid(yes - no).cpu().numpy()
            for k, v in zip(chunk, p):
                scores[k] = v
        return scores
