"""Загрузка VLM и сборка входов. Отличие от прошлогоднего кода — НЕСКОЛЬКО фото на товар.

У товара 1-5 изображений, и маркировка может быть на любом из них (маркировка БАД —
на упаковке, а упаковка не всегда первый кадр). Поэтому промпт содержит столько
плейсхолдеров, сколько подано картинок, а батч собирается вложенными списками.

Семейства: Qwen3-VL — основной путь; InternVL и Gemma поддержаны так же, как в архиве
подготовки (кастомный чекпоинт InternVL требует ручной сборки процессора и image_flags).
"""
from __future__ import annotations

import json
import os

import torch

INTERNVL_EXTRA_TOKENS = {
    "start_image_token": "<img>",
    "end_image_token": "</img>",
    "context_image_token": "<IMG_CONTEXT>",
    "video_token": "<video>",
}


def resolve_model_id(model_id: str) -> str:
    """Путь к весам: в контейнере модели из списка лежат в /shared_models.

    Проверяющая система монтирует их заранее, скачивать ничего нельзя — интернета нет.
    Локально того же каталога нет, и идентификатор остаётся как есть, разрешаясь из
    кэша. Одна строка кода вместо двух веток «локально» и «в контейнере».
    """
    shared = os.environ.get("SHARED_MODELS_PATH", "/shared_models")
    if not os.path.isdir(shared):
        return model_id

    short = model_id.split("/")[-1]
    # Разложить модели можно по-разному: с именем организации и без, в подпапке
    # snapshots или с префиксом кэша huggingface. Промах означает попытку скачать
    # веса, а интернета в контейнере нет — решение теряет ветку целиком.
    direct = [os.path.join(shared, model_id), os.path.join(shared, short),
              os.path.join(shared, model_id.replace("/", "--")),
              os.path.join(shared, "models--" + model_id.replace("/", "--"))]
    for path in direct:
        if os.path.isfile(os.path.join(path, "config.json")):
            return path
    # Если по прямым путям не нашли — ищем вглубь, но неглубоко. Сверяем не имя
    # последней папки, а весь путь: в кэше huggingface веса лежат в snapshots/<хэш>,
    # и по имени листа модель не опознать.
    needle = short.lower()
    for depth_root, dirs, files in os.walk(shared):
        if "config.json" in files:
            rel = os.path.relpath(depth_root, shared).lower().replace("--", "/")
            if needle in rel:
                return depth_root
        if depth_root.count(os.sep) - shared.count(os.sep) >= 4:
            dirs[:] = []
    return model_id


def adapter_base(model_id: str) -> str | None:
    """Если model_id — папка LoRA-адаптера, вернуть id базовой модели, иначе None."""
    cfg = os.path.join(model_id, "adapter_config.json")
    if os.path.isdir(model_id) and os.path.exists(cfg):
        with open(cfg, encoding="utf-8") as f:
            return json.load(f).get("base_model_name_or_path")
    return None


def family(model_id: str) -> str:
    m = (adapter_base(model_id) or model_id).lower()
    if "internvl" in m:
        return "internvl"
    if "gemma" in m:
        return "gemma"
    return "qwen"


def _bnb_config():
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4",
    )


def load_processor(model_id: str, max_pixels: int | None = None):
    model_id = resolve_model_id(adapter_base(model_id) or model_id)
    fam = family(model_id)
    if fam == "internvl":
        from transformers import AutoImageProcessor, AutoTokenizer, AutoVideoProcessor
        from transformers.models.internvl.processing_internvl import InternVLProcessor

        tok = AutoTokenizer.from_pretrained(model_id, extra_special_tokens=INTERNVL_EXTRA_TOKENS)
        return InternVLProcessor(
            image_processor=AutoImageProcessor.from_pretrained(model_id),
            tokenizer=tok,
            video_processor=AutoVideoProcessor.from_pretrained(model_id),
            chat_template=tok.chat_template,
        )
    from transformers import AutoProcessor

    kwargs = {"max_pixels": int(max_pixels)} if max_pixels else {}
    return AutoProcessor.from_pretrained(model_id, **kwargs)


def load_model(model_id: str, processor, load_4bit: bool, for_training: bool,
               base_override: str | None = None):
    """base_override — путь к базовой модели, привезённой в архиве решения.

    ⚠ Нужен потому, что адаптер помнит имя базовой модели («Qwen/Qwen3-VL-4B-Instruct»),
    а её нет в каталоге проверяющей системы: `resolve_model_id` вернёт то же имя,
    загрузчик пойдёт в сеть, а сети в контейнере нет — ветка потеряется целиком.
    Поэтому базу берём из архива по явному пути.
    """
    base = adapter_base(model_id)
    if base:
        model = _load_base(base_override or base, processor, load_4bit)
        from peft import PeftModel

        return PeftModel.from_pretrained(model, model_id)
    return _load_base(base_override or model_id, processor, load_4bit)


def _load_base(model_id: str, processor, load_4bit: bool):
    fam = family(model_id)
    model_id = resolve_model_id(model_id)
    # Модель, привезённая в архиве упакованной: разворачиваем её сами, потензорно.
    if os.path.isdir(model_id) and os.path.isfile(os.path.join(model_id, "packing.json")):
        from .packing import load_packed

        return load_packed(model_id,
                           device="cuda" if torch.cuda.is_available() else "cpu")
    kwargs: dict = {"dtype": torch.bfloat16}
    if load_4bit:
        kwargs["quantization_config"] = _bnb_config()

    if fam == "internvl":
        from transformers import AutoModel
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        cls = get_class_from_dynamic_module("modeling_internvl_chat.InternVLChatModel", model_id)
        cls.all_tied_weights_keys = {}
        model = AutoModel.from_pretrained(
            model_id, trust_remote_code=True, device_map="cuda",
            low_cpu_mem_usage=True, **kwargs,
        )
        model.img_context_token_id = processor.tokenizer.context_image_token_id
        return model

    from transformers import AutoModelForImageTextToText

    model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    if not load_4bit:
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model


def _thinking_kwargs(processor) -> dict:
    """`enable_thinking=False` — только тем шаблонам, которые про него знают.

    ⚠ Зачем. У рассуждающих моделей (Qwen3.5) шаблон заканчивается открытым «<think>»,
    и обученная так модель ставила бы «Да»/«Нет» ВНУТРИ блока рассуждений, против
    собственной привычки. С выключенным блоком шаблон сам закрывает «<think></think>»,
    и ответ оказывается там, где модель его и ждёт.

    Определяем по самому шаблону, а не по имени модели: список моделей меняется, а
    признак «шаблон знает про рассуждения» проверяется прямо. Для Qwen3-VL и прочих
    без этого ключа возвращается пустой словарь, и промпт остаётся ПРЕЖНИМ ДО БАЙТА.
    """
    tpl = getattr(processor, "chat_template", None)
    if tpl is None:
        tok = getattr(processor, "tokenizer", None)
        tpl = getattr(tok, "chat_template", None)
    return {"enable_thinking": False} if tpl and "enable_thinking" in str(tpl) else {}


def render_prompt(processor, text: str, model_id: str, n_images: int = 1,
                  add_generation_prompt: bool = True) -> str:
    """Chat-промпт с n_images плейсхолдерами изображений."""
    content = [{"type": "image"} for _ in range(max(0, n_images))]
    content.append({"type": "text", "text": text})
    messages = [{"role": "user", "content": content}]
    s = processor.apply_chat_template(messages, tokenize=False,
                                      add_generation_prompt=add_generation_prompt,
                                      **_thinking_kwargs(processor))
    if family(model_id) == "internvl":
        s = s.replace("<image>", "<IMG_CONTEXT>")
    return s


def build_inputs(processor, texts: list[str], images: list[list], model_id: str):
    """Токенизация батча. images — список списков PIL (1-5 фото на товар)."""
    fam = family(model_id)
    if fam == "internvl":
        flat = [im for group in images for im in group]
        enc = processor(text=texts, images=flat, return_tensors="pt", padding=True,
                        crop_to_patches=False)
        n_tiles = enc["pixel_values"].shape[0]
        enc["pixel_values"] = enc["pixel_values"].to(torch.bfloat16)
        enc["image_flags"] = torch.ones(n_tiles, 1, dtype=torch.long)
        return enc
    # ⚠ Пустые списки кадров процессор не принимает (IndexError на images[0]), а
    # текстовому режиму картинки не нужны вовсе: передавать их аргументом нельзя.
    if not any(images):
        return processor(text=texts, return_tensors="pt", padding=True)
    return processor(text=texts, images=images, return_tensors="pt", padding=True)


def forward_input_keys(model_id: str) -> list[str] | None:
    if family(model_id) == "internvl":
        return ["input_ids", "attention_mask", "pixel_values", "image_flags", "labels"]
    return None


def lora_task_type(model_id: str):
    return None if family(model_id) == "internvl" else "CAUSAL_LM"


def lora_targets(model_id: str, configured) -> list[str]:
    """У Gemma линейные слои обёрнуты — целимся во внутренний .linear."""
    if family(model_id) == "gemma":
        return [f"{t}.linear" for t in configured]
    return list(configured)


def auto_batch(base_for_80gb: int, min_batch: int = 1) -> int:
    """Размер батча по ФАКТИЧЕСКОЙ видеопамяти: база задаётся под H100 80 ГБ.

    ⚠ Константу ставить нельзя ни ту, ни другую. Батч под 8 ГБ держит H100 в простое,
    пока процессор декодирует кадры; батч под 80 ГБ на нашей карте уходит в OOM и
    дробление — замер показал 2.6 с/товар вместо 0.376, то есть в семь раз хуже.
    Здесь один и тот же код даёт 64 на H100 и 6 на RTX 3060 Ti, поэтому архив, который
    мы проверяем локально, и архив, который поедет, — один и тот же.
    """
    import torch

    if not torch.cuda.is_available():
        return max(min_batch, 2)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    # ⚠ Деление ПРОПОРЦИОНАЛЬНО видеопамяти занижает батч на маленькой карте: вес самой
    # модели занимает фиксированные несколько гигабайт, и под данные остаётся не доля
    # от 80 ГБ, а остаток. Поэтому min_batch — известное рабочее локальное значение,
    # ниже которого опускаться нельзя: иначе «оптимизация» делает локально ХУЖЕ, чем
    # было (замер: auto_batch дал 3 там, где раньше работало 4).
    return max(min_batch, int(base_for_80gb * total_gb / 80.0))
