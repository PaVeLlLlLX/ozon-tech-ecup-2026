"""Вердикт дообученной VLM на инференсе. Раскладка обязана совпадать с обучением.

Веса базовой модели в архив НЕ кладём: `Qwen/Qwen3-VL-2B-Instruct` есть в списке
организаторов и монтируется в /shared_models. В архиве едет только адаптер LoRA
(26 МБ).

⚠ Промпт, число кадров, разрешение и разбор ответа обязаны совпадать с обучением
ДО СИМВОЛА. Иначе модель получит вход из другого распределения, и вердикт станет
случайным — причём молча. Совпадение проверяется тестом, который сверяет константы
с `scripts/sft_vlm_lora.py`.

Любой сбой обязан возвращать None, а не бросать исключение: решение без VLM хуже
решения с ней, но несравнимо лучше упавшего.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

MAX_PIXELS = 100352          # пресет S, как при обучении
# ⚠ Значение по умолчанию. Число кадров ОБЯЗАНО совпадать с обучением адаптера,
# иначе модель получит вход из другого распределения и вердикт поедет молча.
# Задаётся из конфига отправки (vlm_images) и берётся из train_log.json адаптера,
# если он есть рядом. Адаптер rank32 обучен на ЧЕТЫРЁХ кадрах, прежние — на двух.
MAX_IMAGES = 2
MAX_DESC = 800
MAX_NAME = 300
YES, NO = " Да", " Нет"
BASE_MODEL = "Qwen/Qwen3-VL-2B-Instruct"

RULES = {
    "Легковоспламеняющиеся": (
        "Товар ОТНОСИТСЯ к легковоспламеняющимся, если он сам является источником огня "
        "(спички, зажигалки), содержит горючее вещество, жидкость или газ, либо горючее "
        "входит в комплект.\n"
        "Товар НЕ относится, если это устройство для использования с огнём без топлива "
        "(мангал, гриль, плита), пустая тара, встроенный источник огня или горючий "
        "материал лишь как часть изделия."),
    "БАД": (
        "Товар ОТНОСИТСЯ к биологически активным добавкам, если в описании или на "
        "изображении есть прямое указание на это (БАД, dietary supplement).\n"
        "Товар НЕ относится, если это спортивное питание (аминокислоты, BCAA, "
        "L-карнитин, протеин), если прямо указано что это не БАД, либо если маркировки "
        "БАД нет вовсе."),
}

QUESTION = (
    "Ты модератор маркетплейса. Товар заявлен в категории «{cat}».\n"
    "{rules}\n\n"
    "Название: {name}\n"
    "Описание: {desc}\n\n"
    "Относится ли товар к категории «{cat}»? Ответь одним словом: Да или Нет.\n"
    "Ответ:"
)

# ⚠ Второй шаблон. Формулировка должна СОВПАДАТЬ с обучением адаптера посимвольно —
# иначе вход из другого распределения, а ошибки не будет, только тихо поехавший
# вердикт. Ровно так мы уже попались на числе кадров, поэтому шаблон выбирается
# не из конфига отправки, а читается из train_log.json рядом с адаптером.
QUESTION_PACKAGING = (
    "Ты модератор маркетплейса и проверяешь карточку товара.\n\n"
    "Продавец указал категорию: «{cat}».\n"
    "Правила этой категории:\n"
    "{rules}\n\n"
    "Карточка товара.\n"
    "Название: {name}\n"
    "Описание: {desc}\n\n"
    "Изображения товара приложены выше. Смотри на упаковку: маркировка часто "
    "напечатана на ней, а не написана в описании.\n\n"
    "Вопрос: товар действительно относится к категории «{cat}»?\n"
    "Ответь одним словом — Да или Нет."
)

# ⚠ НАШ шаблон. Текст обязан совпадать с обучением ПОСИМВОЛЬНО — сверяется тестом.
# Гнездо {facts} заполняется блоком наблюдений из qc26.facts, и включён он или нет
# решает train_log.json рядом с адаптером, а не конфиг отправки.
QUESTION_QC = (
    "Ты модератор маркетплейса и проверяешь, верно ли продавец выбрал категорию.\n\n"
    "Заявленная категория: «{cat}».\n"
    "{rules}\n\n"
    "Карточка.\n"
    "Название: {name}\n"
    "Описание: {desc}\n\n"
    "{facts}"
    "Решает не то, связан ли товар с темой категории, а то, ЧЕМ ОН ЯВЛЯЕТСЯ САМ. "
    "Изделие, которое работает от вещества, и само вещество, выставленное на "
    "продажу, — разные случаи. Смотри на фотографии: состав и маркировка чаще "
    "напечатаны на упаковке, чем написаны в описании.\n\n"
    "Товар действительно относится к категории «{cat}»? Ответь одним словом: "
    "Да или Нет.\n"
    "Ответ:"
)

# ⚠ Шаблон адаптера vlm_rare2_a, выигравшего редкую категорию: 21 верный из 24 при
# 22 названных. От «packaging» отличается ОДНИМ переносом строки в предпоследнем
# абзаце — и этого достаточно, чтобы вход стал другим. Текст сверен с
# configs/prompts/verdict_ru.txt того решения посимвольно.
QUESTION_RARE2 = (
    "Ты модератор маркетплейса и проверяешь карточку товара.\n\n"
    "Продавец указал категорию: «{cat}».\n"
    "Правила этой категории:\n"
    "{rules}\n\n"
    "Карточка товара.\n"
    "Название: {name}\n"
    "Описание: {desc}\n\n"
    "Изображения товара приложены выше. Смотри на упаковку: маркировка часто напечатана\n"
    "на ней, а не написана в описании.\n\n"
    "Вопрос: товар действительно относится к категории «{cat}»?\n"
    "Ответь одним словом — Да или Нет."
)

PROMPTS = {"ours": QUESTION, "packaging": QUESTION_PACKAGING, "qc": QUESTION_QC,
           "rare2": QUESTION_RARE2}


def _grid_step(proc) -> int:
    """Шаг сетки = патч x слияние. Жёсткая 28 (соглашение Qwen2-VL) роняет раскладку."""
    ip = getattr(proc, "image_processor", proc)
    return int(getattr(ip, "patch_size", 16) or 16) * int(getattr(ip, "merge_size", 2) or 2)


def side_used_in_training(adapter_dir) -> int:
    """Ужималась ли длинная сторона кадра при обучении адаптера — из train_log.json.

    ⚠ Ноль означает «нет такого шага»: кадр идёт прямо под бюджет пикселей с
    выравниванием по сетке. Так работают адаптеры vl4_*, и менять им это нельзя —
    их публичные числа получены именно так.
    """
    return int(_train_log(adapter_dir).get("image_max_side") or 0)


def _load_image(path, step: int, max_pixels: int | None = None,
                max_side: int = 0):
    from PIL import Image

    mp = int(max_pixels or MAX_PIXELS)
    img = Image.open(path).convert("RGB")
    if max_side:
        # ⚠ Ровно подготовка редкой половины: только ужать длинную сторону и отдать
        # процессору, а он сам приведёт кадр к сетке под бюджет пикселей. Никакого
        # второго ужимания здесь быть не должно — в редкой половине его не было.
        w, h = img.size
        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                             Image.LANCZOS)
        return img
    w, h = img.size
    if w * h <= mp and w % step == 0 and h % step == 0:
        return img
    scale = (mp / (w * h)) ** 0.5 if w * h > mp else 1.0
    floor = step * 2
    nw = max(floor, (int(w * scale) // step) * step)
    nh = max(floor, (int(h * scale) // step) * step)
    return img.resize((nw, nh), Image.LANCZOS)


def build_prompt(proc, row, cfg: dict, n_imgs: int, template: str | None = None,
                 with_facts: bool = False, max_desc: int | None = None) -> str:
    """Плейсхолдеров ровно столько же, сколько кадров реально передано.

    ⚠ max_desc — обрезка описания, на которой обучался АДАПТЕР. Она разная у разных
    дообучений: у vl4_* это 800 символов, у vlm_rare2_a — 900. При том же шаблоне
    другая обрезка даёт другой вход, и ошибки не будет, только поехавший вердикт.
    Поэтому число читается из train_log.json рядом с адаптером, а не из конфига.
    """
    d = cfg["data"]
    cat = str(row[d["category_col"]])
    tpl = template or QUESTION
    fields = {"cat": cat, "rules": RULES.get(cat, ""),
              "name": str(row[d["name_col"]])[:MAX_NAME],
              "desc": str(row.get(d["desc_col"]) or "")[:int(max_desc or MAX_DESC)]}
    if "{facts}" in tpl:
        from qc26.facts import facts_block

        fields["facts"] = (facts_block(row[d["name_col"]], row.get(d["desc_col"]), cat)
                           if with_facts else "")
    text = tpl.format(**fields)
    content = [{"type": "image"} for _ in range(n_imgs)] + [{"type": "text", "text": text}]
    # ⚠ enable_thinking=False. Qwen3.5 — рассуждающая модель: её шаблон открывает
    # после хода ассистента блок <think>, и модель на этой позиции собирается
    # писать рассуждение, а не ответ. Мы же берём логит ПОСЛЕДНЕЙ позиции и
    # сравниваем «Да» против «Нет» — на открытом <think> это сравнение бессмысленно.
    # Шаблонам, которые такого ключа не знают (Qwen3-VL), он безвреден: неизвестные
    # аргументы просто уходят в переменные шаблона и не используются.
    # Проверка, что ключ подействовал, — в assert_no_open_thinking ниже.
    return proc.apply_chat_template([{"role": "user", "content": content}],
                                    tokenize=False, add_generation_prompt=True,
                                    enable_thinking=False)


def assert_no_open_thinking(prompt: str, where: str = "") -> None:
    """Промпт не должен обрываться на открытом блоке рассуждения.

    ⚠ Qwen3.5 — рассуждающая модель. Её шаблон после хода ассистента открывает
    <think>, и если он остался открытым, модель на последней позиции собирается
    писать рассуждение, а не ответ. Мы же читаем логит именно последней позиции и
    сравниваем «Да» против «Нет» — при открытом <think> это сравнение ничего не
    значит, а ошибки не будет: получатся правдоподобные, но случайные числа.

    Лечится ключом enable_thinking=False при сборке промпта. Он безвреден для
    шаблонов, которые о нём не знают, но и подействовать может не везде — поэтому
    результат проверяется здесь, а не принимается на веру.
    """
    tail = prompt[-400:]
    opened = tail.rfind("<think>")
    if opened == -1:
        return
    closed = tail.rfind("</think>")
    if closed > opened:
        return
    place = f" {where}" if where else ""
    # ⚠ В контейнере падать нельзя: решение без VLM хуже, но несравнимо лучше
    # упавшего. Поэтому здесь громкое предупреждение, а не остановка —
    # в отличие от обучения, где остановиться дешевле, чем учиться впустую.
    print(f"⚠⚠ промпт{place} обрывается на ОТКРЫТОМ <think>: модель будет "
          f"рассуждать, а не отвечать, и оценка «Да/Нет» станет случайной. "
          f"Хвост: ...{tail[-100:]!r}", flush=True)


# ⚠ Глушилка живёт в qc26/__init__.py и срабатывает ПРИ ИМПОРТЕ пакета: правило
# «не забыть позвать» дважды не сработало. Здесь только ре-экспорт, чтобы прежние
# импорты из этого модуля продолжали работать.
from qc26 import quiet_known_noise  # noqa: E402,F401

def first_token_ids(tok, word: str) -> list[int]:
    out = set()
    bare = word.strip()
    for s in (word, bare, bare[0], " " + bare[0]):
        enc = tok.encode(s, add_special_tokens=False)
        if enc:
            out.add(enc[0])
    return sorted(out)


def find_adapter_dir(path) -> Path | None:
    """Каталог с adapter_config.json: сам путь либо ЕДИНСТВЕННАЯ подпапка с ним.

    ⚠ Скрипт обучения дописывает к --out подпапку «{split}_f{fold}», поэтому адаптер
    оказывается на уровень глубже, чем ожидает конфиг отправки. Сборка из-за этого
    падала с «в адаптере нет файлов». Ищем в обоих местах, но при нескольких
    кандидатах отказываем: молча взять «какой-то» адаптер — это отправка не той модели.
    """
    p = Path(path)
    if (p / "adapter_config.json").exists():
        return p
    if not p.is_dir():
        return None
    found = sorted(x.parent for x in p.glob("*/adapter_config.json"))
    if len(found) == 1:
        return found[0]
    if len(found) > 1:
        print(f"в {p} несколько адаптеров: {[f.name for f in found]} — "
              f"укажите нужный явно", flush=True)
    return None


def apply_lora(model, adapter_dir: Path) -> int:
    """Вживляет LoRA в веса БЕЗ библиотеки peft. Возвращает число изменённых слоёв.

    ⚠ Почему без peft. Образ решения предсобран организаторами
    (`odsai/ecup26-quality-baseline:1.0`), и наш requirements-inference.txt в нём
    может не устанавливаться. Тогда `from peft import PeftModel` падает, наш же
    try/except это ловит, и решение МОЛЧА отдаёт вердикт без VLM. Именно так три
    отправки подряд дали значения, совпавшие до десятого знака с вариантами без VLM:
    0.725200676 дважды и 0.7618931824 против прежнего 0.7618931824.

    Математика LoRA простая и своей библиотеки не требует:
        W <- W + (B @ A) * (alpha / r)
    где A имеет форму (r, вход), B — (выход, r). Складываем прямо в веса, после
    чего модель обычная и никаких дополнительных зависимостей не нужно.
    """
    import json

    import torch
    from safetensors.torch import load_file

    # ⚠ Принимаем и строку, и Path. Раньше строка доходила до «adapter_dir / ...»
    # и падала с «unsupported operand type(s) for /: str and str» — сообщение,
    # по которому не догадаться, что дело в типе аргумента.
    adapter_dir = Path(adapter_dir)
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    scale = float(cfg.get("lora_alpha", 16)) / float(cfg.get("r", 16))
    if cfg.get("use_dora") or cfg.get("use_rslora"):
        raise RuntimeError("адаптер использует DoRA/rsLoRA — простое сложение неверно")

    weights = load_file(str(adapter_dir / "adapter_model.safetensors"))
    by_module: dict[str, dict[str, "torch.Tensor"]] = {}
    for key, tensor in weights.items():
        if ".lora_A" in key:
            base, part = key.split(".lora_A", 1)[0], "A"
        elif ".lora_B" in key:
            base, part = key.split(".lora_B", 1)[0], "B"
        else:
            continue
        base = base.replace("base_model.model.", "", 1)
        by_module.setdefault(base, {})[part] = tensor

    named = dict(model.named_modules())
    done = 0
    for path, pair in by_module.items():
        if "A" not in pair or "B" not in pair:
            raise RuntimeError(f"у слоя {path} нет пары lora_A/lora_B")
        mod = named.get(path)
        if mod is None or not hasattr(mod, "weight"):
            raise RuntimeError(f"слой {path} не найден в модели — адаптер от другой сети")
        w = mod.weight
        delta = (pair["B"].to(torch.float32) @ pair["A"].to(torch.float32)) * scale
        if tuple(delta.shape) != tuple(w.shape):
            raise RuntimeError(f"{path}: форма поправки {tuple(delta.shape)} против "
                               f"{tuple(w.shape)} у весов")
        with torch.no_grad():
            w.add_(delta.to(device=w.device, dtype=w.dtype))
        done += 1
    if done == 0:
        raise RuntimeError("адаптер не изменил ни одного слоя")
    return done


def adapter_fingerprint(adapter_dir: Path) -> str:
    """Короткий отпечаток адаптера: хэш файла весов + суммарная норма поправок.

    ⚠ Зачем. Три отправки с ТРЕМЯ РАЗНЫМИ адаптерами дали побитово одинаковое
    публичное значение 0.6921257405606154 (разложение: БАД 459/524, редкая 11 из 19).
    Отличить «модели действительно согласны» от «поехал не тот адаптер» по логу было
    нечем. Теперь отпечаток печатается и при сборке, и на прогоне: если у двух
    отправок он совпал — это одна и та же модель, и обсуждать нечего.
    """
    import hashlib
    import json

    import torch
    from safetensors.torch import load_file

    p = Path(adapter_dir) / "adapter_model.safetensors"
    h = hashlib.sha256(p.read_bytes()).hexdigest()[:12]
    cfg = json.loads((Path(adapter_dir) / "adapter_config.json").read_text(
        encoding="utf-8"))
    scale = float(cfg.get("lora_alpha", 16)) / float(cfg.get("r", 16))
    w = load_file(str(p))
    total = 0.0
    for key, tensor in w.items():
        if ".lora_B" in key:
            a_key = key.replace(".lora_B", ".lora_A")
            if a_key in w:
                total += float((tensor.to(torch.float32)
                                @ w[a_key].to(torch.float32)).norm()) * scale
    return f"{h} | сумма норм поправок {total:.2f} | слоёв {len(w) // 2}"


def _resolve_base(name: str = BASE_MODEL) -> str:
    """Путь к базовой модели. В списке организаторов она с маленькой «i», у нас с большой."""
    from .embed import has_weights, resolve_local_model

    root = Path(os.environ.get("SHARED_MODELS_PATH", "/shared_models"))
    tried = []
    for cand in (name, name.replace("-Instruct", "-instruct"),
                 name.split("/")[-1], name.split("/")[-1].replace("-Instruct", "-instruct")):
        p = root / cand
        tried.append(p)
        if has_weights(p):
            print(f"базовая модель из общего каталога: {p}", flush=True)
            return str(p)
    # ⚠ Сюда попадаем, если модели нет в /shared_models. Локально спасёт кэш
    # HuggingFace, но В КОНТЕЙНЕРЕ ЕГО НЕТ — значит там будет молчаливый откат
    # на решение без VLM. Кричим об этом, чтобы не узнать по совпавшему до
    # десятого знака публичному значению.
    print("⚠ ВНИМАНИЕ: базовой модели нет в общем каталоге, пробовали: "
          + ", ".join(str(p) for p in tried)
          + ". Беру кэш HuggingFace — В КОНТЕЙНЕРЕ ЭТО НЕ СРАБОТАЕТ.", flush=True)
    return resolve_local_model(name)


def images_used_in_training(adapter_dir) -> int | None:
    """Сколько кадров видел адаптер при обучении — из train_log.json рядом с ним.

    ⚠ Единственный надёжный источник. Число кадров в конфиге отправки может
    разойтись с обучением, и тогда модель получит чужое распределение входа, а
    ошибки не будет — только тихо поехавший вердикт.
    """
    import json

    p = Path(adapter_dir) / "train_log.json"
    if not p.exists():
        return None
    try:
        return int(json.loads(p.read_text(encoding="utf-8")).get("n_images"))
    except Exception:
        return None


def _train_log(adapter_dir) -> dict:
    """Журнал обучения рядом с адаптером. Пустой словарь, если его нет."""
    import json

    p = Path(adapter_dir) / "train_log.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def assert_reproducible_input(adapter_dir) -> bool:
    """Сможем ли мы воспроизвести вход, на котором учился адаптер.

    ⚠ В прогон едет только ОДИН набор правил (полный) и только режим «текст и кадры».
    Если адаптер учился на сокращённых или официальных правилах, либо в режиме
    «только текст» / «только кадры», воспроизвести его вход нечем — а подставить
    что-то похожее значит дать модели чужое распределение молча.

    Ровно так мы уже попадались на числе кадров и шаблоне промпта. Здесь лучше
    отказаться вслух: решение без VLM хуже, но его слабость видна, а тихо поехавший
    вердикт неотличим от правильного до самого лидерборда.
    """
    log = _train_log(adapter_dir)
    rules = str(log.get("rules") or "on")
    modality = str(log.get("modality") or "both")
    ok = True
    if rules != "on":
        print(f"⚠⚠ адаптер обучен на правилах «{rules}», а прогон умеет только "
              f"полные — вход не воспроизвести, VLM пропускается", flush=True)
        ok = False
    if modality != "both":
        print(f"⚠⚠ адаптер обучен в режиме «{modality}», а прогон умеет только "
              f"«текст и кадры» — вход не воспроизвести, VLM пропускается", flush=True)
        ok = False
    return ok


def pixels_used_in_training(adapter_dir) -> int:
    """Разрешение кадра при обучении адаптера — из train_log.json рядом с ним.

    ⚠ Та же ловушка, что с числом кадров, шаблоном промпта и базовой моделью.
    Подать кадр другого размера — значит дать модели вход из другого распределения,
    и ошибки не будет: поедет вердикт. Прежние адаптеры записи не имеют, им 100352 —
    другого разрешения тогда не существовало.
    """
    return int(_train_log(adapter_dir).get("max_pixels") or MAX_PIXELS)


def answer_ids_used_in_training(tok, adapter_dir):
    """Токены «да» и «нет», на которых считалась шкала адаптера.

    ⚠ Списки написаний берутся из train_log.json рядом с адаптером. Их отсутствие
    означает поведение по умолчанию — ровно то, на котором получены публичные баллы
    адаптеров vl4_*, поэтому менять его нельзя.
    """
    log = _train_log(adapter_dir)
    yv, nv = log.get("yes_variants"), log.get("no_variants")
    if not (yv and nv):
        return first_token_ids(tok, YES), first_token_ids(tok, NO)

    def ids(variants):
        out = set()
        for v in variants:
            enc = tok.encode(v, add_special_tokens=False)
            if enc:
                out.add(enc[0])
        return sorted(out)

    return ids(yv), ids(nv)


def desc_used_in_training(adapter_dir) -> int:
    """Обрезка описания при обучении адаптера — из train_log.json рядом с ним."""
    return int(_train_log(adapter_dir).get("desc_max") or MAX_DESC)


def base_used_in_training(adapter_dir) -> str:
    """На каких весах обучен адаптер — из train_log.json рядом с ним.

    ⚠ Адаптер ранга 32 от 4B и от 2B на диске выглядят одинаково, а формы слоёв
    у них разные. Вживление не в те веса либо упадёт, либо — если размерности
    случайно сойдутся — тихо испортит вердикт. Источник истины один: журнал.
    Старые адаптеры записи не имеют, им 2B: другой базы тогда не существовало.
    """
    return str(_train_log(adapter_dir).get("base_model") or BASE_MODEL)


def facts_used_in_training(adapter_dir) -> bool:
    """Был ли в промпте блок эвристических наблюдений (qc26.facts).

    ⚠ Добавить его только на прогоне — значит подать вход из другого
    распределения; убрать, если он был при обучении, — то же самое наоборот.
    """
    return str(_train_log(adapter_dir).get("facts") or "off") == "on"


def prompt_used_in_training(adapter_dir) -> tuple[str, str]:
    """Какой шаблон промпта видел адаптер — из train_log.json рядом с ним.

    ⚠ Та же ловушка, что с числом кадров, и она уже сработала: адаптер, обученный
    с формулировкой «packaging», а размеченный на прогоне нашей — это вход из
    другого распределения. Абляция показала, насколько модель чувствительна к
    структуре промпта: удаление блока правил роняет попадания с 9 из 30 до 1.
    Поэтому источник истины — журнал обучения, а не конфиг отправки.

    Возвращает (имя, шаблон). Старые адаптеры журнала не имеют — им «ours»,
    потому что до появления флага другого шаблона не существовало.
    """
    name = str(_train_log(adapter_dir).get("prompt") or "ours")
    if name not in PROMPTS:
        print(f"⚠ адаптер обучен на неизвестном шаблоне «{name}» — беру «ours», "
              f"вердикт может поехать", flush=True)
        name = "ours"
    return name, PROMPTS[name]


def _load_from_packed(src, dtype, device):
    """Модель из каталога с весами в int8. Скелет по конфигу, веса развёрнуты.

    ⚠ Почему не from_pretrained: он ищет обычные тензоры по именам, а у нас вместо
    каждой матрицы лежат три — «.q», «.s» и «.d». Он бы их не узнал, оставил слои
    неинициализированными и предупредил в лог, который никто не читает. Вердикт
    стал бы случайным.

    ⚠ Скелет собирается на meta-устройстве: иначе torch сначала выделит 8.3 ГБ под
    случайную инициализацию, а потом мы поверх положим свои веса — двойной расход
    памяти на ровном месте.
    """
    import torch
    from transformers import AutoConfig, AutoModelForImageTextToText

    from qc26.models.packing import load_packed

    from qc26.models.packing import load_packed_into

    import torch

    cfg = AutoConfig.from_pretrained(src, local_files_only=True)
    # ⚠ БЕЗ meta-устройства. Transformers в новых версиях создаёт скелет на meta
    # по умолчанию. Если модель на meta, .to(dtype) падает с «Cannot copy out of
    # meta tensor». Решение PyTorch: использовать to_empty(device) вместо to().
    model = AutoModelForImageTextToText.from_config(cfg,
                                                    attn_implementation="sdpa")
    if str(model.device) == "meta":
        # Переносим со скелета на CPU без копирования данных (их всё равно нет)
        model = model.to_empty(device="cpu", recurse=True)
    model = model.to(dtype)
    n, missing = load_packed_into(model, src, dtype=dtype)
    # Связанные веса (lm_head к embed_tokens) в файле лежат один раз — попадают
    # в недостающие законно. Всё остальное — нет.
    real = [k for k in missing if "lm_head" not in k]
    if real:
        raise RuntimeError(
            f"после распаковки не хватило {len(real)} тензоров из {len(missing) + n}, "
            f"например {real[:3]}. Веса и конфиг из разных версий модели?")
    print(f"развёрнуто тензоров: {n}", flush=True)
    model.tie_weights()
    return model.to(device)


def auto_batch(device, n_images: int, asked: int | None = None,
               max_pixels: int | None = None) -> int:
    """Батч прогона ПО ФАКТИЧЕСКОЙ видеопамяти, а не числом из конфига.

    ⚠ Это грабля второй половины, стоившая ему замера: батч, подобранный под свою
    карту на 8-24 ГБ, оставляет H100 проверяющей системы (80 ГБ) полупустой,
    и «ускорение» выходит 1.3x вместо разов. Поэтому в конфиге стоит батч 32
    с пометкой «под H100 80 ГБ», а локально он ужимается сам.

    Считаем от того, что реально свободно после загрузки весов. Модель 4B в bf16
    занимает около 8.3 ГБ, ей нужен запас на активации: при пяти кадрах и длине
    около 1200 токенов один товар стоит примерно 0.35 ГБ пиково.

    asked — значение из конфига отправки. Оно работает как ВЕРХНЯЯ граница:
    поднять батч выше него авторасчёт не станет, а опустить при нехватке памяти —
    обязан, иначе прогон упадёт по памяти уже в контейнере.
    """
    import torch

    hard_cap = int(asked) if asked else 64
    if getattr(device, "type", str(device)) != "cuda" or not torch.cuda.is_available():
        return min(hard_cap, 4)          # процессор и MPS: батч ничего не ускоряет

    free, total = torch.cuda.mem_get_info()
    free_gb = free / 2 ** 30
    # ⚠ Оставляем четверть свободного про запас: длины последовательностей разные,
    # и один длинный товар не должен ронять прогон на середине.
    # ⚠ 0.07 ГБ на кадр замерено при 100352 пикселях. Память под активации растёт
    # линейно с числом зрительных токенов, а оно пропорционально пикселям: при
    # 200704 один товар стоит вдвое дороже. Без этого множителя расчёт на 24 ГБ
    # выдавал батч 31, и КАЖДЫЙ батч падал по памяти — а падение ловится как
    # «сбой батча», вердикт молча уходил на текстовую модель. Именно так вышло
    # 100% сбоев на 300 товарах при том, что на 10 всё считалось.
    # ⚠ КВАДРАТ, а не просто отношение. Замерено: при 200704 расчёт с линейным
    # множителем дал батч 15, и половина батчей не влезла — настоящая ёмкость
    # около семи. Память внимания растёт быстрее числа токенов, а токенов
    # пропорционально пикселям. Квадрат совпал с наблюдением.
    px_scale = max(1.0, float(max_pixels or MAX_PIXELS) / MAX_PIXELS) ** 2
    per_item_gb = 0.07 * max(1, n_images) * px_scale
    fit = int(free_gb * 0.75 / per_item_gb)
    batch = max(1, min(hard_cap, fit))
    print(f"батч прогона: {batch} (свободно {free_gb:.1f} ГБ из "
          f"{total / 2 ** 30:.0f}, кадров {n_images}, "
          f"разрешение x{px_scale:.2g}, потолок из конфига {hard_cap})",
          flush=True)
    return batch


_MODELS: dict = {}


def load_with_adapter(src, adapter, dtype, device):
    """Модель с уже вживлённым адаптером. Одна на пару (веса, адаптер).

    ⚠ Кэш нужен по времени, а не по памяти. Каждый адаптер запрашивается дважды —
    для вердикта и для объяснения, — и без кэша это четыре распаковки int8 подряд.
    На стадии проверки лимит три минуты, и четыре в него не помещаются: отправка
    снималась по времени ещё до публичной стадии.
    """
    from transformers import AutoModelForImageTextToText

    key = (str(src), str(adapter), str(dtype), str(device))
    hit = _MODELS.get(key)
    if hit is not None:
        print(f"адаптер {Path(adapter).name}: модель уже собрана — переиспользую, "
              f"распаковка не повторяется", flush=True)
        return hit
    if (Path(src) / "packing.json").exists():
        print("веса упакованы в int8 — разворачиваю", flush=True)
        model = _load_from_packed(src, dtype, device)
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            src, dtype=dtype, local_files_only=True,
            attn_implementation="sdpa").to(device)
    print(f"отпечаток адаптера: {adapter_fingerprint(adapter)}", flush=True)
    n = apply_lora(model, adapter)
    print(f"LoRA вживлена в {n} слоёв (без peft)", flush=True)
    model.eval()
    model.config.use_cache = False
    _MODELS[key] = model
    return model


def score_frame(df: pd.DataFrame, cfg: dict, images_root, adapter_path,
                *, batch: int = 8, n_images: int | None = None,
                verbose: bool = True, base_override: str | None = None):
    """Оценка «товар относится к заявленной категории» на каждую строку.

    ⚠ Шкала — ЛОГИТ-РАЗНОСТЬ log P(Да) - log P(Нет), а НЕ вероятность: у вероятности
    в float32 нет разрешения выше z = 17, и треть выборки слипается в ровно 1.0.
    Величина монотонна вероятности, поэтому отбор по доле и по рангу работает как
    прежде, но ничьих не остаётся. Порог в ВЕРОЯТНОСТНОЙ шкале к ней неприменим.

    Возвращает массив длиной len(df) либо None, если VLM недоступна.
    """
    try:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        adapter = find_adapter_dir(adapter_path)
        if adapter is None:
            print(f"адаптер не найден в {adapter_path} — VLM пропускается", flush=True)
            return None
        # ⚠ Базовые веса — те же, на которых обучался адаптер. Формы слоёв 2B и 4B
        # различаются, и вживление не в те веса испортит вердикт. Источник один:
        # журнал обучения рядом с адаптером, а не константа и не конфиг отправки.
        quiet_known_noise()
        base = base_used_in_training(adapter)
        # ⚠ base_override — веса не из каталога жюри, а привезённые с решением.
        # Нужен, когда модели нет в /shared_models: Qwen3-VL-4B туда не входит,
        # и она едет в архиве упакованной в int8. Задаётся из конфига отправки.
        src = str(base_override) if base_override else _resolve_base(base)
        if base_override:
            if not Path(src).is_dir():
                print(f"⚠⚠ НЕТ КАТАЛОГА ВЕСОВ {src} — VLM пропускается", flush=True)
                return None
            print(f"веса из архива решения: {src}", flush=True)

        # ⚠ MPS обязателен для ЛОКАЛЬНОЙ проверки сборки. В контейнере CUDA, но
        # проверять архив приходится на Mac, а на процессоре в fp32 полный прогон
        # редкой категории идёт часами — сборка выглядит зависшей. На MPS это минуты.
        if torch.cuda.is_available():
            device, dtype = "cuda", torch.bfloat16
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device, dtype = "mps", torch.bfloat16
        else:
            device, dtype = "cpu", torch.float32
        proc = AutoProcessor.from_pretrained(src, local_files_only=True)
        # ⚠ Упакованные веса грузятся иначе: from_pretrained не знает про наш формат
        # int8 с масштабом на строку. Сборка и вживка адаптера вынесены в
        # load_with_adapter — она же держит кэш, чтобы проход объяснений не
        # распаковывал те же веса заново.
        model = load_with_adapter(src, adapter, dtype, device)

        tok = proc.tokenizer
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"             # логит берём с последней позиции
        yes_ids, no_ids = answer_ids_used_in_training(tok, adapter)
        print(f"токены ответа: «да» {len(yes_ids)} шт., «нет» {len(no_ids)} шт.",
              flush=True)
        if set(yes_ids) & set(no_ids):
            print("классы делят первый токен — VLM пропускается", flush=True)
            return None
        step = _grid_step(proc)
        trained = images_used_in_training(adapter)
        n_img = int(n_images or trained or MAX_IMAGES)
        if trained is not None and n_img != trained:
            print(f"⚠ адаптер обучен на {trained} кадрах, подаём {n_img} — "
                  f"беру {trained}, иначе вход из другого распределения", flush=True)
            n_img = trained
        # ⚠ Батч считаем ПОСЛЕ загрузки весов: до неё свободная память ничего
        # не говорит о том, сколько останется под активации.
        prompt_name, template = prompt_used_in_training(adapter)
        desc_train = desc_used_in_training(adapter)
        side_train = side_used_in_training(adapter)
        if side_train:
            print(f"подготовка кадра как в редкой половине: длинная сторона до "
                  f"{side_train}, дальше процессор", flush=True)
        with_facts = facts_used_in_training(adapter)
        # ⚠ Разрешение — тоже из журнала обучения, а не из константы модуля.
        if not assert_reproducible_input(adapter):
            return None
        px = pixels_used_in_training(adapter)
        # ⚠ Батч считается ПОСЛЕ разрешения: при удвоенном разрешении товар стоит
        # вдвое больше памяти, и расчёт без этого знания даёт батч, который не
        # влезает. Порядок строк здесь существенный.
        batch = auto_batch(model.device, n_img, asked=batch, max_pixels=px)
        if px != MAX_PIXELS:
            print(f"⚠ адаптер обучен при {px} пикселях на кадр вместо {MAX_PIXELS} — "
                  f"беру {px}, иначе вход из другого распределения", flush=True)
        if verbose:
            print(f"VLM: {src} + адаптер {adapter.name}, устройство {device}, "
                  f"шаг сетки {step}, кадров {n_img}, пикселей {px}, "
                  f"промпт «{prompt_name}»"
                  + (", блок наблюдений включён" if with_facts else ""),
                  flush=True)
    except Exception as e:
        print(f"VLM недоступна ({type(e).__name__}: {str(e)[:160]}) — пропускается",
              flush=True)
        return None

    idc = cfg["data"]["id_col"]
    out: list[float] = []
    fails = 0
    t0 = time.time()
    for start in range(0, len(df), batch):
        chunk = df.iloc[start:start + batch]
        # ⚠ Промпт и его кадры храним ПАРОЙ. Раньше кадры складывались в отдельный
        # список только для товаров, у которых они нашлись, и два списка
        # разъезжались. Разделить такой батч пополам без ошибки нельзя, а делить
        # приходится — см. _run ниже.
        items: list = []
        for _, row in chunk.iterrows():
            imgs = []
            for p in _paths(images_root, str(row[idc]), n_img):
                try:
                    imgs.append(_load_image(p, step, px, side_train))
                except Exception:
                    pass
            prompt = build_prompt(proc, row, cfg, len(imgs), template,
                                  with_facts, max_desc=desc_train)
            if start == 0 and not items:
                assert_no_open_thinking(prompt, "прогона")
            items.append((prompt, imgs))
        images = [im for _, im in items if im]
        try:
            import torch

            def _forward(pr, im):
                enc = proc(text=pr, images=im or None, padding=True,
                           return_tensors="pt").to(model.device)
                with torch.no_grad():
                    return model(**enc).logits[:, -1, :].float()

            def _is_oom(e):
                return (type(e).__name__ == "OutOfMemoryError"
                        or "out of memory" in str(e).lower())

            def _run(part):
                """Проход с ДРОБЛЕНИЕМ при нехватке памяти.

                ⚠ Зачем: батч считается по свободной памяти ДО прогона, но длины
                описаний разные, и отдельный батч всё равно может не влезть.
                Раньше это ловилось как «сбой батча», и товары молча теряли
                оценку: при удвоенном разрешении так пропало сначала 100%
                выборки, потом 50%. Уполовинивание надёжнее любого множителя в
                оценке — оно смотрит на факт, а не на предсказание, и потому
                одинаково верно на нашей карте и на чужой.

                Сдаёмся, только если не влезает ОДИН товар: это уже настоящая
                беда, и знать о ней надо.
                """
                pr = [t for t, _ in part]
                im = [i for _, i in part if i]
                try:
                    return _forward(pr, im)
                except Exception as e:
                    if not _is_oom(e) or len(part) == 1:
                        raise
                    torch.cuda.empty_cache()
                    half = len(part) // 2
                    return torch.cat([_run(part[:half]), _run(part[half:])], dim=0)

            lg = _run(items)
            if not torch.isfinite(lg).all():
                raise RuntimeError("нечисловые логиты")
            # ⚠ ЛОГИТ-РАЗНОСТЬ, а не вероятность. sigma(z) в float32 при z > 17
            # неотличима от единицы, и треть выборки слипается в ровно 1.0.
            # Замерено на 100 товарах: среднее оценок 0.3298, разброс 0.4692 — это
            # ровно f и sqrt(f(1-f)) при f = 0.33, то есть распределение двугорбое.
            # Отбор «топ-19 по доле» из такой ничьей выбирает по ошибке округления
            # последнего бита. Именно поэтому три РАЗНЫХ адаптера дали одинаковые
            # 11 попаданий из 19. Логит-разность монотонна той же вероятности,
            # но разрешения ей хватает.
            yes = torch.logsumexp(lg[:, yes_ids], dim=-1)
            no = torch.logsumexp(lg[:, no_ids], dim=-1)
            out.extend((yes - no).cpu().numpy().tolist())
        except Exception as e:
            # ⚠ строку терять нельзя: 0.5 значит «нет мнения», решит порог
            fails += len(chunk)
            if fails <= len(chunk) * 2:
                print(f"сбой батча VLM ({type(e).__name__}): {str(e)[:140]}", flush=True)
            out.extend([0.0] * len(chunk))   # 0 в логит-разности = «нет мнения»
        for im in images:
            for x in im:
                try:
                    x.close()
                except Exception:
                    pass

    arr = np.asarray(out, dtype=np.float32)
    if len(arr) != len(df):
        print(f"VLM вернула {len(arr)} оценок на {len(df)} строк — пропускается",
              flush=True)
        return None
    if verbose:
        print(f"VLM готова: {len(arr)} оценок, "
              f"{(time.time() - t0) / max(1, len(df)) * 1000:.0f} мс/товар"
              + (f", сбоев {fails}" if fails else ""), flush=True)
    if fails > len(df) * 0.05:
        print(f"сбоев {fails / len(df):.1%} — больше 5%, VLM пропускается", flush=True)
        return None
    return arr


def _paths(images_root, item_id: str, n_images: int = MAX_IMAGES):
    from ..data import image_paths

    return image_paths(images_root, item_id)[:n_images]
