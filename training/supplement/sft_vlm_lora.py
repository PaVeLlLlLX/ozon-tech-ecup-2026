# scripts/sft_vlm_lora_mac.py
# LoRA-дообучение VLM на редкой категории для MacBook M-series (Apple Silicon)
#
# Запуск:
#   python scripts/sft_vlm_lora_mac.py --stage train --split fold_seed42 --fold 0
#   python scripts/sft_vlm_lora_mac.py --stage eval  --split fold_seed42 --fold 0

import argparse
import json
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import torch
from PIL import Image

from qc26.config import load_config, resolve_path
from qc26.data import image_paths, images_root, load_cards

# ⚠ Базовая модель — ГЛОБАЛ, а не константа: задаётся из --base-model перед
# загрузкой весов и записывается в train_log.json рядом с адаптером. Инференс
# читает её оттуда же. Разойтись здесь значит вживить адаптер 4B в 2B: формы
# слоёв не совпадут, и это как раз тот редкий случай, когда ошибка будет громкой.
# Но если совпадут (одно семейство, разный размер) — вердикт поедет молча.
MODEL = "Qwen/Qwen3-VL-2B-Instruct"
KNOWN_BASES = {
    # имя -> (параметров, вес в bf16 ГБ, есть ли в каталоге жюри)
    "Qwen/Qwen3-VL-2B-Instruct": (2.2e9, 4.4, True),
    "Qwen/Qwen3-VL-4B-Instruct": (4.5e9, 8.3, False),
    "Qwen/Qwen3.5-4B": (4.0e9, 8.8, True),
}

# настройки LoRA: задаются из аргументов перед загрузкой модели
_lora_r, _lora_alpha, _lora_dropout = 16, 32, 0.05
_lora_targets = "q_proj,k_proj,v_proj,o_proj"
CATEGORY = "Легковоспламеняющиеся"
YES, NO = " Да", " Нет"
# ⚠ Разрешение кадра — ГЛОБАЛ, задаётся из --max-pixels и пишется в train_log.json.
# Прогон читает его оттуда же. Разойтись здесь значит подать модели вход из другого
# распределения БЕЗ ЕДИНОЙ ОШИБКИ — ровно как было с числом кадров и шаблоном промпта.
#
# Замерено 27.08 на 400 товарах: при 100352 модель не читает упаковку, а выдумывает
# правдоподобные русские слова («БАПЛОН» вместо «БАЛЛОН», «ПЕРЕДОЗНА» вместо
# «ПЕРЕХОДНИК»). Пятая часть «прочитанного» исчезает при вчетверо большем разрешении.
MAX_PIXELS = 100352
PRIOR = 0.0339

# Правила у категорий РАЗНЫЕ, поэтому в промпт идёт блок своей категории.
# Обучение на обеих сразу — ключевое отличие от трёх наших прежних попыток:
# на редкой всего 158 обучающих позитивов, на всей выборке около 4600.
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

# ⚠ Аудит разметки: в рабочей точке 12 из 30 ложных срабатываний — товары, которые
# РЕГЛАМЕНТ относит к категории, а разметка нет (бензин Zippo, брикеты для розжига,
# сухое горючее). Модель, идеально выучившая правила, может проиграть по лидерборду,
# поэтому правила здесь не константа, а переключатель для замера.
RULES_SHORT = {
    "Легковоспламеняющиеся": ("Важно, продаётся ли само горючее (или входит ли оно "
                              "в комплект), а не устройство для огня без топлива."),
    "БАД": "Важно наличие маркировки БАД, а не польза для здоровья.",
}

# Дословная формулировка организаторов. Отличие от RULES не косметическое:
# у нас «горючее входит в комплект», у них «в комплект входит ЛЕГКОВОСПЛАМЕНЯЮЩИЙСЯ
# ТОВАР» — второе уже. Плюс отдельным пунктом «источник воспламенения встроен
# в изделие», которого у нас нет вовсе.
RULES_OFFICIAL = {
    "Легковоспламеняющиеся": (
        "Товар является легковоспламеняющимся, если: товар является самостоятельным "
        "источником воспламенения (изделия, основное назначение которых — создание или "
        "поддержание открытого огня: спички, зажигалки); товар содержит горючее "
        "вещество, легковоспламеняющиеся вещества или горючие газы; в комплект товара "
        "входит легковоспламеняющийся товар.\n"
        "Товар не является легковоспламеняющимся, если: товар не содержит источника "
        "воспламенения или горючего вещества (устройство для использования с огнём само "
        "по себе не является таковым — мангалы, грили, газовые плиты); "
        "легковоспламеняющимся является содержимое, а не сама конструкция, и "
        "содержимого нет; источник воспламенения встроен в изделие; потенциально "
        "легковоспламеняющийся материал используется лишь как компонент другого изделия "
        "(активированный уголь в фильтрах, уголь для рисования); легковоспламеняющийся "
        "предмет не входит в комплект."),
    "БАД": (
        "Товар является биологически активной добавкой, если в описании или на "
        "изображении содержится прямое указание на это (БАД, dietary supplement).\n"
        "Товар не является биологически активной добавкой, если это спортивное питание "
        "(аминокислоты, BCAA, L-карнитин, протеин или иной товар с прямым указанием на "
        "принадлежность к спортивному питанию); если в описании явно указано, что товар "
        "не является биологически активной добавкой; если товар не содержит маркировок "
        "биологически активной добавки."),
}

QUESTION = (
    "Ты модератор маркетплейса. Товар заявлен в категории «{cat}».\n"
    "{rules}\n\n"
    "Название: {name}\n"
    "Описание: {desc}\n\n"
    "Относится ли товар к категории «{cat}»? Ответь одним словом: Да или Нет.\n"
    "Ответ:"
)

# ⚠ Формулировка, на которой во второй половине получено публично 0.7809 на ТОЙ ЖЕ
# базовой модели 2B. Отличий от нашей четыре, и все содержательные:
#   1. прямое указание смотреть на упаковку — у нас разбор ошибок показал, что
#      маркировка БАД у 725 товаров есть только на фото;
#   2. «Продавец указал категорию» вместо «Товар заявлен» — категория подана как
#      утверждение, которое надо проверить, а не как факт;
#   3. «товар ДЕЙСТВИТЕЛЬНО относится» — смещает задачу к проверке;
#   4. нет суффикса «Ответ:» — ответ начинает генеративная часть шаблона чата.
# Абляция показала, что модель реагирует на НАЛИЧИЕ блока правил, а не на его
# содержание (чужой текст правил даёт 8 из 30 против родных 9, пустой — 1).
# Значит структура промпта весит больше смысла, и переформулировка небезразлична.
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

# ⚠ НАШ шаблон. Собран из собственных замеров, а не переписан с чужого:
#
#   1. Название категории вводит в заблуждение. Разбор фолда 0 показал: 27 из 41
#      позитива — хлопушки, которые в обиходе никто не назовёт легковоспламеняющимися,
#      а 511 из 1060 негативов — газовые горелки с пьезоподжигом, которые назовёт
#      каждый. Базовая модель на этой паре даёт AUC 0.345, то есть ранжирует
#      ОБРАТНО разметке. Поэтому критерий назван прямо: чем товар является сам.
#   2. Из 22 достижимых промахов 11 — комплектация: «продаётся вместе с» против
#      «работает от». Это и есть решающее различие, и оно сказано явно.
#   3. Только фотографии дают 8 из 30 против 4 у чистого текста, поэтому указание
#      смотреть на упаковку сохранено — но своими словами.
#   4. Гнездо {facts} — блок эвристических наблюдений из qc26.facts. Пустая строка,
#      когда сказать нечего: заголовок без содержания добавил бы структуру без
#      информации, а к структуре промпта модель чувствительна (удаление блока
#      правил роняет попадания с 9 из 30 до 1).
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

PROMPTS = {"ours": QUESTION, "packaging": QUESTION_PACKAGING, "qc": QUESTION_QC}


# без названия и описания: сколько несёт ОДИН зрительный канал
QUESTION_IMAGE_ONLY = (
    "Ты модератор маркетплейса. Товар заявлен в категории «{cat}».\n"
    "{rules}\n\n"
    "Относится ли товар к категории «{cat}»? Ответь одним словом: Да или Нет.\n"
    "Ответ:"
)


def per_item_ce(logits, labels):
    """Средний лосс по токенам ОТВЕТА для каждого товара батча.

    ⚠ Считается только на позициях ответа, а не на всей последовательности.
    Прежний код делал logits[:, :-1].float() — копию всей матрицы [батч, длина,
    151936] в float32. При батче 8 это 6.8 ГБ поверх 3.4 ГБ самих логитов, и
    упиралось оно в память раньше, чем модель: на Mac пришлось держать батч 2.
    Размеченных позиций около двух на товар, остальное маскировано -100, поэтому
    выборка нужных позиций даёт ровно тот же результат за мегабайты.
    """
    lg = logits[:, :-1]
    tgt = labels[:, 1:]
    keep = tgt != -100
    sel = keep.nonzero(as_tuple=True)
    ce = torch.zeros(tgt.shape, dtype=torch.float32, device=tgt.device)
    if sel[0].numel():
        ce[sel] = torch.nn.functional.cross_entropy(
            lg[sel].float(), tgt[sel], reduction="none")
    kf = keep.float()
    return (ce * kf).sum(1) / kf.sum(1).clamp(min=1)


def get_device() -> torch.device:
    """Устройство обучения: CUDA, иначе MPS на Mac, иначе процессор.

    ⚠ CUDA сюда добавлена не для полноты. Без неё на арендованной A100 функция
    возвращала ПРОЦЕССОР: прогон не падал, а шёл в сотни раз медленнее, и аренда
    сгорала впустую. Модуль инференса CUDA знал, обучение — нет.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device: torch.device, args=None) -> None:
    """Громко печатает, на чём считаем. Молчаливый откат на процессор — это
    потерянные сутки аренды, и заметить его по логу иначе нечем."""
    if device.type == "cuda":
        pr = torch.cuda.get_device_properties(0)
        print(f"устройство: CUDA — {pr.name}, память {pr.total_memory / 2**30:.0f} ГБ, "
              f"вычислительная способность {pr.major}.{pr.minor}", flush=True)
        if pr.major < 8:
            print("  ⚠ карта старше Ampere: bfloat16 будет эмулироваться, "
                  "обучение замедлится", flush=True)
    elif device.type == "mps":
        print("устройство: MPS (Apple)", flush=True)
    else:
        print("устройство: ПРОЦЕССОР", flush=True)
        if torch.cuda.is_available():
            print("  ⚠⚠ видеокарта видна, но не выбрана — это ошибка, остановитесь",
                  flush=True)
        steps = getattr(args, "max_steps", 0) or 0
        if steps > 100 and not getattr(args, "dry_run", False):
            raise SystemExit(
                f"отказ: {steps} шагов на процессоре — это недели счёта. "
                f"Если это осознанно, запустите с --max-steps 100 для пробы.")


def clear_cache(device: torch.device) -> None:
    """Освобождает кэш выделений. У MPS и CUDA вызовы разные, на процессоре нечего."""
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def grid_step(proc) -> int:
    ip = getattr(proc, "image_processor", proc)
    return int(getattr(ip, "patch_size", 16) or 16) * int(getattr(ip, "merge_size", 2) or 2)


def load_image(path, step: int):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if w * h <= MAX_PIXELS and w % step == 0 and h % step == 0:
        return img
    scale = (MAX_PIXELS / (w * h)) ** 0.5 if w * h > MAX_PIXELS else 1.0
    floor = step * 2
    nw = max(floor, (int(w * scale) // step) * step)
    nh = max(floor, (int(h * scale) // step) * step)
    return img.resize((nw, nh), Image.LANCZOS)


def build_prompt(proc, row, cfg, imgs, max_desc: int, rules_mode: str = "on",
                 modality: str = "both", prompt_mode: str = "ours",
                 facts_mode: str = "off") -> str:
    """Промпт под НЕСКОЛЬКО кадров: плейсхолдеров ровно столько же, сколько картинок.

    ⚠ Несовпадение числа плейсхолдеров и картинок — это не ошибка формата, а тихий
    сдвиг: процессор разложит кадры по чужим карточкам батча, и обучение пойдёт по
    перепутанным парам. Поэтому список строится из самих кадров, а не из аргумента.

    ⚠ facts_mode добавляет блок эвристических наблюдений (см. qc26.facts). Он ДОЛЖЕН
    совпадать между обучением и прогоном, поэтому пишется в train_log.json, а
    инференс читает его оттуда — как число кадров и шаблон вопроса.
    """
    d = cfg["data"]
    imgs = list(imgs or [])
    if modality == "text":
        imgs = []                       # кадры не подаём совсем
    cat = str(row[d["category_col"]])
    rules = {"on": RULES.get(cat, ""), "off": "",
             "short": RULES_SHORT.get(cat, ""),
             "official": RULES_OFFICIAL.get(cat, "")}[rules_mode]
    if modality == "image":
        text = QUESTION_IMAGE_ONLY.format(cat=cat, rules=rules)
    else:
        tpl = PROMPTS[prompt_mode]
        fields = {"cat": cat, "rules": rules,
                  "name": str(row[d["name_col"]])[:300],
                  "desc": str(row.get(d["desc_col"]) or "")[:max_desc]}
        if "{facts}" in tpl:
            from qc26.facts import facts_block

            fields["facts"] = (facts_block(row[d["name_col"]],
                                           row.get(d["desc_col"]), cat)
                               if facts_mode == "on" else "")
        elif facts_mode == "on":
            raise SystemExit(
                f"шаблон «{prompt_mode}» не имеет гнезда {{facts}} — блок наблюдений "
                f"вставить некуда. Возьмите --prompt qc либо --facts off.")
        text = tpl.format(**fields)
    content = [{"type": "image"} for _ in imgs] + [{"type": "text", "text": text}]
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
    raise SystemExit(
        f"промпт{place} обрывается на ОТКРЫТОМ <think>.\n"
        f"  Модель будет рассуждать, а не отвечать, и оценка «Да/Нет» станет "
        f"случайной — молча.\n"
        f"  Хвост промпта: ...{tail[-120:]!r}\n"
        f"  Ключ enable_thinking=False не подействовал: смотрите chat_template "
        f"модели, там может быть другое имя переменной.")


_IMG_WORKERS = 1          # ⚠ ставится из --img-workers перед обучением


def _safe_load(path, step):
    """Кадр или None. Битый файл не должен ронять весь батч."""
    try:
        return load_image(path, step)
    except Exception:
        return None

def _other_training_pids() -> list[int]:
    """Другие живые процессы этого же скрипта. Пусто, если считаем одни.

    Читаем /proc напрямую: ps есть не в каждом образе, а /proc есть всегда там,
    где вообще имеет смысл гасить машину.
    """
    import os
    import pathlib

    me, out = os.getpid(), []
    proc = pathlib.Path("/proc")
    if not proc.is_dir():
        return []
    for d in proc.iterdir():
        if not d.name.isdigit() or int(d.name) == me:
            continue
        try:
            cmd = (d / "cmdline").read_bytes().decode("utf-8", "replace")
        except Exception:
            continue
        if "sft_vlm_lora.py" in cmd:
            out.append(int(d.name))
    return sorted(out)


def poweroff_if_asked(args, ok: bool) -> None:
    """Гасит машину после прогона, если просили. Только при УСПЕХЕ.

    ⚠ Зачем: арендованная карта стоит около 106 руб/час на пару, и забытый на ночь
    сервер съедает 850 руб. Прогон на три эпохи кончается в непредсказуемый час,
    сидеть и ждать его дорого.

    ⚠ Чего этот флаг НЕ гарантирует: у некоторых провайдеров биллинг считает
    состояние ВИРТУАЛЬНОЙ МАШИНЫ, а не операционной системы, и погашенная изнутри
    машина продолжает тарифицироваться. Убедиться можно только по балансу в
    панели. Поэтому флаг — страховка, а не замена будильнику.

    ⚠ Гасим ТОЛЬКО после успеха: при падении машина обязана остаться живой,
    иначе разбираться будет негде, а адаптер останется недокачанным.
    Всё, что нужно, уже на диске — диск переживает остановку.
    """
    if not getattr(args, "poweroff", False):
        return
    if not ok:
        print("⚠ прогон завершился неуспешно — машину НЕ гашу, разбирайтесь на месте",
              flush=True)
        return

    import subprocess

    # ⚠ Две карты — два прогона одновременно. Тот, что закончит первым, погасил бы
    # машину ПОД ВТОРЫМ, и вторая ночь ушла бы впустую. Проверяем, не считает ли
    # ещё кто-то то же самое.
    others = _other_training_pids()
    if others:
        print(f"⚠ выключение ОТМЕНЕНО: ещё считают процессы {others}. "
              f"Гасить машину под ними значит потерять их работу.\n"
              f"  Погасите вручную, когда закончат все: shutdown -h +5", flush=True)
        return

    delay = max(1, int(args.poweroff))
    print(f"\n=== ВЫКЛЮЧЕНИЕ ЧЕРЕЗ {delay} МИН ===", flush=True)
    print("  отменить:  shutdown -c", flush=True)
    print("  ⚠ проверьте по балансу в панели, что списания прекратились: "
          "погашенная изнутри машина у части провайдеров продолжает тарифицироваться",
          flush=True)
    try:
        subprocess.run(["shutdown", "-h", f"+{delay}"], check=True)
    except Exception as e:
        print(f"⚠ выключить не удалось ({type(e).__name__}: {e}). "
              f"ГАСИТЕ ВРУЧНУЮ ИЗ ПАНЕЛИ.", flush=True)

def quiet_known_noise() -> None:
    """Глушит ровно два повторяющихся сообщения transformers. Не больше.

    ⚠ Глушим ТОЧЕЧНО, а не set_verbosity_error(): предупреждение про
    «fast path is not available» стоило нам целого разбора и едва не увело
    в прогон на 2000 руб мимо бюджета. Затыкать весь канал значит однажды
    не увидеть такое же важное.

    1) «Kwargs passed to processor.__call__ have to be in processor_kwargs» —
       печатается на КАЖДЫЙ вызов процессора, то есть десятки тысяч строк за
       прогон. Совет из него проверен и оказался ВРЕДНЫМ: с processor_kwargs
       процессор возвращает списки вместо выровненного тензора. Мы намеренно
       зовём по-старому, защёлка стоит в check_batch_padded.
    2) «torch_dtype is deprecated» — печатается один раз на загрузку модели.
    """
    import logging

    drop = ("processor_kwargs", "torch_dtype` is deprecated")

    class _Filter(logging.Filter):
        def filter(self, record):
            msg = str(record.getMessage())
            return not any(d in msg for d in drop)

    f = _Filter()
    for name in ("transformers", "transformers.processing_utils",
                 "transformers.modeling_utils"):
        logging.getLogger(name).addFilter(f)
    # transformers держит собственный корневой логгер библиотеки
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.get_logger().addFilter(f)
        hf_logging._get_library_root_logger().addFilter(f)
    except Exception:
        pass


def check_batch_padded(enc, n_rows: int) -> None:
    """Процессор обязан вернуть выровненный двумерный тензор, а не списки.

    ⚠ Защёлка против тихой смены поведения. transformers 5.14 предупреждает, что
    ключи padding и return_tensors положено передавать в processor_kwargs, — но
    проверка показала, что при таком вызове возвращаются СПИСКИ, то есть совет
    из предупреждения ломает батч. Мы оставили прежнюю форму, и она работает.

    Если следующая версия начнёт наши ключи игнорировать, обучение пойдёт по
    невыровненным входам. Дешевле узнать об этом на первом же батче.
    """
    ids = enc.get("input_ids") if hasattr(enc, "get") else None
    if not isinstance(ids, torch.Tensor) or ids.ndim != 2 or ids.shape[0] != n_rows:
        got = type(ids).__name__ + (f" {tuple(ids.shape)}"
                                    if isinstance(ids, torch.Tensor) else "")
        raise SystemExit(
            f"процессор вернул {got}, а нужен тензор [{n_rows}, длина].\n"
            f"  Ключи padding/return_tensors перестали действовать — вероятно, "
            f"сменилась версия transformers.\n"
            f"  ⚠ НЕ переносите их в processor_kwargs: проверено, оттуда "
            f"возвращаются списки. Смотрите сигнатуру processor.__call__.")


def check_answer_mask(tok, prompt: str, answer: str) -> None:
    """Токены, на которых стоит лосс, обязаны быть ровно ответом.

    ⚠ Маска строится вычитанием: n_ans токенов с конца — это ответ, всё до них
    закрыто. Работает это ровно тогда, когда ответ в контексте токенизируется
    так же, как отдельно. А хвост промпта у Qwen3.5 теперь закрытый блок
    рассуждения с двумя переводами строки, и токенизатор вполне может склеить
    его с началом ответа иначе — тогда граница уедет, и обучение пойдёт по чужим
    позициям. Ошибки при этом не будет.

    Проверяем прямо: раскодируем ровно те позиции, на которые встанет лосс, и
    сверяем с ответом.
    """
    full = tok.encode(prompt + answer + tok.eos_token, add_special_tokens=False)
    n_ans = len(tok.encode(answer + tok.eos_token, add_special_tokens=False))
    got = tok.decode(full[len(full) - n_ans:])
    want = answer + tok.eos_token
    if got != want:
        raise SystemExit(
            f"граница ответа поехала: под лоссом окажется {got!r}, "
            f"а должно быть {want!r}.\n"
            f"  Токенизатор склеил хвост промпта с началом ответа. Обучение "
            f"пошло бы по чужим позициям, и ошибки бы не было.\n"
            f"  Хвост промпта: {prompt[-40:]!r}")
    print(f"  граница ответа проверена: под лоссом ровно {got!r}", flush=True)

def first_token_ids(tok, word: str) -> list[int]:
    out = set()
    bare = word.strip()
    for s in (word, bare, bare[0], " " + bare[0]):
        enc = tok.encode(s, add_special_tokens=False)
        if enc:
            out.add(enc[0])
    return sorted(out)


def select(df, split, fold, neg_per_pos, seed=42):
    tr = df[df[split] != fold]
    te = df[df[split] == fold]
    pos = tr[tr.label == 1]
    n_neg = min(len(pos) * neg_per_pos, int((tr.label == 0).sum()))
    neg = tr[tr.label == 0].sample(n=n_neg, random_state=seed)
    train = pd.concat([pos, neg]).sample(frac=1.0, random_state=seed)
    return train, te


_grad_checkpoint = True    # ⚠ ставится из args перед load_model
_thinking_checked = False  # открытый <think> проверяем один раз за прогон


def load_model(device: torch.device, adapter=None, train_mode=False,
               resume_from=None):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    proc = AutoProcessor.from_pretrained(MODEL)
    
    # ⚠ bfloat16 и на MPS, и на CUDA. Раньше здесь стояло «bf16 только для MPS,
    # иначе float32», и на A100 модель грузилась в float32: вдвое больше памяти,
    # мимо тензорных ядер. Процессор оставляем на float32 — bf16 там медленнее.
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL,
        torch_dtype=dtype,
        attn_implementation="sdpa",  # стабильно и на MPS, и на CUDA
    ).to(device)
    print(f"  веса загружены в {dtype}", flush=True)
    
    model.config.use_cache = False
    
    if adapter:
        from peft import PeftModel

        # ⚠ Скрипт САМ дописывает к --out подпапку «{split}_f{fold}». Если адаптер
        # перекладывали руками, файлы часто оказываются на уровень выше, и peft
        # трактует путь как имя репозитория на HuggingFace — сообщение про
        # «Repo id must be in the form...» уводит совсем не туда. Проверяем явно.
        adapter_dir = Path(adapter)
        if not (adapter_dir / "adapter_config.json").exists():
            base = adapter_dir.parent
            found = (sorted(x.parent for x in base.rglob("adapter_config.json"))
                     if base.is_dir() else [])
            hint = ""
            if found:
                hint = (f"\n  Похоже, адаптер лежит здесь: {found[0]}"
                        f"\n  Тогда запускайте с --out {found[0].parent}")
            raise FileNotFoundError(
                f"нет adapter_config.json в {adapter_dir}."
                f"\n  Путь собирается как <--out>/<split>_f<fold>.{hint}")
        model = PeftModel.from_pretrained(model, str(adapter))
        print(f"адаптер загружен: {adapter}", flush=True)
    elif train_mode and resume_from:
        # ⚠ Продолжение обучения СУЩЕСТВУЮЩЕГО адаптера, а не создание нового.
        # Ранг, alpha и целевые слои берутся из его собственного конфига —
        # аргументы --lora-* здесь игнорируются намеренно: изменить их на лету
        # нельзя, а молча создать другой адаптер значит потерять всё обучение.
        from peft import PeftModel

        from qc26.inference.vlm import find_adapter_dir

        src_dir = find_adapter_dir(Path(resume_from))
        if src_dir is None:
            raise FileNotFoundError(f"адаптер для продолжения не найден: {resume_from}")
        for name, param in model.named_parameters():
            if "visual" in name:
                param.requires_grad = False
        model = PeftModel.from_pretrained(model, str(src_dir), is_trainable=True)
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if n_train == 0:
            raise RuntimeError("после загрузки адаптера нет обучаемых параметров — "
                               "is_trainable не сработал, обучение было бы пустым")
        import json as _json

        acfg = _json.loads((src_dir / "adapter_config.json").read_text(encoding="utf-8"))
        print(f"ПРОДОЛЖЕНИЕ обучения адаптера {src_dir}", flush=True)
        print(f"  его конфигурация: ранг {acfg.get('r')}, alpha {acfg.get('lora_alpha')}, "
              f"слои {acfg.get('target_modules')}", flush=True)
        print(f"  обучаемых параметров: {n_train:,}", flush=True)
        if _grad_checkpoint:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
            model.enable_input_require_grads()
    elif train_mode:
        from peft import LoraConfig, get_peft_model

        # Заморозка зрительной башни — стандартным способом, без хаков
        for name, param in model.named_parameters():
            if "visual" in name:
                param.requires_grad = False

        # ⚠ prepare_model_for_kbit_training здесь НЕ вызывается. Она предназначена
        # для квантованных моделей и поднимает часть слоёв в fp32; на bf16-пути это
        # только замедляет и ест память, ничего не давая. Всё, что от неё было нужно,
        # вызывается ниже явно: gradient_checkpointing_enable и
        # enable_input_require_grads. Базовые веса замораживает сам get_peft_model.
        # ⚠ Ранг, слои и dropout вынесены в аргументы: до сих пор это были
        # константы, то есть НЕ выбор, а то, что написали изначально. Ёмкость
        # адаптера ни разу не проверялась замером.
        targets = [t.strip() for t in _lora_targets.split(",") if t.strip()]
        model = get_peft_model(model, LoraConfig(
            r=_lora_r, lora_alpha=_lora_alpha, lora_dropout=_lora_dropout,
            bias="none", task_type="CAUSAL_LM", target_modules=targets))
        print(f"  LoRA: ранг {_lora_r}, alpha {_lora_alpha}, dropout {_lora_dropout}, "
              f"слои {targets}", flush=True)
        
        model.print_trainable_parameters()
        if _grad_checkpoint:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
            model.enable_input_require_grads()
    
    return proc, model


def encode_batch(proc, rows, cfg, root, step, max_desc, answers=None, device="mps",
                 n_images: int = 1, rules_mode: str = "on", modality: str = "both",
                 prompt_mode: str = "ours", facts_mode: str = "off"):
    """⚠ Кадров берём n_images, а не только главный. Замер на OOF показал, что
    решающий признак редкой категории — КОМПЛЕКТАЦИЯ («горелка + баллон»,
    «мангал в комплекте с углём»), а она видна на 2-м–5-м кадре, тогда как узел
    «горючее в комплекте» по тексту даёт AUC всего 0.68."""
    prompts, images = [], []

    # ⚠ Кадры читаем ПОТОКАМИ, а не по одному. При пяти кадрах на товар и батче 8
    # это сорок картинок подряд в одном процессе: декодирование JPEG занимает
    # десятки миллисекунд каждое, и карта всё это время простаивает. Протокол
    # калибровки прямо про это: загрузка карты ниже 85% значит, что упираемся
    # в чтение, а не в вычисления. Декодирование отпускает GIL, поэтому потоки
    # здесь работают, отдельные процессы не нужны.
    per_row = []
    for _, row in rows.iterrows():
        if modality == "text":
            paths = []                  # ⚠ не открываем файлы вовсе, иначе замер
        else:                           #   «только текст» платил бы за чтение кадров
            paths = image_paths(root, str(row[cfg["data"]["id_col"]]))[:max(0, n_images)]
        per_row.append((row, paths))

    flat = [(i, pth) for i, (_, paths) in enumerate(per_row) for pth in paths]
    loaded: dict[int, list] = {i: [] for i in range(len(per_row))}
    if flat and _IMG_WORKERS > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(_IMG_WORKERS, len(flat))) as pool:
            for (i, _), img in zip(flat, pool.map(
                    lambda t: _safe_load(t[1], step), flat)):
                if img is not None:
                    loaded[i].append(img)
    else:
        for i, pth in flat:
            img = _safe_load(pth, step)
            if img is not None:
                loaded[i].append(img)

    for i, (row, _paths) in enumerate(per_row):
        imgs = loaded[i]
        prompts.append(build_prompt(proc, row, cfg, imgs, max_desc, rules_mode,
                                    modality, prompt_mode, facts_mode))
        # ⚠ Один раз за прогон, на первом промпте: открытый <think> в конце делает
        # оценку «Да/Нет» случайной, и без этой проверки узнать об этом нечем.
        global _thinking_checked
        if not _thinking_checked:
            _thinking_checked = True
            assert_no_open_thinking(prompts[-1], "обучения")
            if answers is not None:
                check_answer_mask(proc.tokenizer, prompts[-1], answers[len(prompts) - 1])
        if imgs:
            images.append(imgs)
    if answers is not None:
        prompts = [p + a + proc.tokenizer.eos_token for p, a in zip(prompts, answers)]
    # ⚠ padding и return_tensors передаём именно так, обычными ключами.
    # transformers 5.14 на это ругается: «Kwargs passed to processor.__call__ have
    # to be in processor_kwargs dict». Совет ПРОВЕРЕН и оказался вредным: с
    # processor_kwargs={...} процессор возвращает СПИСКИ вместо выровненного
    # тензора, то есть последовать предупреждению значило бы сломать батч.
    # Нынешняя форма работает: (2, 24) на двух строках разной длины.
    enc = proc(text=prompts, images=images or None, padding=True, return_tensors="pt")
    check_batch_padded(enc, len(prompts))
    for im in images:
        for x in im:
            x.close()
    return enc.to(device)


def main() -> None:
    # ⚠ Объявление здесь, а не у присваивания: MODEL используется уже в значении
    # по умолчанию для --base-model, а Python требует global ДО первого упоминания.
    global MODEL, MAX_PIXELS
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="train", choices=["train", "eval"])
    ap.add_argument("--split", default="fold_seed42")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--train-categories", default="both", choices=["both", "rare", "bad"],
                    help="⚠ both — ключевое отличие: на редкой категории всего 158 "
                         "обучающих позитивов, а на всей выборке около 4600. Три наши "
                         "прежние попытки учили редкую изолированно и провалились "
                         "(PR-AUC 0.07-0.15); во второй половине обучение на всём объёме "
                         "дало публично 0.7809.")
    ap.add_argument("--eval-category", default="Легковоспламеняющиеся")
    # ⚠ Защита от меморизации. Проект уже дважды на ней обжёгся: наша попытка на
    # Windows дала лосс 0.06 и запоминание, DPO v2 — AUC 0.5000 при лоссе 0.0002.
    # Оба раза это выяснялось ПОСЛЕ обучения, потому что проверки во время не было.
    ap.add_argument("--eval-every", type=int, default=50,
                    help="каждые N шагов считать PR-AUC на внутренней проверке")
    ap.add_argument("--patience", type=int, default=4,
                    help="сколько замеров без улучшения до остановки")
    ap.add_argument("--val-size", type=int, default=200,
                    help="устарел: размер задаётся --val-pos / --val-neg")
    ap.add_argument("--val-pos", type=int, default=40,
                    help="позитивов НА КАТЕГОРИЮ во внутренней проверке")
    ap.add_argument("--val-neg", type=int, default=100,
                    help="негативов НА КАТЕГОРИЮ. Доля позитивов в проверке при этом "
                         "НЕ равна доле в выборке: ячейки взвешиваются обратно частоте, "
                         "и F1 считается при естественной доле. Точность оценки тратится "
                         "там, где её мало, — на редких позитивах.")
    ap.add_argument("--n-images", type=int, default=1,
                    help="кадров на товар. 1 — как было (только главный кадр). "
                         "Замер на OOF: промахи редкой категории — это комплектация "
                         "(«горелка + баллон», «в комплекте с углём»), которой на "
                         "главном кадре нет. Эксперимент 1 / 2 / 3 при прочих равных.")
    ap.add_argument("--prompt", default="ours", choices=["ours", "packaging", "qc"],
                    help="шаблон вопроса. packaging — формулировка второй половины, на "
                         "той же базовой 2B дала публично 0.7809. qc — наша: называет "
                         "критерий прямо (чем товар является сам, а не связан ли он "
                         "с темой) и имеет гнездо для блока наблюдений")
    ap.add_argument("--rules", default="on",
                    choices=["on", "off", "short", "official"],
                    help="правила категории в промпте. ⚠ Аудит показал 12 конфликтов "
                         "метка-регламент из 30 ложных в рабочей точке: следование "
                         "регламенту может УХУДШАТЬ лидерборд. Замер обязателен.")
    ap.add_argument("--modality", default="both", choices=["both", "text", "image"],
                    help="both — текст и кадры; text — без кадров; image — без "
                         "названия и описания. Разбор каналов на OOF: у редкой "
                         "категории PR-AUC текста 0.51 против 0.62 у зрения.")
    ap.add_argument("--train-probe", action="store_true",
                    help="дополнительно мерить те же метрики на ОБУЧАЮЩЕМ фолде. "
                         "Без этого есть только лосс, а он падает и при запоминании, "
                         "и при обобщении. Расхождение AUC обучение-проверка и есть "
                         "прямая мера меморизации. Замер дорожает вдвое.")
    ap.add_argument("--resume-from", default=None,
                    help="продолжить обучение существующего адаптера вместо создания "
                         "нового. Ранг, alpha и слои берутся из его конфига, "
                         "аргументы --lora-* игнорируются: сменить их на лету нельзя.")
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=0,
                    help="0 = взять вдвое больше ранга, как было")
    ap.add_argument("--lora-dropout", type=float, default=0.05,
                    help="выше 0.05 регуляризует против шума разметки, а он у нас "
                         "измерен: 12 из 30 ложных в рабочей точке — товары, которые "
                         "регламент относит к категории, а разметка нет")
    ap.add_argument("--lora-targets",
                    default="q_proj,k_proj,v_proj,o_proj",
                    help="слои для адаптера. Добавление gate_proj,up_proj,down_proj "
                         "подключает блоки прямого распространения — заметно больше "
                         "ёмкости и заметно медленнее")
    ap.add_argument("--select-by", default="recall",
                    choices=["recall", "auc", "f1", "pr_auc"],
                    help="по какой величине выбирать чекпоинт. По умолчанию полнота "
                         "в рабочей точке — она равна метрике соревнования там, где "
                         "модель принимает решения, и не зависит от подбора порога. "
                         "AUC и PR-AUC при менее чем 50 позитивах определяются "
                         "хвостом ранжирования и предсказывают с обратным знаком.")
    ap.add_argument("--no-val", action="store_true",
                    help="обучаться на 100% данных, отложенной выборки нет. Тогда "
                         "единственный источник метрики — лидерборд, а чекпоинт "
                         "берётся последний. Подразумевает --use-all-folds.")
    ap.add_argument("--use-all-folds", action="store_true",
                    help="учиться на всех фолдах кроме проверочного. Для отправки "
                         "свой экзамен не нужен — закрытая выборка внешняя, — а он "
                         "съедает пятую часть данных. Даёт 157 редких позитивов "
                         "вместо 118. ⚠ После этого офлайн-сравнение с прежними "
                         "прогонами невалидно: экзаменационного фолда больше нет.")
    ap.add_argument("--sampler-alpha", type=float, default=0.0,
                    help="сила выравнивания ячеек категория x класс. 0 — поровну "
                         "(2.00 редких позитива на шаг), 1 — как в данных (0.12), "
                         "между ними плавно. Нужен, чтобы разводить два фактора: "
                         "число шагов и частоту показа одних и тех же 118 редких "
                         "позитивов.")
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--loss-balance", default="cells",
                    choices=["mean", "cells", "focal"],
                    help="mean — как было, среднее по токенам батча; cells — равный "
                         "вклад четырёх ячеек категория x класс; focal — то же плюс "
                         "гашение лёгких примеров")
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--balanced-sampler", action="store_true", default=True,
                    help="набирать батч поровну из ячеек категория x класс. Без этого "
                         "редкий позитив попадается реже раза в восемь шагов: их 157 "
                         "из 10376, то есть 1.5 процента")
    ap.add_argument("--no-balanced-sampler", dest="balanced_sampler",
                    action="store_false")
    ap.add_argument("--desc-jitter", type=float, default=0.3,
                    help="доля случайной обрезки описания: мешает запоминать карточку "
                         "целиком, не трогая сам признак")
    ap.add_argument("--no-save-resume", dest="save_resume", action="store_false",
                    help="не писать точку продолжения. ⚠ На арендованной карте НЕ "
                         "отключать: обрыв на третьем часу стоит около 320 ₽, а "
                         "точка весит около 0.5 ГБ и пишется за секунды")
    ap.add_argument("--base-model", default=MODEL,
                    help="базовые веса. 2B — в каталоге жюри, архив не тратит; "
                         "4B даёт +7.9 п.п. по замеру на второй половине, но 4.15 ГБ "
                         "архива из пяти доступных")
    ap.add_argument("--facts", default="off", choices=["on", "off"],
                    help="блок эвристических наблюдений из qc26.facts в промпте. "
                         "⚠ требует шаблона с гнездом {facts} (--prompt qc) и "
                         "ОБЯЗАН совпадать на прогоне — читается из train_log.json")
    ap.add_argument("--no-grad-checkpoint", dest="grad_checkpoint",
                    action="store_false",
                    help="не пересчитывать активации. Экономит ~30%% времени, но "
                         "требует памяти. На A100 80 ГБ модель 2B помещается и без "
                         "пересчёта; на Mac НЕ включать")
    ap.add_argument("--no-keep-all", dest="keep_all", action="store_false",
                    help="не сохранять снимок каждого замера. ⚠ По умолчанию "
                         "сохраняются ВСЕ: «лучший» выбирается метрикой, которая "
                         "на 39 позитивах даёт ничьи, и какой чекпоинт окажется "
                         "лучшим на лидерборде — заранее неизвестно")
    ap.add_argument("--poweroff", type=int, default=0, metavar="МИН",
                    help="погасить машину через N минут ПОСЛЕ УСПЕШНОГО прогона. "
                         "0 — не гасить. ⚠ Проверяйте по балансу: у части "
                         "провайдеров биллинг считает состояние машины, а не "
                         "системы, и гашение изнутри его не останавливает")
    ap.add_argument("--img-workers", type=int, default=8,
                    help="потоков на чтение кадров. При пяти кадрах и батче 8 это "
                         "сорок JPEG на шаг: последовательно карта простаивает. "
                         "На 38 vCPU можно 12-16; 1 отключает потоки")
    ap.add_argument("--dry-run", action="store_true",
                    help="разобрать данные, показать разбиение и готовый промпт "
                         "и выйти ДО загрузки модели. Для подготовки на дешёвой "
                         "машине перед арендой видеокарты")
    ap.add_argument("--inner-fold", type=int, default=None,
                    help="фолд внутренней проверки. По умолчанию следующий за --fold. "
                         "⚠ фолд 0 непригоден: 66%% его позитивов редкой категории — "
                         "один кластер почти-дублей. Лучший для проверки — 4 (15%%)")
    ap.add_argument("--neg-per-pos", type=int, default=6)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=2)      # На 24 ГБ можно увеличить
    ap.add_argument("--accum", type=int, default=4)       # Эффективный батч = 8
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-pixels", type=int, default=MAX_PIXELS,
                    help="пикселей на кадр после ужатия. 100352 — наш прежний, "
                         "200704 — вдвое. ⚠ Больше x2 не брать: по замеру "
                         "второй половины вердикт при x4 хуже, чем при x1")
    ap.add_argument("--max-desc", type=int, default=800)  # Можно вернуть полную длину
    ap.add_argument("--eval-batch", type=int, default=8)
    ap.add_argument("--eval-neg", type=int, default=0, help="0 = весь фолд целиком")
    ap.add_argument("--out", default="artifacts/sft_vlm")
    args = ap.parse_args()
    if args.no_val:
        args.use_all_folds = True

    torch.manual_seed(42)
    np.random.seed(42)
    
    quiet_known_noise()
    device = get_device()
    describe_device(device, args)
    
    cfg = load_config("configs/data.yaml")
    idc = cfg["data"]["id_col"]
    cards = load_cards(cfg)
    cards[idc] = cards[idc].astype(str)
    groups = pd.read_parquet(resolve_path("artifacts/splits/groups.parquet"))
    groups["id"] = groups["id"].astype(str)
    fold_cols = [c for c in groups.columns if c.startswith("fold_")]
    # ⚠ «group» тянем обязательно: без него не отличить «нашли новый тип товара» от
    # «подняли один кластер почти-дубликатов». В редкой категории 198 позитивов
    # лежат всего в 91 группе, а один кластер доходит до 27 строк.
    df = cards.merge(groups[["id", "group", *fold_cols]], left_on=idc, right_on="id")
    catc = cfg["data"]["category_col"]
    keep = {"both": None, "rare": "Легковоспламеняющиеся", "bad": "БАД"}[args.train_categories]
    train_pool = df if keep is None else df[df[catc] == keep]
    # ⚠ Полная выборка сохраняется ОТДЕЛЬНО. Проверка обязана видеть ОБЕ категории,
    # даже когда обучаем на одной: вторая ступень на редкой может испортить БАД
    # («забывание»), и заметить это можно только замером на ней. Раньше проверка
    # нарезалась из train_pool и при --train-categories rare теряла БАД целиком.
    full_pool = df
    # ⚠ оценка ВСЕГДА на одной категории: метрика считается по категориям отдельно,
    # и смешивать их в одном числе значит потерять смысл
    eval_pool = df[df[catc] == args.eval_category].reset_index(drop=True)
    train, _ = select(train_pool.reset_index(drop=True), args.split, args.fold,
                      args.neg_per_pos)
    _, test = select(eval_pool, args.split, args.fold, args.neg_per_pos)
    df = eval_pool
    root = images_root(cfg)
    dst = resolve_path(args.out) / f"{args.split}_f{args.fold}"
    dst.mkdir(parents=True, exist_ok=True)
    print(f"обучение: {len(train)} товаров, позитивов {int(train.label.sum())} "
          f"(категории: {args.train_categories})", flush=True)
    print(f"оценка: {args.eval_category}, {len(test)} товаров, позитивов "
          f"{int(test.label.sum())}", flush=True)

    global _lora_r, _lora_alpha, _lora_dropout, _lora_targets
    _lora_r = args.lora_rank
    _lora_alpha = args.lora_alpha or args.lora_rank * 2
    _lora_dropout = args.lora_dropout
    _lora_targets = args.lora_targets
    global _grad_checkpoint
    _grad_checkpoint = args.grad_checkpoint
    global _IMG_WORKERS
    _IMG_WORKERS = max(1, int(args.img_workers))
    MODEL = args.base_model
    MAX_PIXELS = int(args.max_pixels)
    if MAX_PIXELS != 100352:
        print(f"⚠ разрешение кадра {MAX_PIXELS} вместо прежних 100352 — прогон "
              f"обязан взять то же число, оно пишется в train_log.json", flush=True)
    if MODEL in KNOWN_BASES:
        n_par, gb, shared = KNOWN_BASES[MODEL]
        print(f"базовая модель: {MODEL} — {n_par / 1e9:.1f} млрд, {gb} ГБ в bf16, "
              + ("есть в каталоге жюри (архив не тратит)" if shared
                 else "⚠ НЕТ в каталоге жюри: поедет в архиве решения"), flush=True)
    else:
        print(f"⚠ базовая модель {MODEL} не из известных — проверьте, что она есть "
              f"в каталоге жюри, иначе решение не соберётся", flush=True)
    if args.dry_run:
        # ⚠ Модель не грузим, но разбиение ниже строится тем же кодом, что и в бою:
        # оно чистая работа с таблицами и ни процессора, ни весов не требует.
        # Пересобрать его «как в обучении» отдельной веткой значит однажды разойтись.
        proc = model = tok = None
        step, yes_ids, no_ids = 32, [], []
        print("сухой прогон: модель пропущена, разбиение строится настоящее",
              flush=True)
    else:
        proc, model = load_model(device, adapter=dst if args.stage == "eval" else None,
                                 resume_from=args.resume_from,
                                 train_mode=args.stage == "train")
        tok = proc.tokenizer
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        step = grid_step(proc)
        yes_ids, no_ids = first_token_ids(tok, YES), first_token_ids(tok, NO)
        print(f"шаг сетки {step} | первые токены «Да» {yes_ids}, «Нет» {no_ids}",
              flush=True)
        assert not (set(yes_ids) & set(no_ids)), \
            "классы делят первый токен — счёт невозможен"

    if args.stage == "train":
        # ⚠ Внутренняя проверка берётся из ОБУЧАЮЩИХ фолдов, а не из оцениваемого:
        # иначе ранняя остановка подглядывает в собственный экзамен и все числа
        # оказываются завышены.
        # ⚠ Фолд проверки ЗАДАЁТСЯ, а не выводится из экзаменационного. Замерено:
        # позитивы редкой категории распределены по фолдам крайне неровно, кластер
        # из 27 почти-дублей падает в фолд 0 при всех трёх сидах и даёт 66% его
        # позитивов. Проверять там нельзя — метрика меряет одну товарную семью.
        # Пригодность фолдов (крупнейший кластер в долях позитивов):
        #     фолд 0 — 66%, фолд 1 — 31%, фолд 2 — 23%, фолд 3 — 25%, фолд 4 — 15%.
        inner = args.inner_fold if args.inner_fold is not None else (args.fold + 1) % 5
        if inner == args.fold:
            raise SystemExit(f"проверочный фолд {inner} совпал с экзаменационным")
        # ⚠ Проверка вырезается из СЫРОГО фолда, ДО андерсэмплинга, и по ОБЕИМ
        # категориям: метрика соревнования — среднее F1 двух категорий, и ни доля
        # позитивов, ни вторая категория не должны теряться по дороге.
        # ⚠ Экзаменационный фолд нужен, только пока мы сравниваем модели офлайн.
        # Для ОТПРАВКИ закрытая выборка организаторов и так внешняя, а свой экзамен
        # просто съедает данные: замерено, что AUC на обучающем фолде выходит на
        # 1.000 к шагу 900 — 118 редких позитивов исчерпаны, и единственный
        # оставшийся рычаг это их количество. Фолды 1-4 дают 157 вместо 118, +33%.
        if args.use_all_folds:
            # ⚠ Проверочный фолд НЕ меняем: он остаётся тем же (inner), что и в
            # прошлых прогонах, а в обучение уходит бывший экзаменационный. Иначе
            # эксперимент «что дал объём данных» смешался бы со сменой выборки,
            # на которой мы меряем, — и особенно скверно, что фолд 0 патологический
            # (27 из 41 позитива в нём это один кластер почти-дубликатов).
            # НОЧНОЙ ПРОГОН: гиперпараметры фиксируем — это даёт чистый замер вклада
            # данных (+39 редких позитивов). После находим этот вклад (+33%), потом
            # на зафиксированной выборке варьируем гиперпараметры (А/Б). Метрики
            # выводим только train-side для контроля меморизации.
            raw_pool = train_pool
            print(f"⚠ обучение на ВСЕХ фолдах кроме проверочного {inner}: "
                  f"экзаменационного фолда больше нет, метрика идёт только "
                  f"с проверки", flush=True)
        else:
            raw_pool = train_pool[train_pool[args.split] != args.fold]
        # ⚠ --no-val: обучение на 100% данных, отложенной выборки нет вовсе.
        # Тогда единственный источник метрики — лидерборд, а чекпоинт берётся
        # последний (выбирать не по чему). Проверку всё равно нарезаем, но ИЗ
        # ОБУЧАЮЩИХ строк: она уже не оценка качества, а датчик того, что обучение
        # вообще идёт, и её числа НЕЛЬЗЯ трактовать как качество.
        # ⚠ Проверку нарезаем из ПОЛНОЙ выборки тех же фолдов, а не из обучающей:
        # иначе при обучении на одной категории вторая исчезает из замера.
        val_pool = full_pool[full_pool[args.split].isin(raw_pool[args.split].unique())]
        if args.no_val:
            val = carve_validation(val_pool, args.split, inner, catc,
                                   args.val_pos, args.val_neg)
            train_raw = raw_pool.reset_index(drop=True)
            print("⚠ --no-val: обучение на ВСЕХ данных, отложенной выборки НЕТ. "
                  "Числа «проверки» ниже считаются на ОБУЧАЮЩИХ строках и качеством "
                  "не являются. Чекпоинт сохраняется последний.", flush=True)
        else:
            val = carve_validation(val_pool, args.split, inner, catc,
                                   args.val_pos, args.val_neg)
            # обучение — из оставшихся фолдов, андерсэмплинг ТОЛЬКО к нему
            train_raw = raw_pool[raw_pool[args.split] != inner].reset_index(drop=True)
        pos = train_raw[train_raw.label == 1]
        n_neg = min(len(pos) * args.neg_per_pos, int((train_raw.label == 0).sum()))
        train = pd.concat([pos, train_raw[train_raw.label == 0].sample(
            n=n_neg, random_state=42)]).sample(frac=1.0, random_state=42)

        print(f"внутренняя проверка: фолд {inner}, {len(val)} товаров", flush=True)
        for cat, part in val.groupby(catc):
            nat = float((part.label * part.cell_weight).sum() / part.cell_weight.sum())
            print(f"    {cat}: {len(part)} строк, позитивов {int(part.label.sum())}, "
                  f"естественная доля {nat:.4f}", flush=True)
            if not 0.001 < nat < 0.95:
                raise SystemExit(f"доля позитивов {nat:.4f} в «{cat}» неправдоподобна")
        if val[catc].nunique() < 2:
            # ⚠ Ворота против однобокого отбора чекпоинта. Но они применимы ТОЛЬКО
            # когда отбор вообще идёт: при --no-val берётся последний чекпоинт, а при
            # обучении на одной категории вторая в проверке взяться не может по
            # построению. Останавливать такой прогон значит блокировать законный
            # замер — именно так был сорван зонд на градиент второй ступени.
            msg = ("в проверке одна категория — метрика соревнования усредняет ДВЕ")
            if args.no_val or args.train_categories != "both":
                print(f"⚠ {msg}. Отбор чекпоинта здесь не работает "
                      f"(no_val={args.no_val}, категории={args.train_categories}), "
                      f"поэтому продолжаю.", flush=True)
            else:
                raise SystemExit(f"{msg}, отбор чекпоинта будет однобоким")
        print(f"обучение сокращено до {len(train)} товаров", flush=True)
        train_probe = None
        if args.train_probe:
            # берём ОБУЧАЮЩИЙ фолд той же процедурой и того же размера — иначе
            # числа несопоставимы с проверкой и сравнивать их бессмысленно
            probe_fold = (args.fold + 2) % 5
            if probe_fold in (args.fold, inner):
                probe_fold = (args.fold + 3) % 5
            train_probe = carve_validation(raw_pool, args.split, probe_fold, catc,
                                           args.val_pos, args.val_neg, seed=7)
            print(f"зонд обучения: фолд {probe_fold} (входит в обучение), "
                  f"{len(train_probe)} строк — замер станет вдвое дороже", flush=True)
        if args.dry_run:
            # ⚠ Стоим ровно здесь: разбиение, проверка и зонд уже построены
            # настоящим кодом, а весов ещё нет. Всё, что ниже, требует модели.
            print("\n=== СУХОЙ ПРОГОН: обучение не начинается ===", flush=True)
            for tag, part in (("обучение", train), ("проверка", val)):
                per = part.groupby([catc, "label"]).size()
                print(f"  {tag}: {len(part)} строк")
                for (c, l), n in per.items():
                    print(f"      {c} / метка {l}: {n}")
            eff = args.batch * args.accum
            print(f"  {args.max_steps} шагов x эффективный батч {eff} = "
                  f"{args.max_steps * eff} показов = "
                  f"{args.max_steps * eff / len(train):.2f} эпохи")
            # ⚠ Промпт печатаем ТЕМ ЖЕ кодом, что подаёт его в обучение.
            # Пересобрать его здесь отдельной веткой значит однажды разойтись.
            row = train[train[catc] == args.eval_category].iloc[0]
            print(f"\n  --- промпт «{args.prompt}», правила «{args.rules}», "
                  f"кадров {args.n_images}, метка {int(row.label)} ---")
            try:
                from transformers import AutoProcessor

                pr = AutoProcessor.from_pretrained(MODEL)
                # ⚠ facts передаём ОБЯЗАТЕЛЬНО. Без него сухой прогон печатал
                # промпт без блока наблюдений даже при --facts on, то есть врал
                # ровно о том, ради чего его и смотрят.
                shown = build_prompt(pr, row, cfg, [None] * args.n_images,
                                     args.max_desc, args.rules, args.modality,
                                     args.prompt, args.facts)
                assert_no_open_thinking(shown, "сухого прогона")
                print(shown)
            except Exception as e:
                # на машине подготовки может не быть ни сети, ни кэша весов —
                # это не повод ронять сухой прогон, чат-шаблон здесь не главное
                print(f"  (процессор недоступен: {type(e).__name__} — "
                      f"{str(e)[:80]}; чат-шаблон пропущен)")
            raise SystemExit("\nсухой прогон завершён, ошибок нет")

        run_train(args, proc, model, tok, train, val, cfg, root, step, dst, device,
                  yes_ids, no_ids, train_probe=train_probe)
    else:
        run_eval(args, proc, model, tok, test, cfg, root, step, dst, yes_ids, no_ids, device)


def save_checked(model, dst) -> None:
    bad = [n for n, p in model.named_parameters()
           if p.requires_grad and not torch.isfinite(p).all()]
    if bad:
        raise RuntimeError(f"в адаптере нечисловые веса ({len(bad)} тензоров)")
    model.save_pretrained(str(dst))


def train_meta(args, device, step_i: int, extra: dict | None = None) -> str:
    """Журнал обучения. Кладётся РЯДОМ С КАЖДЫМ чекпоинтом, а не только с финальным.

    ⚠ Без него прогон не узнает ни базовую модель, ни число кадров, ни шаблон
    промпта, ни режим наблюдений — и разойдётся с обучением молча. Раз чекпоинтов
    теперь много, journal нужен у каждого: иначе выбранный из середины окажется
    непригоден к отправке.
    """
    d = {"args": vars(args), "device": str(device),
         "n_images": int(args.n_images), "rules": args.rules,
         "prompt": args.prompt, "facts": args.facts,
         "base_model": MODEL,
         "max_pixels": int(MAX_PIXELS), "modality": args.modality,
         "steps_done": int(step_i)}
    d.update(extra or {})
    return json.dumps(d, ensure_ascii=False)


def save_step_checkpoint(dst, model, args, device, step_i, f1, per_cat, thr, rec,
                         sel) -> None:
    """Снимок каждого замера — отдельной папкой, чтобы ничего не терять.

    ⚠ Зачем, если лучший и последний уже сохраняются. На прогоне pack1 шаги 600,
    1400 и 1600 дали по метрике отбора ровно одно и то же (полнота на 39 позитивах
    квантуется шагом 1/39), сохранился ранний — а на 1600 метрика соревнования
    была выше. Вернуть было нечего: пересчёт стоил бы суток.

    Адаптер весит 126 МБ, замеров за прогон около двенадцати — три гигабайта на
    оба прогона при диске в 120. Цена несопоставима с потерянной ночью.
    """
    snap = Path(dst) / f"step{step_i:05d}"
    save_checked(model, snap)
    (snap / "thresholds.json").write_text(
        json.dumps({"thresholds": thr, "per_category": per_cat, "objective": f1,
                    "step": step_i, "recall": rec, "select_by": args.select_by,
                    "select_score": sel,
                    "tuned_on": "внутренняя проверка (сырой фолд)"},
                   ensure_ascii=False), encoding="utf-8")
    (snap / "train_log.json").write_text(
        train_meta(args, device, step_i, {"objective": float(f1)}), encoding="utf-8")


def check_disk_for_checkpoints(dst, args, n_trainable: int) -> None:
    """Хватит ли места на все снимки. Считаем ДО обучения, а не по ходу.

    Кончившееся место на седьмом часу — это упавший прогон и оплаченная ночь
    впустую, причём упадёт он на записи, то есть уже после полезной работы.
    """
    import shutil

    if not args.keep_all:
        return
    n_snaps = args.max_steps // max(1, args.eval_every) + 3
    need = n_snaps * n_trainable * 2 / 2 ** 30          # bf16
    need += n_trainable * 8 / 2 ** 30                    # точка продолжения
    free = shutil.disk_usage(Path(dst).parent).free / 2 ** 30
    print(f"  снимков будет ~{n_snaps}, нужно ~{need:.1f} ГБ, свободно {free:.1f} ГБ",
          flush=True)
    if free < need * 1.5:
        raise SystemExit(
            f"мало места: нужно ~{need:.1f} ГБ на снимки, свободно {free:.1f} ГБ.\n"
            f"  Нужно освободить диск: сохраняем ВСЕ чекпоинты (лучший по recall может "
            f"быть не последний), рисковать нельзя.")


def save_resume(dst, model, opt, sched, step_i, best_sel, best_f1) -> None:
    """Точка продолжения: веса адаптера И состояние оптимизатора со шкалой шага.

    ⚠ Без состояния Adam продолжение выходит холодным: моменты обнулены, скорость
    обучения начинает расписание заново. Это НЕ то же самое, что продолжить прогон,
    и на оплаченной карте разница видна деньгами — по плану аренды падение на
    третьем часу стоит около 320 ₽, и лечится оно только настоящим продолжением.

    Пишем через временный файл и переименование: обрыв посреди записи оставил бы
    битую точку продолжения вместо рабочей, а узнали бы мы об этом уже при падении.
    """
    tmp = Path(dst) / "resume.pt.tmp"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"opt": opt.state_dict(), "sched": sched.state_dict(),
                "step": int(step_i), "best_sel": float(best_sel),
                "best_f1": float(best_f1)}, tmp)
    tmp.replace(Path(dst) / "resume.pt")
    save_checked(model, Path(dst) / "resume_adapter")


def load_resume(dst, opt, sched):
    """Возвращает (шаг, лучший отбор, лучшая метрика) или None, если точки нет."""
    p = Path(dst) / "resume.pt"
    if not p.exists():
        return None
    st = torch.load(p, map_location="cpu", weights_only=False)
    opt.load_state_dict(st["opt"])
    sched.load_state_dict(st["sched"])
    print(f"продолжение с шага {st['step']}: состояние оптимизатора и расписание "
          f"восстановлены, лучший отбор {st['best_sel']:.4f}", flush=True)
    return int(st["step"]), float(st["best_sel"]), float(st["best_f1"])


def carve_validation(pool, split, inner, cat_col, n_pos, n_neg, seed=42):
    """Внутренняя проверка из СЫРОГО фолда, с весами ячеек.

    ⚠ Раньше проверка вырезалась из УЖЕ андерсэмплированного обучения: доля
    позитивов в ней выходила 0.195 против 0.034 при развёртывании. Порог,
    подобранный при такой доле, на боевой выборке смещён, а число несопоставимо
    с метрикой соревнования.

    Честная доля целиком означала бы 5500 товаров на замер — на MPS это часы.
    Поэтому из каждой ячейки (категория, метка) берём подвыборку и запоминаем
    ВЕС: сколько строк сырого фолда она представляет. Взвешенные TP/FP/FN дают
    несмещённую оценку F1 при ЕСТЕСТВЕННОЙ доле, а точность оценки тратится там,
    где её не хватает, — на редких позитивах.
    """
    rng = np.random.default_rng(seed)
    parts = []
    raw = pool[pool[split] == inner]
    for cat in sorted(raw[cat_col].unique()):
        sub = raw[raw[cat_col] == cat]
        for lab, want in ((1, n_pos), (0, n_neg)):
            cell = sub[sub.label == lab]
            if len(cell) == 0:
                continue
            take = min(want, len(cell))
            idx = rng.choice(cell.index.to_numpy(), size=take, replace=False)
            part = cell.loc[idx].copy()
            part["cell_weight"] = len(cell) / take   # одна строка представляет столько
            parts.append(part)
    return pd.concat(parts).reset_index(drop=True)


def weighted_f1(y, score, weight, thr):
    """F1 по классу 1 при естественной доле: веса восстанавливают состав фолда."""
    pred = score >= thr
    tp = float(weight[(y == 1) & pred].sum())
    fp = float(weight[(y == 0) & pred].sum())
    fn = float(weight[(y == 1) & ~pred].sum())
    return 0.0 if tp <= 0 else 2 * tp / (2 * tp + fp + fn)


# доли отбора на закрытой выборке: БАД 528/921, редкая 19/707. Полнота на этих
# долях сравнима с фактом «14 верных из 24» и потому информативнее F1 при 39 позитивах.
OPERATING_RATES = {"БАД": [0.5733, 0.62], "Легковоспламеняющиеся": [0.0269, 0.034, 0.042, 0.057]}


def recall_at_rate(y, score, weight, rate):
    """Полнота, если отобрать долю rate от ВЗВЕШЕННОЙ популяции.

    Порог ищется по взвешенному числу отобранных, а не по числу строк: строки
    представляют разное количество товаров, и «топ-K строк» тут не имеет смысла.
    """
    order = np.argsort(-np.asarray(score))
    w = np.asarray(weight)[order]
    yy = np.asarray(y)[order]
    budget = rate * w.sum()
    taken = np.cumsum(w) <= budget
    if not taken.any():
        taken[0] = True
    pos_total = w[yy == 1].sum()
    return float(w[taken & (yy == 1)].sum() / pos_total) if pos_total > 0 else float("nan")


def group_recall_at_rate(y, score, weight, groups, rate):
    """Доля ПОЗИТИВНЫХ ГРУПП, у которых хоть одна строка попала в отбор.

    ⚠ Построчная полнота обманывает: в редкой категории 198 позитивов лежат в 91
    группе почти-дубликатов, и один кластер доходит до 27 строк. Модель, поднявшая
    один такой кластер, выглядит как модель, научившаяся новому типу товара.
    Групповая полнота этой разницы не скрывает.
    """
    order = np.argsort(-np.asarray(score))
    w = np.asarray(weight)[order]
    yy = np.asarray(y)[order]
    gg = np.asarray(groups)[order]
    taken = np.cumsum(w) <= rate * w.sum()
    if not taken.any():
        taken[0] = True
    pos_groups = set(gg[yy == 1].tolist())
    if not pos_groups:
        return float("nan")
    hit = set(gg[taken & (yy == 1)].tolist())
    return float(len(hit) / len(pos_groups))


def cluster_stats(y, score, groups, rates):
    """Крупнейший дубль-кластер позитивов: сколько его строк попадает в отбор.

    ⚠ Кластер 445 (27 хлопушек) целиком лежит в ФОЛДЕ 0, то есть в экзамене, а не в
    проверке — метрику именно по нему во время обучения не посчитать. Поэтому берём
    крупнейший кластер ТОЙ выборки, на которой считаем: вопрос «модель тащит один
    кластер или разные товары» от номера кластера не зависит.
    """
    y, score, groups = np.asarray(y), np.asarray(score), np.asarray(groups)
    pos = y == 1
    if not pos.any():
        return {}
    vals, cnt = np.unique(groups[pos], return_counts=True)
    big = vals[int(np.argmax(cnt))]
    n_big = int(cnt.max())
    order = np.argsort(-score)
    rank = np.empty(len(score), dtype=int)
    rank[order] = np.arange(1, len(score) + 1)
    m = pos & (groups == big)
    out = {"cluster::размер": float(n_big),
           "cluster::ранг_медиана": float(np.median(rank[m]))}
    for r in rates:
        k = max(1, int(round(r * len(score))))
        out[f"cluster@{r}::попало"] = float((rank[m] <= k).sum())
    return out


def competition_objective(val, scores, cat_col, grid=None):
    """Ровно метрика соревнования: порог СВОЙ у категории, F1 усредняется по ним.

    ⚠ Раньше отбор чекпоинта шёл по PR-AUC одной категории. PR-AUC усредняет по
    всем порогам, а решает один; и половина метрики (вторая категория) в число
    вовсе не входила. Здесь считается то, что показывает лидерборд.
    """
    if grid is None:
        grid = np.unique(np.round(np.linspace(0.01, 0.99, 197), 4))
    y = val["label"].to_numpy()
    w = val["cell_weight"].to_numpy()
    cats = val[cat_col].to_numpy()
    per_cat, thr, rec = {}, {}, {}
    for cat in sorted(set(cats)):
        m = cats == cat
        if len(np.unique(y[m])) < 2:
            continue
        best = max(((weighted_f1(y[m], scores[m], w[m], t), t) for t in grid),
                   key=lambda x: x[0])
        per_cat[str(cat)], thr[str(cat)] = float(best[0]), float(best[1])
        rates = OPERATING_RATES.get(str(cat), [])
        for r in rates:
            rec[f"recall@{r}::{cat}"] = recall_at_rate(y[m], scores[m], w[m], r)
        if "group" in val.columns:
            gr = val["group"].to_numpy()
            for r in rates:
                rec[f"grouprecall@{r}::{cat}"] = group_recall_at_rate(
                    y[m], scores[m], w[m], gr[m], r)
            if str(cat) == "Легковоспламеняющиеся":
                for k, v in cluster_stats(y[m], scores[m], gr[m], rates).items():
                    rec[f"{k}::{cat}"] = v
        # ⚠ Качество РАНЖИРОВАНИЯ отдельно от порога. При 39 позитивах F1 скачет от
        # одного товара, а AUC устойчивее и не зависит от доли классов вовсе —
        # поэтому считается без весов. Для PR-AUC доля важна, там веса нужны.
        # Именно по этой паре видно меморизацию: обучающий лосс падает, ранжирование
        # на отложенных данных стоит на месте (замерено: адаптер 400 шагов дал на
        # настоящем фолде AUC 0.5525 при внутренней PR-AUC 0.7477).
        from sklearn.metrics import average_precision_score, roc_auc_score

        rec[f"auc::{cat}"] = float(roc_auc_score(y[m], scores[m]))
        rec[f"pr_auc::{cat}"] = float(
            average_precision_score(y[m], scores[m], sample_weight=w[m]))
    mean = float(np.mean(list(per_cat.values()))) if per_cat else 0.0
    return mean, per_cat, thr, rec


@torch.no_grad()          # ⚠ ОБЯЗАТЕЛЕН: enable_input_require_grads() ставит хук на
                          # слой эмбеддингов, и логиты требуют градиент даже в eval().
                          # Без этого .numpy() падает, а активации всей проверки
                          # копятся в памяти. Проверяется тестом по разбору AST.
def score_rows(proc, model, tok, rows, cfg, root, step, max_desc, device, yes_ids, no_ids,
               batch=8, n_images: int = 1, rules_mode: str = "on",
               modality: str = "both", prompt_mode: str = "ours",
               facts_mode: str = "off"):
    """Оценка по логиту первого токена. Генерации нет — это в десятки раз дешевле."""
    side = tok.padding_side
    tok.padding_side = "left"          # логит берём с последней позиции
    was_training = model.training
    model.eval()
    out = []
    for i in range(0, len(rows), batch):
        enc = encode_batch(proc, rows.iloc[i:i + batch], cfg, root, step, max_desc,
                           device=device, n_images=n_images,
                           rules_mode=rules_mode, modality=modality,
                           prompt_mode=prompt_mode, facts_mode=facts_mode)
        lg = model(**enc).logits[:, -1, :].float()
        if not torch.isfinite(lg).all():
            out.extend([0.5] * len(rows.iloc[i:i + batch]))
            continue
        p = torch.softmax(lg, dim=-1)
        s = p[:, yes_ids].sum(-1) / (p[:, yes_ids].sum(-1) + p[:, no_ids].sum(-1) + 1e-9)
        out.extend(s.cpu().numpy().tolist())
    tok.padding_side = side
    if was_training:
        model.train()
    return np.array(out)


def selection_score(f1_avg, rec, how: str) -> float:
    """Величина, по которой сравниваются чекпоинты.

    ⚠ По умолчанию — ПОЛНОТА В РАБОЧЕЙ ТОЧКЕ, а не AUC и не PR-AUC.

    Замерено вторая половина на одном холдауте (37 позитивов):
        2B, 3 кадра:  PR-AUC 0.7757  ->  публично 0.7809
        4B, 5 кадров: PR-AUC 0.6738  ->  публично 0.8602
    PR-AUC предсказала с ОБРАТНЫМ знаком, и бутстрап это не поймал — он воспроизвёл
    тот же шум и выдал «надёжность 95%» на пустом месте. Причина: при малом числе
    позитивов PR-AUC и AUC определяются хвостом ранжирования, а решения модели
    сосредоточены в верхушке.

    Полнота на замеренной доле от этого свободна. При фиксированной доле k
    точность = попаданий/k, полнота = попаданий/P, поэтому
        F1 = 2*попаданий/(k+P)
    — строго монотонна полноте. То есть это и есть метрика соревнования в рабочей
    точке, только без шума подбора порога.
    """
    if how == "f1":
        return float(f1_avg)
    if how == "recall":
        # доля отбора закрытой выборки: 528/921 у БАД, 19/707 у редкой
        vals = [v for k, v in rec.items()
                if k.startswith("recall@0.0269") or k.startswith("recall@0.5733")]
        return float(np.mean(vals)) if vals else float("nan")
    key = "auc" if how == "auc" else "pr_auc"
    vals = [v for k, v in rec.items() if k.split("::")[0] == key]
    return float(np.mean(vals)) if vals else float("nan")


def beats_best(sel, f1, best_sel, best_f1, eps=1e-6) -> tuple[bool, bool]:
    """Побил ли чекпоинт лучший. Возвращает (брать ли, ничья ли по отбору).

    ⚠ Ничья разрешается по метрике соревнования, а не «кто раньше».

    Замерено на прогоне pack1: шаги 600, 1400 и 1600 дали по отбору ровно 0.7083 —
    полнота считается на 39 позитивах и квантуется шагом 1/39 = 0.0256, поэтому
    точные совпадения неизбежны, а не редки. Побеждал ранний. Между тем на шаге
    1600 метрика была 0.8757 против 0.8674, F1 редкой категории 0.813 против 0.800,
    а лосс продолжал падать — то есть отбор не ошибся в сравнении, он ПЕРЕСТАЛ
    РАЗЛИЧАТЬ, и разрыв ничьи достался случайности.
    """
    if sel > best_sel + eps:
        return True, False
    tie = abs(sel - best_sel) <= eps
    return (tie and f1 > best_f1 + eps), tie


def run_train(args, proc, model, tok, train, val, cfg, root, step, dst, device,
              yes_ids, no_ids, train_probe=None) -> None:
    """Обучение с проверкой по ходу, лучшим чекпоинтом и ранней остановкой.

    ⚠ Без этого скрипт учился фиксированное число шагов и сохранял ПОСЛЕДНИЙ
    адаптер. Меморизация в этом проекте случалась дважды (лосс 0.06 и 0.0002),
    и оба раза её замечали уже после обучения.
    """
    tok.padding_side = "right"
    catc = cfg["data"]["category_col"]

    def measure(frame=None):
        f = val if frame is None else frame
        sc = score_rows(proc, model, tok, f, cfg, root, step, args.max_desc,
                        device, yes_ids, no_ids, n_images=args.n_images,
                        rules_mode=args.rules, modality=args.modality,
                        prompt_mode=args.prompt, facts_mode=args.facts)
        return competition_objective(f, sc, catc)

    # база до обучения: без неё непонятно, что вообще дало дообучение
    base_f1, base_cat, base_thr, base_rec = measure()
    print(f"  база до обучения: метрика {base_f1:.4f} "
          + " | ".join(f"{k} {v:.4f} (порог {base_thr[k]:.2f})"
                       for k, v in base_cat.items()), flush=True)
    print("  редкая, полнота на рабочих долях: "
          + " | ".join(f"{k.split('::')[0]} {v:.3f}" for k, v in base_rec.items()
                       if k.startswith("recall@") and "Легковоспламеняющиеся" in k),
          flush=True)
    print("  редкая, ранжирование: "
          + " | ".join(f"{k.split('::')[0]} {v:.4f}" for k, v in base_rec.items()
                       if k.split("::")[0] in ("auc", "pr_auc")
                       and "Легковоспламеняющиеся" in k), flush=True)

    # ⚠ Траектория пишется на КАЖДОМ замере, а не только для лучшего чекпоинта.
    # При 118 обучающих редких позитивах форма кривой (где пик, как быстро спад)
    # информативнее самого победителя: она говорит, не перегрет ли бюджет шагов.
    traj = dst / "trajectory.csv"
    # ⚠ Замер на ОБУЧАЮЩЕМ фолде той же процедурой и того же размера. Без него не
    # отличить «запомнила и всё равно обобщает» от «выучила правило»: у нас есть
    # только лосс, а он падает в обоих случаях. Стоит ровно вдвое дороже замера.
    train_cols = []
    if train_probe is not None:
        train_cols = ["train_f1_avg"] + [f"train_auc::{k}" for k in sorted(base_cat)]

    traj_cols = (["step", "loss", "lr", "f1_avg"]
                 + [f"f1::{k}" for k in sorted(base_cat)]
                 + [f"thr::{k}" for k in sorted(base_thr)]
                 + sorted(base_rec) + train_cols)

    def log_row(step_i, loss, f1, per_cat, thr, rec, lr=float("nan"), tr=None):
        new = not traj.exists()
        with traj.open("a", encoding="utf-8") as fh:
            if new:
                fh.write(",".join(traj_cols) + "\n")
            vals = {"step": step_i, "loss": f"{loss:.5f}", "lr": f"{lr:.3e}",
                    "f1_avg": f"{f1:.5f}"}
            vals.update({f"f1::{k}": f"{v:.5f}" for k, v in per_cat.items()})
            vals.update({f"thr::{k}": f"{v:.4f}" for k, v in thr.items()})
            vals.update({k: f"{v:.5f}" for k, v in rec.items()})
            if tr is not None:
                t_f1, t_cat, _, t_rec = tr
                vals["train_f1_avg"] = f"{t_f1:.5f}"
                for k in t_cat:
                    vals[f"train_auc::{k}"] = f"{t_rec.get(f'auc::{k}', float('nan')):.5f}"
            fh.write(",".join(str(vals.get(c, "")) for c in traj_cols) + "\n")

    log_row(0, float("nan"), base_f1, base_cat, base_thr, base_rec,
            tr=measure(train_probe) if train_probe is not None else None)

    model.train()
    rng = np.random.default_rng(42)
    base_sel = selection_score(base_f1, base_rec, args.select_by)
    check_disk_for_checkpoints(dst, args,
                               sum(p.numel() for p in model.parameters()
                                   if p.requires_grad))
    best_sel, best_f1, waited, saved_any = base_sel, base_f1, 0, False
    last_measure = None      # последний замер — для thresholds.json у «last»
    print(f"  отбор чекпоинта по «{args.select_by}», база {base_sel:.4f}", flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=args.weight_decay)
    from transformers import get_linear_schedule_with_warmup
    sched = get_linear_schedule_with_warmup(opt, int(args.max_steps * 0.06), args.max_steps)

    # ⚠ Пулы по ячейкам категория x класс. Обычная перестановка даёт 0.12 редкого
    # позитива на шаг при батче 2 и накоплении 4 — их просто нет в градиенте.
    catc = cfg["data"]["category_col"]
    cells = {}
    for cc in train[catc].astype(str).unique():
        for ll in (0, 1):
            idx = np.where((train[catc].astype(str).to_numpy() == cc)
                           & (train.label.to_numpy() == ll))[0]
            if len(idx):
                cells[(cc, ll)] = idx
    cell_keys = sorted(cells)
    print("  пулы для отбора: "
          + ", ".join(f"{c}/{l}: {len(cells[(c, l)])}" for c, l in cell_keys), flush=True)

    cursor = {"i": 0}
    # ⚠ Насколько сильно выравнивать ячейки. Вероятность ячейки ∝ размер^alpha:
    #   alpha = 0   — поровну, 2.00 редких позитива на шаг (было единственным режимом);
    #   alpha = 1   — как в данных, 0.12 на шаг (редкий позитив раз в восемь шагов);
    #   между ними — плавно.
    # Зачем промежуток. При 118 уникальных редких позитивах и 2.00 на шаг за 800 шагов
    # каждый показывается около 13.6 раза, за 1600 — около 27. Это режим «быстро выучи
    # редкие примеры, потом многократно их перепиши». Уменьшив давление и удлинив
    # обучение, можно держать то же число показов, но дать модели больше разных
    # негативов между ними.
    alpha = float(getattr(args, "sampler_alpha", 0.0))
    sizes = np.array([len(cells[k]) for k in cell_keys], dtype=float)
    cell_p = sizes ** alpha
    cell_p = cell_p / cell_p.sum()

    def draw(n):
        """n индексов. При alpha=0 ячейки чередуются циклически, иначе разыгрываются.

        ⚠ Наивный вариант (взять по одному из каждой ячейки и обрезать до n)
        при батче 2 и четырёх ячейках всегда оставлял первые две — обе БАД,
        и редкие позитивы не попадали в градиент вовсе. Циклический обход
        покрывает все ячейки за несколько микро-батчей накопления.
        """
        out = []
        while len(out) < n:
            if alpha == 0.0:
                k = cell_keys[cursor["i"] % len(cell_keys)]
                cursor["i"] += 1
            else:
                k = cell_keys[int(rng.choice(len(cell_keys), p=cell_p))]
            out.append(int(rng.choice(cells[k])))
        return np.array(out)

    exp_rare = args.batch * args.accum * float(
        cell_p[cell_keys.index(("Легковоспламеняющиеся", 1))]
        if ("Легковоспламеняющиеся", 1) in cell_keys else 0.0)
    if alpha == 0.0:
        exp_rare = args.batch * args.accum / len(cell_keys)
    n_rare = len(cells.get(("Легковоспламеняющиеся", 1), []))
    print(f"  сэмплер: alpha {alpha}, редких позитивов ~{exp_rare:.2f} на шаг, "
          f"за {args.max_steps} шагов каждый из {n_rare} покажется "
          f"~{exp_rare * args.max_steps / max(1, n_rare):.1f} раза", flush=True)

    order = np.random.permutation(len(train))
    log, t0, step_i, pos_in_epoch, skipped = [], time.time(), 0, 0, 0

    # ⚠ Продолжение с точки: восстанавливаем шаг, моменты Adam и расписание скорости.
    # Веса адаптера при этом уже подняты в load_model через --resume-from, иначе
    # продолжение вышло бы «холодным» и от нового прогона почти не отличалось.
    if args.resume_from:
        got = load_resume(Path(args.resume_from), opt, sched)
        if got:
            step_i, best_sel, best_f1 = got
            print(f"  осталось шагов: {args.max_steps - step_i}", flush=True)
        else:
            print(f"⚠ в {args.resume_from} нет resume.pt — продолжаем с холодного "
                  f"оптимизатора и с начала расписания скорости", flush=True)
    
    while step_i < args.max_steps:
        opt.zero_grad()
        total = 0.0
        
        for _ in range(args.accum):
            if args.balanced_sampler:
                rows = train.iloc[draw(args.batch)]
            else:
                if pos_in_epoch + args.batch > len(order):
                    order = np.random.permutation(len(train))
                    pos_in_epoch = 0
                rows = train.iloc[order[pos_in_epoch:pos_in_epoch + args.batch]]
                pos_in_epoch += args.batch
            answers = [YES if v == 1 else NO for v in rows.label.values]
            # ⚠ случайная обрезка описания: мешает запоминать карточку дословно,
            # но не трогает сам признак — он в первых предложениях и на фото
            md = args.max_desc
            if args.desc_jitter > 0:
                md = int(args.max_desc * (1.0 - rng.random() * args.desc_jitter))
            enc = encode_batch(proc, rows, cfg, root, step, md, answers, device,
                               n_images=args.n_images, rules_mode=args.rules,
                               modality=args.modality, prompt_mode=args.prompt,
                               facts_mode=args.facts)

            labels = enc["input_ids"].clone()
            labels[enc["attention_mask"] == 0] = -100
            for j, a in enumerate(answers):
                n_ans = len(tok.encode(a + tok.eos_token, add_special_tokens=False))
                seq = int(enc["attention_mask"][j].sum())
                labels[j, : seq - n_ans] = -100
            
            out = model(**enc, labels=labels)
            if args.loss_balance == "mean":
                loss = (out.loss / args.accum).float()
            else:
                # Средний по токенам лосс отдаёт градиент той ячейке, которой в
                # батче больше. У нас доли противоположны: БАД 75% позитивов,
                # редкая 3.6%, а категория написана в промпте — модель может
                # выучить ярлык, отвечая по названию категории и не глядя на товар.
                # Равный вес ячеек этот путь закрывает.
                per_item = per_item_ce(out.logits, labels)
                if args.loss_balance == "focal":
                    prob = torch.exp(-per_item.detach().clamp(max=20))
                    per_item = ((1 - prob) ** args.focal_gamma) * per_item
                cats = rows[cfg["data"]["category_col"]].astype(str).to_numpy()
                labs = rows.label.to_numpy()
                terms = []
                for cc in set(cats):
                    for ll in (0, 1):
                        idx = [i for i in range(len(cats))
                               if cats[i] == cc and labs[i] == ll]
                        if idx:
                            terms.append(per_item[idx].mean())
                loss = (torch.stack(terms).mean() / args.accum).float()
            
            if not torch.isfinite(loss):
                print("  шаг пропущен: лосс не число", flush=True)
                continue
            
            loss.backward()
            total += loss.detach().item()
        
        params = [p for p in model.parameters() if p.requires_grad]
        bad_grad = any(p.grad is not None and not torch.isfinite(p.grad).all()
                       for p in params)
        
        if bad_grad:
            skipped += 1
            opt.zero_grad(set_to_none=True)
            clear_cache(device)
            for group in opt.param_groups:
                for p in group['params']:
                    state = opt.state.get(p)
                    if state:
                        if 'exp_avg' in state: state['exp_avg'].zero_()
                        if 'exp_avg_sq' in state: state['exp_avg_sq'].zero_()
            if skipped <= 3 or skipped % 25 == 0:
                print(f"  шаг {step_i + 1} пропущен: нечисловой градиент "
                      f"(всего пропущено {skipped})", flush=True)
            if skipped > max(20, args.max_steps // 5):
                raise RuntimeError("слишком много нечисловых градиентов")
            step_i += 1
            sched.step()
            continue
        
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        
        # Периодическая очистка кэша MPS для дефрагментации
        if step_i % 50 == 0:
            clear_cache(device)
            import gc
            gc.collect()
        
        if not all(torch.isfinite(p).all() for p in params):
            raise RuntimeError(f"веса стали нечисловыми на шаге {step_i + 1}")
        step_i += 1
        log.append(total)
        
        # ⚠ На калибровочных прогонах в 20-30 шагов шаг 25 может не наступить, и
        # тогда ни скорости, ни памяти не увидеть — а замеряем мы ровно их.
        report_every = 25 if args.max_steps > 60 else 5
        if step_i % report_every == 0 or step_i == 1:
            done = step_i / args.max_steps
            # ⚠ Пиковая память на видеокарте — единственный способ понять, можно ли
            # поднять батч, не доводя до отказа на середине оплаченного прогона.
            mem = ""
            if device.type == "cuda":
                peak = torch.cuda.max_memory_allocated() / 2**30
                total = torch.cuda.get_device_properties(0).total_memory / 2**30
                mem = f" | память {peak:.1f}/{total:.0f} ГБ ({peak / total:.0%})"
            print(f"  шаг {step_i}/{args.max_steps} | лосс {np.mean(log[-25:]):.4f} | "
                  f"{(time.time() - t0) / 60:.1f} мин, осталось ~"
                  f"{(time.time() - t0) / max(done, 1e-9) * (1 - done) / 60:.0f} мин"
                  + mem, flush=True)
        if step_i % args.eval_every == 0:
            f1, per_cat, thr, rec = measure()
            last_measure = (f1, per_cat, thr, rec)
            tr = measure(train_probe) if train_probe is not None else None
            log_row(step_i, float(np.mean(log[-args.eval_every:])), f1, per_cat, thr,
                    rec, lr=float(sched.get_last_lr()[0]), tr=tr)
            if tr is not None:
                print(f"    обучающий фолд: метрика {tr[0]:.4f} | "
                      + " ".join(f"AUC {k} {tr[3].get(f'auc::{k}', float('nan')):.3f}"
                                 for k in tr[1]), flush=True)
            mark = ""
            sel = selection_score(f1, rec, args.select_by)
            # ⚠ Ничью разрешаем по метрике соревнования, а не «кто раньше». Замерено
            # на pack1: шаги 600, 1400 и 1600 дали ровно 0.7083 — полнота на 39
            # позитивах квантуется шагом 1/39, поэтому совпадения неизбежны. Победил
            # ранний, хотя на 1600 метрика была 0.8757 против 0.8674.
            take, tie = beats_best(sel, f1, best_sel, best_f1)
            if args.no_val or take:
                best_sel, best_f1, waited = sel, f1, 0
                mark = (f"  ← {'ничья, но метрика выше' if tie else 'лучший'} "
                        f"по {args.select_by} ({sel:.4f}), сохранён")
                save_checked(model, dst)     # ЛУЧШИЙ чекпоинт, а не последний
                saved_any = True
                (dst / "thresholds.json").write_text(
                    json.dumps({"thresholds": thr, "per_category": per_cat,
                                "objective": f1, "step": step_i, "recall": rec,
                                "select_by": args.select_by, "select_score": sel,
                                "tuned_on": "внутренняя проверка (сырой фолд), "
                                            "порог НЕ подбирался на экзамене"},
                               ensure_ascii=False), encoding="utf-8")
            else:
                waited += 1
            # ⚠ Снимок КАЖДОГО замера. Лучший и последний уже есть, но «лучший»
            # выбирается метрикой, которая на 39 позитивах квантуется грубо и
            # регулярно даёт ничьи — а какой из них окажется лучшим на лидерборде,
            # заранее неизвестно. 126 МБ за снимок против суток пересчёта.
            if args.keep_all:
                save_step_checkpoint(dst, model, args, device, step_i,
                                     f1, per_cat, thr, rec, sel)
            # ⚠ Точка продолжения пишется на КАЖДОМ замере, а не только при
            # улучшении: она страхует не от плохого чекпоинта, а от обрыва.
            if args.save_resume:
                save_resume(dst, model, opt, sched, step_i, best_sel, best_f1)
            print(f"  шаг {step_i}: метрика {f1:.4f} (к базе {f1 - base_f1:+.4f}, "
                  f"лучшая {best_f1:.4f}) "
                  + " | ".join(f"{k} {v:.3f}@{thr[k]:.2f}"
                               for k, v in per_cat.items())
                  + " | редкая: полнота "
                  + " ".join(f"{k.split('@')[1].split('::')[0]}:{v:.2f}"
                             for k, v in rec.items()
                             if k.startswith("recall@")
                             and "Легковоспламеняющиеся" in k)
                  + f" AUC {rec.get('auc::Легковоспламеняющиеся', float('nan')):.3f}"
                  + f" PR {rec.get('pr_auc::Легковоспламеняющиеся', float('nan')):.3f}"
                  + mark, flush=True)
            if waited >= args.patience:
                print(f"  ранняя остановка: {waited} замеров без улучшения. "
                      f"Дальше модель запоминает, а не учится.", flush=True)
                break
    
    # ⚠ НЕ сохраняем финальную модель поверх лучшей: именно последний чекпоинт и
    # оказывается переобученным. Сохраняем только если ни один замер не побил базу —
    # тогда хоть что-то, но с громким предупреждением.
    # ⚠ Последний чекпоинт сохраняется ВСЕГДА, рядом, в подпапку «last». Прогон
    # pack1 показал цену отказа: на шаге 1600 метрика была 0.8757 против 0.8674 у
    # сохранённого, лосс продолжал падать — а вернуть было нечего, переобучать сутки.
    # Место на диске стоит 26 МБ, повторный прогон — двадцать пять часов.
    last = dst / "last"
    last.mkdir(parents=True, exist_ok=True)
    save_checked(model, last)
    if last_measure is not None:
        (last / "thresholds.json").write_text(json.dumps(
            {"thresholds": last_measure[2], "per_category": last_measure[1],
             "objective": last_measure[0], "step": int(step_i),
             "recall": last_measure[3], "select_by": args.select_by,
             "tuned_on": "внутренняя проверка (сырой фолд), ПОСЛЕДНИЙ чекпоинт"},
            ensure_ascii=False), encoding="utf-8")
    print(f"ПОСЛЕДНИЙ чекпоинт (шаг {step_i}) сохранён отдельно в {last}"
          + (f", метрика {last_measure[0]:.4f}" if last_measure else ""), flush=True)

    if not saved_any:
        save_checked(model, dst)
        print("⚠ ни один замер не побил базу до обучения — сохранён ПОСЛЕДНИЙ адаптер. "
              "Скорее всего дообучение не дало ничего, проверьте оценку отдельно.",
              flush=True)
    else:
        print(f"сохранён ЛУЧШИЙ чекпоинт по «{args.select_by}»: {best_sel:.4f} "
              f"(база {base_sel:.4f}, прирост {best_sel - base_sel:+.4f}); "
              f"метрика на нём {best_f1:.4f}", flush=True)
    tl = train_meta(args, device, step_i,
                    {"loss": log, "base_objective": float(base_f1),
                     "best_objective": float(best_f1),
                     "saved_best": bool(saved_any)})
    (dst / "train_log.json").write_text(tl, encoding="utf-8")
    # ⚠ Тот же журнал кладём и к последнему чекпоинту: без него инференс не узнает
    # ни числа кадров, ни шаблона промпта, а разойдутся они молча.
    (last / "train_log.json").write_text(tl, encoding="utf-8")
    print(f"обучение завершено за {(time.time() - t0) / 60:.1f} мин, адаптер в {dst}")
    print(f"  чтобы отправить последний вместо лучшего, укажите {last}", flush=True)

    # ⚠ Гасим ЗДЕСЬ: обе точки уже на диске, журнал записан. Сюда исполнение
    # доходит только при успешном прогоне — при падении машина остаётся живой,
    # иначе разбираться было бы негде, а результаты остались бы недокачанными.
    poweroff_if_asked(args, ok=True)


@torch.no_grad()
def run_eval(args, proc, model, tok, test, cfg, root, step, dst, yes_ids, no_ids, device) -> None:
    from sklearn.metrics import average_precision_score, roc_auc_score

    tok.padding_side = "left"
    model.eval()
    if args.eval_neg:
        pos = test[test.label == 1]
        neg = test[test.label == 0].sample(n=min(args.eval_neg, int((test.label == 0).sum())),
                                           random_state=42)
        test = pd.concat([pos, neg])
    
    scores, failed, t0 = [], [], time.time()
    for i in range(0, len(test), args.eval_batch):
        rows = test.iloc[i:i + args.eval_batch]
        enc = encode_batch(proc, rows, cfg, root, step, args.max_desc, device=device,
                           n_images=args.n_images, rules_mode=args.rules,
                           modality=args.modality, prompt_mode=args.prompt,
                           facts_mode=args.facts)
        logits = model(**enc).logits[:, -1, :].float()
        
        if not torch.isfinite(logits).all():
            fixed = []
            for j in range(len(rows)):
                one = encode_batch(proc, rows.iloc[[j]], cfg, root, step,
                                   args.max_desc, device=device,
                                   n_images=args.n_images, rules_mode=args.rules,
                                   modality=args.modality, prompt_mode=args.prompt,
                               facts_mode=args.facts)
                lg = model(**one).logits[:, -1, :].float()
                if torch.isfinite(lg).all():
                    fixed.append(lg[0])
                else:
                    failed.append(str(rows.iloc[j][cfg["data"]["id_col"]]))
                    fixed.append(torch.zeros_like(logits[0]))
            logits = torch.stack(fixed)
        
        p = torch.softmax(logits, dim=-1)
        s = p[:, yes_ids].sum(-1) / (p[:, yes_ids].sum(-1) + p[:, no_ids].sum(-1) + 1e-9)
        s = torch.nan_to_num(s, nan=0.5)
        for j, item in enumerate(rows[cfg["data"]["id_col"]].astype(str)):
            if item in failed:
                s[j] = 0.5
        scores.extend(s.cpu().numpy().tolist())
        
        if (i // args.eval_batch) % 25 == 0:
            print(f"  оценено {i + len(rows)}/{len(test)} "
                  f"({(time.time() - t0) / 60:.1f} мин)", flush=True)
    
    y = test.label.to_numpy()
    s = np.array(scores)
    out = pd.DataFrame({"id": test[cfg["data"]["id_col"]].to_numpy(), "label": y, "score": s})
    out.to_parquet(dst / "scores.parquet", index=False)

    auc, pr = roc_auc_score(y, s), average_precision_score(y, s)
    cand = np.unique(np.quantile(s, np.linspace(0.5, 0.9995, 200)))
    best = 0.0
    for t in cand:
        p = (s >= t).astype(int)
        tp, fp, fn = int(((p == 1) & (y == 1)).sum()), int(((p == 1) & (y == 0)).sum()), \
            int(((p == 0) & (y == 1)).sum())
        best = max(best, 2 * tp / max(1, 2 * tp + fp + fn))
    
    if failed:
        share = len(failed) / len(test)
        print(f"\n⚠ товаров без оценки: {len(failed)} из {len(test)} ({share:.1%})", flush=True)
        if share > 0.02:
            raise RuntimeError(f"отказов {share:.1%} — больше 2%")
    
    print(f"\nФОЛД {args.fold} ({args.split}): AUC={auc:.4f} PR-AUC={pr:.4f} "
          f"F1(лучший порог)={best:.4f}  n={len(test)}, позитивов {int(y.sum())}")
    if args.eval_neg:
        print(f"⚠ негативы подрезаны до {args.eval_neg}: PR-AUC ЗАВЫШЕН")
    else:
        print("ориентир на полном фолде — эмбеддинги+голова: PR-AUC 0.6225 / 0.6152")
    print(f"оценка заняла {(time.time() - t0) / 60:.1f} мин, оценки в {dst / 'scores.parquet'}")


if __name__ == "__main__":
    main()