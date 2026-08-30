"""Правка модуля прогона под адаптер vlm_rare2_a.

    python patch_vlm.py

Две вещи, без которых наша половина решения тихо поехала бы.

1. ШАБЛОН. Адаптер обучен на configs/prompts/verdict_ru.txt того решения. От готового
   «packaging» тот текст отличается ОДНИМ переносом строки в предпоследнем абзаце —
   и этого достаточно: последовательность токенов другая. Гнездо {ocr} при обучении
   оставалось пустым (в конфиге стоит use_ocr: false), поэтому распознавание убрано
   без последствий для входа.

2. ОБРЕЗКА ОПИСАНИЯ. В модуле она константа 800, а адаптер обучен на 900. Разная
   обрезка даёт другой вход при том же шаблоне, ошибки при этом не будет — только
   поехавший вердикт. Делаем её свойством адаптера, как уже сделаны кадры, пиксели
   и сам шаблон.
"""
from pathlib import Path

SRC = Path(__file__).resolve().parent / "build/src/qc26/inference/vlm.py"

TEMPLATE = '''# ⚠ Шаблон адаптера vlm_rare2_a, выигравшего редкую категорию: 21 верный из 24 при
# 22 названных. От «packaging» отличается ОДНИМ переносом строки в предпоследнем
# абзаце — и этого достаточно, чтобы вход стал другим. Текст сверен с
# configs/prompts/verdict_ru.txt того решения посимвольно.
QUESTION_RARE2 = (
    "Ты модератор маркетплейса и проверяешь карточку товара.\\n\\n"
    "Продавец указал категорию: «{cat}».\\n"
    "Правила этой категории:\\n"
    "{rules}\\n\\n"
    "Карточка товара.\\n"
    "Название: {name}\\n"
    "Описание: {desc}\\n\\n"
    "Изображения товара приложены выше. Смотри на упаковку: маркировка часто напечатана\\n"
    "на ней, а не написана в описании.\\n\\n"
    "Вопрос: товар действительно относится к категории «{cat}»?\\n"
    "Ответь одним словом — Да или Нет."
)

PROMPTS = {"ours": QUESTION, "packaging": QUESTION_PACKAGING, "qc": QUESTION_QC,
           "rare2": QUESTION_RARE2}'''

OLD_BP = '''def build_prompt(proc, row, cfg: dict, n_imgs: int, template: str | None = None,
                 with_facts: bool = False) -> str:
    """Плейсхолдеров ровно столько же, сколько кадров реально передано."""
    d = cfg["data"]
    cat = str(row[d["category_col"]])
    tpl = template or QUESTION
    fields = {"cat": cat, "rules": RULES.get(cat, ""),
              "name": str(row[d["name_col"]])[:MAX_NAME],
              "desc": str(row.get(d["desc_col"]) or "")[:MAX_DESC]}'''

NEW_BP = '''def build_prompt(proc, row, cfg: dict, n_imgs: int, template: str | None = None,
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
              "desc": str(row.get(d["desc_col"]) or "")[:int(max_desc or MAX_DESC)]}'''

READER = '''def desc_used_in_training(adapter_dir) -> int:
    """Обрезка описания при обучении адаптера — из train_log.json рядом с ним."""
    return int(_train_log(adapter_dir).get("desc_max") or MAX_DESC)


def base_used_in_training(adapter_dir) -> str:'''


def main() -> None:
    s = SRC.read_text(encoding="utf-8")
    old_map = ('PROMPTS = {"ours": QUESTION, "packaging": QUESTION_PACKAGING, '
               '"qc": QUESTION_QC}')
    for old, new in ((old_map, TEMPLATE),
                     (OLD_BP, NEW_BP),
                     ("def base_used_in_training(adapter_dir) -> str:", READER)):
        if old not in s:
            raise SystemExit(f"не нашёл кусок: {old[:60]}")
        s = s.replace(old, new, 1)

    call = "prompt = build_prompt(proc, row, cfg, len(imgs), template, with_facts)"
    if call not in s:
        raise SystemExit("не нашёл вызов build_prompt в прогоне")
    s = s.replace(call, "prompt = build_prompt(proc, row, cfg, len(imgs), template,\n"
                        "                                  with_facts, max_desc=desc_train)", 1)
    pn = "        prompt_name, template = prompt_used_in_training(adapter)"
    if pn not in s:
        raise SystemExit("не нашёл чтение шаблона")
    s = s.replace(pn, pn + "\n        desc_train = desc_used_in_training(adapter)", 1)
    SRC.write_text(s, encoding="utf-8")
    print("правка применена")


if __name__ == "__main__":
    main()
