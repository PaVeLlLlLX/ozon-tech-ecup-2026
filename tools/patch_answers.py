"""Набор написаний ответа — свойство адаптера, а не константа модуля.

    python patch_answers.py

⚠⚠ Зачем. Оценка товара это разность двух логарифмов сумм: по токенам «да» и по
токенам «нет». Какие именно написания входят в каждую сумму, решает не код прогона, а
то, на чём считалась шкала, под которую подобран порог.

    рекорд     ["Да", " Да", "да", "Yes", " Yes"] / ["Нет", " Нет", "нет", "No", " No"]
    новый код  " Да" плюс «Д» и « Д» / " Нет" плюс «Н» и « Н»

Разные слагаемые дают разное число, а порог 0.9699 — срез абсолютный, он этого сдвига
не переживает. У адаптера сокомандника набор остаётся прежним: его доля 0.5733 получена
именно на нём.

⚠ Чего эта правка НЕ чинит: рекорд выравнивал пачку вправо (`padding_side: right` по
умолчанию) и читал логит с последней позиции, то есть ПОСЛЕ добивочных токенов. При
таком чтении оценка товара зависит от того, с кем он попал в одну пачку из тридцати
двух. Это невоспроизводимо в принципе — ни здесь, ни в любом другом коде. Поэтому читаем
правильно (выравнивание влево, логит сразу после вопроса), а состоятельность переноса
порога проверяем замером.
"""
from pathlib import Path

SRC = Path(__file__).resolve().parent / "build/src/qc26/inference/vlm.py"

READER = '''def answer_ids_used_in_training(tok, adapter_dir):
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


def desc_used_in_training(adapter_dir) -> int:'''

OLD_CALL = ("        yes_ids, no_ids = first_token_ids(tok, YES), "
            "first_token_ids(tok, NO)")
NEW_CALL = '''        yes_ids, no_ids = answer_ids_used_in_training(tok, adapter)
        print(f"токены ответа: «да» {len(yes_ids)} шт., «нет» {len(no_ids)} шт.",
              flush=True)'''


def main() -> None:
    s = SRC.read_text(encoding="utf-8")
    if "answer_ids_used_in_training" in s:
        raise SystemExit("правка уже применена")
    anchor = "def desc_used_in_training(adapter_dir) -> int:"
    if anchor not in s:
        raise SystemExit("не нашёл соседнюю функцию — порядок файла изменился")
    s = s.replace(anchor, READER, 1)
    if OLD_CALL not in s:
        raise SystemExit("не нашёл место, где берутся токены ответа")
    s = s.replace(OLD_CALL, NEW_CALL, 1)
    SRC.write_text(s, encoding="utf-8")
    print("набор ответов стал свойством адаптера")


if __name__ == "__main__":
    main()
