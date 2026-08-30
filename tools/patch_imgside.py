"""Подготовка кадра — свойство адаптера, как шаблон, длина описания и написания ответа.

    python patch_imgside.py

⚠ Расхождение, которое закрывается. Рекорд vlm_blend_rare2_a_t24 готовил кадр так:

    открыть -> если длинная сторона больше 640, ужать LANCZOS -> отдать процессору

Код сокомандника — иначе: сразу ужимает под бюджет в 100352 пикселя и выравнивает обе
стороны по сетке патчей. Оба приходят примерно к 316x316, но рекорд пересэмплирует
ДВАЖДЫ (1000 -> 640 -> 316), а тут это один проход. Пиксели выходят разные.

⚠⚠ Это ГИПОТЕЗА о причине, почему порог отбирает 3.72% вместо 3.11%, а не факт. Проверять
её обязан замер на фолде 0, а не рассуждение: правка ставится рядом с прежней версией,
и остаётся та, у которой числа ближе к рекордным.

Ключ `image_max_side` в train_log.json включает поведение рекорда для КОНКРЕТНОГО
адаптера. Без ключа всё остаётся как было — путь адаптеров vl4_* не трогается, их доля
0.5733 получена именно на нём.
"""
from pathlib import Path

SRC = Path(__file__).resolve().parent / "build/src/qc26/inference/vlm.py"

OLD = '''def _load_image(path, step: int, max_pixels: int | None = None):
    from PIL import Image

    mp = int(max_pixels or MAX_PIXELS)
    img = Image.open(path).convert("RGB")
    w, h = img.size'''

NEW = '''def side_used_in_training(adapter_dir) -> int:
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
        # ⚠ Ровно подготовка рекорда: только ужать длинную сторону и отдать
        # процессору, а он сам приведёт кадр к сетке под бюджет пикселей. Никакого
        # второго ужимания здесь быть не должно — в рекорде его не было.
        w, h = img.size
        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                             Image.LANCZOS)
        return img
    w, h = img.size'''

OLD_CALL = "                    imgs.append(_load_image(p, step, px))"
NEW_CALL = "                    imgs.append(_load_image(p, step, px, side_train))"

OLD_READ = "        desc_train = desc_used_in_training(adapter)"
NEW_READ = ("        desc_train = desc_used_in_training(adapter)\n"
            "        side_train = side_used_in_training(adapter)\n"
            "        if side_train:\n"
            "            print(f\"подготовка кадра как в рекорде: длинная сторона до \"\n"
            "                  f\"{side_train}, дальше процессор\", flush=True)")


def main() -> None:
    s = SRC.read_text(encoding="utf-8")
    if "side_used_in_training" in s:
        raise SystemExit("правка уже применена")
    for old, new, what in ((OLD, NEW, "загрузка кадра"),
                           (OLD_READ, NEW_READ, "чтение журнала"),
                           (OLD_CALL, NEW_CALL, "вызов загрузки")):
        if old not in s:
            raise SystemExit(f"не нашёл место для правки «{what}»")
        s = s.replace(old, new, 1)
    SRC.write_text(s, encoding="utf-8")
    print("подготовка кадра стала свойством адаптера")


if __name__ == "__main__":
    main()
