"""Упаковка весов модели в int8 — чтобы модель влезла в архив решения.

## Зачем

Qwen3-VL-4B-Instruct нет в каталоге проверяющей системы, значит её надо везти с собой.
В bf16 она весит 8.3 ГБ, а лимиты такие:

    архив решения   5 ГБ
    образ          15 ГБ, из них ~12 занимает базовый образ организаторов

То есть в bf16 модель не помещается НИ КУДА. Единственный проход — квантование:
int8 с масштабом на строку даёт ровно вдвое, 8.3 -> 4.15 ГБ, и в архив она входит
с запасом около 0.7 ГБ под адаптер и код.

## Как

Для каждой двумерной матрицы весов W формы (выход, вход):

    масштаб = max|W| по строке / 127
    q       = round(W / масштаб)          int8
    W'      = q * масштаб                 при распаковке

Масштаб свой у каждой строки, а не один на тензор: у строк линейного слоя разброс
величин отличается на порядки, и общий масштаб съел бы малые строки целиком.

⚠ Одномерные тензоры (смещения, нормировки) НЕ пакуются. Их суммарный вес — единицы
мегабайт, а чувствительность к шуму у нормировок высокая: экономия копеечная, риск
непропорциональный.

## Что проверяется

Упаковка без замера ошибки бессмысленна, поэтому pack_model возвращает среднюю и
наибольшую относительную ошибку по тензорам.

Ориентир выводится, а не берётся на глаз. Ошибка округления равномерна в пределах
половины масштаба, масштаб равен max|строки|/127, а для гауссовой строки длины n
максимум примерно sigma*sqrt(2*ln n) при среднем модуле sigma*sqrt(2/pi). Отсюда

    относительная ошибка ~ 0.25 * sqrt(2*ln n) / 127 / sqrt(2/pi)

    строка   256 -> 0.0082
    строка   512 -> 0.0087
    строка  2560 -> 0.0098
    строка  9728 -> 0.0106

То есть **0.008-0.011 — это норма, а не дефект**. Замер на синтетике дал 0.0078 при
строках 256 и 512 — совпадает. Заметно большее значение означает выбросы в весах:
у настоящих моделей они бывают, и тогда масштаб на строку растягивается одним
элементом, а остальные теряют разрядность.

⚠ Относительная ошибка ВЕСА — не то же самое, что потеря качества. Единственная
честная проверка: посчитать одни и те же товары упакованной и неупакованной моделью
и сравнить вердикты. Она делается там, где лежат веса, — на сервере.
"""
from __future__ import annotations

import json
from pathlib import Path

# ⚠ Порог на размер: мелкие двумерные тензоры паковать невыгодно — заголовок
# safetensors и массив масштабов съедят выигрыш, а ошибку добавят.
MIN_NUMEL = 1 << 16

# ⚠ Что не пакуем НИКОГДА, независимо от размера.
#
# Замер на Qwen3-VL-4B: из 358 упакованных тензоров шесть дали ошибку выше 0.02,
# и худший — model.visual.pos_embed.weight, 0.0528 при выбросе 46.6. Позиционные
# вложения СКЛАДЫВАЮТСЯ с вложениями заплаток на самом входе зрительной башни:
# ошибка там не усредняется по слоям, а сразу смещает всё, что идёт дальше.
# А весит он 4.7 МБ, то есть отказ от упаковки стоит 2.4 МБ из 850 свободных.
#
# Обратите внимание: embed_tokens и lm_head в список НЕ входят — замер показал,
# что они квантуются хорошо и в худшую дюжину не попадают вовсе.
NEVER_PACK = ("pos_embed",)


def _pack_tensor(w):
    """(int8-тензор, масштабы). Масштаб на строку, вдоль последней оси."""
    import torch

    f = w.detach().to(torch.float32)
    scale = f.abs().amax(dim=-1, keepdim=True) / 127.0
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = torch.clamp(torch.round(f / scale), -127, 127).to(torch.int8)
    return q, scale.squeeze(-1).to(torch.float32)


def _unpack_tensor(q, scale, dtype):
    import torch

    return (q.to(torch.float32) * scale.unsqueeze(-1)).to(dtype)


def pack_model(src, dst, keep=()) -> dict:
    """Пакует каталог модели. Возвращает статистику и отчёт об ошибке.

    keep — дополнительные подстроки имён, которые паковать не надо. Кладутся
    поверх NEVER_PACK. Пример: keep=("visual",) оставит всю зрительную башню
    в исходной точности, если проверка вердиктов покажет, что она того стоит.
    """
    import torch
    from safetensors.torch import save_file

    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)

    shards = sorted(src.glob("*.safetensors"))
    if not shards:
        raise SystemExit(f"в {src} нет файлов safetensors")

    packed_names: list[str] = []
    errs: list[tuple[float, str]] = []
    n_packed = n_kept = 0
    kept_bytes = [0]

    from safetensors import safe_open

    for shard in shards:
        out: dict = {}
        # ⚠ Читаем ТЕНЗОР ЗА ТЕНЗОРОМ, а не кусок целиком. У Qwen3-VL-4B куски по
        # 4 ГБ, и загрузка куска вместе с упакованной копией давала пик около 6 ГБ.
        # На подготовительной машине с 8 ГБ это отказ по памяти — ровно там, где
        # упаковку и удобнее всего делать.
        with safe_open(str(shard), framework="pt") as fh:
            for name in fh.keys():
                t = fh.get_tensor(name)
                # Пакуем только двумерные и достаточно крупные: на остальном
                # экономия несоразмерна риску.
                skip = tuple(NEVER_PACK) + tuple(keep)
                if (t.ndim != 2 or t.numel() < MIN_NUMEL
                        or any(w in name for w in skip)):
                    out[name] = t
                    n_kept += 1
                    kept_bytes[0] += t.numel() * t.element_size()
                    continue
                dtype_code = _DTYPE_CODE[str(t.dtype)]
                q, scale = _pack_tensor(t)
                back = _unpack_tensor(q, scale, torch.float32)
                orig = t.detach().to(torch.float32)
                denom = orig.abs().mean().clamp(min=1e-12)
                errs.append((float((back - orig).abs().mean() / denom), name))
                del back, orig, t          # освобождаем до следующего тензора
                out[name + ".q"] = q
                out[name + ".s"] = scale
                out[name + ".d"] = torch.tensor([dtype_code], dtype=torch.int32)
                packed_names.append(name)
                n_packed += 1
        save_file(out, str(dst / shard.name))
        del out

    # конфиги, токенизатор, шаблон чата — всё, что не веса
    for f in src.iterdir():
        if f.is_file() and f.suffix != ".safetensors":
            (dst / f.name).write_bytes(f.read_bytes())

    (dst / "packing.json").write_text(json.dumps(
        {"format": "int8-rowscale-v1", "packed": sorted(packed_names),
         "min_numel": MIN_NUMEL}, ensure_ascii=False), encoding="utf-8")

    size = lambda p: sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file())
    vals = [e for e, _ in errs]
    worst = sorted(errs, reverse=True)[:8]
    return {"packed_tensors": n_packed, "kept_tensors": n_kept,
            "ratio": size(dst) / max(1, size(src)),
            "err_mean_rel": sum(vals) / max(1, len(vals)),
            "err_max_rel": max(vals) if vals else 0.0,
            "worst": worst,
            "n_over_002": sum(1 for v in vals if v > 0.02),
            "kept_mb": kept_bytes[0] / 2 ** 20}


def load_packed_into(model, path, dtype=None) -> tuple[int, list]:
    """Кладёт упакованные веса В УЖЕ СОЗДАННУЮ модель, по одному куску.

    ⚠ Почему не через meta-устройство. Первая версия собирала скелет на meta и
    клала веса поверх — и падала с «Cannot copy out of meta tensor; no data!».
    Причина: БУФЕРЫ (таблицы поворотов и подобное) в safetensors не хранятся, они
    вычисляются при создании модели. На meta их нет, в весах их нет, и перенос на
    устройство спотыкается именно о них.

    Здесь модель создаётся обычным образом (буферы считаются сами), а веса
    вкладываются кусок за куском с освобождением по ходу: пик по памяти — модель
    плюс ОДИН кусок, а не модель плюс весь словарь весов.

    Возвращает (сколько тензоров вложено, чего не хватило).
    """
    import torch
    from safetensors import safe_open

    own = dict(model.state_dict())
    filled: set[str] = set()
    path = Path(path)

    for shard in sorted(path.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt") as fh:
            keys = list(fh.keys())
            packed = {k.rsplit(".", 1)[0] for k in keys if k.endswith(".q")}
            for name in packed:
                q = fh.get_tensor(name + ".q")
                sc = fh.get_tensor(name + ".s")
                code = int(fh.get_tensor(name + ".d")[0])
                dt = dtype or getattr(torch, _CODE_DTYPE[code])
                if name in own:
                    own[name].data.copy_(_unpack_tensor(q, sc, dt))
                    filled.add(name)
                del q, sc
            for k in keys:
                if k.endswith((".q", ".s", ".d")):
                    continue
                if k in own:
                    t = fh.get_tensor(k)
                    own[k].data.copy_(t.to(dtype) if dtype and t.is_floating_point()
                                      else t)
                    filled.add(k)

    missing = [k for k in own if k not in filled]
    return len(filled), missing


_DTYPE_CODE = {"torch.bfloat16": 0, "torch.float16": 1, "torch.float32": 2}
_CODE_DTYPE = {0: "bfloat16", 1: "float16", 2: "float32"}


def load_packed(path, device="cpu") -> dict:
    """Разворачивает упакованный каталог обратно в обычный словарь весов.

    ⚠ Разворачивается в исходный тип (bf16), то есть в памяти модель займёт полные
    8.3 ГБ. Экономия здесь только на РАЗМЕРЕ АРХИВА, не на памяти прогона.
    """
    import torch
    from safetensors.torch import load_file

    path = Path(path)
    state: dict = {}
    for shard in sorted(path.glob("*.safetensors")):
        raw = load_file(str(shard))
        names = {k.rsplit(".", 1)[0] for k in raw if k.endswith((".q", ".s", ".d"))}
        for k, v in raw.items():
            if not k.endswith((".q", ".s", ".d")):
                state[k] = v.to(device)
        for name in names:
            q, s = raw[name + ".q"], raw[name + ".s"]
            code = int(raw[name + ".d"][0])
            dt = getattr(torch, _CODE_DTYPE[code])
            state[name] = _unpack_tensor(q.to(device), s.to(device), dt)
    if not state:
        raise SystemExit(f"в {path} не нашлось весов")
    return state
