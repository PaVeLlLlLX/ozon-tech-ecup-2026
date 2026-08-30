"""Упаковка весов модели в архив решения: int8 с покомпонентными масштабами.

Зачем. Qwen3-VL-4B-Instruct нет в каталоге моделей проверяющей системы, значит его надо
везти с собой. В bf16 он занимает **8.28 ГБ** — в архив на 5 ГБ не влезает, а в образ
влезает только впритык (базовый образ 6.16 ГБ сжатым при лимите 15) и требует заливки
пятнадцати гигабайт в реестр.

Здесь веса хранятся в int8 с отдельным масштабом на каждую строку матрицы, а при
загрузке разворачиваются обратно в bf16. Размер падает вдвое, до ~4.2 ГБ, и архив
укладывается в лимит.

⚠ Библиотек для этого не нужно НИКАКИХ сверх тех, что уже есть в образе: torch,
safetensors и accelerate. Это принципиально — 13.08 отправка `vlmtext_shippable` упала
с `ImportError: bitsandbytes`, потому что квантование в 4 бита требует пакета, которого
нет ни в базовом образе, ни в нашем. Здесь всё считается средствами torch.

⚠ Разворачивание идёт ПОТЕНЗОРНО, через `set_module_tensor_to_device`: держать в памяти
одновременно и упакованные веса, и развёрнутую копию не нужно.

Точность. Масштаб свой на каждую строку (то есть на каждый выходной канал линейного
слоя), поэтому потеря заметно меньше, чем при одном масштабе на тензор. Величина ошибки
замеряется `scripts/pack_model.py --check` и записывается в отчёт: без замера упаковку
в отправку не ставить.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

# Тензоры мельче этого порога оставляем как есть: выигрыш копеечный, а риск потери
# точности на нормировках и смещениях — нет.
MIN_NUMEL = 1_000_000
SCALE_SUFFIX = ".qscale"
PACK_MANIFEST = "packing.json"


def _quantize_row_wise(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """int8 + масштаб на строку. Возвращает (значения, масштабы)."""
    w32 = w.to(torch.float32)
    scale = w32.abs().amax(dim=1, keepdim=True) / 127.0
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = torch.clamp(torch.round(w32 / scale), -127, 127).to(torch.int8)
    return q, scale.squeeze(1).to(torch.float32)


def _dequantize_row_wise(q: torch.Tensor, scale: torch.Tensor,
                         dtype=torch.bfloat16) -> torch.Tensor:
    return (q.to(torch.float32) * scale.unsqueeze(1)).to(dtype)


def _should_pack(name: str, t: torch.Tensor) -> bool:
    return (t.dtype in (torch.bfloat16, torch.float16, torch.float32)
            and t.ndim == 2 and t.numel() >= MIN_NUMEL)


def pack_model(src_dir: str | Path, dst_dir: str | Path) -> dict:
    """Читает safetensors модели и пишет упакованную копию рядом с её конфигами."""
    import shutil

    from safetensors.torch import load_file, save_file

    src, dst = Path(src_dir), Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    shards = sorted(src.glob("*.safetensors"))
    if not shards:
        raise RuntimeError(f"в {src} нет файлов safetensors")

    packed_names: list[str] = []
    errors: list[float] = []
    bytes_before = bytes_after = 0
    for shard in shards:
        tensors = load_file(str(shard))
        out: dict[str, torch.Tensor] = {}
        for name, t in tensors.items():
            bytes_before += t.numel() * t.element_size()
            if _should_pack(name, t):
                q, scale = _quantize_row_wise(t)
                out[name] = q
                out[name + SCALE_SUFFIX] = scale
                packed_names.append(name)
                bytes_after += q.numel() + scale.numel() * 4
                # Ошибка считается СРАЗУ: тензор уже в памяти, и это бесплатно.
                # Упаковка без замера ошибки в отправку не ставится.
                back = _dequantize_row_wise(q, scale, t.dtype)
                denom = t.to(torch.float32).abs().mean().clamp_min(1e-12)
                errors.append(float(((back - t).to(torch.float32).abs().mean() / denom)))
                del back
            else:
                out[name] = t
                bytes_after += t.numel() * t.element_size()
        save_file(out, str(dst / shard.name), metadata={"format": "pt"})
        del tensors, out

    # конфиги, токенизатор и прочее — как есть
    for f in src.iterdir():
        if f.is_file() and f.suffix != ".safetensors":
            shutil.copy2(f, dst / f.name)

    info = {"packed_tensors": len(packed_names), "min_numel": MIN_NUMEL,
            "bytes_before": bytes_before, "bytes_after": bytes_after,
            "ratio": round(bytes_after / max(1, bytes_before), 4),
            "err_mean_rel": round(sum(errors) / max(1, len(errors)), 6),
            "err_max_rel": round(max(errors) if errors else 0.0, 6)}
    (dst / PACK_MANIFEST).write_text(json.dumps(info, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    return info


def load_packed(pack_dir: str | Path, device: str = "cuda",
                dtype=torch.bfloat16):
    """Собирает модель из упакованных весов, разворачивая тензоры по одному."""
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForImageTextToText

    pack = Path(pack_dir)
    if not (pack / PACK_MANIFEST).exists():
        raise RuntimeError(f"{pack} не похож на упакованную модель: нет {PACK_MANIFEST}")
    config = AutoConfig.from_pretrained(str(pack))
    # ⚠ include_buffers=False обязателен. Непостоянные буферы (например, inv_freq
    # поворотных эмбеддингов) в файле весов НЕ лежат — они вычисляются при создании
    # модели. Под обычным init_empty_weights они остаются пустышками, и прогон падает
    # с «tensors on cuda:0 and cpu» уже в момент первого forward.
    with init_empty_weights(include_buffers=False):
        model = AutoModelForImageTextToText.from_config(config)
    model.tie_weights()

    seen: set[str] = set()
    for shard in sorted(pack.glob("*.safetensors")):
        tensors = load_file(str(shard))
        for name, t in tensors.items():
            if name.endswith(SCALE_SUFFIX):
                continue
            scale = tensors.get(name + SCALE_SUFFIX)
            value = (_dequantize_row_wise(t, scale, dtype) if scale is not None
                     else t.to(dtype) if t.is_floating_point() else t)
            set_module_tensor_to_device(model, name, device, value=value)
            seen.add(name)
        del tensors

    missing = [n for n, _ in model.named_parameters() if n not in seen]
    # Связанные веса (голова языковой модели с таблицей эмбеддингов) в файлах не лежат —
    # их восстанавливает tie_weights. Всё остальное отсутствовать не должно.
    if missing:
        model.tie_weights()
        still = [n for n in missing
                 if getattr(model.get_parameter(n), "is_meta", False)
                 or model.get_parameter(n).device.type == "meta"]
        if still:
            raise RuntimeError(f"не заполнены веса: {still[:5]} (всего {len(still)})")
    # Буферы созданы на процессоре — переносим их туда же, где веса.
    for name, buf in list(model.named_buffers()):
        if buf is not None and buf.device.type != device.split(":")[0]:
            mod = model.get_submodule(name.rpartition(".")[0]) if "." in name else model
            setattr(mod, name.rpartition(".")[2], buf.to(device))
    model.eval()
    return model
