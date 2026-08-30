"""Загрузка YAML-конфигов с оверрайдами из командной строки (key.sub=value).

Все относительные пути в конфигах считаются от корня репозитория, поэтому скрипты
можно запускать из любой папки.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

try:  # PyYAML может отсутствовать в базовом образе организаторов
    import yaml
except ImportError:  # pragma: no cover — путь только для контейнера
    yaml = None


def _read_config_file(path: Path) -> dict:
    """Читает YAML; если PyYAML недоступен — рядом лежащую копию в JSON.

    Сборщик архива (`scripts/build_submission.py`) кладёт `*.json` рядом с каждым
    конфигом, поэтому решение работает в образе без PyYAML.
    """
    if yaml is not None and path.exists():
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    alt = path.with_suffix(".json")
    if alt.exists():
        with open(alt, encoding="utf-8") as f:
            return json.load(f) or {}
    if path.exists():
        raise RuntimeError(f"нужен PyYAML либо копия конфига в JSON рядом с {path}")
    raise FileNotFoundError(path)


def _parse_value(raw: str) -> Any:
    if yaml is not None:
        return yaml.safe_load(raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"true": True, "false": False, "null": None}.get(raw.strip().lower(), raw)


def _deep_merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(*paths: str | Path, overrides: list[str] | None = None) -> dict:
    """Сливает несколько YAML (позже — приоритетнее) и применяет оверрайды a.b.c=значение."""
    cfg: dict = {}
    for p in paths:
        path = Path(p)
        if not path.is_absolute():
            path = REPO_ROOT / path
        cfg = _deep_merge(cfg, _read_config_file(path))
    for ov in overrides or []:
        key, _, raw = ov.partition("=")
        node = cfg
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _parse_value(raw)
    return cfg


def resolve_path(cfg_path: str | Path) -> Path:
    p = Path(cfg_path)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()
