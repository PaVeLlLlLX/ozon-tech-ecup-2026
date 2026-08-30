"""Чтение карточек товаров и поиск их изображений.

Путь к папке с картинками проверяющая система задаёт неявно: он выводится из пути к
csv (`<папка csv>/images`). У выданного архива внутри оказался лишний уровень
(`images/images/<id>/`) плюс служебная папка `__MACOSX`, поэтому корень определяем
пробой, а не жёстко. В контейнере это же спасёт от неверной догадки о раскладке.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
_SERVICE_DIRS = {"__MACOSX", ".ipynb_checkpoints"}


def load_cards(cfg: dict, path: str | Path | None = None) -> pd.DataFrame:
    """Читает csv карточек. Служебный безымянный индекс отбрасывается."""
    from .config import resolve_path

    p = Path(path) if path is not None else resolve_path(cfg["paths"]["data_csv"])
    df = pd.read_csv(p)
    df = df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed")])
    idc = cfg["data"]["id_col"]
    df[idc] = df[idc].astype(str)
    for col in ("name", "description"):
        if col in df.columns:
            df[col] = df[col].fillna("")
    return df


def resolve_images_root(data_path: str | Path) -> Path | None:
    """Корень папки с картинками: та папка, в которой лежат подпапки-идентификаторы.

    Спускаемся вглубь, пока видим единственную содержательную подпапку (`images/images`),
    и останавливаемся, как только подпапок стало много — это и есть уровень товаров.
    """
    root = Path(data_path).parent / "images"
    if not root.is_dir():
        return None
    for _ in range(4):
        subs = [d for d in root.iterdir() if d.is_dir() and d.name not in _SERVICE_DIRS]
        if len(subs) == 1 and subs[0].name.lower() == "images":
            root = subs[0]
            continue
        return root
    return root


@lru_cache(maxsize=8)
def _images_root_cached(data_path: str) -> Path | None:
    return resolve_images_root(data_path)


def images_root(cfg: dict, data_path: str | Path | None = None) -> Path | None:
    from .config import resolve_path

    p = Path(data_path) if data_path is not None else resolve_path(cfg["paths"]["data_csv"])
    return _images_root_cached(str(p))


def image_paths(root: Path | None, item_id: str) -> list[Path]:
    """Отсортированные пути к фото товара (1-5 штук). Пустой список, если фото нет."""
    if root is None:
        return []
    d = root / str(item_id)
    if not d.is_dir():
        return []
    files = [f for f in d.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
    return sorted(files, key=lambda f: (len(f.stem), f.stem))


def load_image(path: str | Path, max_side: int = 768):
    """Открывает картинку и ужимает длинную сторону — держим память под контролем."""
    from PIL import Image

    img = Image.open(path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    return img


def attach_image_counts(cfg: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет колонку с числом найденных фото (диагностика, не признак)."""
    root = images_root(cfg)
    idc = cfg["data"]["id_col"]
    df = df.copy()
    df["n_images"] = [len(image_paths(root, i)) for i in df[idc]]
    return df
