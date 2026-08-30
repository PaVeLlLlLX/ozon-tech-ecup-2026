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
    parent = Path(data_path).parent
    # Штатный путь из условий — рядом с csv. Остальные проверяем на случай, если
    # проверяющая система разложит данные иначе: промах здесь лишает решение зрения.
    candidates = [parent / "images", parent / "image", parent, parent.parent / "images"]
    root = next((c for c in candidates if c.is_dir()
                 and (any(d.is_dir() for d in c.iterdir())
                      or any(f.suffix.lower() in IMAGE_EXTS for f in c.iterdir()))), None)
    if root is None:
        return None
    for _ in range(4):  # спускаемся через лишнюю вложенность вида images/images
        subs = [d for d in root.iterdir() if d.is_dir() and d.name not in _SERVICE_DIRS]
        if len(subs) == 1 and subs[0].name.lower() in ("images", "image"):
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


@lru_cache(maxsize=4)
def _flat_index(root: str) -> dict[str, tuple[Path, ...]]:
    """Индекс «идентификатор → файлы» для раскладки без подпапок.

    Проверяющая система может хранить фото не папками (`images/<id>/1.jpg`), а файлами
    (`images/<id>.jpg`, `images/<id>_2.jpg`). Отправка показала, что угадывать раскладку
    нельзя: при промахе решение молча теряет зрение и откатывается на текст. Поэтому
    строим индекс по тому, что реально лежит на диске.
    """
    out: dict[str, list[Path]] = {}
    base = Path(root)
    for f in base.iterdir():
        if not f.is_file() or f.suffix.lower() not in IMAGE_EXTS:
            continue
        stem = f.stem
        key = stem.split("_")[0].split("-")[0]
        out.setdefault(key, []).append(f)
        if key != stem:
            out.setdefault(stem, []).append(f)
    return {k: tuple(sorted(v, key=lambda f: (len(f.stem), f.stem))) for k, v in out.items()}


def image_paths(root: Path | None, item_id: str) -> list[Path]:
    """Отсортированные пути к фото товара (1-5 штук). Пустой список, если фото нет.

    Поддерживаются обе раскладки: подпапка на товар и плоский набор файлов.
    """
    if root is None:
        return []
    item_id = str(item_id)
    d = root / item_id
    if d.is_dir():
        files = [f for f in d.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
        return sorted(files, key=lambda f: (len(f.stem), f.stem))
    try:
        return list(_flat_index(str(root)).get(item_id, ()))
    except OSError:
        return []


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
