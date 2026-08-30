"""Запись прогонов в MLflow. Без mlflow пайплайн продолжает работать.

⚠ Имена метрик обязаны совпадать во ВСЕХ прогонах. MLflow показывает метрики
столбцами: если обучение пишет `metric_mean`, а отправка `local_metric_binary`, в
таблице получаются два столбца с дырами, и прогоны нельзя сравнить взглядом. Поэтому
имена собраны здесь константами, а не набираются строками по месту вызова.

Разделение по смыслу:
- МЕТРИКА — число, которое сравнивают между прогонами (по ней сортируют);
- ПАРАМЕТР — чем прогон отличается по настройке (модель, разрешение, пороги);
- ТЕГ — по чему прогоны фильтруют (этап, тип выборки, вариант отправки).
Одно и то же поле не должно быть одновременно параметром и тегом.
"""
from __future__ import annotations

import warnings

from .config import resolve_path

# метрика соревнования в двух прочтениях — считаем и храним оба всегда
M_BINARY = "metric_binary"      # F1 класса «не бан» в категории, среднее по категориям
M_MACRO = "metric_macro"        # F1 macro по обоим классам, среднее по категориям
M_AUC = "auc_mean"
M_PR_AUC = "pr_auc_mean"
M_PUBLIC = "public_f1"          # с публичного лидерборда, вписывается руками
M_PRIVATE = "private_f1"


def f1_metric_name(category: str, average: str = "binary") -> str:
    return metric_name(f"f1_{average}__{category}")


# MLflow допускает в именах метрик только латиницу, цифры и _-./ и пробел, а у нас
# категории называются по-русски — переводим имя метрики в латиницу.
_TRANSLIT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e", "ж": "zh",
    "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o",
    "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f", "х": "h", "ц": "c",
    "ч": "ch", "ш": "sh", "щ": "sch", "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu",
    "я": "ya",
}


def metric_name(name: str) -> str:
    out = []
    for ch in str(name):
        low = ch.lower()
        if low in _TRANSLIT:
            t = _TRANSLIT[low]
            out.append(t.upper() if ch.isupper() else t)
        elif ch.isalnum() or ch in "_-./ ":
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        elif isinstance(v, (list, tuple)):
            out[key] = str(v)[:250]
        else:
            out[key] = v
    return out


def log_run(cfg: dict, run_name: str, params: dict, metrics: dict,
            tags: dict | None = None, note: str | None = None) -> None:
    """note — короткая суть прогона: что именно поменялось и что здесь важно.

    Кладётся в штатное поле описания MLflow (`mlflow.note.content`), поэтому видно
    прямо в списке прогонов, а не только внутри карточки. Тот же текст идёт
    комментарием к отправке на платформе соревнования.
    """
    mc = cfg.get("mlflow", {})
    if not mc.get("enabled", False):
        return
    try:
        import mlflow
    except ImportError:
        warnings.warn("mlflow не установлен — запись прогона пропущена")
        return

    uri = str(mc["tracking_uri"])
    if uri.startswith("sqlite:///"):
        p = resolve_path(uri.removeprefix("sqlite:///"))
        uri = "sqlite:///" + str(p).replace("\\", "/")
    elif not uri.startswith(("http://", "https://")):
        uri = resolve_path(uri).as_uri()
    try:
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(mc.get("experiment", "default"))
        with mlflow.start_run(run_name=run_name):
            if tags:
                mlflow.set_tags(tags)
            if note:
                mlflow.set_tag("mlflow.note.content", note)
                mlflow.set_tag("comment", note[:490])
            mlflow.log_params({metric_name(k): v for k, v in _flatten(params).items()
                               if v is not None})
            mlflow.log_metrics({metric_name(k): float(v) for k, v in _flatten(metrics).items()
                                if isinstance(v, (int, float)) and v == v})
    except Exception as e:  # трекинг не должен ронять прогон
        warnings.warn(f"mlflow: запись не удалась: {e}")
