"""Набор классических моделей для честного сравнения на одних и тех же признаках.

Смысл не в том, чтобы найти «лучший алгоритм вообще», а в том, чтобы понять природу
задачи: если линейная модель по разреженному тексту бьёт бустинг по эмбеддингам,
значит сигнал лексический, и вкладываться надо в текст и распознавание; если наоборот —
в зрение. Один прогон отвечает на этот вопрос вместо десяти догадок.

Все модели дают вероятность. Там, где её нет (линейный SVM), берётся решающая функция
и приводится сигмоидой: для AUC и PR-AUC важен порядок, а не калибровка, а калибровка
делается отдельным этапом.
"""
from __future__ import annotations

import numpy as np

# Модели, которым нужна плотная матрица (деревья, бустинги, соседи, перцептрон),
# и модели, которые нормально работают с разреженным tf-idf.
DENSE_ONLY = {"random_forest", "extra_trees", "hist_gb", "catboost", "lightgbm",
              "xgboost", "knn", "mlp", "gaussian_nb"}


def _logreg(seed: int, **kw):
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(C=kw.get("C", 4.0), max_iter=2000,
                              class_weight="balanced", random_state=seed)


def _linear_svm(seed: int, **kw):
    from sklearn.svm import LinearSVC

    return LinearSVC(C=kw.get("C", 0.5), class_weight="balanced", random_state=seed)


def _sgd(seed: int, **kw):
    from sklearn.linear_model import SGDClassifier

    return SGDClassifier(loss="modified_huber", alpha=1e-5, max_iter=3000,
                         class_weight="balanced", random_state=seed)


def _complement_nb(seed: int, **kw):
    from sklearn.naive_bayes import ComplementNB

    return ComplementNB(alpha=0.3)


def _random_forest(seed: int, **kw):
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(n_estimators=400, min_samples_leaf=2, n_jobs=-1,
                                  class_weight="balanced_subsample", random_state=seed)


def _extra_trees(seed: int, **kw):
    from sklearn.ensemble import ExtraTreesClassifier

    return ExtraTreesClassifier(n_estimators=500, min_samples_leaf=2, n_jobs=-1,
                                class_weight="balanced_subsample", random_state=seed)


def _hist_gb(seed: int, **kw):
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06,
                                          class_weight="balanced", random_state=seed)


def _catboost(seed: int, **kw):
    """CatBoost. Настройки видеокарты подобраны под RTX 3060 Ti (8 ГБ).

    Что именно даёт скорость на этой карте:
    - `border_count=32` вместо 128 по умолчанию. Число порогов дискретизации признака
      прямо определяет объём работы на каждом расщеплении, а на плотных вещественных
      признаках вроде эмбеддингов 32 порога качества почти не отнимают;
    - `boosting_type="Plain"`. На маленьких выборках CatBoost склонен выбирать
      упорядоченный режим, который на видеокарте заметно дороже;
    - `gpu_ram_part=0.95` — карта отдаётся обучению целиком. Этапы прогона идут
      последовательно, одновременно учится одна модель, делить видеопамять не с чем.
      Единица недостижима: контекст CUDA и рабочий стол уже занимают часть карты
      (замерено 740 МБ из 8192 в простое), поэтому 0.95 и есть практический максимум —
      это же значение CatBoost использует по умолчанию;
    - `max_ctr_complexity=1` — категориальных признаков у нас нет, комбинации считать
      незачем.

    Устройство и число итераций читаются из окружения: QC_CB_GPU=1 и QC_CB_ITERS.
    """
    import os

    from catboost import CatBoostClassifier

    on_gpu = os.environ.get("QC_CB_GPU") == "1"
    params = dict(
        iterations=int(kw.get("iterations", os.environ.get("QC_CB_ITERS", 600))),
        depth=6, learning_rate=0.06, auto_class_weights="Balanced", verbose=0,
        random_seed=seed, allow_writing_files=False,
        task_type="GPU" if on_gpu else "CPU")
    if on_gpu:
        params.update(devices="0", border_count=32, boosting_type="Plain",
                      gpu_ram_part=0.95, max_ctr_complexity=1)
    return CatBoostClassifier(**params)


def _lightgbm(seed: int, **kw):
    from lightgbm import LGBMClassifier

    return LGBMClassifier(n_estimators=600, learning_rate=0.06, num_leaves=63,
                          class_weight="balanced", random_state=seed, n_jobs=-1,
                          verbose=-1)


def _xgboost(seed: int, pos_weight: float = 1.0, **kw):
    from xgboost import XGBClassifier

    return XGBClassifier(n_estimators=600, learning_rate=0.06, max_depth=6,
                         subsample=0.9, colsample_bytree=0.8, n_jobs=-1,
                         scale_pos_weight=pos_weight, random_state=seed,
                         eval_metric="logloss", tree_method="hist")


def _knn(seed: int, **kw):
    from sklearn.neighbors import KNeighborsClassifier

    return KNeighborsClassifier(n_neighbors=15, weights="distance", metric="cosine",
                                n_jobs=-1)


def _mlp(seed: int, **kw):
    from sklearn.neural_network import MLPClassifier

    return MLPClassifier(hidden_layer_sizes=(256, 64), alpha=1e-4, max_iter=300,
                         early_stopping=True, random_state=seed)


FACTORIES = {
    "logreg": _logreg,
    "linear_svm": _linear_svm,
    "sgd": _sgd,
    "complement_nb": _complement_nb,
    "random_forest": _random_forest,
    "extra_trees": _extra_trees,
    "hist_gb": _hist_gb,
    "catboost": _catboost,
    "lightgbm": _lightgbm,
    "xgboost": _xgboost,
    "knn": _knn,
    "mlp": _mlp,
}


def build(name: str, seed: int, y_train=None):
    """Создаёт модель. Для xgboost вес позитива считается по обучающей выборке."""
    kw = {}
    if name == "xgboost" and y_train is not None:
        pos = float(np.sum(y_train == 1))
        neg = float(np.sum(y_train == 0))
        kw["pos_weight"] = max(1.0, neg / max(pos, 1.0))
    return FACTORIES[name](seed, **kw)


def predict_scores(model, X) -> np.ndarray:
    """Непрерывный скор: вероятность, а при её отсутствии — решающая функция."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1].astype(np.float32)
    raw = model.decision_function(X)
    return (1.0 / (1.0 + np.exp(-raw))).astype(np.float32)


def supports_sparse(name: str) -> bool:
    return name not in DENSE_ONLY
