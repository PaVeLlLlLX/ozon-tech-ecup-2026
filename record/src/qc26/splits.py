"""Сплиты без утечки: группы почти-дубликатов целиком уходят в один фолд.

Дополнительно держим стратификацию по паре (категория, метка): позитивов в
легковоспламеняющихся всего 198 на 143 группы, и случайное деление легко оставит
фолд вовсе без них.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def stratified_group_folds(df: pd.DataFrame, n_folds: int, seed: int,
                           group_col: str = "group", strat_cols: tuple = ("category", "label"),
                           ) -> np.ndarray:
    """Номер фолда для каждой строки. Группа неделима, страта балансируется жадно.

    Жадная раскладка: группы одной страты идут по фолдам, начиная с самого «пустого» —
    так редкий класс размазывается равномерно даже при 143 группах на всю категорию.
    """
    rng = np.random.default_rng(seed)
    key = df[list(strat_cols)].astype(str).agg("|".join, axis=1)

    grp = pd.DataFrame({"group": df[group_col].values, "key": key.values})
    # страта группы — самая частая страта внутри неё (группы почти всегда однородны)
    gkey = grp.groupby("group")["key"].agg(lambda s: s.value_counts().index[0])
    gsize = grp.groupby("group").size()

    fold_of_group: dict[int, int] = {}
    total_load = np.zeros(n_folds, dtype=np.int64)  # общий счётчик, чтобы фолды не перекосило
    for _, groups in gkey.groupby(gkey):
        ids = groups.index.to_numpy()
        rng.shuffle(ids)
        # крупные группы раскладываем первыми — иначе последняя перекосит фолд
        ids = ids[np.argsort(-gsize.loc[ids].to_numpy(), kind="stable")]
        strat_load = np.zeros(n_folds, dtype=np.int64)
        for gid in ids:
            # сначала выравниваем страту, при равенстве — общий размер фолда
            f = int(np.lexsort((total_load, strat_load))[0])
            fold_of_group[int(gid)] = f
            size = int(gsize.loc[gid])
            strat_load[f] += size
            total_load[f] += size

    return df[group_col].map(fold_of_group).to_numpy()


def holdout_mask(df: pd.DataFrame, frac: float, seed: int, group_col: str = "group",
                 strat_cols: tuple = ("category", "label")) -> np.ndarray:
    """Отложенная выборка теми же правилами: доля frac, группы неделимы."""
    n_folds = max(2, int(round(1 / frac)))
    folds = stratified_group_folds(df, n_folds, seed, group_col, strat_cols)
    return folds == 0
