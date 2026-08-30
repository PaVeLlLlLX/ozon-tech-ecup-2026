"""Накопление текстового отчёта EDA: каждый блок пишет свой markdown-файл.

Результаты нужны в читаемом виде и без ноутбука — скрипты гоняются автономно, а
выводы потом переносятся в рабочие заметки.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd


class Report:
    def __init__(self, title: str, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lines: list[str] = [f"# {title}", "",
                                 f"_Сформировано {datetime.now():%d.%m.%Y %H:%M}_", ""]

    def h(self, text: str, level: int = 2) -> None:
        self.lines += ["", "#" * level + " " + text, ""]

    def p(self, text: str) -> None:
        self.lines += [text, ""]

    def kv(self, pairs: dict) -> None:
        for k, v in pairs.items():
            self.lines.append(f"- **{k}**: {v}")
        self.lines.append("")

    def table(self, df: pd.DataFrame, floatfmt: str = "{:.3f}", index: bool = False) -> None:
        show = df.copy()
        for c in show.columns:
            if pd.api.types.is_float_dtype(show[c]):
                show[c] = show[c].map(lambda x: "—" if pd.isna(x) else floatfmt.format(x))
        cols = ([show.index.name or ""] if index else []) + [str(c) for c in show.columns]
        self.lines.append("| " + " | ".join(cols) + " |")
        self.lines.append("|" + "|".join(["---"] * len(cols)) + "|")
        for idx, row in show.iterrows():
            cells = ([str(idx)] if index else []) + [str(v) for v in row.tolist()]
            self.lines.append("| " + " | ".join(cells) + " |")
        self.lines.append("")

    def code(self, text: str) -> None:
        self.lines += ["```", text.rstrip(), "```", ""]

    def save(self) -> Path:
        self.path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")
        print(f"отчёт: {self.path}", flush=True)
        return self.path
