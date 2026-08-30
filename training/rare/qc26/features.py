"""Плотная матрица признаков для классических моделей.

Деревья и бустинги не работают с разреженным tf-idf на сотни тысяч колонок, поэтому
текст сюда попадает сжатым усечённым разложением, а рядом кладутся признаки, которые
логично назвать явно: улики по правилам, находки в распознанном с фото тексте, длины
и число кадров. Явные признаки нужны ещё и для объяснений — по ним видно, на что
опиралось решение, чего из эмбеддинга не достать.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from .rules import SUPPLEMENT_MARK, SUPPLEMENT_WORD, evidence_frame

# Находки в распознанном с фото тексте. Маркировка часто печатается на упаковке и в
# описание не попадает — это ровно та слепая зона, где текст карточки бессилен.
OCR_PATTERNS = {
    "ocr_supplement_mark": SUPPLEMENT_MARK,
    "ocr_supplement_word": SUPPLEMENT_WORD,
    "ocr_dietary": re.compile(r"d\w{0,2}etary\s+su\w+", re.I),  # распознаётся с ошибками
    "ocr_hazard": re.compile(r"огнеопас|легковоспламен|взрывоопас|hazard|flammable", re.I),
    "ocr_gost": re.compile(r"\bгост\b|\bту\b\s*\d", re.I),
    "ocr_sgr": re.compile(r"свидетельств\w+\s+о\s+гос|\bсгр\b|au\.\d{2}\.", re.I),
}


def ocr_features(ocr: pd.Series | None, index) -> pd.DataFrame:
    """Признаки из распознанного текста: что найдено и сколько текста вообще есть."""
    # ⚠ Ширина блока НЕ должна зависеть от того, было ли распознавание. Пустая таблица
    # при ocr=None ломала модели, обученные с этими признаками: CatBoost падал с
    # «Feature 34 is present in model but not in pool» уже на прогоне, а линейные модели
    # молча получали матрицу другой формы. Возвращаем те же колонки, заполненные нулями.
    text = (pd.Series("", index=index, dtype=object) if ocr is None
            else ocr.reindex(index).fillna("").astype(str))
    out = {name: text.str.contains(pat, regex=True).astype(np.float32)
           for name, pat in OCR_PATTERNS.items()}
    out["ocr_len"] = np.log1p(text.str.len()).astype(np.float32)
    out["ocr_n_frames"] = text.str.count(r"\|").astype(np.float32)
    return pd.DataFrame(out, index=index)


# Тип товара по названию. EDA-9 показал, что доля позитивов внутри типа меняется от
# 45.8% (спички) до 0.1% (плиты и камины) при фоне 3.6% — это и есть граница класса,
# выраженная устойчивее отдельных слов и переносимая на другие формулировки.
PRODUCT_TYPES = [
    ("accessory_empty", r"чехол|футляр|подставк|держател|брелок|спичечниц"
                        r"|без спичек|без газа|газ не вход|не входит в комплект"),
    ("souvenir_toy", r"сувенир|прикол|игрушк|шокер|фонарик"),
    ("pyro", r"хлопушк|петард|фейерверк|салют|бенгальск|дымов|цветной дым|страйкбол|граната"),
    ("matches", r"\bспичк|\bспичек|\bспички"),
    ("lighter", r"зажигалк"),
    ("firestarter", r"розжиг|растопк|сухое горюч|таблетк\w* горюч|супер[- ]?ролл|роллы"),
    ("gas_container", r"баллон|мапп|цанговы|пропан|бутан"),
    ("flammable_liquid", r"бензин|керосин|уайт[- ]?спирит|спирт|топлив|жидкост\w* для розжиг"),
    ("charcoal", r"\bугол|\bуголь|брикет"),
    ("candle", r"свеч"),
    ("burner", r"горелк|резак|паяльн"),
    ("grill", r"мангал|гриль|барбекю|жаровн|коптильн|таган|шампур"),
    ("stove", r"плит|печь|печк|камин|котёл|котел"),
]
_TYPE_RE = [(name, re.compile(pat, re.I)) for name, pat in PRODUCT_TYPES]


def assign_product_type(df: pd.DataFrame, name_col: str = "name",
                        desc_col: str | None = None) -> pd.Series:
    """Тип товара одной колонкой — для обучения отдельных моделей по типам.

    Тип присваивается ПЕРВЫМ подошедшим правилом. Описание подмешивается по желанию:
    у части карточек тип виден только там («в комплекте угольные брикеты»), но брать
    его целиком нельзя — длинный текст цепляет чужие правила, поэтому берётся начало.
    """
    text = df[name_col].astype(str)
    if desc_col is not None:
        text = text + " " + df[desc_col].astype(str).str.slice(0, 200)
    assigned = pd.Series("other", index=df.index, dtype=object)
    unset = pd.Series(True, index=df.index)
    for label, rx in _TYPE_RE:
        hit = unset & text.str.contains(rx, na=False)
        assigned[hit] = label
        unset &= ~hit
    return assigned


def product_type_features(df: pd.DataFrame, name_col: str = "name") -> pd.DataFrame:
    """Однозначный тип товара по названию, развёрнутый в бинарные колонки.

    Тип присваивается ПЕРВЫМ подошедшим правилом, поэтому «мангал с углём» попадает в
    мангалы, а не в уголь: иначе один товар оказался бы сразу в трёх группах и признак
    перестал бы что-либо разделять.
    """
    names = df[name_col].astype(str)
    assigned = pd.Series("other", index=df.index, dtype=object)
    unset = pd.Series(True, index=df.index)
    for label, rx in _TYPE_RE:
        hit = unset & names.str.contains(rx, na=False)
        assigned[hit] = label
        unset &= ~hit
    out = pd.DataFrame(index=df.index)
    for label, _ in _TYPE_RE:
        out[f"type_{label}"] = (assigned == label).astype(np.float32)
    out["type_other"] = (assigned == "other").astype(np.float32)
    return out


def numeric_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Простые числовые признаки карточки."""
    name = df[cfg["data"]["name_col"]].astype(str)
    desc = df[cfg["data"]["desc_col"]].astype(str)
    out = pd.DataFrame(index=df.index)
    out["name_len"] = np.log1p(name.str.len()).astype(np.float32)
    out["desc_len"] = np.log1p(desc.str.len()).astype(np.float32)
    out["name_words"] = np.log1p(name.str.count(r"\s+") + 1).astype(np.float32)
    out["desc_empty"] = (desc.str.len() < 5).astype(np.float32)
    out["digits_in_name"] = name.str.count(r"\d").astype(np.float32)
    out["upper_share"] = (name.str.count(r"[А-ЯA-Z]")
                          / name.str.len().clip(lower=1)).astype(np.float32)
    if "n_images" in df.columns:
        out["n_images"] = df["n_images"].astype(np.float32)
    return out


class DenseFeatureBuilder:
    """Собирает плотную матрицу: эмбеддинги + сжатый текст + явные признаки.

    Усечённое разложение обучается ТОЛЬКО на обучающей части фолда — иначе информация
    о валидации протекает в представление и оценка завышается.
    """

    def __init__(self, cfg: dict, use_text: bool = True, use_ocr_text: bool = True,
                 svd_text: int = 128, svd_ocr: int = 64, svd_emb: int | None = 256):
        self.cfg = cfg
        self.use_text = use_text
        self.use_ocr_text = use_ocr_text
        self.svd_text = svd_text
        self.svd_ocr = svd_ocr
        # Эмбеддинг из 2048 чисел деревьям только мешает: они перебирают признаки по
        # одному, и сотни почти одинаковых координат раздувают время без пользы.
        self.svd_emb = svd_emb
        self.text_vec: TfidfVectorizer | None = None
        self.text_svd: TruncatedSVD | None = None
        self.ocr_vec: TfidfVectorizer | None = None
        self.ocr_svd: TruncatedSVD | None = None
        self.emb_svd: TruncatedSVD | None = None
        self.columns: list[str] = []

    @staticmethod
    def _norm(s: pd.Series) -> pd.Series:
        return s.astype(str).str.lower().str.replace("ё", "е", regex=False)

    def _card_text(self, df: pd.DataFrame) -> pd.Series:
        c = self.cfg["data"]
        return self._norm(df[c["name_col"]] + " \n "
                          + df[c["desc_col"]].astype(str).str.slice(0, 2500))

    def fit_transform(self, df: pd.DataFrame, emb: np.ndarray | None,
                      ocr: pd.Series | None) -> np.ndarray:
        blocks, names = [], []
        if emb is not None and emb.size:
            if self.svd_emb and emb.shape[1] > self.svd_emb:
                self.emb_svd = TruncatedSVD(n_components=self.svd_emb, random_state=0)
                block = self.emb_svd.fit_transform(emb).astype(np.float32)
            else:
                block = emb.astype(np.float32)
            blocks.append(block)
            names += [f"emb_{i}" for i in range(block.shape[1])]

        if self.use_text:
            self.text_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                                            min_df=3, max_features=120000, sublinear_tf=True)
            x = self.text_vec.fit_transform(self._card_text(df))
            k = min(self.svd_text, max(2, x.shape[1] - 1))
            self.text_svd = TruncatedSVD(n_components=k, random_state=0)
            blocks.append(self.text_svd.fit_transform(x).astype(np.float32))
            names += [f"txt_{i}" for i in range(k)]

        if self.use_ocr_text and ocr is not None:
            self.ocr_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                                           min_df=3, max_features=60000, sublinear_tf=True)
            x = self.ocr_vec.fit_transform(self._norm(ocr.reindex(df.index).fillna("")))
            k = min(self.svd_ocr, max(2, x.shape[1] - 1))
            self.ocr_svd = TruncatedSVD(n_components=k, random_state=0)
            blocks.append(self.ocr_svd.fit_transform(x).astype(np.float32))
            names += [f"ocrtxt_{i}" for i in range(k)]

        extra = self._explicit(df, ocr)
        blocks.append(extra.to_numpy(dtype=np.float32))
        names += list(extra.columns)

        self.columns = names
        return np.hstack(blocks)

    def transform(self, df: pd.DataFrame, emb: np.ndarray | None,
                  ocr: pd.Series | None) -> np.ndarray:
        blocks = []
        if emb is not None and emb.size:
            blocks.append(self.emb_svd.transform(emb).astype(np.float32)
                          if self.emb_svd is not None else emb.astype(np.float32))
        if self.use_text and self.text_vec is not None:
            blocks.append(self.text_svd.transform(
                self.text_vec.transform(self._card_text(df))).astype(np.float32))
        if self.use_ocr_text and self.ocr_vec is not None and ocr is not None:
            blocks.append(self.ocr_svd.transform(self.ocr_vec.transform(
                self._norm(ocr.reindex(df.index).fillna("")))).astype(np.float32))
        blocks.append(self._explicit(df, ocr).to_numpy(dtype=np.float32))
        return np.hstack(blocks)

    def _explicit(self, df: pd.DataFrame, ocr: pd.Series | None) -> pd.DataFrame:
        ev = evidence_frame(df, ocr).astype(np.float32)
        num = numeric_features(df, self.cfg)
        oc = ocr_features(ocr, df.index)
        types = product_type_features(df, self.cfg["data"]["name_col"])
        return pd.concat([ev, num, oc, types], axis=1).fillna(0.0)


def sparse_text_matrix(cfg: dict, df: pd.DataFrame, ocr: pd.Series | None,
                       vectorizers: dict | None = None) -> tuple:
    """Разреженный tf-idf для линейных моделей (им плотное сжатие только вредит)."""
    c = cfg["data"]
    text = (df[c["name_col"]].astype(str) + " \n "
            + df[c["desc_col"]].astype(str).str.slice(0, 2500))
    if ocr is not None:
        text = text + " \n НА ФОТО: " + ocr.reindex(df.index).fillna("").astype(str)
    text = text.str.lower().str.replace("ё", "е", regex=False)

    fit = vectorizers is None
    if fit:
        vectorizers = {
            "word": TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                                    max_features=200000, sublinear_tf=True),
            "char": TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                                    max_features=200000, sublinear_tf=True),
        }
        blocks = [v.fit_transform(text) for v in vectorizers.values()]
    else:
        blocks = [v.transform(text) for v in vectorizers.values()]
    extra = pd.concat([evidence_frame(df, ocr),
                       product_type_features(df, c["name_col"])], axis=1)
    ev = sparse.csr_matrix(extra.to_numpy(dtype=np.float32))
    return sparse.hstack(blocks + [ev]).tocsr(), vectorizers
