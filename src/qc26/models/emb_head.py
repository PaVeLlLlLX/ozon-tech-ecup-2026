"""Гейтовая голова поверх мультимодальных эмбеддингов: обучение и инференс.

Замерено (docs/emb_head_results.local.md): в категории «Легковоспламеняющиеся»
смесь с tf-idf даёт +7.5…+16.2 п.п. F1 при весе 0.45, на БАДах прироста нет.
Решающим оказался не вид нелинейности (reglu, geglu, обычный MLP различаются в
пределах шума), а сам факт нелинейности: линейная голова даёт 0.42 PR-AUC против
0.60+ у любой нелинейной.

Ансамбль вместо одной модели. Обучаем K голов, каждая на своих 4/5 данных, с ранней
остановкой по отложенному фолду, и усредняем вероятности. Это ровно то, что мерилось
на OOF: одна голова на всех данных не имеет честной точки остановки, а при 158
позитивах переобучение наступает быстро.

Артефакт кладётся в архив решения (единицы мегабайт) — веса энкодера в архив НЕ идут,
он берётся из /shared_models.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

GATES = {"reglu": F.relu, "geglu": F.gelu, "swiglu": F.silu}

class GatedHead(nn.Module):
    """LayerNorm → гейт → dropout → логит.

    Нормировка входа обязательна: без неё гейт вырождается, одна из проекций
    уводит произведение в ноль.

    ⚠ НЕ УДАЛЯТЬ. Этой головой обучен `artifacts/models/emb_head.pt` — артефакт
    отправки `tfidf_emb_blend`, публично 0.7619, наш лучший результат. Когда класс
    заменили на вариационный (эксперименты с контрастными потерями, все замеры
    которых провалили порог), артефакт перестал загружаться: в state_dict лежат
    `w_gate.weight/bias`, а вариационный класс ждёт `w_gate_mu/w_gate_logvar`.
    Сборка отправки при этом уходила на запасной текстовый путь. Поймано тестом
    `test_emb_head_artifact_sane`.
    """

    def __init__(self, d_in: int, d_hidden: int = 256, gate: str = "reglu",
                 dropout: float = 0.2):
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        self.w_gate = nn.Linear(d_in, d_hidden)
        self.w_value = nn.Linear(d_in, d_hidden)
        self.act = GATES[gate]
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(d_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.act(self.w_gate(h)) * self.w_value(h)
        return self.out(self.drop(h)).squeeze(-1)


def head_for_state(state, d_in, d_hidden, gate, dropout):
    """Класс головы ВЫВОДИТСЯ из сохранённых весов, а не задаётся снаружи.

    Иначе смена класса в коде молча ломает старые артефакты: загрузка падает, а
    сборка отправки уходит на запасной путь и отдаёт правдоподобные, но чужие
    числа. Ровно это и произошло между 15 и 18 августа.
    """
    if "w_gate_mu.weight" in state:
        return VariationalGatedHead(d_in, d_hidden, gate, dropout)
    if "w_gate.weight" in state:
        return GatedHead(d_in, d_hidden, gate, dropout)
    raise RuntimeError(
        f"неизвестная раскладка весов головы: {sorted(state)[:6]}. "
        f"Ожидались w_gate.* (GatedHead) или w_gate_mu.* (VariationalGatedHead)")


class VariationalGatedHead(nn.Module):
    """LayerNorm → Вариационный гейт (Mu/LogVar) → Dropout → Логит.
    
    Использует трюк репараметризации для мягкого сжатия признаков Qwen,
    чтобы убрать шум и выделить скрытые факторы риска.
    """
    def __init__(self, d_in: int, d_hidden: int = 256, gate: str = "reglu", dropout: float = 0.2):
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        
        # Проекции для гейта (распределения)
        self.w_gate_mu = nn.Linear(d_in, d_hidden)
        self.w_gate_logvar = nn.Linear(d_in, d_hidden)
        
        # Проекция для значений (Value)
        self.w_value = nn.Linear(d_in, d_hidden)
        
        self.act = GATES[gate]
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(d_hidden, 1)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.norm(x)
        
        # Вычисляем параметры распределения для гейта
        mu = self.w_gate_mu(h)
        logvar = self.w_gate_logvar(h)
        
        # Сэмплируем активированный гейт
        sampled_gate = self.reparameterize(mu, logvar)
        activated_gate = self.act(sampled_gate)
        
        # Гейтирование
        gated_features = activated_gate * self.w_value(h)
        
        logits = self.out(self.drop(gated_features)).squeeze(-1)
        return logits, mu, logvar


def _fit_one(xtr, ytr, xva, yva, *, d_hidden, gate, dropout, epochs, lr,
             weight_decay, seed, device, patience=8, max_kl_weight=1e-3):
    """Улучшенная голова с циклическим отжигом KL и стабилизацией батчей."""
    from sklearn.metrics import roc_auc_score
    import numpy as np

    torch.manual_seed(seed)
    model = VariationalGatedHead(xtr.shape[1], d_hidden, gate, dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Расчет веса для компенсации дисбаланса
    pos = float(ytr.sum())
    pw = torch.tensor([(len(ytr) - pos) / max(pos, 1.0)], device=device)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)

    xtr_t = torch.tensor(xtr, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=device)
    xva_t = torch.tensor(xva, device=device)

    # Разделяем индексы для сбалансированного батчевания
    pos_idx = np.where(ytr == 1)[0]
    neg_idx = np.where(ytr == 0)[0]

    best_auc, best_state, waited = -1.0, None, 0
    batch_size = 256
    
    # Сколько позитивных примеров должно быть в каждом батче (сохраняем пропорцию ~1:27)
    pos_per_batch = max(1, int(batch_size * (len(pos_idx) / len(ytr)))) 
    neg_per_batch = batch_size - pos_per_batch

    for epoch in range(epochs):
        model.train()
        
        # --- ХАК 1: Расчет текущего веса KL (Циклический отжиг) ---
        # Первые 20% эпох KL-лосс равен 0, затем плавно растет до max_kl_weight
        if epoch < int(epochs * 0.2):
            kl_weight = 0.0
        else:
            progress = (epoch - int(epochs * 0.2)) / (epochs * 0.8)
            kl_weight = max_kl_weight * min(1.0, progress)
            
        # Перемешиваем индексы классов перед эпохой
        np.random.shuffle(pos_idx)
        np.random.shuffle(neg_idx)
        
        # Считаем, сколько шагов нужно сделать, чтобы пройтись по всем негативным примерам
        steps = len(neg_idx) // neg_per_batch
        
        for step in range(steps):
            # Набираем стабильный батч из позитивных и негативных примеров
            n_idx = neg_idx[step * neg_per_batch : (step + 1) * neg_per_batch]
            # Позитивные зацикливаем, если они кончились
            p_idx = pos_idx[(step * pos_per_batch) % len(pos_idx) : (step * pos_per_batch) % len(pos_idx) + pos_per_batch]
            
            # Объединяем и переносим в тензор индексов
            idx = torch.tensor(np.concatenate([n_idx, p_idx]), device=device)
            
            opt.zero_grad()
            logits, mu, logvar = model(xtr_t[idx])
            
            # Лосс классификации
            bce_loss = lossf(logits, ytr_t[idx])
            
            # Вариационный лосс
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
            
            # Общий динамический лосс
            total_loss = bce_loss + (kl_weight * kl_loss)
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            
        # Валидация
        model.eval()
        with torch.no_grad():
            va, _, _ = model(xva_t)
            va = va.cpu().numpy()
            
        if not np.isfinite(va).all():
            break
            
        auc = roc_auc_score(yva, va) if len(np.unique(yva)) > 1 else 0.5
        
        if auc > best_auc:
            best_auc, waited = auc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= patience:
                break
                
    return best_state, best_auc


class CategoryEmbHead:
    """Полностью сохраняет оригинальный API для интеграции в fit_emb_head_final.py"""
    def __init__(self, d_in: int, d_hidden: int = 256, gate: str = "reglu", dropout: float = 0.2):
        self.d_in, self.d_hidden, self.gate, self.dropout = d_in, d_hidden, gate, dropout
        self.members: list[dict] = []

    def fit(self, x: np.ndarray, y: np.ndarray, folds: np.ndarray, *, seeds=(0, 1, 2),
            epochs=60, lr=3e-4, weight_decay=1e-2, device="cpu") -> "CategoryEmbHead":
        uniq = sorted(np.unique(folds))
        for k in uniq:
            va = folds == k
            tr = ~va
            mu, sd = x[tr].mean(0), x[tr].std(0) + 1e-6
            xs = ((x - mu) / sd).astype(np.float32)
            for s in seeds:
                state, auc = _fit_one(
                    xs[tr], y[tr], xs[va], y[va], d_hidden=self.d_hidden, gate=self.gate,
                    dropout=self.dropout, epochs=epochs, lr=lr,
                    weight_decay=weight_decay, seed=s, device=device, kl_weight=1e-4)
                if state is None:
                    continue
                self.members.append({"state": state, "mean": mu.astype(np.float32),
                                     "std": sd.astype(np.float32), "auc": float(auc)})
        if not self.members:
            raise RuntimeError("ни одна голова не обучилась")
        return self

    @torch.no_grad()
    def predict_proba(self, x: np.ndarray, device: str = "cpu", batch: int = 512) -> np.ndarray:
        acc = np.zeros(len(x), dtype=np.float64)
        for m in self.members:
            model = head_for_state(m["state"], self.d_in, self.d_hidden,
                                   self.gate, self.dropout).to(device)
            model.load_state_dict(m["state"])
            model.eval()
            xs = ((x - m["mean"]) / m["std"]).astype(np.float32)
            out = np.empty(len(xs), dtype=np.float32)
            for i in range(0, len(xs), batch):
                chunk = torch.tensor(xs[i:i + batch], device=device)
                res = model(chunk)
                # вариационная голова отдаёт (логит, mu, logvar), обычная — логит
                logits = res[0] if isinstance(res, tuple) else res
                out[i:i + batch] = torch.sigmoid(logits).cpu().numpy()
            acc += out
        return acc / len(self.members)


class EmbHeadBundle:
    """Головы по категориям + веса смеси + пороги. Сохраняется одним файлом."""

    def __init__(self):
        self.heads: dict[str, CategoryEmbHead] = {}
        self.weights: dict[str, float] = {}      # вес эмбеддингов в смеси
        self.thresholds: dict[str, float] = {}

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        blob = {
            "weights": self.weights, "thresholds": self.thresholds,
            "heads": {cat: {"d_in": h.d_in, "d_hidden": h.d_hidden, "gate": h.gate,
                            "dropout": h.dropout, "members": h.members}
                      for cat, h in self.heads.items()},
        }
        torch.save(blob, path)

    @staticmethod
    def load(path) -> "EmbHeadBundle":
        # ⚠ weights_only=False: внутри numpy-массивы нормировки, а не только тензоры
        blob = torch.load(Path(path), map_location="cpu", weights_only=False)
        b = EmbHeadBundle()
        b.weights = blob.get("weights", {})
        b.thresholds = blob.get("thresholds", {})
        for cat, h in blob.get("heads", {}).items():
            head = CategoryEmbHead(h["d_in"], h["d_hidden"], h["gate"], h["dropout"])
            head.members = h["members"]
            b.heads[cat] = head
        return b
