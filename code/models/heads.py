"""Classifier heads fitted on cached frozen embeddings.

All heads expose the same ``fit``/``predict_proba`` surface so the protocol
comparison holds the classifier fixed while the splitter varies.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-8)


class LinearHead:
    """L2-regularised multinomial logistic regression, the standard probe for
    judging a frozen representation."""

    name = "linear"

    def __init__(self, C: float = 1.0, class_weight: str | None = "balanced",
                 seed: int = 0, max_iter: int = 3000):
        self.C, self.class_weight, self.seed, self.max_iter = C, class_weight, seed, max_iter

    def fit(self, X, y):
        self.clf_ = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=self.C, max_iter=self.max_iter,
                               class_weight=self.class_weight,
                               random_state=self.seed),
        ).fit(X, y)
        self.classes_ = self.clf_.classes_
        return self

    def predict_proba(self, X):
        return self.clf_.predict_proba(X)


class MLPHead:
    """One hidden layer with dropout; fitted in torch so class weights and
    early stopping are explicit."""

    name = "mlp"

    def __init__(self, hidden: int = 256, epochs: int = 200, lr: float = 1e-3,
                 wd: float = 1e-4, dropout: float = 0.3, seed: int = 0,
                 class_weight: bool = True):
        self.__dict__.update(hidden=hidden, epochs=epochs, lr=lr, wd=wd,
                             dropout=dropout, seed=seed, class_weight=class_weight)

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        self.classes_ = np.unique(y)
        yi = np.searchsorted(self.classes_, y)
        self.mu_, self.sd_ = X.mean(0), X.std(0) + 1e-6
        Xt = torch.tensor((X - self.mu_) / self.sd_, dtype=torch.float32)
        yt = torch.tensor(yi, dtype=torch.long)

        self.net_ = nn.Sequential(
            nn.Linear(X.shape[1], self.hidden), nn.BatchNorm1d(self.hidden),
            nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(self.hidden, len(self.classes_)),
        )
        w = None
        if self.class_weight:
            cnt = np.bincount(yi, minlength=len(self.classes_)).astype(float)
            w = torch.tensor(len(yi) / (len(self.classes_) * np.maximum(cnt, 1)),
                             dtype=torch.float32)
        opt = torch.optim.AdamW(self.net_.parameters(), lr=self.lr, weight_decay=self.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.epochs)
        lossf = nn.CrossEntropyLoss(weight=w)
        self.net_.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            lossf(self.net_(Xt), yt).backward()
            opt.step()
            sched.step()
        self.net_.eval()
        return self

    def predict_proba(self, X):
        with torch.no_grad():
            Xt = torch.tensor((X - self.mu_) / self.sd_, dtype=torch.float32)
            return torch.softmax(self.net_(Xt), dim=1).numpy()


class _GatedAttention(nn.Module):
    """Gated attention pooling (Ilse et al., 2018), the aggregation the dataset
    descriptor's MIL baseline used, here over the full tissue tile grid rather
    than 20 random crops."""

    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.V = nn.Linear(dim, hidden)
        self.U = nn.Linear(dim, hidden)
        self.w = nn.Linear(hidden, 1)

    def forward(self, h, mask):
        a = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))).squeeze(-1)
        a = a.masked_fill(~mask, float("-inf"))
        a = torch.softmax(a, dim=-1)
        return torch.einsum("bn,bnd->bd", a, h), a


class AttentionMILHead:
    """Attention-MIL over per-image tile bags of cached embeddings."""

    name = "attnmil"

    def __init__(self, proj: int = 256, hidden: int = 128, epochs: int = 120,
                 lr: float = 1e-3, wd: float = 1e-4, dropout: float = 0.25,
                 seed: int = 0, batch: int = 32, class_weight: bool = True):
        self.__dict__.update(proj=proj, hidden=hidden, epochs=epochs, lr=lr, wd=wd,
                             dropout=dropout, seed=seed, batch=batch,
                             class_weight=class_weight)

    @staticmethod
    def _pad(bags: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
        n = max(len(b) for b in bags)
        d = bags[0].shape[1]
        H = torch.zeros(len(bags), n, d)
        M = torch.zeros(len(bags), n, dtype=torch.bool)
        for i, b in enumerate(bags):
            H[i, : len(b)] = torch.tensor(b, dtype=torch.float32)
            M[i, : len(b)] = True
        return H, M

    def fit(self, bags: list[np.ndarray], y):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.classes_ = np.unique(y)
        yi = np.searchsorted(self.classes_, y)
        allv = np.concatenate(bags, axis=0)
        self.mu_, self.sd_ = allv.mean(0), allv.std(0) + 1e-6
        bags = [(b - self.mu_) / self.sd_ for b in bags]
        d = bags[0].shape[1]

        self.enc_ = nn.Sequential(nn.Linear(d, self.proj), nn.ReLU(), nn.Dropout(self.dropout))
        self.att_ = _GatedAttention(self.proj, self.hidden)
        self.cls_ = nn.Linear(self.proj, len(self.classes_))
        params = list(self.enc_.parameters()) + list(self.att_.parameters()) + list(self.cls_.parameters())

        w = None
        if self.class_weight:
            cnt = np.bincount(yi, minlength=len(self.classes_)).astype(float)
            w = torch.tensor(len(yi) / (len(self.classes_) * np.maximum(cnt, 1)),
                             dtype=torch.float32)
        opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.epochs)
        lossf = nn.CrossEntropyLoss(weight=w)

        for _ in range(self.epochs):
            order = rng.permutation(len(bags))
            for s in range(0, len(order), self.batch):
                sel = order[s : s + self.batch]
                H, M = self._pad([bags[i] for i in sel])
                logits = self.cls_(self.att_(self.enc_(H), M)[0])
                opt.zero_grad()
                lossf(logits, torch.tensor(yi[sel], dtype=torch.long)).backward()
                opt.step()
            sched.step()
        for m in (self.enc_, self.att_, self.cls_):
            m.eval()
        return self

    def predict_proba(self, bags: list[np.ndarray], return_attention: bool = False):
        bags = [(b - self.mu_) / self.sd_ for b in bags]
        out, atts = [], []
        with torch.no_grad():
            for s in range(0, len(bags), self.batch):
                H, M = self._pad(bags[s : s + self.batch])
                z, a = self.att_(self.enc_(H), M)
                out.append(torch.softmax(self.cls_(z), dim=1).numpy())
                if return_attention:
                    for i in range(len(a)):
                        atts.append(a[i][M[i]].numpy())
        proba = np.concatenate(out, axis=0)
        return (proba, atts) if return_attention else proba


HEADS = {h.name: h for h in (LinearHead, MLPHead, AttentionMILHead)}
