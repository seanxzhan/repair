"""The learned prior: a small CNN that predicts which repair template the oracle would
pick from the rasterized (member, damage) image. This amortizes the oracle -- at inference
the prior picks the template and the optimizer only fits that one template's parameters.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .templates import TEMPLATE_NAMES


class _Net(nn.Module):
    def __init__(self, in_ch: int, n_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 64
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),      # 32
            nn.Conv2d(32, 48, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(48, 64), nn.ReLU(),
                                  nn.Linear(64, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


class TorchCNNPrior:
    def __init__(self, in_ch: int = 4, classes=TEMPLATE_NAMES, seed: int = 0):
        torch.manual_seed(seed)
        self.classes = list(classes)
        self.net = _Net(in_ch, len(self.classes))
        self.history: list[float] = []

    # -- training -------------------------------------------------------------
    def fit(self, samples, epochs: int = 40, lr: float = 1e-3, batch: int = 32,
            verbose: bool = True):
        X = torch.tensor(np.stack([s.raster for s in samples]), dtype=torch.float32)
        y = torch.tensor(np.array([s.label for s in samples]), dtype=torch.long)
        # class weights counter the dutchman-heavy central-rot imbalance
        counts = torch.bincount(y, minlength=len(self.classes)).float()
        w = (counts.sum() / (counts + 1e-6))
        w = w / w.mean()
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        n = len(samples)
        for ep in range(epochs):
            perm = torch.randperm(n)
            tot = 0.0
            self.net.train()
            for i in range(0, n, batch):
                bi = perm[i:i + batch]
                opt.zero_grad()
                logits = self.net(X[bi])
                loss = F.cross_entropy(logits, y[bi], weight=w)
                loss.backward()
                opt.step()
                tot += float(loss.detach()) * len(bi)
            self.history.append(tot / n)
            if verbose and (ep % 5 == 0 or ep == epochs - 1):
                print(f"    epoch {ep:3d}  loss={self.history[-1]:.4f}")
        return self

    # -- inference ------------------------------------------------------------
    @torch.no_grad()
    def predict_proba(self, raster: np.ndarray) -> np.ndarray:
        self.net.eval()
        x = torch.tensor(raster[None], dtype=torch.float32)
        return F.softmax(self.net(x), dim=1).numpy()[0]

    def predict_template(self, raster: np.ndarray) -> str:
        return self.classes[int(self.predict_proba(raster).argmax())]

    def rank_templates(self, raster: np.ndarray) -> list[str]:
        p = self.predict_proba(raster)
        return [self.classes[i] for i in np.argsort(p)[::-1]]

    def predict_theta_init(self, raster: np.ndarray):
        return None  # v1 warm-starts from each template's default_theta

    # -- persistence ----------------------------------------------------------
    def save(self, path):
        torch.save({"state": self.net.state_dict(), "classes": self.classes,
                    "history": self.history}, path)

    @classmethod
    def load(cls, path, in_ch: int = 4):
        blob = torch.load(path, weights_only=False)
        obj = cls(in_ch=in_ch, classes=blob["classes"])
        obj.net.load_state_dict(blob["state"])
        obj.history = blob.get("history", [])
        return obj


def accuracy(prior: TorchCNNPrior, samples) -> float:
    correct = sum(prior.predict_template(s.raster) == s.label_name for s in samples)
    return correct / max(1, len(samples))
