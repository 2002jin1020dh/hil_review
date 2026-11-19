"""Stand-in reward classifier for SYNTHETIC experiments: L2-regularized logistic regression (numpy).

The real study uses LeRobot's ResNet10-based reward classifier, trained on the GPU laptop. This stand-in
only exercises the pipeline (labels -> classifier -> threshold -> audit metrics). Like LeRobot's classifier,
it is trained with plain (unweighted) binary cross-entropy.
"""

from __future__ import annotations

import hashlib

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))


class LogisticRegressionSGD:
    def __init__(self, l2: float = 1e-3, lr: float = 0.1, epochs: int = 30, batch_size: int = 64, seed: int = 0):
        self.l2, self.lr, self.epochs, self.batch_size, self.seed = l2, lr, epochs, batch_size, seed
        self.w: np.ndarray | None = None
        self.b: float = 0.0
        self.mu: np.ndarray | None = None
        self.sd: np.ndarray | None = None
        self.degenerate = False

    def fit(self, X, y) -> "LogisticRegressionSGD":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2 or len(X) != len(y) or len(y) == 0:
            raise ValueError("X must be 2-D with len(X) == len(y) > 0")
        if not np.all((y == 0) | (y == 1)):
            raise ValueError("y must be 0/1")
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0) + 1e-8
        rng = np.random.default_rng(self.seed)
        self.w = rng.normal(0.0, 0.01, X.shape[1])
        if len(np.unique(y)) < 2:  # single-class training data: predict the constant prevalence
            self.degenerate = True
            self.w = np.zeros(X.shape[1])
            prev = float(np.clip(y.mean(), 1e-3, 1 - 1e-3))
            self.b = float(np.log(prev / (1 - prev)))
            return self
        Z = (X - self.mu) / self.sd
        n = len(y)
        for _ in range(self.epochs):
            perm = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                bi = perm[start:start + self.batch_size]
                p = _sigmoid(Z[bi] @ self.w + self.b)
                g = p - y[bi]
                self.w -= self.lr * (Z[bi].T @ g / len(bi) + self.l2 * self.w)
                self.b -= self.lr * float(g.mean())
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("model is not fitted")
        Z = (np.asarray(X, dtype=float) - self.mu) / self.sd
        return _sigmoid(Z @ self.w + self.b)

    def param_hash(self) -> str:
        """Fingerprint of the fitted parameters (used to prove audit data never influences training)."""
        if self.w is None:
            return "unfitted"
        h = hashlib.sha256()
        for arr in (self.w, np.array([self.b]), self.mu, self.sd):
            h.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
        return h.hexdigest()[:16]
