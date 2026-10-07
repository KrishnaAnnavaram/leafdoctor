"""Evaluation metrics: macro-F1 with a bootstrap interval, per-class recall, calibration, confusion."""
from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, log_loss,
                             precision_recall_fscore_support)


def expected_calibration_error(y_true_idx: np.ndarray, proba: np.ndarray, n_bins: int = 15) -> float:
    """Top-label ECE: the weighted gap between confidence and accuracy over equal-width bins."""
    if len(y_true_idx) == 0:
        return 0.0
    conf = proba.max(axis=1)
    correct = (proba.argmax(axis=1) == y_true_idx).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def bootstrap_macro_f1(y_true_idx, y_pred_idx, labels, n_boot: int = 200, seed: int = 0) -> tuple[float, float]:
    """95% percentile interval of macro-F1 over bootstrap resamples of the evaluation images."""
    n = len(y_true_idx)
    if n == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        scores.append(f1_score(y_true_idx[idx], y_pred_idx[idx], labels=labels, average="macro", zero_division=0))
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return (float(lo), float(hi))


@dataclass
class EvalReport:
    split: str
    n: int
    accuracy: float
    macro_f1: float
    macro_f1_ci: tuple[float, float]
    balanced_accuracy: float
    log_loss: float
    ece: float
    per_class: dict[str, dict[str, float]]
    confusion: list[list[float]]
    classes: list[str]

    def headline(self) -> dict[str, float]:
        return {"accuracy": self.accuracy, "macro_f1": self.macro_f1, "balanced_accuracy": self.balanced_accuracy,
                "log_loss": self.log_loss, "ece": self.ece}

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate(split: str, true_labels, proba: np.ndarray, classes: list[str], n_boot: int = 200,
             seed: int = 0) -> EvalReport:
    """Score probabilities (columns in `classes` order) against true labels."""
    with warnings.catch_warnings():
        # small or one-class evaluation sets make scikit-learn warn; the numbers are still defined
        warnings.simplefilter("ignore", UserWarning)
        return _evaluate(split, true_labels, proba, classes, n_boot, seed)


def _evaluate(split, true_labels, proba, classes, n_boot, seed) -> EvalReport:
    pos = {c: i for i, c in enumerate(classes)}
    unknown = sorted({t for t in true_labels if t not in pos})
    if unknown:
        raise ValueError(f"labels not in the class list: {unknown}")
    y = np.array([pos[t] for t in true_labels], dtype=np.int64)
    proba = np.asarray(proba, dtype=np.float64)
    if proba.shape != (len(y), len(classes)):
        raise ValueError(f"proba shape {proba.shape} does not match ({len(y)}, {len(classes)})")
    pred = proba.argmax(axis=1)
    labels = list(range(len(classes)))
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y, pred, labels=labels).astype(np.float64)
    rows = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, rows, out=np.zeros_like(cm), where=rows > 0)
    clipped = np.clip(proba, 1e-12, 1.0)
    clipped = clipped / clipped.sum(axis=1, keepdims=True)
    return EvalReport(
        split=split,
        n=int(len(y)),
        accuracy=float(accuracy_score(y, pred)),
        macro_f1=float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
        macro_f1_ci=bootstrap_macro_f1(y, pred, labels, n_boot=n_boot, seed=seed),
        balanced_accuracy=float(balanced_accuracy_score(y, pred)) if len(y) else 0.0,
        log_loss=float(log_loss(y, clipped, labels=labels)) if len(y) else 0.0,
        ece=expected_calibration_error(y, proba),
        per_class={c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                   for i, c in enumerate(classes)},
        confusion=cm_norm.round(4).tolist(),
        classes=list(classes),
    )
