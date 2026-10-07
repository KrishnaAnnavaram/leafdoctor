"""Classifiers on cached features. Each one is a scikit-learn Pipeline, so scaling is fit on train only."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

CLASSIFIERS = ("majority", "logreg", "rf", "hgb", "xgb")
IMBALANCE_POLICIES = ("balanced", "none")


def build_classifier(name: str, seed: int) -> Pipeline:
    if name == "majority":
        clf = DummyClassifier(strategy="prior")
    elif name == "logreg":
        clf = LogisticRegression(max_iter=3000, C=0.5, random_state=seed)
    elif name == "rf":
        clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=1, n_jobs=-1, random_state=seed)
    elif name == "hgb":
        clf = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, early_stopping=False, random_state=seed)
    elif name == "xgb":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:  # pragma: no cover - depends on the optional extra
            raise ImportError("classifier 'xgb' needs the extra: pip install 'leafdoctor[xgb]'") from exc
        clf = XGBClassifier(tree_method="hist", n_estimators=400, learning_rate=0.1, max_depth=6,
                            eval_metric="mlogloss", random_state=seed, n_jobs=-1)
    else:
        raise ValueError(f"unknown classifier {name!r}; choose one of {CLASSIFIERS}")
    return Pipeline([("scale", StandardScaler()), ("clf", clf)])


def sample_weights(y: np.ndarray, policy: str) -> np.ndarray | None:
    if policy not in IMBALANCE_POLICIES:
        raise ValueError(f"imbalance policy must be one of {IMBALANCE_POLICIES}")
    return compute_sample_weight("balanced", y) if policy == "balanced" else None


@dataclass
class TrainedModel:
    """Everything that `predict` needs: the fitted pipeline, the classes and the feature recipe."""

    pipeline: Pipeline
    classes: list[str]
    extractor: str
    image_size: int
    classifier: str
    seed: int
    info: dict = field(default_factory=dict)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.classes)[self.predict_proba(X).argmax(axis=1)]

    def save(self, path: str | Path) -> Path:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: str | Path) -> "TrainedModel":
        """Load a bundle. joblib uses pickle: load only files that you made yourself."""
        import joblib

        obj = joblib.load(path)
        if not isinstance(obj, TrainedModel):
            raise TypeError(f"{path} is not a leafdoctor model bundle")
        return obj


def fit_classifier(name: str, X: np.ndarray, labels: np.ndarray, seed: int, imbalance: str,
                   extractor: str = "", image_size: int = 0) -> TrainedModel:
    enc = LabelEncoder().fit(labels)
    y = enc.transform(labels)
    pipe = build_classifier(name, seed)
    w = sample_weights(y, imbalance)
    t0 = time.perf_counter()
    if w is None:
        pipe.fit(X, y)
    else:
        pipe.fit(X, y, clf__sample_weight=w)
    info = {"fit_seconds": round(time.perf_counter() - t0, 3), "n_train": int(len(y))}
    return TrainedModel(pipe, list(enc.classes_), extractor, image_size, name, seed, info)


def align_proba(model: TrainedModel, proba: np.ndarray, classes: list[str]) -> np.ndarray:
    """Re-order model probabilities into `classes` order (zeros for classes the model never saw)."""
    out = np.zeros((proba.shape[0], len(classes)))
    pos = {c: i for i, c in enumerate(classes)}
    for j, c in enumerate(model.classes):
        if c in pos:
            out[:, pos[c]] = proba[:, j]
    return out
