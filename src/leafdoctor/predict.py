"""Batch inference with a saved model bundle."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .features.backbones import build_extractor
from .images import load_image
from .models import TrainedModel


@dataclass
class Prediction:
    path: str
    label: str
    confidence: float
    top3: list[tuple[str, float]]


def predict_paths(model: TrainedModel, paths: Sequence[str | Path], batch_size: int = 32,
                  device: str = "cpu") -> list[Prediction]:
    extractor = build_extractor(model.extractor, device=device, seed=model.seed)
    out: list[Prediction] = []
    for i in range(0, len(paths), batch_size):
        chunk = [str(p) for p in paths[i:i + batch_size]]
        X = extractor.extract([load_image(p, model.image_size) for p in chunk])
        proba = model.predict_proba(X)
        for p, row in zip(chunk, proba):
            order = np.argsort(row)[::-1][:3]
            out.append(Prediction(p, model.classes[order[0]], float(row[order[0]]),
                                  [(model.classes[j], round(float(row[j]), 4)) for j in order]))
    return out
