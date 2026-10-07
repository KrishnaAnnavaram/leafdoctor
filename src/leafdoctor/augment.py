"""Train-time augmentation. It is applied to the train split only, never to val, test or field."""
from __future__ import annotations

import numpy as np


def augment(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random flip, 90-degree rotation, brightness and contrast change, and light noise."""
    out = np.rot90(img, int(rng.integers(0, 4)))
    if rng.random() < 0.5:
        out = np.fliplr(out)
    a = out.astype(np.float64)
    mean = a.mean()
    a = (a - mean) * rng.uniform(0.8, 1.2) + mean * rng.uniform(0.8, 1.2)
    a += rng.normal(0, 3.0, a.shape)
    return np.clip(a, 0, 255).astype(np.uint8)


def view_rng(seed: int, index: int, view: int) -> np.random.Generator:
    """One independent, reproducible generator for each (image, view)."""
    return np.random.default_rng([seed, index, view])
