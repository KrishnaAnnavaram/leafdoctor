"""Handcrafted colour and texture features, in vectorised NumPy.

- HSV histogram: 8 x 8 x 8 bins, L1-normalised (512 values).
- GLCM: grey image quantised to 32 levels, 3 distances x 4 angles. For each distance and each
  property, the mean over the angles (rotation-invariant) and the range over the angles (36 values).
- LBP: 8 neighbours at radius 1, rotation-invariant uniform codes (10 bins).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ..images import rgb_to_hsv, to_gray

GLCM_PROPS = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "asm")
ANGLE_OFFSETS = {0: (0, 1), 45: (-1, 1), 90: (-1, 0), 135: (-1, -1)}


def hsv_histogram(img: np.ndarray, bins: tuple[int, int, int] = (8, 8, 8)) -> np.ndarray:
    hsv = rgb_to_hsv(img).reshape(-1, 3)
    idx = [np.minimum((hsv[:, c] * bins[c]).astype(np.int64), bins[c] - 1) for c in range(3)]
    flat = (idx[0] * bins[1] + idx[1]) * bins[2] + idx[2]
    hist = np.bincount(flat, minlength=bins[0] * bins[1] * bins[2]).astype(np.float64)
    return hist / max(hist.sum(), 1.0)


def quantise(gray: np.ndarray, levels: int) -> np.ndarray:
    return np.minimum((gray / 256.0 * levels).astype(np.int64), levels - 1)


def glcm(q: np.ndarray, dy: int, dx: int, levels: int) -> np.ndarray:
    """Symmetric, normalised grey-level co-occurrence matrix for one offset."""
    h, w = q.shape
    y0, y1 = max(0, -dy), h - max(0, dy)
    x0, x1 = max(0, -dx), w - max(0, dx)
    a = q[y0:y1, x0:x1].ravel()
    b = q[y0 + dy:y1 + dy, x0 + dx:x1 + dx].ravel()
    m = np.bincount(a * levels + b, minlength=levels * levels).reshape(levels, levels).astype(np.float64)
    m = m + m.T
    total = m.sum()
    return m / total if total > 0 else m


def glcm_props(p: np.ndarray) -> np.ndarray:
    levels = p.shape[0]
    i, j = np.mgrid[0:levels, 0:levels].astype(np.float64)
    diff = i - j
    contrast = float((p * diff ** 2).sum())
    dissim = float((p * np.abs(diff)).sum())
    homog = float((p / (1.0 + diff ** 2)).sum())
    asm = float((p ** 2).sum())
    energy = float(np.sqrt(asm))
    mu_i, mu_j = (p * i).sum(), (p * j).sum()
    sd_i = np.sqrt((p * (i - mu_i) ** 2).sum())
    sd_j = np.sqrt((p * (j - mu_j) ** 2).sum())
    corr = float((p * (i - mu_i) * (j - mu_j)).sum() / (sd_i * sd_j)) if sd_i > 0 and sd_j > 0 else 1.0
    return np.array([contrast, dissim, homog, energy, corr, asm])


def glcm_features(img: np.ndarray, levels: int = 32, distances: Sequence[int] = (1, 2, 4)) -> np.ndarray:
    q = quantise(to_gray(img), levels)
    out = []
    for d in distances:
        per_angle = np.stack([glcm_props(glcm(q, dy * d, dx * d, levels)) for dy, dx in ANGLE_OFFSETS.values()])
        out.append(per_angle.mean(axis=0))
        out.append(per_angle.max(axis=0) - per_angle.min(axis=0))
    return np.concatenate(out)


_NEIGHBOURS = ((-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1))


def lbp_histogram(img: np.ndarray) -> np.ndarray:
    """Rotation-invariant uniform LBP (P=8, R=1): codes 0..8 plus one bin for non-uniform patterns."""
    g = to_gray(img)
    c = g[1:-1, 1:-1]
    h, w = g.shape
    bits = np.stack([(g[1 + dy:h - 1 + dy, 1 + dx:w - 1 + dx] >= c) for dy, dx in _NEIGHBOURS]).astype(np.int64)
    transitions = np.abs(bits - np.roll(bits, 1, axis=0)).sum(axis=0)
    ones = bits.sum(axis=0)
    codes = np.where(transitions <= 2, ones, 9)
    hist = np.bincount(codes.ravel(), minlength=10).astype(np.float64)
    return hist / max(hist.sum(), 1.0)


class HandcraftedExtractor:
    """HSV histogram + multi-angle GLCM + LBP for each image."""

    name = "handcrafted"

    def __init__(self, levels: int = 32, distances: Sequence[int] = (1, 2, 4), bins=(8, 8, 8)):
        self.levels = levels
        self.distances = tuple(distances)
        self.bins = tuple(bins)
        self.dim = int(np.prod(self.bins)) + len(GLCM_PROPS) * 2 * len(self.distances) + 10

    def extract_one(self, img: np.ndarray) -> np.ndarray:
        return np.concatenate([hsv_histogram(img, self.bins), glcm_features(img, self.levels, self.distances),
                               lbp_histogram(img)])

    def extract(self, images: Sequence[np.ndarray]) -> np.ndarray:
        if len(images) == 0:
            return np.zeros((0, self.dim))
        return np.stack([self.extract_one(im) for im in images])
