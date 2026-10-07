"""Batched feature extraction with an on-disk cache.

Images are read and featurised `batch_size` at a time, so memory stays flat for large splits.
The result for each (split rows, extractor, image size, augmentation views, seed) is saved once as
`.npz`. The cache key is a hash of these inputs, so a changed manifest gives a new file.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from ..augment import augment, view_rng
from ..images import load_image
from ..manifest import Row


@dataclass
class FeatureSet:
    X: np.ndarray
    labels: np.ndarray
    paths: np.ndarray
    groups: np.ndarray

    def __len__(self) -> int:
        return int(self.X.shape[0])


def cache_key(rows: Sequence[Row], extractor_name: str, image_size: int, views: int, seed: int) -> str:
    h = hashlib.sha1()
    h.update(json.dumps([extractor_name, image_size, views, seed]).encode())
    for r in rows:
        p = Path(r.path)
        stat = p.stat() if p.exists() else None
        h.update(f"{r.path}|{r.label}|{stat.st_size if stat else -1}|{int(stat.st_mtime) if stat else -1}\n".encode())
    return h.hexdigest()[:16]


class FeatureStore:
    def __init__(self, cache_dir: str | Path | None, batch_size: int = 32,
                 loader: Callable[[str, int], np.ndarray] = lambda p, s: load_image(p, s)):
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.batch_size = batch_size
        self.loader = loader
        self.hits = 0
        self.misses = 0

    def features(self, rows: Sequence[Row], extractor, split_name: str, image_size: int,
                 train_views: int = 0, seed: int = 0) -> FeatureSet:
        """Features for each row. With train_views > 0, add that many augmented views for each image."""
        key = cache_key(rows, extractor.name, image_size, train_views, seed)
        path = None
        if self.cache_dir is not None:
            safe = extractor.name.replace(":", "_").replace("+", "-")
            path = self.cache_dir / f"{split_name}__{safe}__{key}.npz"
            if path.exists():
                self.hits += 1
                d = np.load(path, allow_pickle=False)
                return FeatureSet(d["X"], d["labels"], d["paths"], d["groups"])
        self.misses += 1
        blocks: list[np.ndarray] = []
        labels: list[str] = []
        paths: list[str] = []
        groups: list[str] = []
        for start in range(0, len(rows), self.batch_size):
            chunk = rows[start:start + self.batch_size]
            imgs = []
            for offset, r in enumerate(chunk):
                img = self.loader(r.path, image_size)
                variants = [img] + [augment(img, view_rng(seed, start + offset, v)) for v in range(1, train_views + 1)]
                imgs.extend(variants)
                labels.extend([r.label] * len(variants))
                paths.extend([r.path] * len(variants))
                groups.extend([r.group_id] * len(variants))
            for b in range(0, len(imgs), self.batch_size):
                blocks.append(np.asarray(extractor.extract(imgs[b:b + self.batch_size]), dtype=np.float32))
        X = np.concatenate(blocks) if blocks else np.zeros((0, extractor.dim), dtype=np.float32)
        fs = FeatureSet(X, np.array(labels, dtype=str), np.array(paths, dtype=str), np.array(groups, dtype=str))
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, X=fs.X, labels=fs.labels, paths=fs.paths, groups=fs.groups)
            tmp.replace(path)
        return fs
