"""Synthetic leaf images for the offline demo and the tests.

Each synthetic leaf is an ellipse with veins. Each class adds its own lesion pattern (colour and
texture). A leaf gets several "views" (flips, rotations, brightness changes) with the same leaf id,
which copies the near-duplicate structure of PlantVillage and its augmented derivative.
The `field` style adds a cluttered background and uneven light, to imitate a domain shift.
These images are not plant photographs. Results on them only prove that the pipeline works.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from .images import save_image

SYNTHETIC_CLASSES: tuple[str, ...] = (
    "Tomato___healthy",
    "Tomato___Early_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Tomato_mosaic_virus",
    "Potato___Late_blight",
    "Corn_(maize)___Common_rust_",
)


def _grid(size: int) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    return yy, xx


def _ellipse_mask(size, cy, cx, ry, rx, angle) -> np.ndarray:
    yy, xx = _grid(size)
    ca, sa = np.cos(angle), np.sin(angle)
    dy, dx = yy - cy, xx - cx
    u = (dx * ca + dy * sa) / rx
    v = (-dx * sa + dy * ca) / ry
    return (u * u + v * v) <= 1.0


def _paint(img: np.ndarray, mask: np.ndarray, color, alpha: float = 1.0) -> None:
    c = np.asarray(color, dtype=np.float64)
    img[mask] = (1 - alpha) * img[mask] + alpha * c


def _spots(img, leaf, rng, n_range, r_range, color, ring=None, jitter=18.0):
    size = img.shape[0]
    ys, xs = np.nonzero(leaf)
    if len(ys) == 0:
        return
    for _ in range(int(rng.integers(*n_range))):
        k = int(rng.integers(len(ys)))
        r = float(rng.uniform(*r_range)) * size
        col = np.clip(np.asarray(color) + rng.normal(0, jitter, 3), 0, 255)
        m = _ellipse_mask(size, ys[k], xs[k], r, r * rng.uniform(0.7, 1.3), rng.uniform(0, np.pi)) & leaf
        _paint(img, m, col)
        if ring is not None:
            inner = _ellipse_mask(size, ys[k], xs[k], r * 0.5, r * 0.5, 0.0) & leaf
            _paint(img, inner, ring)


def _healthy(img, leaf, rng):
    return None


def _early_blight(img, leaf, rng):
    _spots(img, leaf, rng, (3, 7), (0.04, 0.08), (120, 80, 40), ring=(70, 45, 25))


def _leaf_mold(img, leaf, rng):
    size = img.shape[0]
    noise = rng.random((size // 8 + 1, size // 8 + 1))
    big = np.kron(noise, np.ones((8, 8)))[:size, :size]
    m = leaf & (big > 0.6)
    _paint(img, m, (185, 175, 70), alpha=0.75)


def _mosaic(img, leaf, rng):
    size = img.shape[0]
    cell = int(rng.integers(3, 6))
    noise = rng.random((size // cell + 1, size // cell + 1))
    big = np.kron(noise, np.ones((cell, cell)))[:size, :size]
    _paint(img, leaf & (big > 0.5), (150, 200, 90), alpha=0.7)
    _paint(img, leaf & (big < 0.2), (30, 90, 30), alpha=0.7)


def _late_blight(img, leaf, rng):
    _spots(img, leaf, rng, (1, 3), (0.12, 0.2), (55, 45, 35), jitter=10.0)


def _rust(img, leaf, rng):
    _spots(img, leaf, rng, (25, 45), (0.01, 0.02), (210, 100, 30), jitter=20.0)


PAINTERS: dict[str, Callable] = {
    "Tomato___healthy": _healthy,
    "Tomato___Early_blight": _early_blight,
    "Tomato___Leaf_Mold": _leaf_mold,
    "Tomato___Tomato_mosaic_virus": _mosaic,
    "Potato___Late_blight": _late_blight,
    "Corn_(maize)___Common_rust_": _rust,
}


def make_leaf(label: str, rng: np.random.Generator, size: int = 64, field: bool = False) -> np.ndarray:
    """Draw one synthetic leaf image (uint8, size x size x 3) with a random lesion severity."""
    if label not in PAINTERS:
        raise KeyError(f"no synthetic recipe for {label!r}")
    yy, xx = _grid(size)
    if field:
        base = rng.uniform(60, 160, 3)
        grad = (yy / size)[..., None] * rng.uniform(-60, 60, 3)
        img = np.clip(base + grad + rng.normal(0, 25, (size, size, 3)), 0, 255)
        for _ in range(int(rng.integers(3, 7))):
            m = _ellipse_mask(size, rng.uniform(0, size), rng.uniform(0, size), rng.uniform(4, size / 3),
                              rng.uniform(4, size / 3), rng.uniform(0, np.pi))
            _paint(img, m, rng.uniform(30, 200, 3), alpha=0.8)
    else:
        # each leaf has its own background tint, as each photo session has its own light
        img = np.ones((size, size, 3)) * rng.uniform(150, 230, 3) + rng.normal(0, 4, (size, size, 3))
    cy, cx = size / 2 + rng.normal(0, size * 0.05, 2)
    ry, rx = size * rng.uniform(0.32, 0.42), size * rng.uniform(0.2, 0.3)
    angle = rng.uniform(0, np.pi)
    leaf = _ellipse_mask(size, cy, cx, ry, rx, angle)
    green = np.array([60, 140, 50]) + rng.normal(0, 12, 3)
    _paint(img, leaf, np.clip(green, 0, 255))
    vein_dir = np.array([np.cos(angle + np.pi / 2), np.sin(angle + np.pi / 2)])
    dist = np.abs((xx - cx) * vein_dir[1] - (yy - cy) * vein_dir[0])
    _paint(img, leaf & (dist < 0.8), np.clip(green + 40, 0, 255))
    healthy_img = img.copy()
    PAINTERS[label](img, leaf, rng)
    severity = rng.uniform(0.35, 1.0)  # mild and severe cases
    img = healthy_img + severity * (img - healthy_img)
    if field:
        gain = rng.uniform(0.6, 1.3)
        light = 1.0 + 0.35 * ((xx - size / 2) / size)[..., None] * rng.choice([-1, 1])
        img = img * gain * light
    return np.clip(img + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)


def make_view(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """An offline-augmented copy: flip or rotate by 90 degrees, then change the brightness a little."""
    out = np.rot90(img, int(rng.integers(0, 4)))
    if rng.random() < 0.5:
        out = np.fliplr(out)
    gain = rng.uniform(0.95, 1.05)
    return np.clip(out.astype(np.float64) * gain, 0, 255).astype(np.uint8)


def generate_dataset(out_dir: str | Path, leaves_per_class: int | dict[str, int] = 30, views_per_leaf: int = 3,
                     size: int = 64, seed: int = 0, field: bool = False,
                     classes: Sequence[str] = SYNTHETIC_CLASSES) -> int:
    """Write out_dir/<class>/<leafid>___v<k>.png and return the number of images."""
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    count = 0
    for label in classes:
        n = leaves_per_class[label] if isinstance(leaves_per_class, dict) else leaves_per_class
        for _ in range(n):
            leaf_id = "".join(f"{b:02x}" for b in rng.integers(0, 256, 8))
            base = make_leaf(label, rng, size=size, field=field)
            for k in range(max(1, views_per_leaf)):
                img = base if k == 0 else make_view(base, rng)
                save_image(img, out / label / f"{leaf_id}___v{k}.png")
                count += 1
    return count
