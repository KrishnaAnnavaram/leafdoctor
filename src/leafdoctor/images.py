"""Image input and output, colour conversion and the dihedral-invariant difference hash."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


class ImageError(OSError):
    """An image file is missing, corrupt or not an image."""


def load_image(path: str | Path, size: int | None = None) -> np.ndarray:
    """Load an image as RGB uint8 (H, W, 3). Resize to size x size when `size` is given."""
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            if size is not None:
                im = im.resize((size, size), Image.BILINEAR)
            return np.asarray(im, dtype=np.uint8).copy()
    except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
        raise ImageError(f"cannot read image {path}: {exc}") from exc


def save_image(arr: np.ndarray, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.ascontiguousarray(arr, dtype=np.uint8)).save(path)


def is_grayscale(arr: np.ndarray, tol: int = 2) -> bool:
    """True when the three channels are (almost) equal."""
    a = arr.astype(np.int16)
    return bool(np.abs(a[..., 0] - a[..., 1]).max() <= tol and np.abs(a[..., 1] - a[..., 2]).max() <= tol)


def to_gray(arr: np.ndarray) -> np.ndarray:
    """ITU-R 601 luma as float in [0, 255]."""
    a = arr.astype(np.float64)
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def rgb_to_hsv(arr: np.ndarray) -> np.ndarray:
    """Vectorised RGB uint8 -> HSV float in [0, 1] for each channel."""
    rgb = arr.astype(np.float64) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = rgb.max(axis=-1)
    minc = rgb.min(axis=-1)
    delta = maxc - minc
    v = maxc
    s = np.where(maxc > 0, delta / np.where(maxc > 0, maxc, 1), 0.0)
    safe = np.where(delta > 0, delta, 1.0)
    rc, gc, bc = (maxc - r) / safe, (maxc - g) / safe, (maxc - b) / safe
    h = np.where(r == maxc, bc - gc, np.where(g == maxc, 2.0 + rc - bc, 4.0 + gc - rc))
    h = np.where(delta > 0, (h / 6.0) % 1.0, 0.0)
    return np.stack([h, s, v], axis=-1)


def dihedral_views(arr: np.ndarray) -> list[np.ndarray]:
    """The 8 flips and 90-degree rotations of an image."""
    views = []
    for k in range(4):
        r = np.rot90(arr, k)
        views.append(r)
        views.append(np.fliplr(r))
    return views


def dhash(gray: np.ndarray, hash_size: int = 8) -> int:
    """Difference hash of a 2-D grey image: 64 bits for hash_size 8."""
    im = Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8)).resize((hash_size + 1, hash_size), Image.BILINEAR)
    px = np.asarray(im, dtype=np.int16)
    bits = (px[:, 1:] > px[:, :-1]).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def canonical_hash(arr: np.ndarray) -> int:
    """Smallest dHash over the 8 dihedral views. Flipped or rotated copies get the same value."""
    gray = to_gray(arr)
    return min(dhash(v) for v in dihedral_views(gray))


def hamming(a: int, b: int) -> int:
    return int(bin(a ^ b).count("1"))
