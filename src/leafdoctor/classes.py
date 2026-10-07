"""Canonical class names and the folder-name harmonisation for PlantVillage-style datasets.

Each source names its folders a little differently ("Pepper__bell___Bacterial_spot" against
"Pepper,_bell___Bacterial_spot"). `canonical_label` compares names by a compact key (lower case,
letters and digits only), so all known spellings map to one canonical label. A name that does not
map raises `UnknownClassError`: the code never invents a class.
"""
from __future__ import annotations

import re

CANONICAL_CLASSES: tuple[str, ...] = (
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
)

# Spellings that the compact key cannot join on its own. Key: compact key of the alias.
EXTRA_ALIASES: dict[str, str] = {
    "tomatospidermites": "Tomato___Spider_mites Two-spotted_spider_mite",
    "tomatoyellowleafcurlvirus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "tomatomosaicvirus": "Tomato___Tomato_mosaic_virus",
    "pepperbacterialspot": "Pepper,_bell___Bacterial_spot",
    "pepperhealthy": "Pepper,_bell___healthy",
}


class UnknownClassError(KeyError):
    """A folder name does not match a canonical class."""


def compact_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


_INDEX: dict[str, str] = {compact_key(c): c for c in CANONICAL_CLASSES}
_INDEX.update(EXTRA_ALIASES)


def canonical_label(name: str) -> str:
    """Return the canonical class for a folder name, or raise UnknownClassError."""
    key = compact_key(name)
    if key in _INDEX:
        return _INDEX[key]
    raise UnknownClassError(f"unknown class folder {name!r} (key {key!r}); add it to EXTRA_ALIASES")


def crop_of(label: str) -> str:
    return label.split("___", 1)[0]


def is_healthy(label: str) -> bool:
    return label.endswith("___healthy")
