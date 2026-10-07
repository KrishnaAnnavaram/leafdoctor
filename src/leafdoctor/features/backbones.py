"""Deep-feature extractors behind one small interface.

- `TorchBackbone`: a frozen ImageNet CNN from torchvision (extra `cnn`). torch is imported lazily.
- `PixelProjection`: an offline stand-in with no download. It is NOT a CNN. It projects a 16 x 16
  thumbnail with a fixed, seeded random matrix, so the fusion code runs in CI with no weights.
- `Fusion`: concatenates the outputs of several extractors.
"""
from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np
from PIL import Image

from .handcrafted import HandcraftedExtractor


class Extractor(Protocol):
    name: str
    dim: int

    def extract(self, images: Sequence[np.ndarray]) -> np.ndarray: ...


class PixelProjection:
    def __init__(self, dim: int = 128, thumb: int = 16, seed: int = 0):
        self.dim = dim
        self.thumb = thumb
        self.seed = seed
        self.name = f"pixelproj{dim}"
        rng = np.random.default_rng(seed)
        self._w = rng.normal(0, 1.0 / np.sqrt(thumb * thumb * 3), (thumb * thumb * 3, dim))

    def extract(self, images: Sequence[np.ndarray]) -> np.ndarray:
        if len(images) == 0:
            return np.zeros((0, self.dim))
        thumbs = np.stack([
            np.asarray(Image.fromarray(im).resize((self.thumb, self.thumb), Image.BILINEAR), dtype=np.float64).ravel()
            for im in images
        ]) / 255.0
        return np.maximum(thumbs @ self._w, 0.0)


TORCH_ARCHS = {
    # arch: (torchvision builder name, weights enum name, feature dim)
    "efficientnet_b3": ("efficientnet_b3", "EfficientNet_B3_Weights", 1536),
    "convnext_tiny": ("convnext_tiny", "ConvNeXt_Tiny_Weights", 768),
    "resnet50": ("resnet50", "ResNet50_Weights", 2048),
    "densenet201": ("densenet201", "DenseNet201_Weights", 1920),
}


class TorchBackbone:
    """Frozen torchvision CNN with the classifier head removed. Batched inference, no gradients."""

    def __init__(self, arch: str = "efficientnet_b3", device: str = "cpu", input_size: int = 224):
        if arch not in TORCH_ARCHS:
            raise ValueError(f"unknown arch {arch!r}; choose one of {sorted(TORCH_ARCHS)}")
        import torch  # noqa: F401  (lazy: the core package must import without torch)
        import torchvision.models as tvm

        builder, weights_name, dim = TORCH_ARCHS[arch]
        weights = getattr(tvm, weights_name).DEFAULT
        model = getattr(tvm, builder)(weights=weights)
        if hasattr(model, "classifier"):
            if arch.startswith("convnext"):
                model.classifier[-1] = torch.nn.Identity()
            else:
                model.classifier = torch.nn.Identity()
        else:
            model.fc = torch.nn.Identity()
        self.model = model.eval().to(device)
        self.device = device
        self.input_size = input_size
        self.dim = dim
        self.name = f"torch_{arch}"
        self._mean = np.array([0.485, 0.456, 0.406])
        self._std = np.array([0.229, 0.224, 0.225])

    def extract(self, images: Sequence[np.ndarray]) -> np.ndarray:
        import torch

        if len(images) == 0:
            return np.zeros((0, self.dim))
        batch = np.stack([
            np.asarray(Image.fromarray(im).resize((self.input_size, self.input_size), Image.BILINEAR),
                       dtype=np.float32) / 255.0 for im in images
        ])
        batch = ((batch - self._mean) / self._std).astype(np.float32).transpose(0, 3, 1, 2)
        with torch.inference_mode():
            out = self.model(torch.from_numpy(batch).to(self.device))
        return out.cpu().numpy().reshape(len(images), -1)


class Fusion:
    def __init__(self, parts: Sequence[Extractor]):
        if not parts:
            raise ValueError("Fusion needs at least one extractor")
        self.parts = list(parts)
        self.dim = sum(p.dim for p in self.parts)
        self.name = "+".join(p.name for p in self.parts)

    def extract(self, images: Sequence[np.ndarray]) -> np.ndarray:
        return np.concatenate([p.extract(images) for p in self.parts], axis=1)


def build_extractor(spec: str, device: str = "cpu", seed: int = 0) -> Extractor:
    """Build an extractor from a spec: `handcrafted`, `pixelproj`, `torch:<arch>`, or parts joined by `+`."""
    parts = [s.strip() for s in spec.split("+") if s.strip()]
    if not parts:
        raise ValueError("empty extractor spec")
    built: list[Extractor] = []
    for p in parts:
        if p == "handcrafted":
            built.append(HandcraftedExtractor())
        elif p == "pixelproj":
            built.append(PixelProjection(seed=seed))
        elif p.startswith("torch:"):
            built.append(TorchBackbone(p.split(":", 1)[1], device=device))
        else:
            raise ValueError(f"unknown extractor {p!r}; use handcrafted, pixelproj or torch:<arch>")
    return built[0] if len(built) == 1 else Fusion(built)
