"""End-to-end CNN baseline (extra `cnn`). torch is imported only inside these functions.

`arch="tiny"` is a 3-block CNN that needs only torch and trains on a CPU in seconds (used by the
tests). Any arch from `features.backbones.TORCH_ARCHS` fine-tunes an ImageNet model (needs torchvision).
The loss uses class weights from the train split, and augmentation runs on train images only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .augment import augment
from .images import load_image
from .manifest import Row

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _tiny_cnn(n_classes: int):
    import torch.nn as nn

    def block(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))

    return nn.Sequential(block(3, 16), block(16, 32), block(32, 64), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                         nn.Linear(64, n_classes))


def _pretrained(arch: str, n_classes: int):
    import torch.nn as nn
    import torchvision.models as tvm

    from .features.backbones import TORCH_ARCHS

    builder, weights_name, dim = TORCH_ARCHS[arch]
    model = getattr(tvm, builder)(weights=getattr(tvm, weights_name).DEFAULT)
    if arch.startswith("convnext"):
        model.classifier[-1] = nn.Linear(dim, n_classes)
    elif hasattr(model, "classifier"):
        model.classifier = nn.Linear(dim, n_classes)
    else:
        model.fc = nn.Linear(dim, n_classes)
    return model


def _to_tensor(imgs: Sequence[np.ndarray]):
    import torch

    arr = (np.stack(imgs).astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(arr.transpose(0, 3, 1, 2).copy())


@dataclass
class FineTuned:
    model: object
    classes: list[str]
    image_size: int
    device: str

    def predict_proba_images(self, imgs: Sequence[np.ndarray], batch_size: int = 32) -> np.ndarray:
        import torch

        self.model.eval()
        out = []
        with torch.inference_mode():
            for i in range(0, len(imgs), batch_size):
                logits = self.model(_to_tensor(imgs[i:i + batch_size]).to(self.device))
                out.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, len(self.classes)))

    def predict_proba_rows(self, rows: Sequence[Row]) -> np.ndarray:
        return self.predict_proba_images([load_image(r.path, self.image_size) for r in rows])


def finetune(train_rows: Sequence[Row], arch: str = "tiny", epochs: int = 5, lr: float = 1e-3,
             batch_size: int = 32, image_size: int = 64, seed: int = 42, device: str = "cpu") -> FineTuned:
    import torch

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    classes = sorted({r.label for r in train_rows})
    pos = {c: i for i, c in enumerate(classes)}
    images = [load_image(r.path, image_size) for r in train_rows]
    y = np.array([pos[r.label] for r in train_rows], dtype=np.int64)
    counts = np.bincount(y, minlength=len(classes)).astype(np.float32)
    weights = torch.tensor(len(y) / (len(classes) * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    model = (_tiny_cnn(len(classes)) if arch == "tiny" else _pretrained(arch, len(classes))).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    for _ in range(epochs):
        model.train()
        order = rng.permutation(len(images))
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            xb = _to_tensor([augment(images[j], rng) for j in idx]).to(device)
            yb = torch.from_numpy(y[idx]).to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
    return FineTuned(model, classes, image_size, device)
