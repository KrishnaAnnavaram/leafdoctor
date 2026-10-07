"""The evaluation protocol: select on val, report once on test, then on an external field set.

`select_on_val` gets only the train and val features. It cannot see the test features, because
`run_experiment` computes them after the selection is finished.
"""
from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from .features.backbones import build_extractor
from .features.store import FeatureSet, FeatureStore
from .manifest import Row
from .metrics import EvalReport, evaluate
from .models import CLASSIFIERS, IMBALANCE_POLICIES, TrainedModel, align_proba, fit_classifier
from .split import LeakageError, check_leakage, group_split, random_image_split
from .tracking import NullTracker

REGISTERED_MODEL = "leafdoctor-classifier"


@dataclass
class ExperimentConfig:
    name: str = "hybrid"
    extractor: str = "pixelproj+handcrafted"
    classifiers: list[str] = field(default_factory=lambda: ["logreg", "rf", "hgb"])
    imbalance: str = "balanced"
    train_views: int = 1
    seed: int = 42
    image_size: int = 64
    n_boot: int = 200

    def validate(self) -> "ExperimentConfig":
        bad = [c for c in self.classifiers if c not in CLASSIFIERS]
        if not self.classifiers or bad:
            raise ValueError(f"classifiers must be a non-empty subset of {CLASSIFIERS}, bad: {bad}")
        if self.imbalance not in IMBALANCE_POLICIES:
            raise ValueError(f"imbalance must be one of {IMBALANCE_POLICIES}")
        if self.train_views < 0 or self.image_size < 32:
            raise ValueError("train_views must be >= 0 and image_size >= 32")
        return self

    @classmethod
    def from_toml(cls, path: str | Path) -> "ExperimentConfig":
        data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"unknown config keys {unknown}")
        return cls(**data).validate()


@dataclass
class RunResult:
    config: ExperimentConfig
    run_id: str
    selected: str
    val_scores: dict[str, float]
    reports: dict[str, EvalReport]
    model_path: Path | None

    def summary(self) -> dict:
        out = {"name": self.config.name, "extractor": self.config.extractor, "selected": self.selected,
               "val_macro_f1_by_classifier": {k: round(v, 4) for k, v in self.val_scores.items()}}
        for split, rep in self.reports.items():
            out[split] = {k: round(v, 4) for k, v in rep.headline().items()}
            out[split]["macro_f1_ci"] = [round(x, 4) for x in rep.macro_f1_ci]
            out[split]["n"] = rep.n
        return out


def _proba(model: TrainedModel, fs: FeatureSet, classes: list[str]) -> np.ndarray:
    return align_proba(model, model.predict_proba(fs.X), classes)


def select_on_val(train: FeatureSet, val: FeatureSet, cfg: ExperimentConfig,
                  classes: list[str]) -> tuple[TrainedModel, dict[str, float], EvalReport]:
    """Fit each candidate on train and keep the best val macro-F1. Test data never enters here."""
    best: tuple[float, TrainedModel, EvalReport] | None = None
    scores: dict[str, float] = {}
    for name in cfg.classifiers:
        model = fit_classifier(name, train.X, train.labels, cfg.seed, cfg.imbalance, cfg.extractor, cfg.image_size)
        rep = evaluate("val", val.labels, _proba(model, val, classes), classes, n_boot=cfg.n_boot, seed=cfg.seed)
        scores[name] = rep.macro_f1
        if best is None or rep.macro_f1 > best[0]:
            best = (rep.macro_f1, model, rep)
    assert best is not None
    return best[1], scores, best[2]


def run_experiment(cfg: ExperimentConfig, splits: dict[str, Sequence[Row]], store: FeatureStore,
                   tracker=None, field_rows: Sequence[Row] | None = None, models_dir: Path | None = None,
                   device: str = "cpu") -> RunResult:
    cfg.validate()
    tracker = tracker or NullTracker()
    extractor = build_extractor(cfg.extractor, device=device, seed=cfg.seed)
    classes = sorted({r.label for r in splits["train"]})
    train = store.features(splits["train"], extractor, "train", cfg.image_size, cfg.train_views, cfg.seed)
    val = store.features(splits["val"], extractor, "val", cfg.image_size, 0, cfg.seed)
    model, scores, val_report = select_on_val(train, val, cfg, classes)

    # The selection is final. Only now are the test and field features computed and scored, once.
    reports = {"val": val_report}
    test = store.features(splits["test"], extractor, "test", cfg.image_size, 0, cfg.seed)
    reports["test"] = evaluate("test", test.labels, _proba(model, test, classes), classes, cfg.n_boot, cfg.seed)
    if field_rows:
        known = [r for r in field_rows if r.label in set(classes)]
        fs = store.features(known, extractor, "field", cfg.image_size, 0, cfg.seed)
        reports["field"] = evaluate("field", fs.labels, _proba(model, fs, classes), classes, cfg.n_boot, cfg.seed)

    run_id = tracker.start_run(cfg.name)
    tracker.log_params({"extractor": cfg.extractor, "classifier": model.classifier, "imbalance": cfg.imbalance,
                        "train_views": cfg.train_views, "seed": cfg.seed, "image_size": cfg.image_size,
                        "selected_on": "val_macro_f1"})
    for rep in reports.values():
        tracker.log_eval(rep)
    model.info.update({"run_id": run_id, "config": asdict(cfg), "val_scores": scores})
    model_path = None
    if models_dir is not None:
        model_path = model.save(Path(models_dir) / f"{cfg.name}-{run_id}.joblib")
        tracker.log_model(model_path, register_as=REGISTERED_MODEL)
    tracker.end_run()
    return RunResult(cfg, run_id, model.classifier, scores, reports, model_path)


ABLATION_EXTRACTORS = ("cnn", "handcrafted", "cnn+handcrafted")


def ablation_configs(base: ExperimentConfig, cnn_spec: str = "pixelproj") -> list[ExperimentConfig]:
    """Three runs that differ only by the feature set: CNN only, handcrafted only, both."""
    specs = {"cnn": cnn_spec, "handcrafted": "handcrafted", "cnn+handcrafted": f"{cnn_spec}+handcrafted"}
    out = []
    for key in ABLATION_EXTRACTORS:
        d = asdict(base)
        d.update(name=f"{base.name}-{key.replace('+', '-')}", extractor=specs[key])
        out.append(ExperimentConfig(**d))
    return out


def leakage_comparison(rows: Sequence[Row], cfg: ExperimentConfig, store: FeatureStore,
                       max_distance: int = 3) -> dict:
    """Val macro-F1 for an image-level random split against the group-aware split, same classifier."""
    out = {}
    for name, result in (("image_level", random_image_split(rows, seed=cfg.seed)),
                         ("group_aware", group_split(rows, seed=cfg.seed))):
        try:
            leak = check_leakage({k: v for k, v in result.splits.items() if k != "test"}, max_distance)
        except LeakageError:
            leak = None
        shared = len({r.group_id for r in result.splits["train"]} & {r.group_id for r in result.splits["val"]})
        extractor = build_extractor(cfg.extractor, seed=cfg.seed)
        train = store.features(result.splits["train"], extractor, f"lk_{name}_train", cfg.image_size, 0, cfg.seed)
        val = store.features(result.splits["val"], extractor, f"lk_{name}_val", cfg.image_size, 0, cfg.seed)
        classes = sorted(set(train.labels.tolist()))
        model = fit_classifier(cfg.classifiers[0], train.X, train.labels, cfg.seed, cfg.imbalance)
        rep = evaluate("val", val.labels, _proba(model, val, classes), classes, n_boot=cfg.n_boot, seed=cfg.seed)
        out[name] = {"val_macro_f1": round(rep.macro_f1, 4), "val_accuracy": round(rep.accuracy, 4),
                     "train_val_shared_groups": shared, "leakage_check_passed": leak is not None}
    return out

