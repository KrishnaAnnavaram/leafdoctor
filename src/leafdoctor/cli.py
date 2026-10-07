"""The `leafdoctor` command line."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .config import Settings
from .experiment import ExperimentConfig, ablation_configs, leakage_comparison, run_experiment
from .features.store import FeatureStore
from .manifest import read_manifest, scan_folder, write_manifest
from .split import attach_augmented, check_leakage, group_split, merge_groups, read_splits, write_splits
from .synthetic import generate_dataset
from .tracking import build_tracker, rows_to_csv


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def _config(args, settings: Settings) -> ExperimentConfig:
    cfg = ExperimentConfig.from_toml(args.config) if getattr(args, "config", None) else ExperimentConfig(
        seed=settings.seed, image_size=settings.image_size)
    if getattr(args, "extractor", None):
        cfg.extractor = args.extractor
    if getattr(args, "classifiers", None):
        cfg.classifiers = args.classifiers
    return cfg.validate()


def cmd_synth(args, settings):
    n = generate_dataset(args.out, args.leaves, args.views, args.size, args.seed, field=args.field)
    _print({"images": n, "out": str(args.out), "style": "field" if args.field else "lab"})


def cmd_manifest(args, settings):
    rep = scan_folder(args.root, args.source, compute_hash=not args.no_hash)
    write_manifest(rep.rows, args.out)
    _print({"out": str(args.out), "images": len(rep.rows), "classes": rep.class_counts(),
            "imbalance_ratio": round(rep.imbalance_ratio(), 2), "unknown_folders": rep.unknown_folders,
            "corrupt": rep.corrupt, "grayscale": rep.grayscale})


def cmd_split(args, settings):
    rows = merge_groups(read_manifest(args.manifest), args.max_distance)
    result = group_split(rows, tuple(args.fractions), seed=args.seed if args.seed is not None else settings.seed)
    if args.augmented:
        aug = read_manifest(args.augmented)
        result = attach_augmented(result, aug, args.max_distance)
    leak = check_leakage(result.splits, args.max_distance)
    out = write_splits(result, args.out or settings.splits_dir)
    _print({"out": str(out), "leakage": leak, **result.summary()})


def _run(cfgs, args, settings):
    splits = read_splits(args.splits or settings.splits_dir)
    field_rows = read_manifest(args.field) if args.field else None
    store = FeatureStore(settings.cache_dir, settings.batch_size)
    tracker = build_tracker(settings.tracker, settings.runs_dir, settings.experiment, settings.mlflow_uri)
    results = [run_experiment(c, splits, store, tracker, field_rows, settings.models_dir, settings.device)
               for c in cfgs]
    _print([r.summary() | {"model": str(r.model_path)} for r in results])


def cmd_train(args, settings):
    _run([_config(args, settings)], args, settings)


def cmd_ablate(args, settings):
    _run(ablation_configs(_config(args, settings), cnn_spec=args.cnn), args, settings)


def cmd_leakage(args, settings):
    rows = merge_groups(read_manifest(args.manifest), args.max_distance)
    cfg = _config(args, settings)
    _print(leakage_comparison(rows, cfg, FeatureStore(None, settings.batch_size), args.max_distance))


def cmd_finetune(args, settings):
    from .finetune import finetune
    from .metrics import evaluate

    splits = read_splits(args.splits or settings.splits_dir)
    ft = finetune(splits["train"], arch=args.arch, epochs=args.epochs, image_size=args.size, seed=settings.seed,
                  device=settings.device)
    val = evaluate("val", [r.label for r in splits["val"]], ft.predict_proba_rows(splits["val"]), ft.classes)
    test = evaluate("test", [r.label for r in splits["test"]], ft.predict_proba_rows(splits["test"]), ft.classes)
    _print({"arch": args.arch, "val": val.headline(), "test": test.headline()})


def cmd_predict(args, settings):
    from .models import TrainedModel
    from .predict import predict_paths

    model = TrainedModel.load(args.model)
    _print([asdict(p) for p in predict_paths(model, args.images, settings.batch_size, settings.device)])


def cmd_runs(args, settings):
    tracker = build_tracker(settings.tracker, settings.runs_dir, settings.experiment, settings.mlflow_uri)
    text = rows_to_csv(tracker.export_rows())
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)


def cmd_demo(args, settings):
    out = Path(args.out)
    lab, field = out / "images" / "lab", out / "images" / "field"
    leaves = {c: args.leaves for c in ("Tomato___healthy", "Tomato___Early_blight", "Tomato___Leaf_Mold",
                                       "Tomato___Tomato_mosaic_virus", "Potato___Late_blight",
                                       "Corn_(maize)___Common_rust_")}
    leaves["Tomato___healthy"] = args.leaves * 3  # imbalance on purpose
    generate_dataset(lab, leaves, views_per_leaf=3, size=64, seed=settings.seed)
    generate_dataset(field, max(4, args.leaves // 3), views_per_leaf=1, size=64, seed=settings.seed + 1, field=True)
    lab_rows = merge_groups(scan_folder(lab, "synthetic-lab").rows, 3)
    field_rows = scan_folder(field, "synthetic-field").rows
    write_manifest(lab_rows, out / "manifest_lab.csv")
    write_manifest(field_rows, out / "manifest_field.csv")
    result = group_split(lab_rows, seed=settings.seed)
    leak = check_leakage(result.splits, 3)
    write_splits(result, out / "splits")
    store = FeatureStore(out / "features", settings.batch_size)
    tracker = build_tracker("json", out / "runs", "demo")
    base = ExperimentConfig(name="demo", seed=settings.seed, image_size=64, classifiers=["majority", "logreg", "rf"])
    results = [run_experiment(c, result.splits, store, tracker, field_rows, out / "models")
               for c in ablation_configs(base)]
    lk = leakage_comparison(lab_rows, ExperimentConfig(seed=settings.seed, image_size=64, classifiers=["rf"],
                                                       extractor="handcrafted"), store, 3)
    summary = {"images": {"lab": len(lab_rows), "field": len(field_rows)},
               "split": result.summary(), "leakage": leak, "ablation": [r.summary() for r in results],
               "leakage_comparison": lk}
    (out / "demo_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _print(summary)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="leafdoctor", description="Leakage-safe plant disease classification.")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synth", help="write synthetic leaf images")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--leaves", type=int, default=30)
    s.add_argument("--views", type=int, default=3)
    s.add_argument("--size", type=int, default=64)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--field", action="store_true", help="cluttered background and uneven light")
    s.set_defaults(func=cmd_synth)

    s = sub.add_parser("manifest", help="scan <root>/<class>/<image> into a validated manifest")
    s.add_argument("root", type=Path)
    s.add_argument("--source", required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--no-hash", action="store_true")
    s.set_defaults(func=cmd_manifest)

    s = sub.add_parser("split", help="group-aware train/val/test split with a leakage check")
    s.add_argument("manifest", type=Path)
    s.add_argument("--augmented", type=Path, help="manifest of offline-augmented copies (train only)")
    s.add_argument("--out", type=Path)
    s.add_argument("--fractions", type=float, nargs=3, default=[0.7, 0.15, 0.15])
    s.add_argument("--max-distance", type=int, default=3)
    s.add_argument("--seed", type=int)
    s.set_defaults(func=cmd_split)

    for name, func, helptext in (("train", cmd_train, "select a classifier on val, score test once"),
                                 ("ablate", cmd_ablate, "CNN only, handcrafted only, and both")):
        s = sub.add_parser(name, help=helptext)
        s.add_argument("--config", type=Path)
        s.add_argument("--splits", type=Path)
        s.add_argument("--field", type=Path, help="manifest of an external field test set")
        s.add_argument("--extractor")
        s.add_argument("--classifiers", nargs="+")
        if name == "ablate":
            s.add_argument("--cnn", default="pixelproj", help="CNN part, for example torch:efficientnet_b3")
        s.set_defaults(func=func)

    s = sub.add_parser("leakage-check", help="image-level split against group-aware split")
    s.add_argument("manifest", type=Path)
    s.add_argument("--config", type=Path)
    s.add_argument("--extractor", default="handcrafted")
    s.add_argument("--classifiers", nargs="+", default=["rf"])
    s.add_argument("--max-distance", type=int, default=3)
    s.set_defaults(func=cmd_leakage)

    s = sub.add_parser("finetune", help="end-to-end CNN baseline (extra cnn)")
    s.add_argument("--splits", type=Path)
    s.add_argument("--arch", default="tiny")
    s.add_argument("--epochs", type=int, default=5)
    s.add_argument("--size", type=int, default=64)
    s.set_defaults(func=cmd_finetune)

    s = sub.add_parser("predict", help="classify image files with a saved model")
    s.add_argument("model", type=Path)
    s.add_argument("images", nargs="+")
    s.set_defaults(func=cmd_predict)

    s = sub.add_parser("runs", help="export the tracked runs as CSV")
    s.add_argument("--out", type=Path)
    s.set_defaults(func=cmd_runs)

    s = sub.add_parser("demo", help="full offline pipeline on synthetic images")
    s.add_argument("--out", type=Path, default=Path("artifacts/demo"))
    s.add_argument("--leaves", type=int, default=20)
    s.set_defaults(func=cmd_demo)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings.from_env()
    try:
        args.func(args, settings)
    except (ValueError, OSError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
