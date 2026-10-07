"""The evaluation protocol, the ablation, the CLI and the optional torch baseline (problems 2, 10 and 11)."""
import json

import numpy as np
import pytest

from leafdoctor import experiment as exp_mod
from leafdoctor.cli import main
from leafdoctor.experiment import ExperimentConfig, ablation_configs, leakage_comparison, run_experiment
from leafdoctor.features.store import FeatureStore
from leafdoctor.manifest import write_manifest
from leafdoctor.models import TrainedModel
from leafdoctor.split import write_splits, SplitResult
from leafdoctor.tracking import JsonTracker

FAST = dict(image_size=48, classifiers=["majority", "logreg"], n_boot=20, extractor="handcrafted", train_views=1)


def test_selection_never_sees_test_and_test_is_scored_once(tmp_path, splits, field_rows, monkeypatch):
    seen = []
    real_select = exp_mod.select_on_val

    def spy_select(train, val, cfg, classes):
        seen.append(set(val.paths.tolist()) | set(train.paths.tolist()))
        return real_select(train, val, cfg, classes)

    calls = []
    real_eval = exp_mod.evaluate

    def spy_eval(split, *a, **k):
        calls.append(split)
        return real_eval(split, *a, **k)

    monkeypatch.setattr(exp_mod, "select_on_val", spy_select)
    monkeypatch.setattr(exp_mod, "evaluate", spy_eval)
    cfg = ExperimentConfig(**FAST)
    res = run_experiment(cfg, splits, FeatureStore(tmp_path / "f"), JsonTracker(tmp_path / "runs", "e"),
                         field_rows, tmp_path / "models")
    test_paths = {r.path for r in splits["test"]}
    assert seen and not (seen[0] & test_paths)
    assert calls.count("test") == 1 and calls.count("field") == 1
    assert set(res.reports) == {"val", "test", "field"}
    assert res.selected in cfg.classifiers
    assert res.model_path.exists()
    assert res.val_scores["logreg"] > res.val_scores["majority"]


def test_ablation_differs_only_by_features():
    cfgs = ablation_configs(ExperimentConfig(**FAST), cnn_spec="pixelproj")
    assert [c.extractor for c in cfgs] == ["pixelproj", "handcrafted", "pixelproj+handcrafted"]
    assert len({(c.seed, c.image_size, tuple(c.classifiers)) for c in cfgs}) == 1


def test_config_from_toml(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text('name = "x"\nextractor = "handcrafted"\nclassifiers = ["rf"]\nseed = 3\n', encoding="utf-8")
    cfg = ExperimentConfig.from_toml(p)
    assert cfg.classifiers == ["rf"] and cfg.seed == 3
    p.write_text('use_label_encoder = false\n', encoding="utf-8")
    with pytest.raises(ValueError):
        ExperimentConfig.from_toml(p)
    with pytest.raises(ValueError):
        ExperimentConfig(classifiers=["svm"]).validate()


def test_leakage_comparison_reports_both_protocols(lab_rows):
    cfg = ExperimentConfig(image_size=48, classifiers=["rf"], extractor="handcrafted", n_boot=10)
    out = leakage_comparison(lab_rows, cfg, FeatureStore(None))
    assert out["image_level"]["train_val_shared_groups"] > 0 and not out["image_level"]["leakage_check_passed"]
    assert out["group_aware"]["train_val_shared_groups"] == 0 and out["group_aware"]["leakage_check_passed"]


def test_cli_end_to_end(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LEAFDOCTOR_WORK_DIR", str(tmp_path / "work"))
    monkeypatch.setenv("LEAFDOCTOR_IMAGE_SIZE", "48")
    assert main(["synth", "--out", "imgs", "--leaves", "6", "--views", "2", "--size", "48"]) == 0
    assert main(["manifest", "imgs", "--source", "syn", "--out", "m.csv"]) == 0
    capsys.readouterr()
    assert main(["split", "m.csv"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["leakage"]["shared_groups"] == 0
    assert main(["train", "--extractor", "handcrafted", "--classifiers", "logreg"]) == 0
    trained = json.loads(capsys.readouterr().out)
    assert trained[0]["selected"] == "logreg" and "test" in trained[0]
    model = trained[0]["model"]
    img = next((tmp_path / "imgs").rglob("*.png"))
    assert main(["predict", model, str(img)]) == 0
    pred = json.loads(capsys.readouterr().out)
    assert pred[0]["label"] in TrainedModel.load(model).classes and len(pred[0]["top3"]) == 3
    assert main(["runs"]) == 0
    assert "val_macro_f1" in capsys.readouterr().out


def test_cli_reports_errors_without_traceback(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["manifest", "nowhere", "--source", "x", "--out", "m.csv"]) == 2
    assert "error:" in capsys.readouterr().err


def test_cli_demo(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("LEAFDOCTOR_SEED", "1")
    assert main(["demo", "--out", str(tmp_path / "demo"), "--leaves", "4"]) == 0
    summary = json.loads((tmp_path / "demo" / "demo_summary.json").read_text())
    assert summary["leakage"] == {"shared_groups": 0, "cross_split_near_duplicates": 0}
    assert [r["extractor"] for r in summary["ablation"]] == ["pixelproj", "handcrafted", "pixelproj+handcrafted"]
    assert all("field" in r for r in summary["ablation"])


def test_finetune_tiny_cnn(splits):
    pytest.importorskip("torch")
    from leafdoctor.finetune import finetune
    ft = finetune(splits["train"][:24], arch="tiny", epochs=1, batch_size=8, image_size=32, seed=0)
    proba = ft.predict_proba_rows(splits["val"][:5])
    assert proba.shape == (5, len(ft.classes)) and np.allclose(proba.sum(axis=1), 1, atol=1e-5)


def test_torchvision_backbone_shape():
    pytest.importorskip("torchvision")
    from leafdoctor.features.backbones import TorchBackbone
    try:
        bb = TorchBackbone("resnet50", input_size=64)
    except Exception as exc:  # weights download needs network
        pytest.skip(f"weights not available: {exc}")
    assert bb.extract([np.zeros((64, 64, 3), dtype=np.uint8)]).shape == (1, 2048)


def test_xgb_classifier_when_installed():
    pytest.importorskip("xgboost")
    from leafdoctor.models import fit_classifier
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 5))
    y = np.array(["Tomato___healthy", "Tomato___Leaf_Mold", "Potato___healthy"] * 20)
    assert fit_classifier("xgb", X, y, 0, "balanced").predict_proba(X).shape == (60, 3)


def test_write_splits_writes_report(tmp_path, splits):
    out = write_splits(SplitResult({k: list(v) for k, v in splits.items()}, 7, (0.7, 0.15, 0.15)), tmp_path)
    rep = json.loads((out / "split_report.json").read_text())
    assert rep["seed"] == 7 and rep["train"]["images"] == len(splits["train"])
    write_manifest(splits["val"], tmp_path / "again.csv")
