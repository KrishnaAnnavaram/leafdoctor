"""Classifiers, metrics and honest tracking (problems 3, 4, 7 and 10)."""
import json
import math

import numpy as np
import pytest

from leafdoctor.config import ConfigError, Settings
from leafdoctor.metrics import EvalReport, evaluate, expected_calibration_error
from leafdoctor.models import TrainedModel, align_proba, build_classifier, fit_classifier, sample_weights
from leafdoctor.tracking import (METRIC_KEYS, PARAM_KEYS, JsonTracker, TrackingError, metrics_from_report,
                                 rows_to_csv)


def _imbalanced(seed=0):
    rng = np.random.default_rng(seed)
    big = rng.normal(0.0, 1.0, (300, 4))
    small = rng.normal(1.2, 1.0, (15, 4))
    X = np.vstack([big, small])
    y = np.array(["Tomato___healthy"] * 300 + ["Tomato___Leaf_Mold"] * 15)
    return X, y


def test_balanced_weights_raise_minority_recall():
    X, y = _imbalanced()
    Xt, yt = _imbalanced(1)
    recalls = {}
    for policy in ("none", "balanced"):
        m = fit_classifier("logreg", X, y, 0, policy)
        pred = m.predict(Xt)
        recalls[policy] = float((pred[yt == "Tomato___Leaf_Mold"] == "Tomato___Leaf_Mold").mean())
    assert recalls["balanced"] > recalls["none"]


def test_sample_weights_policy():
    y = np.array([0, 0, 0, 1])
    w = sample_weights(y, "balanced")
    assert np.isclose(w[y == 0].sum(), w[y == 1].sum())
    assert sample_weights(y, "none") is None
    with pytest.raises(ValueError):
        sample_weights(y, "oversample")


def test_same_seed_same_model():
    X, y = _imbalanced()
    a = fit_classifier("rf", X, y, 5, "balanced").predict_proba(X)
    b = fit_classifier("rf", X, y, 5, "balanced").predict_proba(X)
    assert np.array_equal(a, b)


def test_scaler_is_fit_on_train_only():
    X, y = _imbalanced()
    m = fit_classifier("logreg", X, y, 0, "none")
    assert np.allclose(m.pipeline.named_steps["scale"].mean_, X.mean(axis=0))


def test_unknown_classifier_and_majority_baseline():
    with pytest.raises(ValueError):
        build_classifier("svm9000", 0)
    X, y = _imbalanced()
    m = fit_classifier("majority", X, y, 0, "none")
    assert set(m.predict(X)) == {"Tomato___healthy"}


def test_model_bundle_round_trip(tmp_path):
    X, y = _imbalanced()
    m = fit_classifier("logreg", X, y, 0, "balanced", "handcrafted", 64)
    path = m.save(tmp_path / "m.joblib")
    m2 = TrainedModel.load(path)
    assert m2.classes == m.classes and m2.extractor == "handcrafted"
    assert np.allclose(m2.predict_proba(X), m.predict_proba(X))


def test_align_proba_puts_unseen_classes_at_zero():
    X, y = _imbalanced()
    m = fit_classifier("logreg", X, y, 0, "none")
    out = align_proba(m, m.predict_proba(X[:2]), ["A___x", *m.classes])
    assert out.shape == (2, 3) and (out[:, 0] == 0).all()


def test_evaluate_perfect_and_known_values():
    classes = ["a", "b"]
    proba = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    rep = evaluate("test", ["a", "b", "a", "b"], proba, classes, n_boot=50)
    assert rep.accuracy == 1 and rep.macro_f1 == 1 and rep.ece == 0
    assert rep.confusion == [[1.0, 0.0], [0.0, 1.0]]
    rep2 = evaluate("test", ["a", "a", "a", "b"], np.array([[0.9, 0.1]] * 4), classes, n_boot=50)
    assert rep2.accuracy == 0.75 and rep2.balanced_accuracy == 0.5
    assert rep2.per_class["b"]["recall"] == 0.0
    lo, hi = rep2.macro_f1_ci
    assert lo <= rep2.macro_f1 <= hi


def test_evaluate_rejects_bad_input():
    with pytest.raises(ValueError):
        evaluate("val", ["z"], np.array([[1.0, 0.0]]), ["a", "b"])
    with pytest.raises(ValueError):
        evaluate("val", ["a"], np.array([[1.0, 0.0, 0.0]]), ["a", "b"])


def test_ece_of_overconfident_model():
    y = np.array([0, 1, 0, 1])
    proba = np.array([[0.99, 0.01]] * 4)
    assert math.isclose(expected_calibration_error(y, proba), 0.49, abs_tol=1e-9)


def _report(split="val", acc=0.8):
    return EvalReport(split, 10, acc, 0.7, (0.6, 0.8), 0.75, 0.5, 0.1, {}, [], [])


def test_tracker_rejects_free_and_non_finite_metrics():
    """Problem 3: no hard-coded number can be logged. Metrics come only from an EvalReport."""
    with pytest.raises(TrackingError):
        metrics_from_report(_report(acc=float("nan")))
    with pytest.raises(TrackingError):
        metrics_from_report(_report(split="train"))
    t = JsonTracker("unused", "exp")
    assert not hasattr(t, "log_metric") and not hasattr(t, "log_metrics")
    with pytest.raises(TrackingError):
        t.log_params({"model_type": "XGBoost"})
    with pytest.raises(TrackingError):
        t.log_eval(_report())  # no active run


def test_export_reads_the_same_experiment_and_keys(tmp_path):
    """Problem 4: the exporter uses the writer's experiment name and key names."""
    t = JsonTracker(tmp_path, "leafdoctor")
    t.start_run("r1")
    t.log_params({"extractor": "handcrafted", "classifier": "rf", "seed": 1})
    t.log_eval(_report("val"))
    t.log_eval(_report("test", 0.9))
    t.log_model(tmp_path / "m.joblib", register_as="leafdoctor-classifier")
    t.end_run()
    rows = t.export_rows()
    assert len(rows) == 1
    assert rows[0]["classifier"] == "rf" and rows[0]["val_accuracy"] == 0.8 and rows[0]["test_accuracy"] == 0.9
    assert set(PARAM_KEYS) <= set(rows[0]) and set(METRIC_KEYS) <= set(rows[0])
    header = rows_to_csv(rows).splitlines()[0]
    assert "test_macro_f1" in header and "log_loss" in header
    saved = json.loads(next((tmp_path / "leafdoctor").glob("*.json")).read_text())
    assert saved["alias"] == "leafdoctor-classifier@champion"  # an alias, not a deprecated stage


def test_mlflow_tracker_uses_aliases(tmp_path):
    pytest.importorskip("mlflow")
    from leafdoctor.tracking import MlflowTracker
    t = MlflowTracker("leafdoctor-test", f"file:{tmp_path / 'mlruns'}")
    t.start_run("r")
    t.log_params({"seed": 1})
    t.log_eval(_report())
    t.end_run()
    assert t.export_rows()[0]["val_accuracy"] == 0.8


def test_settings_from_env(tmp_path):
    s = Settings.from_env({"LEAFDOCTOR_SEED": "7", "LEAFDOCTOR_WORK_DIR": str(tmp_path)}, dotenv=None)
    assert s.seed == 7 and s.splits_dir == tmp_path / "splits" and s.tracker == "json"
    with pytest.raises(ConfigError):
        Settings.from_env({"LEAFDOCTOR_SEED": "x"}, dotenv=None)
    with pytest.raises(ConfigError):
        Settings.from_env({"LEAFDOCTOR_TRACKER": "wandb"}, dotenv=None)
    with pytest.raises(ConfigError):
        Settings.from_env({"LEAFDOCTOR_IMAGE_SIZE": "8"}, dotenv=None)


def test_dotenv_is_read_and_environment_wins(tmp_path):
    f = tmp_path / ".env"
    f.write_text("LEAFDOCTOR_SEED=3\nLEAFDOCTOR_EXPERIMENT=from-file\n", encoding="utf-8")
    s = Settings.from_env({"LEAFDOCTOR_SEED": "9"}, dotenv=f)
    assert s.seed == 9 and s.experiment == "from-file"
