"""Experiment tracking with one fixed vocabulary of parameter and metric names.

The writer and the exporter use the same constants and the same experiment name, so the exported
table always has the logged values. A metric can only come from an `EvalReport`: there is no
public function that logs a free number. The MLflow tracker uses model aliases, not stages.
"""
from __future__ import annotations

import csv
import io
import json
import math
import time
import uuid
from pathlib import Path
from typing import Protocol

from .metrics import EvalReport

PARAM_KEYS = ("extractor", "classifier", "imbalance", "train_views", "seed", "image_size", "selected_on")
METRIC_NAMES = ("accuracy", "macro_f1", "balanced_accuracy", "log_loss", "ece")
METRIC_SPLITS = ("val", "test", "field")
METRIC_KEYS = tuple(f"{s}_{m}" for s in METRIC_SPLITS for m in METRIC_NAMES)
MODEL_ALIAS = "champion"


class TrackingError(ValueError):
    pass


def _check_params(params: dict) -> dict:
    unknown = sorted(set(params) - set(PARAM_KEYS))
    if unknown:
        raise TrackingError(f"unknown param keys {unknown}; allowed: {PARAM_KEYS}")
    return {k: str(v) for k, v in params.items()}


def metrics_from_report(report: EvalReport) -> dict[str, float]:
    if report.split not in METRIC_SPLITS:
        raise TrackingError(f"split {report.split!r} is not one of {METRIC_SPLITS}")
    out = {}
    for name, value in report.headline().items():
        if not math.isfinite(value):
            raise TrackingError(f"metric {report.split}_{name} is not finite: {value}")
        out[f"{report.split}_{name}"] = float(value)
    return out


class Tracker(Protocol):
    def start_run(self, name: str) -> str: ...
    def log_params(self, params: dict) -> None: ...
    def log_eval(self, report: EvalReport) -> None: ...
    def log_model(self, path: Path, register_as: str | None = None) -> None: ...
    def end_run(self) -> None: ...


class NullTracker:
    def start_run(self, name: str) -> str:
        return uuid.uuid4().hex[:12]

    def log_params(self, params: dict) -> None:
        _check_params(params)

    def log_eval(self, report: EvalReport) -> None:
        metrics_from_report(report)

    def log_model(self, path: Path, register_as: str | None = None) -> None:
        return None

    def end_run(self) -> None:
        return None


class JsonTracker:
    """Writes runs/<experiment>/<run_id>.json. No server needed."""

    def __init__(self, runs_dir: str | Path, experiment: str):
        self.dir = Path(runs_dir) / experiment
        self.experiment = experiment
        self._run: dict | None = None

    def start_run(self, name: str) -> str:
        run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        self._run = {"run_id": run_id, "name": name, "experiment": self.experiment, "params": {}, "metrics": {},
                     "model": None, "alias": None, "reports": {}}
        return run_id

    def _need_run(self) -> dict:
        if self._run is None:
            raise TrackingError("no active run: call start_run first")
        return self._run

    def log_params(self, params: dict) -> None:
        self._need_run()["params"].update(_check_params(params))

    def log_eval(self, report: EvalReport) -> None:
        run = self._need_run()
        run["metrics"].update(metrics_from_report(report))
        run["reports"][report.split] = report.to_dict()

    def log_model(self, path: Path, register_as: str | None = None) -> None:
        run = self._need_run()
        run["model"] = str(path)
        run["alias"] = f"{register_as}@{MODEL_ALIAS}" if register_as else None

    def end_run(self) -> None:
        run = self._need_run()
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / f"{run['run_id']}.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
        self._run = None

    def export_rows(self) -> list[dict]:
        rows = []
        for f in sorted(self.dir.glob("*.json")):
            run = json.loads(f.read_text(encoding="utf-8"))
            row = {"run_id": run["run_id"], "name": run["name"]}
            row.update({k: run["params"].get(k, "") for k in PARAM_KEYS})
            row.update({k: run["metrics"].get(k, "") for k in METRIC_KEYS})
            rows.append(row)
        return rows


class MlflowTracker:  # pragma: no cover - needs the optional mlflow extra
    def __init__(self, experiment: str, tracking_uri: str | None = None):
        import mlflow

        self.mlflow = mlflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)
        self.experiment = experiment

    def start_run(self, name: str) -> str:
        return self.mlflow.start_run(run_name=name).info.run_id

    def log_params(self, params: dict) -> None:
        self.mlflow.log_params(_check_params(params))

    def log_eval(self, report: EvalReport) -> None:
        self.mlflow.log_metrics(metrics_from_report(report))
        self.mlflow.log_dict(report.to_dict(), f"reports/{report.split}.json")

    def log_model(self, path: Path, register_as: str | None = None) -> None:
        self.mlflow.log_artifact(str(path), artifact_path="model")
        if register_as:
            run_id = self.mlflow.active_run().info.run_id
            mv = self.mlflow.register_model(f"runs:/{run_id}/model", register_as)
            self.mlflow.MlflowClient().set_registered_model_alias(register_as, MODEL_ALIAS, mv.version)

    def end_run(self) -> None:
        self.mlflow.end_run()

    def export_rows(self) -> list[dict]:
        df = self.mlflow.search_runs(experiment_names=[self.experiment])
        rows = []
        for _, r in df.iterrows():
            row = {"run_id": r["run_id"], "name": r.get("tags.mlflow.runName", "")}
            row.update({k: r.get(f"params.{k}", "") for k in PARAM_KEYS})
            row.update({k: r.get(f"metrics.{k}", "") for k in METRIC_KEYS})
            rows.append(row)
        return rows


def build_tracker(kind: str, runs_dir: Path, experiment: str, mlflow_uri: str | None = None):
    if kind == "json":
        return JsonTracker(runs_dir, experiment)
    if kind == "mlflow":
        return MlflowTracker(experiment, mlflow_uri)
    if kind == "none":
        return NullTracker()
    raise TrackingError(f"unknown tracker {kind!r}")


def rows_to_csv(rows: list[dict]) -> str:
    buf = io.StringIO()
    cols = ["run_id", "name", *PARAM_KEYS, *METRIC_KEYS]
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow(r)
    return buf.getvalue()
