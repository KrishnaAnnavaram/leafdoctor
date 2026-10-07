"""Settings read from environment variables (and an optional local .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

TRACKERS = ("json", "mlflow", "none")


class ConfigError(ValueError):
    """A setting has a value that leafdoctor cannot use."""


def load_dotenv(path: str | Path = ".env") -> dict[str, str]:
    """Read KEY=VALUE lines from a .env file. Existing environment values win."""
    values: dict[str, str] = {}
    p = Path(path)
    if not p.is_file():
        return values
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _int(env: Mapping[str, str], name: str, default: int, minimum: int) -> int:
    raw = env.get(name, "") or str(default)
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}, got {value}")
    return value


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    work_dir: Path
    seed: int
    image_size: int
    batch_size: int
    tracker: str
    experiment: str
    device: str
    mlflow_uri: str | None

    @property
    def splits_dir(self) -> Path:
        return self.work_dir / "splits"

    @property
    def cache_dir(self) -> Path:
        return self.work_dir / "features"

    @property
    def models_dir(self) -> Path:
        return self.work_dir / "models"

    @property
    def runs_dir(self) -> Path:
        return self.work_dir / "runs"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None, dotenv: str | Path | None = ".env") -> "Settings":
        merged: dict[str, str] = {}
        if dotenv is not None:
            merged.update(load_dotenv(dotenv))
        merged.update(os.environ if env is None else env)
        tracker = (merged.get("LEAFDOCTOR_TRACKER") or "json").lower()
        if tracker not in TRACKERS:
            raise ConfigError(f"LEAFDOCTOR_TRACKER must be one of {TRACKERS}, got {tracker!r}")
        return cls(
            data_dir=Path(merged.get("LEAFDOCTOR_DATA_DIR") or "data"),
            work_dir=Path(merged.get("LEAFDOCTOR_WORK_DIR") or "artifacts"),
            seed=_int(merged, "LEAFDOCTOR_SEED", 42, 0),
            image_size=_int(merged, "LEAFDOCTOR_IMAGE_SIZE", 128, 32),
            batch_size=_int(merged, "LEAFDOCTOR_BATCH_SIZE", 32, 1),
            tracker=tracker,
            experiment=merged.get("LEAFDOCTOR_EXPERIMENT") or "leafdoctor",
            device=merged.get("LEAFDOCTOR_DEVICE") or "cpu",
            mlflow_uri=merged.get("MLFLOW_TRACKING_URI") or None,
        )
