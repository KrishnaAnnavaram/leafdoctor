"""Image manifests: one validated row for each image, written atomically.

A manifest replaces the copied `Merged_Dataset` folder. No image is copied, so a new run can never
mix old and new files. Each run writes complete manifest files with `os.replace`.
"""
from __future__ import annotations

import csv
import os
import re
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from .classes import UnknownClassError, canonical_label
from .images import IMAGE_SUFFIXES, ImageError, canonical_hash, is_grayscale, load_image

COLUMNS = ("path", "label", "source", "group_id", "phash")

# Suffixes that offline augmentation adds to a file stem, for example "_flipLR" or "_270deg".
_AUG_SUFFIX = re.compile(r"(?:_(?:new\d+deg(?:flip(?:lr|tb))?|\d+deg|rot\d+|aug\d+|flip(?:lr|tb)))+$", re.I)


class ManifestError(ValueError):
    """A manifest does not obey the schema. `problems` lists each fault."""

    def __init__(self, problems: Sequence[str]):
        self.problems = list(problems)
        head = "; ".join(self.problems[:5])
        more = f" (+{len(self.problems) - 5} more)" if len(self.problems) > 5 else ""
        super().__init__(f"{len(self.problems)} manifest problem(s): {head}{more}")


@dataclass(frozen=True)
class Row:
    path: str
    label: str
    source: str
    group_id: str
    phash: str = ""

    def with_(self, **changes) -> "Row":
        data = asdict(self)
        data.update(changes)
        return Row(**data)


@dataclass
class ScanReport:
    rows: list[Row]
    unknown_folders: list[str] = field(default_factory=list)
    corrupt: list[str] = field(default_factory=list)
    grayscale: int = 0

    def class_counts(self) -> dict[str, int]:
        return dict(sorted(Counter(r.label for r in self.rows).items()))

    def imbalance_ratio(self) -> float:
        counts = list(self.class_counts().values())
        return max(counts) / min(counts) if counts else 0.0


def group_id_from_name(file_name: str) -> str:
    """Return the leaf id in a file name.

    PlantVillage-style names start with a leaf UUID before "___". Offline-augmented copies keep that
    prefix, or add a suffix such as "_flipLR". Both forms map to the same group id.
    """
    stem = Path(file_name).stem
    if "___" in stem:
        return stem.split("___", 1)[0].lower()
    return _AUG_SUFFIX.sub("", stem).lower() or stem.lower()


def scan_folder(root: str | Path, source: str, compute_hash: bool = True, check_images: bool = True) -> ScanReport:
    """Scan root/<class folder>/<image> into manifest rows.

    Unknown class folders and corrupt images are reported, not included.
    """
    root = Path(root)
    if not root.is_dir():
        raise ManifestError([f"folder {root} does not exist"])
    report = ScanReport(rows=[])
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        try:
            label = canonical_label(class_dir.name)
        except UnknownClassError:
            report.unknown_folders.append(class_dir.name)
            continue
        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            phash = ""
            if check_images or compute_hash:
                try:
                    arr = load_image(img_path)
                except ImageError:
                    report.corrupt.append(str(img_path))
                    continue
                if is_grayscale(arr):
                    report.grayscale += 1
                if compute_hash:
                    phash = f"{canonical_hash(arr):016x}"
            report.rows.append(
                Row(
                    path=str(img_path.resolve()),
                    label=label,
                    source=source,
                    group_id=f"{source}:{group_id_from_name(img_path.name)}",
                    phash=phash,
                )
            )
    return report


def validate_rows(rows: Sequence[Row], check_files: bool = True) -> None:
    """Raise ManifestError if a row breaks the schema."""
    problems: list[str] = []
    if not rows:
        problems.append("manifest is empty")
    seen: set[str] = set()
    for i, r in enumerate(rows):
        where = f"row {i + 1}"
        if not r.path:
            problems.append(f"{where}: empty path")
        if r.path in seen:
            problems.append(f"{where}: duplicate path {r.path}")
        seen.add(r.path)
        try:
            if canonical_label(r.label) != r.label:
                problems.append(f"{where}: label {r.label!r} is not canonical")
        except UnknownClassError:
            problems.append(f"{where}: unknown label {r.label!r}")
        if not r.group_id:
            problems.append(f"{where}: empty group_id")
        if r.phash and not re.fullmatch(r"[0-9a-f]{16}", r.phash):
            problems.append(f"{where}: bad phash {r.phash!r}")
        if check_files and r.path and not Path(r.path).is_file():
            problems.append(f"{where}: file not found {r.path}")
    if problems:
        raise ManifestError(problems)


def write_manifest(rows: Iterable[Row], path: str | Path) -> Path:
    """Write rows as CSV. The file is replaced in one step, never appended."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_", suffix=".csv")
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=COLUMNS)
            w.writeheader()
            for r in rows:
                w.writerow(asdict(r))
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return path


def read_manifest(path: str | Path, check_files: bool = True) -> list[Row]:
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ManifestError([f"{path}: missing column(s) {missing}"])
        rows = [Row(**{c: (rec.get(c) or "") for c in COLUMNS}) for rec in reader]
    validate_rows(rows, check_files=check_files)
    return rows
