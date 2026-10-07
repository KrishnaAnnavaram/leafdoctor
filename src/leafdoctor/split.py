"""Leakage-safe, group-aware, stratified train/val/test split.

1. Rows that share a leaf id are one group.
2. Rows whose dihedral-invariant hashes are near (Hamming distance <= max_distance) join the same
   group (union-find, with a band index so that the search is not quadratic).
3. Whole groups go to one split, stratified by label and seeded.
4. `check_leakage` proves that no group and no near-duplicate hash crosses two splits.
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from .images import hamming
from .manifest import Row, read_manifest, write_manifest

SPLITS = ("train", "val", "test")


class LeakageError(ValueError):
    """A group or a near-duplicate image is in more than one split."""


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def _bands(value: int, n_bands: int) -> list[tuple[int, int]]:
    width = 64 // n_bands
    mask = (1 << width) - 1
    return [(b, (value >> (b * width)) & mask) for b in range(n_bands)]


def near_duplicate_pairs(hashes: Sequence[int], max_distance: int) -> list[tuple[int, int]]:
    """Index pairs with Hamming distance <= max_distance.

    With max_distance + 1 bands, two hashes within the distance share at least one band exactly
    (pigeonhole), so only rows in the same band bucket are compared.
    """
    n_bands = max_distance + 1
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, h in enumerate(hashes):
        for key in _bands(h, n_bands):
            buckets[key].append(i)
    pairs: set[tuple[int, int]] = set()
    for members in buckets.values():
        for a_pos in range(len(members)):
            for b in members[a_pos + 1:]:
                a = members[a_pos]
                if (a, b) not in pairs and hamming(hashes[a], hashes[b]) <= max_distance:
                    pairs.add((a, b))
    return sorted(pairs)


def merge_groups(rows: Sequence[Row], max_distance: int = 4) -> list[Row]:
    """Return rows whose group_id also joins near-duplicate images."""
    uf = _UnionFind(len(rows))
    first_by_group: dict[str, int] = {}
    for i, r in enumerate(rows):
        if r.group_id in first_by_group:
            uf.union(first_by_group[r.group_id], i)
        else:
            first_by_group[r.group_id] = i
    hashed = [i for i, r in enumerate(rows) if r.phash]
    if max_distance >= 0 and hashed:
        values = [int(rows[i].phash, 16) for i in hashed]
        for a, b in near_duplicate_pairs(values, max_distance):
            uf.union(hashed[a], hashed[b])
    return [r.with_(group_id=rows[uf.find(i)].group_id) for i, r in enumerate(rows)]


@dataclass
class SplitResult:
    splits: dict[str, list[Row]]
    seed: int
    fractions: tuple[float, float, float]
    dropped_augmented: int = 0
    mixed_label_groups: int = 0
    notes: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        out = {
            "seed": self.seed,
            "fractions": list(self.fractions),
            "dropped_augmented": self.dropped_augmented,
            "mixed_label_groups": self.mixed_label_groups,
            "notes": self.notes,
        }
        for name, rows in self.splits.items():
            out[name] = {
                "images": len(rows),
                "groups": len({r.group_id for r in rows}),
                "classes": dict(sorted(Counter(r.label for r in rows).items())),
            }
        return out


def group_split(rows: Sequence[Row], fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
                seed: int = 42) -> SplitResult:
    """Assign whole groups to train/val/test. Stratify by the majority label of each group."""
    if len(fractions) != 3 or abs(sum(fractions) - 1.0) > 1e-6 or min(fractions) < 0:
        raise ValueError(f"fractions must be 3 non-negative numbers with sum 1, got {fractions}")
    groups: dict[str, list[Row]] = defaultdict(list)
    for r in rows:
        groups[r.group_id].append(r)
    by_label: dict[str, list[str]] = defaultdict(list)
    mixed = 0
    for gid, members in groups.items():
        labels = Counter(m.label for m in members)
        if len(labels) > 1:
            mixed += 1
        by_label[labels.most_common(1)[0][0]].append(gid)

    rng = random.Random(seed)
    splits: dict[str, list[Row]] = {s: [] for s in SPLITS}
    notes: list[str] = []
    for label in sorted(by_label):
        gids = sorted(by_label[label])
        rng.shuffle(gids)
        total = sum(len(groups[g]) for g in gids)
        target = [f * total for f in fractions]
        have = [0, 0, 0]
        for gid in gids:
            deficits = [target[k] - have[k] for k in range(3)]
            k = max(range(3), key=lambda j: (deficits[j], -j))
            splits[SPLITS[k]].extend(groups[gid])
            have[k] += len(groups[gid])
        for k, name in enumerate(SPLITS):
            if fractions[k] > 0 and have[k] == 0:
                notes.append(f"class {label} has no image in {name} ({len(gids)} group(s))")
    for name in SPLITS:
        splits[name].sort(key=lambda r: r.path)
    return SplitResult(splits=splits, seed=seed, fractions=tuple(fractions), mixed_label_groups=mixed, notes=notes)


def random_image_split(rows: Sequence[Row], fractions=(0.7, 0.15, 0.15), seed: int = 42) -> SplitResult:
    """The unsafe image-level split. Use it only to measure leakage (see `leafdoctor leakage-check`)."""
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    a = int(round(fractions[0] * n))
    b = a + int(round(fractions[1] * n))
    parts = {"train": shuffled[:a], "val": shuffled[a:b], "test": shuffled[b:]}
    return SplitResult(splits=parts, seed=seed, fractions=tuple(fractions))


def attach_augmented(result: SplitResult, augmented: Sequence[Row], max_distance: int = 4) -> SplitResult:
    """Add offline-augmented copies to train only, and only for leaves that are in train.

    An augmented copy of a val or test leaf is dropped. This is how the augmented dataset can help
    training without leakage.
    """
    held_groups = {r.group_id for s in ("val", "test") for r in result.splits[s]}
    held_hashes = [int(r.phash, 16) for s in ("val", "test") for r in result.splits[s] if r.phash]
    keep: list[Row] = []
    dropped = 0
    for r in augmented:
        if r.group_id in held_groups:
            dropped += 1
            continue
        if r.phash and held_hashes:
            h = int(r.phash, 16)
            if any(hamming(h, x) <= max_distance for x in held_hashes):
                dropped += 1
                continue
        keep.append(r)
    result.splits["train"] = sorted(result.splits["train"] + keep, key=lambda r: r.path)
    result.dropped_augmented += dropped
    return result


def check_leakage(splits: dict[str, Sequence[Row]], max_distance: int = 4) -> dict:
    """Count groups and near-duplicate hashes that cross splits. Raise LeakageError if any."""
    owner: dict[str, str] = {}
    shared_groups: set[str] = set()
    for name, rows in splits.items():
        for r in rows:
            prev = owner.setdefault(r.group_id, name)
            if prev != name:
                shared_groups.add(r.group_id)
    hashed = [(name, int(r.phash, 16)) for name, rows in splits.items() for r in rows if r.phash]
    cross_pairs = 0
    if hashed and max_distance >= 0:
        for a, b in near_duplicate_pairs([h for _, h in hashed], max_distance):
            if hashed[a][0] != hashed[b][0]:
                cross_pairs += 1
    report = {"shared_groups": len(shared_groups), "cross_split_near_duplicates": cross_pairs}
    if shared_groups or cross_pairs:
        raise LeakageError(f"leakage between splits: {report}")
    return report


def write_splits(result: SplitResult, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in result.splits.items():
        write_manifest(rows, out / f"{name}.csv")
    (out / "split_report.json").write_text(json.dumps(result.summary(), indent=2), encoding="utf-8")
    return out


def read_splits(split_dir: str | Path, check_files: bool = True) -> dict[str, list[Row]]:
    d = Path(split_dir)
    return {name: read_manifest(d / f"{name}.csv", check_files=check_files) for name in SPLITS}
