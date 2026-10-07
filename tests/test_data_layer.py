"""Class harmonisation, manifests, hashing and the leakage-safe split (reference problems 1, 2 and 5)."""
import numpy as np
import pytest

from leafdoctor.classes import CANONICAL_CLASSES, UnknownClassError, canonical_label
from leafdoctor.images import canonical_hash, hamming, save_image
from leafdoctor.manifest import (ManifestError, Row, group_id_from_name, read_manifest, scan_folder, validate_rows,
                                 write_manifest)
from leafdoctor.split import (LeakageError, attach_augmented, check_leakage, group_split, merge_groups,
                              near_duplicate_pairs, random_image_split)
from leafdoctor.synthetic import make_leaf

PLANTVILLAGE_FOLDERS = [
    "Pepper__bell___Bacterial_spot", "Pepper__bell___healthy", "Potato___Early_blight", "Potato___Late_blight",
    "Potato___healthy", "Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_Late_blight", "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot", "Tomato_Spider_mites_Two_spotted_spider_mite", "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus", "Tomato__Tomato_mosaic_virus", "Tomato_healthy",
]


def test_all_plantvillage_folders_map_to_canonical_classes():
    mapped = {canonical_label(f) for f in PLANTVILLAGE_FOLDERS}
    assert len(mapped) == 15 and mapped <= set(CANONICAL_CLASSES)


def test_canonical_names_map_to_themselves_and_unknown_raises():
    assert len(CANONICAL_CLASSES) == 38
    assert all(canonical_label(c) == c for c in CANONICAL_CLASSES)
    with pytest.raises(UnknownClassError):
        canonical_label("Banana___Panama_disease")


@pytest.mark.parametrize("name, group", [
    ("0a5e9323-dbad___FREC_Scab 3417.JPG", "0a5e9323-dbad"),
    ("0a5e9323-dbad___FREC_Scab 3417_new30degFlipLR.JPG", "0a5e9323-dbad"),
    ("leaf_17_flipLR.png", "leaf_17"),
    ("leaf_17_270deg.png", "leaf_17"),
    ("leaf_17.png", "leaf_17"),
    ("leaf_18.png", "leaf_18"),
])
def test_group_id_joins_augmented_copies_but_not_other_leaves(name, group):
    assert group_id_from_name(name) == group


def test_scan_reports_unknown_folders_and_corrupt_files(tmp_path):
    rng = np.random.default_rng(0)
    save_image(make_leaf("Tomato___healthy", rng, 40), tmp_path / "Tomato_healthy" / "a___1.png")
    (tmp_path / "Tomato_healthy" / "broken.png").write_bytes(b"not an image")
    save_image(make_leaf("Tomato___healthy", rng, 40), tmp_path / "Mystery_class" / "b.png")
    rep = scan_folder(tmp_path, "t")
    assert [r.label for r in rep.rows] == ["Tomato___healthy"]
    assert rep.unknown_folders == ["Mystery_class"]
    assert len(rep.corrupt) == 1


def test_manifest_round_trip_and_validation(tmp_path, lab_rows):
    path = write_manifest(lab_rows, tmp_path / "m.csv")
    assert read_manifest(path) == list(lab_rows)
    bad = [lab_rows[0], lab_rows[0], Row("missing.png", "Not___a_class", "s", "")]
    with pytest.raises(ManifestError) as err:
        validate_rows(bad)
    text = str(err.value)
    assert "duplicate path" in text and "unknown label" in text and "empty group_id" in text


def test_manifest_write_replaces_the_file(tmp_path, lab_rows):
    path = tmp_path / "m.csv"
    write_manifest(lab_rows, path)
    write_manifest(lab_rows[:3], path)
    assert len(read_manifest(path)) == 3  # a re-run never keeps stale rows (problem 5)


def test_canonical_hash_is_invariant_to_flips_and_rotations():
    img = make_leaf("Tomato___Early_blight", np.random.default_rng(3), 64)
    h = canonical_hash(img)
    for view in (np.fliplr(img), np.flipud(img), np.rot90(img), np.rot90(img, 3)):
        assert canonical_hash(view) == h
    other = canonical_hash(make_leaf("Corn_(maize)___Common_rust_", np.random.default_rng(4), 64))
    assert hamming(h, other) > 3


def test_band_index_finds_exactly_the_near_pairs():
    rng = np.random.default_rng(0)
    hashes = [int(x) for x in rng.integers(0, 2**63, 200)]
    hashes.append(hashes[5] ^ 0b101)  # distance 2 from item 5
    brute = {(i, j) for i in range(len(hashes)) for j in range(i + 1, len(hashes))
             if hamming(hashes[i], hashes[j]) <= 3}
    assert set(near_duplicate_pairs(hashes, 3)) == brute
    assert (5, 200) in brute


def test_merge_groups_joins_renamed_duplicates():
    rows = [Row("a.png", "Tomato___healthy", "s", "s:a", "00000000000000ff"),
            Row("b.png", "Tomato___healthy", "s", "s:b", "00000000000000fe"),
            Row("c.png", "Tomato___healthy", "s", "s:c", "ffffffff00000000")]
    merged = merge_groups(rows, 3)
    assert merged[0].group_id == merged[1].group_id != merged[2].group_id


def test_group_split_has_no_leakage_and_is_stratified(splits, lab_rows):
    assert check_leakage(splits, 3) == {"shared_groups": 0, "cross_split_near_duplicates": 0}
    assert sum(len(v) for v in splits.values()) == len(lab_rows)
    labels = {r.label for r in lab_rows}
    for name in ("train", "val", "test"):
        assert {r.label for r in splits[name]} == labels
    assert len(splits["train"]) > len(splits["val"]) > 0


def test_group_split_is_seeded(lab_rows):
    a = group_split(lab_rows, seed=11).splits
    b = group_split(lab_rows, seed=11).splits
    c = group_split(lab_rows, seed=12).splits
    assert a == b
    assert a["val"] != c["val"]


def test_image_level_split_leaks_and_the_check_detects_it(lab_rows):
    bad = random_image_split(lab_rows, seed=0).splits
    with pytest.raises(LeakageError):
        check_leakage(bad, 3)


def test_augmented_copies_of_held_out_leaves_are_dropped(splits):
    held = splits["val"][0]
    train_one = splits["train"][0]
    aug = [held.with_(path=held.path + "_flipLR.png"), train_one.with_(path=train_one.path + "_flipLR.png")]
    from leafdoctor.split import SplitResult
    res = attach_augmented(SplitResult({k: list(v) for k, v in splits.items()}, 0, (0.7, 0.15, 0.15)), aug, 3)
    assert res.dropped_augmented == 1
    assert aug[1] in res.splits["train"] and aug[0] not in res.splits["train"]
    check_leakage(res.splits, 3)


def test_bad_fractions_are_rejected(lab_rows):
    with pytest.raises(ValueError):
        group_split(lab_rows, (0.5, 0.5, 0.5))
