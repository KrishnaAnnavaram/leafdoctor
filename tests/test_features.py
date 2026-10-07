"""Handcrafted features, extractors, augmentation and the feature cache (problems 6, 8 and 9)."""
import numpy as np
import pytest

from leafdoctor.augment import augment, view_rng
from leafdoctor.features.backbones import Fusion, PixelProjection, build_extractor
from leafdoctor.features.handcrafted import (HandcraftedExtractor, glcm, glcm_features, hsv_histogram,
                                             lbp_histogram, quantise)
from leafdoctor.features.store import FeatureStore
from leafdoctor.images import rgb_to_hsv, to_gray
from leafdoctor.synthetic import make_leaf


@pytest.fixture(scope="module")
def leaf():
    return make_leaf("Tomato___Early_blight", np.random.default_rng(5), 64)


def test_hsv_matches_colorsys():
    import colorsys
    px = np.array([[[200, 30, 90], [10, 250, 40], [0, 0, 0], [128, 128, 128]]], dtype=np.uint8)
    ours = rgb_to_hsv(px)[0]
    for i, (r, g, b) in enumerate(px[0] / 255.0):
        assert np.allclose(ours[i], colorsys.rgb_to_hsv(r, g, b), atol=1e-9)


def test_hsv_histogram_is_a_distribution(leaf):
    h = hsv_histogram(leaf)
    assert h.shape == (512,) and np.isclose(h.sum(), 1.0) and (h >= 0).all()


def test_glcm_uses_32_levels_and_is_symmetric(leaf):
    q = quantise(to_gray(leaf), 32)
    assert q.min() >= 0 and q.max() <= 31
    p = glcm(q, 0, 1, 32)
    assert p.shape == (32, 32) and np.allclose(p, p.T) and np.isclose(p.sum(), 1.0)


def test_glcm_on_constant_image():
    flat = np.full((20, 20, 3), 100, dtype=np.uint8)
    f = glcm_features(flat, distances=(1,))
    contrast, dissim, homog, energy, corr, asm = f[:6]
    assert contrast == 0 and dissim == 0 and homog == 1 and energy == 1 and asm == 1


def test_texture_features_are_rotation_invariant(leaf):
    """The old GLCM used one angle (0 degrees). The mean over 4 angles and riu2 LBP do not change."""
    for view in (np.rot90(leaf), np.fliplr(leaf)):
        assert np.allclose(glcm_features(view), glcm_features(leaf))
        assert np.allclose(lbp_histogram(view), lbp_histogram(leaf))


def test_lbp_histogram_shape(leaf):
    h = lbp_histogram(leaf)
    assert h.shape == (10,) and np.isclose(h.sum(), 1.0)


def test_handcrafted_extractor_dim(leaf):
    ex = HandcraftedExtractor()
    X = ex.extract([leaf, leaf])
    assert X.shape == (2, ex.dim) == (2, 512 + 36 + 10)
    assert ex.extract([]).shape == (0, ex.dim)


def test_pixel_projection_is_deterministic_and_fusion_concatenates(leaf):
    a, b = PixelProjection(seed=3), PixelProjection(seed=3)
    assert np.array_equal(a.extract([leaf]), b.extract([leaf]))
    fused = build_extractor("pixelproj+handcrafted")
    assert isinstance(fused, Fusion) and fused.dim == 128 + 558
    assert fused.extract([leaf]).shape == (1, fused.dim)


def test_unknown_extractor_spec():
    with pytest.raises(ValueError):
        build_extractor("sift")


def test_augment_is_reproducible_and_changes_the_image(leaf):
    a = augment(leaf, view_rng(1, 2, 1))
    b = augment(leaf, view_rng(1, 2, 1))
    c = augment(leaf, view_rng(1, 2, 2))
    assert np.array_equal(a, b) and not np.array_equal(a, c) and a.shape == leaf.shape


class CountingExtractor:
    name = "counting"
    dim = 2

    def __init__(self):
        self.calls = []

    def extract(self, images):
        self.calls.append(len(images))
        return np.array([[im.mean(), im.std()] for im in images])


def test_store_batches_and_caches(tmp_path, splits):
    rows = splits["train"][:10]
    store = FeatureStore(tmp_path, batch_size=4)
    ex = CountingExtractor()
    fs = store.features(rows, ex, "train", 32)
    assert len(fs) == 10 and max(ex.calls) <= 4  # streamed in batches (problem 9)
    fs2 = store.features(rows, ex, "train", 32)
    assert store.hits == 1 and len(ex.calls) == 3  # second call reads the cache
    assert np.allclose(fs.X, fs2.X) and list(fs2.labels) == [r.label for r in rows]
    store.features(rows[:9], ex, "train", 32)
    assert store.misses == 2  # a different manifest gives a different cache key


def test_train_views_add_augmented_rows_only_when_asked(tmp_path, splits):
    """Augmentation is part of training (problem 6). Val and test get train_views=0."""
    rows = splits["train"][:5]
    store = FeatureStore(None, batch_size=8)
    plain = store.features(rows, CountingExtractor(), "val", 32, train_views=0)
    aug = store.features(rows, CountingExtractor(), "train", 32, train_views=2, seed=1)
    assert len(plain) == 5 and len(aug) == 15
    assert list(aug.groups[:3]) == [rows[0].group_id] * 3
