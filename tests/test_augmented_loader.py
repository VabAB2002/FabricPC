"""Tests for AugmentedImageLoader (random flips and crops for image batches)."""

import numpy as np

from fabricpc.utils.data.dataloader import AugmentedImageLoader
from tests.conftest import ListLoader


def _batches(n=3, b=4, h=8, w=8, c=3):
    rng = np.random.default_rng(0)
    return [
        (rng.normal(size=(b, h, w, c)).astype(np.float32), np.eye(10)[:b])
        for _ in range(n)
    ]


def _all(loader):
    return [(x.copy(), y.copy()) for x, y in loader]


def test_shapes_labels_and_length_are_unchanged():
    base = ListLoader(_batches())
    aug = AugmentedImageLoader(base, seed=1)
    assert len(aug) == 3
    for (x, y), (bx, by) in zip(aug, base):
        assert x.shape == bx.shape
        assert x.dtype == bx.dtype
        np.testing.assert_array_equal(y, by)


def test_flip_only_mirrors_every_image_left_to_right():
    base = ListLoader(_batches())
    aug = AugmentedImageLoader(base, flip_prob=1.0, crop_pad=0, seed=1)
    for (x, _), (bx, _) in zip(aug, base):
        np.testing.assert_array_equal(x, bx[:, :, ::-1, :])


def test_nothing_on_means_nothing_changes():
    base = ListLoader(_batches())
    aug = AugmentedImageLoader(base, flip_prob=0.0, crop_pad=0, seed=1)
    for (x, _), (bx, _) in zip(aug, base):
        np.testing.assert_array_equal(x, bx)


def test_each_crop_is_a_window_of_the_zero_padded_image():
    base = ListLoader(_batches(n=1))
    aug = AugmentedImageLoader(base, flip_prob=0.0, crop_pad=2, seed=3)
    (x, _), (bx, _) = next(iter(aug)), next(iter(base))
    padded = np.pad(bx, ((0, 0), (2, 2), (2, 2), (0, 0)))
    for i in range(x.shape[0]):
        windows = [
            padded[i, dy : dy + 8, dx : dx + 8] for dy in range(5) for dx in range(5)
        ]
        assert any(np.array_equal(x[i], wnd) for wnd in windows)


def test_same_seed_gives_the_same_stream_and_new_seed_a_different_one():
    a = _all(AugmentedImageLoader(ListLoader(_batches()), seed=5))
    b = _all(AugmentedImageLoader(ListLoader(_batches()), seed=5))
    c = _all(AugmentedImageLoader(ListLoader(_batches()), seed=6))
    for (xa, _), (xb, _) in zip(a, b):
        np.testing.assert_array_equal(xa, xb)
    assert any(not np.array_equal(xa, xc) for (xa, _), (xc, _) in zip(a, c))


def test_every_epoch_gets_fresh_augmentation():
    aug = AugmentedImageLoader(ListLoader(_batches()), seed=5)
    first, second = _all(aug), _all(aug)
    assert any(not np.array_equal(x1, x2) for (x1, _), (x2, _) in zip(first, second))
