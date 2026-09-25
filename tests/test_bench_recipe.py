"""Tests for the training recipe: optimizer schedules and VGG-5 augmentation."""

import dataclasses
import math

import pytest

from fabricpc.bench import registry
from fabricpc.bench.registry import ROWS
from fabricpc.bench.runner import run_trial
from fabricpc.utils.data import AugmentedImageLoader
from tests.test_bench_runner import fake_mnist_loaders


def test_every_rows_optimizer_is_built_from_the_total_step_count():
    for row in ROWS.values():
        assert row.optimizer_factory(1000) is not None


def test_vgg_learning_rate_warms_up_then_decays_to_zero():
    total = 1000
    lr = registry._vgg_lr_schedule(total)
    warmup = int(total * registry._VGG_WARMUP_FRACTION)
    assert float(lr(0)) == pytest.approx(0.0)
    assert float(lr(warmup)) == pytest.approx(registry._VGG_PEAK_LR)
    assert float(lr(warmup // 2)) < registry._VGG_PEAK_LR
    assert float(lr(total)) == pytest.approx(0.0, abs=1e-12)
    assert float(lr(total // 2)) < registry._VGG_PEAK_LR


def test_vgg_training_images_are_augmented_but_test_images_are_not(monkeypatch):
    made = []

    class FakeCifar:
        def __init__(self, split, **kwargs):
            made.append((split, kwargs))

        def __iter__(self):
            return iter([])

        def __len__(self):
            return 0

    import fabricpc.utils.data.dataloader as dl

    monkeypatch.setattr(dl, "Cifar10Loader", FakeCifar)
    train, test = ROWS["cifar10-vgg5-spc"].loader_factory(1000)

    assert isinstance(train, AugmentedImageLoader)
    assert isinstance(train.base_loader, FakeCifar)
    assert not isinstance(test, AugmentedImageLoader)
    assert train.flip_prob == 0.5 and train.crop_pad == 4


def test_runner_passes_the_real_step_count_to_the_optimizer(tmp_path, rng_key):
    seen = []
    base = ROWS["mnist-mlp-backprop"]

    def recording(total_steps):
        seen.append(total_steps)
        return base.optimizer_factory(total_steps)

    row = dataclasses.replace(base, optimizer_factory=recording)
    loaders = fake_mnist_loaders(rng_key, n_batches=3)

    run_trial(
        row, 0, tmp_path, loaders=loaders, num_epochs=1.5, warmup_steps=1, timed_steps=1
    )

    assert seen == [math.ceil(3 * 1.5)]
