"""Tests for the continual-learning rows (fabricpc.bench.continual)."""

import dataclasses
import json

import numpy as np
import pytest

from fabricpc.bench import continual
from fabricpc.bench.registry import ROWS


def fake_split(n_per_class, seed=0):
    """Images whose pixels all equal their class, so a batch says its task."""
    labels = np.repeat(np.arange(10), n_per_class)
    images = np.repeat(labels[:, None, None], 28 * 28, axis=1).reshape(-1, 28, 28)
    order = np.random.default_rng(seed).permutation(len(labels))
    return images[order].astype(np.uint8), labels[order].astype(np.int32)


def _classes_in(x):
    raw = x[:, 0] * continual._STD + continual._MEAN  # undo the normalization
    return set(np.round(raw * 255).astype(int).tolist())


def test_the_same_seed_picks_the_same_images_per_class():
    _, labels = fake_split(40)
    a = continual.pick_per_class(labels, 30, seed=1)
    b = continual.pick_per_class(labels, 30, seed=1)
    c = continual.pick_per_class(labels, 30, seed=2)
    assert (a == b).all() and not (a == c).all()
    assert np.bincount(labels[a]).tolist() == [30] * 10


def test_blocks_alternate_between_the_two_tasks():
    block, _ = continual.loaders_from_arrays(
        fake_split(20), fake_split(5), seed=0, per_class=12
    )
    assert len(block) == -(-60 // continual.BATCH_SIZE)  # 5 classes x 12

    seen = []
    for _ in range(4):
        classes = set()
        for x, y in block:
            classes |= _classes_in(x)
            assert y.shape[1] == 5
        seen.append(classes)

    assert seen[0] == seen[2] == {0, 2, 4, 6, 8}
    assert seen[1] == seen[3] == {1, 3, 5, 7, 9}


def test_every_row_of_a_seed_sees_the_same_stream():
    def first_batches(seed):
        block, _ = continual.loaders_from_arrays(
            fake_split(20), fake_split(5), seed=seed, per_class=12
        )
        return [x for _, (x, _) in zip(range(2), block)]

    assert all((a == b).all() for a, b in zip(first_batches(3), first_batches(3)))


def test_the_test_set_marks_each_image_with_its_task():
    _, test = continual.loaders_from_arrays(
        fake_split(20), fake_split(5), seed=0, per_class=12
    )
    batch = next(iter(test))
    classes = np.round((batch["x"][:, 0] * continual._STD + continual._MEAN) * 255)
    assert (batch["task"] == classes.astype(int) % 2).all()
    assert (batch["y"].argmax(1) == classes.astype(int) // 2).all()


def test_forgetting_by_hand():
    # Block 2 trains B: A drops 0.8 -> 0.5. Block 3 trains A: B drops
    # 0.7 -> 0.4. Block 4 trains B: A drops 0.75 -> 0.6. Mean 0.25.
    points = [
        {"accuracy_task_a": a, "accuracy_task_b": b}
        for a, b in [(0.8, 0.1), (0.5, 0.7), (0.75, 0.4), (0.6, 0.65)]
    ]
    s = continual.curve_summary(points)
    assert s["forgetting"] == pytest.approx(0.25)
    assert (s["final_task_a"], s["final_task_b"]) == (0.6, 0.65)


def test_song_metric_and_gain_by_hand():
    # Song et al.'s headline: mean accuracy over the whole run, here the mean
    # of (A + B) / 2 after every block: (0.45 + 0.6 + 0.575 + 0.625) / 4.
    # Gain: what the trained task got in each block from the second on:
    # B 0.1 -> 0.7, A 0.5 -> 0.75, B 0.4 -> 0.65; mean 0.3667. Forgetting
    # (0.25) per unit gained: 0.25 / 0.3667.
    points = [
        {"accuracy_task_a": a, "accuracy_task_b": b}
        for a, b in [(0.8, 0.1), (0.5, 0.7), (0.75, 0.4), (0.6, 0.65)]
    ]
    s = continual.curve_summary(points)
    assert s["mean_accuracy"] == pytest.approx(0.5625)
    assert s["gain"] == pytest.approx(1.1 / 3)
    assert s["forgetting_per_gain"] == pytest.approx(0.25 / (1.1 / 3))


def test_the_family_is_registered_for_all_three_methods():
    for algo in ("spc", "epc", "backprop"):
        row = ROWS[f"fashionmnist-continual-{algo}"]
        assert row.train_config["num_epochs"] == continual.BLOCKS
        assert row.batch_size == continual.BATCH_SIZE
        assert row.curve_summary is continual.curve_summary


def test_a_run_reports_forgetting_and_each_task(tmp_path, monkeypatch):
    from fabricpc.bench import registry
    from fabricpc.bench.__main__ import main

    tiny = dataclasses.replace(
        ROWS["fashionmnist-continual-spc"],
        id="tinycl-continual-spc",
        loader_factory=lambda seed: continual.loaders_from_arrays(
            fake_split(20), fake_split(5), seed=seed, per_class=12
        ),
        train_config={"num_epochs": 4},
    )
    monkeypatch.setitem(registry.ROWS, tiny.id, tiny)

    code = main(
        [tiny.id, "--out", str(tmp_path), "--trials", "2", "--in-process"]
        + ["--warmup", "1", "--timed", "1", "--no-diagnostics"]
    )

    assert code == 0
    trial = json.loads((tmp_path / tiny.id / "trial0.json").read_text())
    assert trial["status"] == "ok", trial["error"]
    assert len(trial["curve"]) == 4  # one point per block
    for name in ("forgetting", "final_task_a", "final_task_b", "accuracy_task_a"):
        assert name in trial["metrics"]
