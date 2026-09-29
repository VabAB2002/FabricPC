"""Tests for the probe command: a quick time and cost estimate, no training."""

import dataclasses
import json

import pytest

from fabricpc.bench import registry
from fabricpc.bench.__main__ import main
from fabricpc.bench.probe import DEFAULT_PRICE_PER_HOUR, probe, probe_row
from tests.test_bench_runner import fake_mnist_loaders


def tiny_row(rng_key, algorithm="spc", n_batches=6):
    """A copy of an MNIST MLP row that loads a tiny random dataset."""
    base = registry.ROWS[f"mnist-mlp-{algorithm}"]
    loaders = fake_mnist_loaders(rng_key, n_batches=n_batches)
    return dataclasses.replace(
        base,
        id=f"tiny-mlp-{algorithm}",
        dataset="tiny",
        loader_factory=lambda seed: loaders,
    )


def register_tiny_family(monkeypatch, rng_key):
    for algorithm in registry.ALGORITHMS:
        row = tiny_row(rng_key, algorithm)
        monkeypatch.setitem(registry.ROWS, row.id, row)
    monkeypatch.setitem(
        registry.COMPARISONS, "tiny-mlp", registry._family_comparison("tiny-mlp")
    )


def test_probe_row_counts_steps_and_multiplies_them_out(rng_key):
    row = tiny_row(rng_key, "backprop", n_batches=6)

    p = probe_row(row, trials=3, epochs=2, warmup=1, timed=2, price_per_hour=2.0)

    assert p.row_id == "tiny-mlp-backprop"
    assert p.steps_per_epoch == 6
    assert p.epochs == 2.0
    assert p.trials == 3
    assert p.step_time_ms > 0
    assert p.compile_time_s >= 0
    # A real trial compiles the step twice (the timing pass, then train()
    # builds its own) and runs the timing pass's 1 + 5 + 30 steps on top of
    # the training steps.
    assert p.train_s_per_seed == pytest.approx(
        (12 + 36) * p.step_time_ms / 1000.0 + 2 * p.compile_time_s
    )
    assert p.seconds_per_seed == pytest.approx(p.train_s_per_seed + p.eval_s_per_seed)
    assert p.total_seconds == pytest.approx(3 * p.seconds_per_seed)
    assert p.cost_usd == pytest.approx(p.total_seconds / 3600.0 * 2.0)


def test_probe_row_uses_the_rows_own_trials_and_epochs_by_default(rng_key):
    row = tiny_row(rng_key, "spc")

    p = probe_row(row, warmup=1, timed=2)

    assert p.trials == row.n_trials
    assert p.epochs == float(row.train_config["num_epochs"])
    assert p.eval_batch_ms is not None and p.eval_batch_ms >= 0
    assert p.test_batches == 1


def test_probe_of_a_family_adds_up_its_rows_and_names_the_device(monkeypatch, rng_key):
    register_tiny_family(monkeypatch, rng_key)

    report = probe("tiny-mlp", trials=2, epochs=1, warmup=1, timed=2)

    assert [r["row_id"] for r in report["rows"]] == [
        "tiny-mlp-spc",
        "tiny-mlp-epc",
        "tiny-mlp-backprop",
    ]
    assert report["total_seconds"] == pytest.approx(
        sum(r["total_seconds"] for r in report["rows"])
    )
    assert report["price_per_hour"] == DEFAULT_PRICE_PER_HOUR == 1.06
    assert report["cost_usd"] == pytest.approx(report["total_seconds"] / 3600.0 * 1.06)
    assert report["device"]["platform"] == "cpu"
    assert report["device"]["kind"]


def test_probe_rejects_an_unknown_target():
    with pytest.raises(KeyError):
        probe("no-such-row")


def test_cli_probe_prints_an_estimate_and_writes_json(
    tmp_path, monkeypatch, rng_key, capsys
):
    register_tiny_family(monkeypatch, rng_key)

    code = main(
        [
            "probe",
            "tiny-mlp-backprop",
            "--trials",
            "4",
            "--epochs",
            "3",
            "--warmup",
            "1",
            "--timed",
            "2",
            "--price-per-hour",
            "0.5",
            "--out",
            str(tmp_path),
        ]
    )

    assert code == 0
    printed = capsys.readouterr().out
    assert "tiny-mlp-backprop" in printed
    assert "$" in printed
    assert "cpu" in printed.lower()
    saved = json.loads((tmp_path / "probe-tiny-mlp-backprop.json").read_text())
    assert saved["price_per_hour"] == 0.5
    assert saved["rows"][0]["trials"] == 4
    assert saved["rows"][0]["epochs"] == 3.0


def test_cli_probe_without_out_writes_nothing(tmp_path, monkeypatch, rng_key):
    register_tiny_family(monkeypatch, rng_key)
    monkeypatch.chdir(tmp_path)

    code = main(["probe", "tiny-mlp-spc", "--warmup", "1", "--timed", "2"])

    assert code == 0
    assert list(tmp_path.iterdir()) == []


def test_cli_probe_unknown_target_exits_2(capsys):
    assert main(["probe", "no-such-row"]) == 2
    assert "no-such-row" in capsys.readouterr().err
