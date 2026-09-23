"""Tests for turning trial files into a summary with mean, SE, and contrasts."""

import dataclasses
import json
import math

import pytest

from fabricpc.bench.runner import TrialResult
from fabricpc.bench.summary import compare_rows, summarize_row


def write_trials(out_dir, row_id, accuracies, *, algorithm="spc", seed_step=1000):
    """Drop fake trial files on disk the way the runner would."""
    (out_dir / row_id).mkdir(parents=True, exist_ok=True)
    for i, acc in enumerate(accuracies):
        r = TrialResult(
            row_id=row_id,
            trial=i,
            seed=i * seed_step,
            algorithm=algorithm,
            n_params=10,
            metrics={"accuracy": acc, "energy": 1.0},
            step_time_ms=10.0 + i,
            compile_time_s=0.5,
            train_time_s=1.0,
            num_epochs=1.0,
        )
        (out_dir / row_id / f"trial{i}.json").write_text(
            json.dumps(dataclasses.asdict(r))
        )


def test_summary_reports_mean_std_and_standard_error(tmp_path):
    write_trials(tmp_path, "mnist-mlp-spc", [0.90, 0.92, 0.94])

    s = summarize_row(tmp_path, "mnist-mlp-spc")

    acc = s.metrics["accuracy"]
    assert acc["n"] == 3
    assert acc["mean"] == pytest.approx(0.92)
    assert acc["std"] == pytest.approx(0.02)  # sample std, ddof=1
    assert acc["se"] == pytest.approx(0.02 / math.sqrt(3))
    assert s.timing["step_time_ms"]["mean"] == pytest.approx(11.0)

    on_disk = json.loads((tmp_path / "mnist-mlp-spc" / "summary.json").read_text())
    assert on_disk["metrics"]["accuracy"]["mean"] == pytest.approx(0.92)
    assert on_disk["seeds"] == [0, 1000, 2000]


def test_summary_refuses_to_average_a_single_trial(tmp_path):
    # One seed is one draw from a distribution. It is not a result.
    write_trials(tmp_path, "mnist-mlp-spc", [0.90])
    with pytest.raises(ValueError, match="at least 2"):
        summarize_row(tmp_path, "mnist-mlp-spc")


def test_summary_skips_failed_trials_but_counts_them(tmp_path):
    write_trials(tmp_path, "mnist-mlp-spc", [0.90, 0.92, 0.94])
    failed = TrialResult(
        row_id="mnist-mlp-spc",
        trial=3,
        seed=3000,
        algorithm="spc",
        n_params=0,
        status="failed",
        error="boom",
    )
    (tmp_path / "mnist-mlp-spc" / "trial3.json").write_text(
        json.dumps(dataclasses.asdict(failed))
    )
    s = summarize_row(tmp_path, "mnist-mlp-spc")
    assert s.metrics["accuracy"]["n"] == 3
    assert s.n_failed == 1


def test_compare_two_rows_gives_a_paired_test(tmp_path):
    write_trials(tmp_path, "mnist-mlp-spc", [0.90, 0.91, 0.92], algorithm="spc")
    write_trials(
        tmp_path, "mnist-mlp-backprop", [0.93, 0.95, 0.94], algorithm="backprop"
    )

    c = compare_rows(tmp_path, "mnist-mlp-spc", "mnist-mlp-backprop", metric="accuracy")

    assert c.n == 3
    assert c.mean_diff == pytest.approx(0.91 - 0.94)
    assert 0.0 <= c.p_value <= 1.0
    assert isinstance(c.significant_at_05, bool)
    assert not math.isnan(c.cohens_d)


def test_compare_identical_rows_has_zero_difference(tmp_path):
    write_trials(tmp_path, "a-mlp-spc", [0.90, 0.91, 0.92])
    write_trials(tmp_path, "a-mlp-epc", [0.90, 0.91, 0.92], algorithm="epc")
    c = compare_rows(tmp_path, "a-mlp-spc", "a-mlp-epc", metric="accuracy")
    assert c.mean_diff == 0.0


def test_compare_refuses_rows_whose_seeds_do_not_match(tmp_path):
    # Pairing only means something when trial i used the same seed in both.
    write_trials(tmp_path, "a-mlp-spc", [0.90, 0.91], seed_step=1000)
    write_trials(tmp_path, "a-mlp-epc", [0.90, 0.91], seed_step=7, algorithm="epc")
    with pytest.raises(ValueError, match="seed"):
        compare_rows(tmp_path, "a-mlp-spc", "a-mlp-epc", metric="accuracy")
