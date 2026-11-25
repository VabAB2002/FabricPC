"""Tests for SFR-031: Compute the Standard Error."""

import json
import math
import statistics

import pytest

from fabricpc.bench.stats.summary import _stats, summarize_row


def test_stats_computes_sample_standard_error():
    """Verify _stats computes SE = sample_std / sqrt(n) with Bessel's correction (ddof=1)."""
    scores = [0.980, 0.982, 0.984, 0.978, 0.986]
    result = _stats(scores)

    expected_mean = statistics.mean(scores)
    expected_std = statistics.stdev(scores)  # ddof=1
    expected_se = expected_std / math.sqrt(len(scores))

    assert result["n"] == len(scores)
    assert result["mean"] == pytest.approx(expected_mean)
    assert result["std"] == pytest.approx(expected_std)
    assert result["se"] == pytest.approx(expected_se)


def test_stats_single_value_returns_zero_se():
    """Verify _stats returns 0.0 standard error when given a single seed."""
    result = _stats([0.95])
    assert result["n"] == 1
    assert result["mean"] == pytest.approx(0.95)
    assert result["std"] == 0.0
    assert result["se"] == 0.0


def test_summarize_row_writes_standard_error_across_seeds(tmp_path):
    """Verify summarize_row computes SE across seed trials and persists it in summary.json."""
    row_id = "mnist-mlp-spc"
    row_dir = tmp_path / row_id
    row_dir.mkdir(parents=True)

    accuracies = [0.9810, 0.9825, 0.9795, 0.9830]
    for trial_idx, acc in enumerate(accuracies):
        trial_payload = {
            "row_id": row_id,
            "trial": trial_idx,
            "seed": trial_idx * 1000,
            "algorithm": "spc",
            "n_params": 218058,
            "metrics": {"accuracy": acc, "loss": 0.06 + trial_idx * 0.005},
            "step_time_ms": 10.0 + trial_idx,
            "compile_time_s": 1.2,
            "train_time_s": 12.5,
            "achieved_tflops": 0.05,
            "num_epochs": 20.0,
            "status": "ok",
        }
        (row_dir / f"trial{trial_idx}.json").write_text(
            json.dumps(trial_payload), encoding="utf-8"
        )

    summary = summarize_row(tmp_path, row_id)
    expected_se = statistics.stdev(accuracies) / math.sqrt(len(accuracies))

    assert summary.n_ok == 4
    assert summary.metrics["accuracy"]["se"] == pytest.approx(expected_se)

    # Confirm summary.json on disk contains the computed standard error.
    saved = json.loads((row_dir / "summary.json").read_text(encoding="utf-8"))
    assert saved["metrics"]["accuracy"]["se"] == pytest.approx(expected_se)


def test_summarize_row_requires_at_least_two_seeds(tmp_path):
    """Verify summarize_row refuses to compute multi-seed stats from a single seed."""
    row_id = "mnist-mlp-spc"
    row_dir = tmp_path / row_id
    row_dir.mkdir(parents=True)

    trial_payload = {
        "row_id": row_id,
        "trial": 0,
        "seed": 0,
        "algorithm": "spc",
        "n_params": 218058,
        "metrics": {"accuracy": 0.981},
        "step_time_ms": 10.0,
        "compile_time_s": 1.2,
        "train_time_s": 12.5,
        "achieved_tflops": 0.05,
        "num_epochs": 20.0,
        "status": "ok",
    }
    (row_dir / "trial0.json").write_text(json.dumps(trial_payload), encoding="utf-8")

    with pytest.raises(ValueError, match="at least 2 successful trials"):
        summarize_row(tmp_path, row_id)