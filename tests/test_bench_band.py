"""Tests for the pass/fail band in fabricpc.bench.band."""

import dataclasses

import pytest

from fabricpc.bench.band import DEFAULT_FLOOR, check_band
from fabricpc.bench.registry import ROWS, Reference
from fabricpc.bench.summary import RowSummary


def _row(value=0.98, floor=DEFAULT_FLOOR, metric="accuracy", n_trials=5):
    ref = Reference(metric=metric, value=value, source="test", floor=floor)
    return dataclasses.replace(ROWS["mnist-mlp-spc"], reference=ref, n_trials=n_trials)


def _summary(mean, se, n=5, epochs=20.0, metric="accuracy"):
    return RowSummary(
        row_id="mnist-mlp-spc",
        algorithm="spc",
        n_ok=n,
        n_failed=0,
        seeds=[i * 1000 for i in range(n)],
        metrics={metric: {"mean": mean, "std": se * n**0.5, "se": se, "n": n}},
        num_epochs=epochs,
    )


def test_a_result_inside_the_band_passes():
    v = check_band(_row(0.98), _summary(mean=0.983, se=0.001))
    assert v.status == "pass"
    assert v.band == pytest.approx(DEFAULT_FLOOR)  # floor beats 2*SE here
    assert v.diff == pytest.approx(0.003)


def test_a_result_outside_the_band_fails():
    v = check_band(_row(0.98), _summary(mean=0.97, se=0.001))
    assert v.status == "fail"
    assert v.diff == pytest.approx(-0.01)


def test_the_band_widens_to_two_standard_errors_when_runs_are_noisy():
    # 2*SE = 0.012 is wider than the 0.005 floor, so a 0.01 miss still passes.
    v = check_band(_row(0.98), _summary(mean=0.97, se=0.006))
    assert v.band == pytest.approx(0.012)
    assert v.status == "pass"


def test_a_row_without_an_expected_score_is_neither_pass_nor_fail():
    row = dataclasses.replace(ROWS["mnist-mlp-spc"], reference=None)
    v = check_band(row, _summary(mean=0.5, se=0.001))
    assert v.status == "no_reference"


def test_a_shortened_run_is_not_compared():
    # The expected score is for the row's full epoch count only.
    v = check_band(_row(0.98), _summary(mean=0.5, se=0.001, epochs=1.0))
    assert v.status == "not_comparable"
    assert "epoch" in v.reason


def test_fewer_seeds_than_the_row_states_is_not_compared():
    v = check_band(_row(0.98), _summary(mean=0.98, se=0.001, n=2))
    assert v.status == "not_comparable"
    assert "seed" in v.reason


def test_a_missing_metric_is_not_compared():
    v = check_band(_row(0.98, metric="perplexity"), _summary(mean=0.98, se=0.001))
    assert v.status == "not_comparable"
    assert "perplexity" in v.reason


def test_every_verdict_says_the_band_rule_is_waiting_for_sign_off():
    v = check_band(_row(0.98), _summary(mean=0.98, se=0.001))
    assert v.pending_signoff is True
    assert v.rule == "max(floor, 2*SE)"
