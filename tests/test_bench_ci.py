"""Tests for the 95% confidence intervals in summaries and comparisons.

The expected numbers are worked out by hand from the Student t table:
t(0.975, 1) = 12.7062, t(0.975, 2) = 4.3027, t(0.975, 4) = 2.7764.
"""

import json
import math

import numpy as np
import pytest
from scipy import stats

from fabricpc.bench.__main__ import main
from fabricpc.bench.ci import mean_ci95, paired_ci95, t_critical
from fabricpc.bench.compare import compare_family
from fabricpc.bench.registry import COMPARISONS
from fabricpc.bench.summary import compare_rows, summarize_row
from tests.test_bench_summary import write_trials


def test_t_critical_matches_the_t_table():
    assert t_critical(2) == pytest.approx(12.7062, abs=1e-4)
    assert t_critical(3) == pytest.approx(4.3027, abs=1e-4)
    assert t_critical(5) == pytest.approx(2.7764, abs=1e-4)


def test_ci_of_one_two_three():
    # mean 2, sd 1, se 1/sqrt(3); half width 4.3027 / sqrt(3) = 2.4841
    low, high = mean_ci95([1.0, 2.0, 3.0])
    assert low == pytest.approx(2.0 - 2.48414, abs=1e-4)
    assert high == pytest.approx(2.0 + 2.48414, abs=1e-4)


def test_ci_of_five_values():
    # mean 0.92, sd 0.0158114, se 0.00707107; half width 2.7764 * se = 0.019632
    low, high = mean_ci95([0.90, 0.91, 0.92, 0.93, 0.94])
    assert low == pytest.approx(0.92 - 0.019632, abs=1e-5)
    assert high == pytest.approx(0.92 + 0.019632, abs=1e-5)


def test_ci_with_two_values_is_wide_but_given():
    # mean 0.5, sd 0.7071, se 0.5; half width 12.7062 * 0.5 = 6.3531
    low, high = mean_ci95([0.0, 1.0])
    assert low == pytest.approx(0.5 - 6.3531, abs=1e-4)
    assert high == pytest.approx(0.5 + 6.3531, abs=1e-4)


def test_ci_needs_two_values():
    with pytest.raises(ValueError, match="at least 2"):
        mean_ci95([0.9])
    with pytest.raises(ValueError, match="at least 2"):
        paired_ci95([0.9], [0.8])


def test_identical_values_give_a_point():
    assert mean_ci95([0.5, 0.5, 0.5]) == (0.5, 0.5)


def test_paired_ci_is_the_ci_of_the_differences():
    a = [1.0, 2.0, 4.0]
    b = [0.0, 0.0, 1.0]  # differences 1, 2, 3
    assert paired_ci95(a, b) == pytest.approx(mean_ci95([1.0, 2.0, 3.0]))


def test_paired_ci_matches_the_paired_t_test():
    # Zero is outside the 95% interval exactly when p < 0.05.
    rng = np.random.default_rng(0)
    for _ in range(50):
        a = rng.normal(size=4)
        b = a + rng.normal(0.8, 1.0, size=4)
        low, high = paired_ci95(a, b)
        p = stats.ttest_rel(a, b).pvalue
        assert (low > 0 or high < 0) == (p < 0.05)


def test_summary_has_a_ci_next_to_mean_and_se(tmp_path):
    write_trials(tmp_path, "mnist-mlp-spc", [0.90, 0.92, 0.94])

    s = summarize_row(tmp_path, "mnist-mlp-spc")

    # sd 0.02, se 0.011547; half width 4.3027 * se = 0.049683
    acc = s.metrics["accuracy"]
    assert acc["ci95_low"] == pytest.approx(0.92 - 0.049683, abs=1e-5)
    assert acc["ci95_high"] == pytest.approx(0.92 + 0.049683, abs=1e-5)
    assert "ci95_low" in s.timing["step_time_ms"]
    on_disk = json.loads((tmp_path / "mnist-mlp-spc" / "summary.json").read_text())
    assert on_disk["metrics"]["accuracy"]["ci95_high"] == pytest.approx(
        acc["ci95_high"]
    )


def test_summary_of_two_trials_has_a_ci(tmp_path):
    write_trials(tmp_path, "mnist-mlp-spc", [0.90, 0.92])
    acc = summarize_row(tmp_path, "mnist-mlp-spc").metrics["accuracy"]
    # sd 0.014142, se 0.01; half width 12.7062 * 0.01 = 0.127062
    assert acc["ci95_low"] == pytest.approx(0.91 - 0.127062, abs=1e-5)
    assert acc["ci95_high"] == pytest.approx(0.91 + 0.127062, abs=1e-5)


def test_compare_rows_has_a_ci_on_the_difference(tmp_path):
    write_trials(tmp_path, "mnist-mlp-spc", [1.0, 2.0, 4.0], algorithm="spc")
    write_trials(tmp_path, "mnist-mlp-backprop", [0.0, 0.0, 1.0], algorithm="backprop")

    c = compare_rows(tmp_path, "mnist-mlp-spc", "mnist-mlp-backprop", metric="accuracy")

    assert c.ci95_low == pytest.approx(2.0 - 2.48414, abs=1e-4)
    assert c.ci95_high == pytest.approx(2.0 + 2.48414, abs=1e-4)
    on_disk = json.loads(
        (tmp_path / "compare-mnist-mlp-spc-vs-mnist-mlp-backprop.json").read_text()
    )
    assert on_disk["ci95_low"] == pytest.approx(c.ci95_low)


def _family_trials(tmp_path):
    write_trials(tmp_path, "mnist-mlp-spc", [0.90, 0.92, 0.94])
    write_trials(tmp_path, "mnist-mlp-epc", [0.92, 0.93, 0.94], algorithm="epc")
    write_trials(
        tmp_path, "mnist-mlp-backprop", [0.91, 0.91, 0.92], algorithm="backprop"
    )


def test_compare_family_gives_every_contrast_a_ci(tmp_path):
    _family_trials(tmp_path)

    out = compare_family(tmp_path, COMPARISONS["mnist-mlp"], metric="accuracy")

    epc_minus_spc = out["contrasts"][2]
    assert (epc_minus_spc["arm_a"], epc_minus_spc["arm_b"]) == (
        "mnist-mlp-epc",
        "mnist-mlp-spc",
    )
    # differences 0.02, 0.01, 0.00: mean 0.01, sd 0.01, half 4.3027*0.01/sqrt(3)
    half = 4.30265 * 0.01 / math.sqrt(3)
    assert epc_minus_spc["ci95_low"] == pytest.approx(0.01 - half, abs=1e-5)
    assert epc_minus_spc["ci95_high"] == pytest.approx(0.01 + half, abs=1e-5)
    on_disk = json.loads((tmp_path / "compare-mnist-mlp.json").read_text())
    assert on_disk == out


def test_cli_compare_prints_the_ci(tmp_path, capsys):
    _family_trials(tmp_path)
    assert main(["compare", "mnist-mlp", "--out", str(tmp_path)]) == 0
    assert "95% CI [" in capsys.readouterr().out
