"""The equivalence test: is a difference inside a stated margin?

"No significant difference" is not evidence that two methods match; a
small, noisy experiment finds no difference too. Two one-sided tests
(TOST, Schuirmann 1987) ask the right question: is the mean difference
above -margin and below +margin? It passes at 5% exactly when the 90%
interval of the difference lies inside (-margin, +margin).
"""

import math

import numpy as np
import pytest

from fabricpc.bench.compare import compare_family
from fabricpc.bench.registry import COMPARISONS
from fabricpc.bench.stats import default_margin, tost_paired
from tests.test_bench_solver_record import _write


def test_a_small_tight_difference_is_equivalent():
    diffs = np.array([0.001, -0.001, 0.0005, 0.0, 0.0015, -0.0005])
    p = tost_paired(diffs, margin=0.005)
    assert p < 0.05


def test_a_noisy_difference_is_not_equivalent_even_with_mean_zero():
    # Mean exactly 0, but the seeds disagree by several points: no
    # significant difference, and no evidence of a match either.
    diffs = np.array([0.04, -0.04, 0.03, -0.03])
    assert tost_paired(diffs, margin=0.005) > 0.05


def test_tost_matches_the_90_percent_interval_rule():
    from scipy import stats

    rng = np.random.default_rng(0)
    for _ in range(200):
        diffs = rng.normal(rng.uniform(-0.01, 0.01), rng.uniform(0.001, 0.01), 6)
        n = len(diffs)
        se = diffs.std(ddof=1) / math.sqrt(n)
        half = stats.t.ppf(0.95, n - 1) * se
        inside = -0.005 < diffs.mean() - half and diffs.mean() + half < 0.005
        assert (tost_paired(diffs, margin=0.005) < 0.05) == inside


def test_differences_with_no_spread():
    assert tost_paired(np.full(4, 0.001), margin=0.005) == 0.0
    assert tost_paired(np.full(4, 0.01), margin=0.005) == 1.0


def test_default_margin_is_half_a_point_for_accuracy_and_none_otherwise():
    assert default_margin("accuracy") == 0.005
    assert default_margin("forward_accuracy") == 0.005
    assert default_margin("reconstruction_mse") is None


def test_a_family_comparison_reports_equivalence_with_its_margin(tmp_path):
    base = [0.90, 0.91, 0.92, 0.93, 0.94]
    _write(tmp_path, "mnist-mlp-backprop", "backprop", [None] * 5, base)
    _write(
        tmp_path,
        "mnist-mlp-spc",
        "spc",
        [None] * 5,
        [b + d for b, d in zip(base, [0.001, -0.001, 0.0, 0.0005, -0.0005])],
    )
    _write(
        tmp_path,
        "mnist-mlp-epc",
        "epc",
        [0.1] * 5,
        [b + d for b, d in zip(base, [0.03, -0.03, 0.02, -0.02, 0.0])],
    )

    out = compare_family(tmp_path, COMPARISONS["mnist-mlp"], metric="accuracy")

    by = {(c["arm_a"], c["arm_b"]): c for c in out["contrasts"]}
    spc_bp = by[("mnist-mlp-spc", "mnist-mlp-backprop")]
    epc_bp = by[("mnist-mlp-epc", "mnist-mlp-backprop")]
    assert out["equivalence_margin"] == 0.005
    assert spc_bp["equivalent_holm_05"] is True
    assert epc_bp["equivalent_holm_05"] is False
    assert epc_bp["significant_holm_05"] is False  # and yet not a match


def test_a_metric_with_no_default_margin_reports_none(tmp_path):
    for algo in ("spc", "epc", "backprop"):
        _write(tmp_path, f"mnist-mlp-{algo}", algo, [None] * 3)

    out = compare_family(
        tmp_path, COMPARISONS["mnist-mlp"], metric="accuracy", margin=0.0
    )

    assert out["equivalence_margin"] is None
    assert "p_equivalence" not in out["contrasts"][0]


@pytest.mark.parametrize("margin", [-0.01, math.nan])
def test_a_bad_margin_is_refused(tmp_path, margin):
    for algo in ("spc", "epc", "backprop"):
        _write(tmp_path, f"mnist-mlp-{algo}", algo, [None] * 3)
    with pytest.raises(ValueError, match="margin"):
        compare_family(
            tmp_path, COMPARISONS["mnist-mlp"], metric="accuracy", margin=margin
        )


def test_the_report_says_whether_each_contrast_was_shown_equivalent(tmp_path):
    from fabricpc.bench.report import run_report

    base = [0.90, 0.91, 0.92, 0.93, 0.94]
    for algo, noise in (
        ("backprop", [0.0] * 5),
        ("spc", [0.001, -0.001, 0.0, 0.0005, -0.0005]),
        ("epc", [0.03, -0.03, 0.02, -0.02, 0.0]),
    ):
        accuracies = [b + d for b, d in zip(base, noise)]
        _write(tmp_path, f"mnist-mlp-{algo}", algo, [None] * 5, accuracies)
    compare_family(tmp_path, COMPARISONS["mnist-mlp"], metric="accuracy")

    page = run_report(tmp_path)

    assert "equivalent at 5%? (TOST)" in page
    assert "yes, within ±0.005" in page
    assert "not shown at ±0.005" in page
