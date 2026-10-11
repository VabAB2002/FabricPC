"""Tests for fabricpc.bench.stats: multiple-test correction and effect size."""

import math

import pytest

from fabricpc.bench.stats import hedges_gz, holm


def test_holm_by_hand():
    # Sorted 0.01, 0.03, 0.04 are multiplied by 3, 2, 1 and kept
    # non-decreasing: 0.03, 0.06, max(0.06, 0.04) = 0.06. Back in input order:
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_holm_never_goes_above_one_and_leaves_one_test_alone():
    assert holm([0.6, 0.7]) == pytest.approx([1.0, 1.0])
    assert holm([0.02]) == pytest.approx([0.02])


def test_holm_counts_a_nan_as_a_test_that_found_nothing():
    # A contrast that gave no p-value was still planned; dropping it from
    # the count would make the others look stronger than they are.
    out = holm([0.01, float("nan")])
    assert out[0] == pytest.approx(0.02)  # two tests, not one
    assert math.isnan(out[1])


def test_hedges_gz_shrinks_d_z_for_few_seeds():
    # J(df) = 1 - 3 / (4 df - 1); five seeds give df 4 and J = 0.8.
    assert hedges_gz(1.0, n=5) == pytest.approx(0.8)
    assert hedges_gz(-2.0, n=10) == pytest.approx(-2.0 * (1 - 3 / 35))
    assert math.isnan(hedges_gz(1.0, n=1))


def test_hedges_gz_is_not_a_number_for_two_seeds():
    # J(1) = 1 - 3 / 3 = 0, which would print every two-seed effect as 0.
    assert math.isnan(hedges_gz(1.5, n=2))
    assert hedges_gz(1.5, n=3) == pytest.approx(1.5 * (1 - 3 / 7))


def _contrast(**kw):
    base = dict(
        arm_a="a",
        arm_b="b",
        mean_diff=0.01,
        se_diff=0.002,
        t_statistic=5.0,
        p_value=0.01,
        significant_at_05=True,
        cohens_d=2.5,
        n=4,
    )
    return {**base, **kw}


def test_a_constant_nonzero_difference_has_no_finite_effect_size():
    # Every seed exactly 1 point apart: the framework's d is mean / 0 and it
    # writes 0, which reads as "no effect".
    from fabricpc.bench.compare import corrected

    (c,) = corrected([_contrast(se_diff=0.0, cohens_d=0.0, p_value=0.0)])
    assert math.isnan(c["cohens_d"]) and math.isnan(c["hedges_gz"])


def test_a_nan_score_has_no_effect_size():
    from fabricpc.bench.compare import corrected

    nan = float("nan")
    (c,) = corrected([_contrast(mean_diff=nan, se_diff=nan, cohens_d=0.0, p_value=nan)])
    assert math.isnan(c["cohens_d"]) and math.isnan(c["hedges_gz"])


def test_identical_rows_still_show_an_effect_of_zero():
    from fabricpc.bench.compare import corrected

    (c,) = corrected(
        [_contrast(mean_diff=0.0, se_diff=0.0, cohens_d=0.0, p_value=float("nan"))]
    )
    assert c["cohens_d"] == 0.0 and c["p_value"] == 1.0
