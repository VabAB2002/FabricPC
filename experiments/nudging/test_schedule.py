"""Tests for pcx's nudging schedule."""

from experiments.nudging.schedule import beta_for_epoch, cn_factors


def test_c10_nn_schedule_matches_pcx_including_the_plus_one_quirk():
    b0, ir = 0.17877062530418192, 0.02
    assert abs(beta_for_epoch(b0, ir, -1.0, 0) - (-b0)) < 1e-12
    assert abs(beta_for_epoch(b0, ir, -1.0, 41) - (-(b0 + 0.82))) < 1e-12
    # pcx sets beta to +1.0 once |beta| >= 1, even when the factor is -1.
    assert beta_for_epoch(b0, ir, -1.0, 42) == 1.0
    assert beta_for_epoch(b0, ir, -1.0, 49) == 1.0


def test_plain_se_is_always_one():
    assert all(beta_for_epoch(1.0, 0.0, 1.0, e) == 1.0 for e in range(50))


def test_cn_factors_are_seeded_and_both_signs():
    a, b = cn_factors(0, 50), cn_factors(0, 50)
    assert a == b and set(a) == {1.0, -1.0}
    assert cn_factors(1, 50) != a
