"""Tests for the CI regression checks (``python -m fabricpc.bench regress``)."""

import dataclasses
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from fabricpc.bench import registry, regress
from fabricpc.bench.__main__ import main as cli_main
from fabricpc.bench.regress import Check, judge, run_regress
from tests.test_bench_runner import fake_mnist_loaders

REPO = Path(__file__).resolve().parents[1]


def at_least(limit):
    return Check("some-row", "accuracy", limit, higher_is_better=True, epochs=0.5)


def at_most(limit):
    return Check("some-row", "reconstruction_mse", limit, False, epochs=0.5)


# --- judging one score ------------------------------------------------------


def test_a_score_above_a_floor_passes():
    verdict = judge(at_least(0.85), 0.90)
    assert verdict.passed
    assert verdict.value == 0.90
    assert ">=" in verdict.reason


def test_a_score_below_a_floor_fails():
    verdict = judge(at_least(0.85), 0.80)
    assert not verdict.passed
    assert "0.8000" in verdict.reason and "0.85" in verdict.reason


def test_a_score_right_on_the_floor_passes():
    assert judge(at_least(0.85), 0.85).passed


def test_a_lower_is_better_score_under_its_ceiling_passes():
    assert judge(at_most(0.055), 0.042).passed
    assert judge(at_most(0.055), 0.055).passed


def test_a_lower_is_better_score_over_its_ceiling_fails():
    verdict = judge(at_most(0.055), 0.07)
    assert not verdict.passed
    assert "0.0700 > 0.055" in verdict.reason


@pytest.mark.parametrize("value", [None, math.nan, math.inf])
def test_a_missing_or_broken_score_fails(value):
    # A NaN compares False with everything, so without a guard it would
    # slip past a lower-is-better ceiling.
    assert not judge(at_least(0.85), value).passed
    assert not judge(at_most(0.055), value).passed


# --- running a list of checks ----------------------------------------------


def fake_summary(**metrics):
    return SimpleNamespace(metrics={k: {"mean": v} for k, v in metrics.items()})


def test_run_checks_reads_the_mean_of_each_rows_metric():
    checks = (at_least(0.85), dataclasses.replace(at_most(0.055), row_id="ae"))
    summaries = {
        "some-row": fake_summary(accuracy=0.9),
        "ae": fake_summary(reconstruction_mse=0.08),
    }

    verdicts = regress.run_checks(checks, lambda c: (0, summaries[c.row_id]))

    assert [v.passed for v in verdicts] == [True, False]
    assert verdicts[1].value == 0.08


def test_a_failed_trial_fails_its_check_even_with_a_good_mean():
    verdicts = regress.run_checks(
        (at_least(0.85),), lambda c: (1, fake_summary(accuracy=0.99))
    )
    assert not verdicts[0].passed
    assert "trial" in verdicts[0].reason


def test_no_summary_fails_the_check():
    verdicts = regress.run_checks((at_least(0.85),), lambda c: (0, None))
    assert not verdicts[0].passed


def test_select_keeps_only_the_named_rows_in_order():
    picked = regress.select(regress.CI_CHECKS, ["patterns64-hopfield-spc"])
    assert [c.row_id for c in picked] == ["patterns64-hopfield-spc"]
    assert regress.select(regress.CI_CHECKS, []) == regress.CI_CHECKS
    with pytest.raises(KeyError):
        regress.select(regress.CI_CHECKS, ["no-such-row"])


# --- the checks CI runs -----------------------------------------------------


def test_every_ci_check_names_a_real_row_and_metric():
    for check in regress.CI_CHECKS:
        row = registry.ROWS[check.row_id]
        known = {"accuracy", *(row.eval_metrics or {})}
        assert check.metric in known, check
        assert check.epochs > 0
        assert check.trials >= 2  # the mean needs a summary, which needs two


def test_every_measured_score_clears_its_line_with_room_to_spare():
    # The lines are meant to be generous: even the worst seed we measured
    # should be well clear, so CI does not flip on a small change of CPU.
    for check in regress.CI_CHECKS:
        (low, high), _ = regress.MEASURED[check.row_id]
        worst = low if check.higher_is_better else high
        assert judge(check, worst).passed
        assert abs(worst - check.limit) >= 0.05 * abs(worst), check


def test_ci_covers_all_three_algorithms_on_three_kinds_of_model():
    ids = {c.row_id for c in regress.CI_CHECKS}
    for family in ("mnist-mlp", "mnist-autoencoder", "patterns64-hopfield"):
        for algo in ("spc", "epc", "backprop"):
            assert f"{family}-{algo}" in ids
    # The old smoke check is part of it, with the same floor.
    smoke = [c for c in regress.CI_CHECKS if c.row_id == "mnist-mlp-spc"][0]
    assert smoke.limit == 0.85 and smoke.epochs == 0.5


def test_the_ci_workflow_runs_the_regression_checks():
    yaml = pytest.importorskip("yaml")
    workflow = yaml.safe_load((REPO / ".github/workflows/test.yml").read_text())
    runs = [s.get("run", "") for s in workflow["jobs"]["bench-smoke"]["steps"]]
    assert any("fabricpc.bench smoke" in r for r in runs)
    assert any("fabricpc.bench regress" in r for r in runs)
    assert any("fabricpc.bench validate regress-results" in r for r in runs)


# --- the whole command, on a tiny fake row ---------------------------------


def register_tiny_row(monkeypatch, rng_key):
    base = registry.ROWS["mnist-mlp-spc"]
    loaders = fake_mnist_loaders(rng_key)
    tiny = dataclasses.replace(
        base, id="tiny-mlp-spc", dataset="tiny", loader_factory=lambda seed: loaders
    )
    monkeypatch.setitem(registry.ROWS, "tiny-mlp-spc", tiny)


def tiny_checks(limit):
    return (Check("tiny-mlp-spc", "accuracy", limit, True, epochs=1),)


def test_regress_passes_writes_verdicts_validates_and_reports(
    tmp_path, monkeypatch, rng_key
):
    register_tiny_row(monkeypatch, rng_key)
    monkeypatch.setattr(regress, "CI_CHECKS", tiny_checks(0.0))

    code = cli_main(["regress", "--out", str(tmp_path), "--in-process"])

    assert code == 0
    saved = json.loads((tmp_path / "regress.json").read_text())
    assert saved["passed"] is True
    assert saved["validate_problems"] == []
    assert saved["checks"][0]["row_id"] == "tiny-mlp-spc"
    assert (tmp_path / "report.md").read_text().strip()


def test_regress_exits_nonzero_when_a_score_misses_its_floor(
    tmp_path, monkeypatch, rng_key, capsys
):
    register_tiny_row(monkeypatch, rng_key)
    monkeypatch.setattr(regress, "CI_CHECKS", tiny_checks(1.01))

    code = cli_main(["regress", "--out", str(tmp_path), "--in-process"])

    assert code == 1
    assert "FAIL" in capsys.readouterr().out
    assert json.loads((tmp_path / "regress.json").read_text())["passed"] is False


def test_run_regress_fails_when_validate_finds_a_problem(tmp_path, monkeypatch):
    monkeypatch.setattr(regress, "validate", lambda out: ["something is missing"])
    monkeypatch.setattr(regress, "run_report", lambda *a, **k: "")
    code = run_regress(
        tmp_path, lambda c: (0, fake_summary(accuracy=0.9)), (at_least(0.85),)
    )
    assert code == 1


def test_regress_with_an_unknown_row_says_so(tmp_path, capsys):
    code = cli_main(["regress", "no-such-row", "--out", str(tmp_path)])
    assert code == 2
    assert "no-such-row" in capsys.readouterr().err
