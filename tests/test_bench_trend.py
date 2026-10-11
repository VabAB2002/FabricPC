"""Tests for trend tracking (issue #54).

One number per night (the median over seeds), divided by the machine's
speed test when every night compared has one. A baseline of earlier nights
that raised no alarm. An alarm only when two nights in a row are both worse
by a practical amount and by more than twice the baseline's own wobble; a
failed seed or a NaN score alarms at once.
"""

import json
import math

import numpy as np
import pytest

from fabricpc.bench import trend
from fabricpc.experiments.statistics import welch_ttest


def test_welch_matches_scipy():
    from scipy import stats

    a, b = np.array([1.0, 2.0, 3.0, 4.0]), np.array([2.5, 3.5, 4.5, 6.0, 7.0])
    r = welch_ttest(a, b)
    t, p = stats.ttest_ind(a, b, equal_var=False)
    assert r.t_statistic == pytest.approx(t) and r.p_value == pytest.approx(p)
    assert r.mean_difference == pytest.approx(a.mean() - b.mean())
    assert r.n_a == 4 and r.n_b == 5


def _night(
    history,
    n,
    *,
    step_ms=100.0,
    speed_ms=None,
    value=0.98,
    metric="accuracy",
    seed=0,
    device="cuda:0 T4",
    flags="",
    epochs=0.5,
    cap=None,
    failed=0,
    values=None,
):
    """One nightly run of one row, 5 seeds, written into the history."""
    rng = np.random.default_rng(seed)
    record = {
        "row_id": "mnist-mlp-spc",
        "metric": metric,
        "created_utc": f"2026-10-{n + 1:02d}T03:00:00Z",
        "git_sha": f"{n:040d}",
        "device": device,
        "xla_flags": flags,
        "num_epochs": epochs,
        "eval_batches_capped": cap,
        "speed_test_ms": speed_ms,
        "n_failed": failed,
        "step_time_ms": (step_ms * (1 + 0.02 * rng.standard_normal(5))).tolist(),
        "values": (
            values
            if values is not None
            else (value * (1 + 0.0005 * rng.standard_normal(5))).tolist()
        ),
    }
    trend.append_record(history, record)


def _quiet_nights(history, count=10, **kw):
    for n in range(count):
        _night(history, n, seed=n, **kw)


def test_ten_ordinary_nights_raise_no_alert(tmp_path):
    rng = np.random.default_rng(1)
    for n in range(12):
        _night(tmp_path, n, step_ms=100.0 * math.exp(rng.normal(0, 0.05)), seed=n)
        assert trend.check(tmp_path) == []


def test_one_slow_night_is_not_enough(tmp_path):
    _quiet_nights(tmp_path)
    _night(tmp_path, 10, step_ms=130.0, seed=99)
    assert trend.check(tmp_path) == []


def test_two_slow_nights_in_a_row_raise_an_alert(tmp_path):
    _quiet_nights(tmp_path)
    _night(tmp_path, 10, step_ms=130.0, seed=98)
    _night(tmp_path, 11, step_ms=130.0, seed=99)

    alerts = trend.check(tmp_path)

    assert [(a.row_id, a.what) for a in alerts] == [("mnist-mlp-spc", "step_time_ms")]
    assert alerts[0].change == pytest.approx(0.30, abs=0.05)
    assert len(alerts[0].recent) == 2
    assert alerts[0].speed_adjusted is False


def test_slow_nights_never_become_the_baseline(tmp_path):
    # By the ninth slow night, seven of the last ten nights before the
    # judged pair are slow; a baseline that took them in would call the
    # slowdown normal and go quiet.
    _quiet_nights(tmp_path)
    for n in range(10, 19):
        _night(tmp_path, n, step_ms=130.0, seed=n)
    assert [a.what for a in trend.check(tmp_path)] == ["step_time_ms"]


def test_a_noisy_row_needs_a_bigger_change_than_a_steady_one(tmp_path):
    # Nights swinging 15% either way: 25% slower is within twice that
    # wobble, so it is not an alarm, although it is past the 15% floor.
    for n in range(10):
        _night(tmp_path, n, step_ms=100.0 * math.exp(0.15 * (-1) ** n), seed=n)
    _night(tmp_path, 10, step_ms=125.0, seed=98)
    _night(tmp_path, 11, step_ms=125.0, seed=99)
    assert trend.check(tmp_path) == []


def test_a_night_when_the_whole_machine_was_slow_is_not_an_alarm(tmp_path):
    _quiet_nights(tmp_path, speed_ms=1.0)
    _night(tmp_path, 10, step_ms=130.0, speed_ms=1.3, seed=98)
    _night(tmp_path, 11, step_ms=130.0, speed_ms=1.3, seed=99)
    assert trend.check(tmp_path) == []


def test_a_slower_fabricpc_on_a_normal_machine_is_an_alarm(tmp_path):
    _quiet_nights(tmp_path, speed_ms=1.0)
    _night(tmp_path, 10, step_ms=130.0, speed_ms=1.0, seed=98)
    _night(tmp_path, 11, step_ms=130.0, speed_ms=1.0, seed=99)

    (alert,) = trend.check(tmp_path)

    assert alert.what == "step_time_ms" and alert.speed_adjusted is True


def test_a_lower_accuracy_on_two_nights_is_flagged(tmp_path):
    _quiet_nights(tmp_path)
    _night(tmp_path, 10, value=0.96, seed=98)
    _night(tmp_path, 11, value=0.96, seed=99)
    assert [a.what for a in trend.check(tmp_path)] == ["accuracy"]


def test_a_tiny_accuracy_change_is_not_flagged(tmp_path):
    _quiet_nights(tmp_path)
    _night(tmp_path, 10, value=0.978, seed=98)
    _night(tmp_path, 11, value=0.978, seed=99)
    assert trend.check(tmp_path) == []


@pytest.mark.parametrize("worse, flagged", [(1.08, True), (1.02, False)])
def test_lower_is_better_scores_use_a_relative_threshold(tmp_path, worse, flagged):
    kw = dict(metric="reconstruction_mse")
    _quiet_nights(tmp_path, value=0.010, **kw)
    _night(tmp_path, 10, value=0.010 * worse, seed=98, **kw)
    _night(tmp_path, 11, value=0.010 * worse, seed=99, **kw)
    whats = [a.what for a in trend.check(tmp_path)]
    assert whats == (["reconstruction_mse"] if flagged else [])


def test_a_failed_seed_alarms_at_once(tmp_path):
    _quiet_nights(tmp_path)
    _night(tmp_path, 10, failed=1, seed=99)
    assert [a.what for a in trend.check(tmp_path)] == ["failed_seeds"]


def test_a_nan_score_alarms_at_once_and_does_not_poison_the_baseline(tmp_path):
    _quiet_nights(tmp_path)
    _night(tmp_path, 10, values=[0.98, float("nan"), 0.98, 0.98, 0.98], seed=99)
    assert [a.what for a in trend.check(tmp_path)] == ["non_finite"]

    _night(tmp_path, 11, seed=100)
    _night(tmp_path, 12, seed=101)
    assert trend.check(tmp_path) == []


@pytest.mark.parametrize(
    "change",
    [
        dict(device="cuda:0 A100"),
        dict(flags="--xla_gpu_autotune_level=0"),
        dict(epochs=1.0),
        dict(cap=2),
        dict(metric="forward_accuracy"),
    ],
)
def test_a_new_setting_starts_a_new_baseline_instead_of_alerting(tmp_path, change):
    _quiet_nights(tmp_path)
    _night(tmp_path, 10, step_ms=40.0, value=0.5, seed=98, **change)
    _night(tmp_path, 11, step_ms=40.0, value=0.5, seed=99, **change)
    assert trend.check(tmp_path) == []


def test_too_few_nights_are_not_judged(tmp_path):
    _quiet_nights(tmp_path, count=trend.MIN_BASELINE - 1)
    _night(tmp_path, 10, step_ms=150.0, seed=98)
    _night(tmp_path, 11, step_ms=150.0, seed=99)
    assert trend.check(tmp_path) == []


def _simulated_flags(rng, nights, *, runner_sd, slowdown=1.0, start=None):
    """One row's nights: a runner factor shared by the night's 2 seeds and
    seen by its speed test (plus the test's own 3% noise), 2% seed jitter."""
    runner = np.exp(rng.normal(0, runner_sd, nights))
    speed = runner * np.exp(rng.normal(0, 0.03, nights))
    seeds = runner[:, None] * np.exp(rng.normal(0, 0.02, (nights, 2)))
    step = np.median(seeds, axis=1)
    if start is not None:
        step[start:] *= slowdown
    return trend.flag_nights(step / speed, higher_is_worse=True, relative=True)


@pytest.mark.parametrize("runner_sd", [0.05, 0.10, 0.15])
def test_issue_54_acceptance_with_the_speed_test(runner_sd):
    # #54: "Normal runner noise over a week produces no alerts" and "an
    # injected slowdown of roughly twice the noise scale (20-30%) ... is
    # flagged within a few nightly runs". Nine rows, as the nightly job runs,
    # assuming the speed test sees the runner's slowness (see guide 18).
    rng = np.random.default_rng(54)
    weeks, rows, base = 100, 9, trend.WINDOW
    false_weeks = sum(
        any(
            any(_simulated_flags(rng, base + 7, runner_sd=runner_sd)[base:])
            for _ in range(rows)
        )
        for _ in range(weeks)
    )
    caught = 0
    for _ in range(weeks):
        flags = _simulated_flags(
            rng, base + 7, runner_sd=runner_sd, slowdown=1.25, start=base
        )
        caught += any(flags[base : base + 3])
    assert false_weeks <= 0.05 * weeks
    assert caught >= 0.9 * weeks


def test_adding_a_results_folder_records_every_row_once(tmp_path):
    from tests.test_bench_summary import write_trials

    results = tmp_path / "results"
    write_trials(results, "mnist-mlp-spc", [0.90, 0.91, 0.92])
    failed = json.loads((results / "mnist-mlp-spc" / "trial2.json").read_text())
    failed["status"] = "failed"
    (results / "mnist-mlp-spc" / "trial2.json").write_text(json.dumps(failed))
    (results / "mnist-mlp-spc" / "manifest.json").write_text(
        json.dumps(
            {
                "created_utc": "2026-10-11T03:00:00Z",
                "git_sha": "a" * 40,
                "devices": ["cpu:0"],
                "xla_flags": "",
                "speed_test_ms": 0.8,
            }
        )
    )
    history = tmp_path / "history"

    assert trend.add_results(results, history) == ["mnist-mlp-spc"]
    assert trend.add_results(results, history) == []  # same run, not twice

    (record,) = trend.read_history(history)["mnist-mlp-spc"]
    assert record["values"] == [0.90, 0.91]
    assert len(record["step_time_ms"]) == 2
    assert record["n_failed"] == 1
    assert record["speed_test_ms"] == 0.8
    assert record["num_epochs"] == 1.0
    assert record["eval_batches_capped"] is None


def test_the_command_adds_runs_and_exits_nonzero_on_an_alert(tmp_path, capsys):
    from fabricpc.bench.__main__ import main

    history = tmp_path / "history"
    _quiet_nights(history)
    assert main(["trend", "check", "--history", str(history)]) == 0
    assert "no regression" in capsys.readouterr().out

    _night(history, 10, step_ms=130.0, seed=98)
    _night(history, 11, step_ms=130.0, seed=99)
    assert main(["trend", "check", "--history", str(history)]) == 1
    assert "mnist-mlp-spc: step_time_ms" in capsys.readouterr().out


def test_the_command_needs_a_history(capsys):
    from fabricpc.bench.__main__ import main

    assert main(["trend", "check"]) == 2
    assert "--history" in capsys.readouterr().err


def _pipes_keep_failures(workflow: dict) -> bool:
    """Every step that pipes a command keeps its failure: bash (pipefail)
    as the step's or the workflow's shell, or `set -o pipefail` in it."""
    default = ((workflow.get("defaults") or {}).get("run") or {}).get("shell")
    for job in workflow["jobs"].values():
        for step in job["steps"]:
            run = step.get("run", "")
            if "|" not in run:
                continue
            shell = step.get("shell", default)
            if shell != "bash" and "set -o pipefail" not in run:
                return False
    return True


def test_the_nightly_job_fails_when_trend_finds_a_regression():
    # GitHub's default shell has no pipefail, so `trend check | tee` would
    # take tee's exit code and a regression would never fail the night.
    from pathlib import Path

    yaml = pytest.importorskip("yaml")
    repo = Path(__file__).resolve().parents[1]
    workflow = yaml.safe_load((repo / ".github/workflows/nightly.yml").read_text())
    assert _pipes_keep_failures(workflow)
    runs = [s.get("run", "") for s in workflow["jobs"]["nightly-cpu"]["steps"]]
    assert any("trend check" in r for r in runs)
