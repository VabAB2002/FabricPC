"""Tests for trend tracking (issue #54): a rolling baseline per row and a
Welch test, so runner noise never alerts and a real slowdown does."""

import json

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


def _night(history, n, *, step_ms, acc=0.98, device="cuda:0 T4", seed=0, flags=""):
    """One nightly run of one row, 5 seeds, written into the history."""
    rng = np.random.default_rng(seed)
    record = {
        "row_id": "mnist-mlp-spc",
        "metric": "accuracy",
        "created_utc": f"2026-10-{n + 1:02d}T03:00:00Z",
        "git_sha": f"{n:040d}",
        "device": device,
        "xla_flags": flags,
        "step_time_ms": (step_ms * (1 + 0.08 * rng.standard_normal(5))).tolist(),
        "values": (acc + 0.002 * rng.standard_normal(5)).tolist(),
    }
    trend.append_record(history, record)


def test_a_week_of_ordinary_noise_raises_no_alert(tmp_path):
    for n in range(7):
        _night(tmp_path, n, step_ms=100.0, seed=n)
        assert trend.check(tmp_path) == []


def test_a_quarter_slower_night_is_flagged(tmp_path):
    for n in range(5):
        _night(tmp_path, n, step_ms=100.0, seed=n)
    _night(tmp_path, 5, step_ms=125.0, seed=99)

    alerts = trend.check(tmp_path)

    assert [(a.row_id, a.what) for a in alerts] == [("mnist-mlp-spc", "step_time_ms")]
    assert alerts[0].change == pytest.approx(0.25, abs=0.08)
    assert alerts[0].p_value < 0.05


def test_a_worse_score_is_flagged_too(tmp_path):
    for n in range(5):
        _night(tmp_path, n, step_ms=100.0, seed=n)
    _night(tmp_path, 5, step_ms=100.0, acc=0.95, seed=99)

    assert [a.what for a in trend.check(tmp_path)] == ["accuracy"]


def test_a_new_device_starts_a_new_baseline_instead_of_alerting(tmp_path):
    for n in range(5):
        _night(tmp_path, n, step_ms=100.0, seed=n)
    _night(tmp_path, 5, step_ms=40.0, device="cuda:0 A100", seed=99)

    assert trend.check(tmp_path) == []


def test_adding_a_results_folder_records_every_row_once(tmp_path):
    from tests.test_bench_summary import write_trials

    results = tmp_path / "results"
    write_trials(results, "mnist-mlp-spc", [0.90, 0.91, 0.92])
    (results / "mnist-mlp-spc" / "manifest.json").write_text(
        json.dumps(
            {
                "created_utc": "2026-10-11T03:00:00Z",
                "git_sha": "a" * 40,
                "devices": ["cpu:0"],
                "xla_flags": "",
            }
        )
    )
    history = tmp_path / "history"

    assert trend.add_results(results, history) == ["mnist-mlp-spc"]
    assert trend.add_results(results, history) == []  # same run, not twice

    (record,) = trend.read_history(history)["mnist-mlp-spc"]
    assert record["values"] == [0.90, 0.91, 0.92]
    assert len(record["step_time_ms"]) == 3


def test_the_command_adds_runs_and_exits_nonzero_on_an_alert(tmp_path, capsys):
    from fabricpc.bench.__main__ import main

    history = tmp_path / "history"
    for n in range(5):
        _night(history, n, step_ms=100.0, seed=n)
    assert main(["trend", "check", "--history", str(history)]) == 0
    assert "no regression" in capsys.readouterr().out

    _night(history, 5, step_ms=130.0, seed=99)
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
