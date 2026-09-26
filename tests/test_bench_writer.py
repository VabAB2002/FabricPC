"""Tests for trials.csv and the ``validate`` check on a results folder."""

import csv
import dataclasses
import json

from fabricpc.bench.manifest import SCHEMA_VERSION, write_manifest
from fabricpc.bench.registry import ROWS
from fabricpc.bench.runner import TrialResult
from fabricpc.bench.summary import summarize_row
from fabricpc.bench.writer import validate, write_trials_csv

ROW = ROWS["mnist-mlp-spc"]


def _write_run(out, *, statuses=("ok", "ok")):
    """A small but complete results folder, written the way a real run does."""
    seeds = [i * 1000 for i in range(len(statuses))]
    write_manifest(out, ROW, seeds=seeds, command=["python", "-m", "fabricpc.bench"])
    for i, status in enumerate(statuses):
        r = TrialResult(
            row_id=ROW.id,
            trial=i,
            seed=seeds[i],
            algorithm="spc",
            n_params=10,
            metrics={"accuracy": 0.9 + i / 100} if status == "ok" else {},
            step_time_ms=10.0 if status == "ok" else 0.0,
            num_epochs=1.0,
            peak_memory_bytes=2048 if status == "ok" else None,
            status=status,
            error=None if status == "ok" else "Traceback...\nRuntimeError: boom",
        )
        path = out / ROW.id / f"trial{i}.json"
        path.write_text(json.dumps(dataclasses.asdict(r)))
    if statuses.count("ok") >= 2:
        summarize_row(out, ROW.id)
    write_trials_csv(out, ROW.id)
    return out / ROW.id


def _edit(path, **changes):
    data = json.loads(path.read_text())
    data.update(changes)
    path.write_text(json.dumps(data))


def _drop(path, key):
    data = json.loads(path.read_text())
    del data[key]
    path.write_text(json.dumps(data))


# --- trials.csv ------------------------------------------------------------


def test_trials_csv_has_one_line_per_trial(tmp_path):
    row_dir = _write_run(tmp_path, statuses=("ok", "ok", "ok"))

    with open(row_dir / "trials.csv") as f:
        lines = list(csv.DictReader(f))

    assert [line["trial"] for line in lines] == ["0", "1", "2"]
    assert [line["seed"] for line in lines] == ["0", "1000", "2000"]
    assert float(lines[1]["accuracy"]) == 0.91
    assert lines[0]["peak_memory_bytes"] == "2048"
    assert lines[0]["status"] == "ok"


def test_trials_csv_keeps_failed_trials_with_the_last_error_line(tmp_path):
    row_dir = _write_run(tmp_path, statuses=("ok", "failed"))

    with open(row_dir / "trials.csv") as f:
        lines = list(csv.DictReader(f))

    assert lines[1]["status"] == "failed"
    assert lines[1]["accuracy"] == ""
    assert lines[1]["error"] == "RuntimeError: boom"


def test_trial_files_carry_the_schema_version(tmp_path):
    row_dir = _write_run(tmp_path)
    trial = json.loads((row_dir / "trial0.json").read_text())
    summary = json.loads((row_dir / "summary.json").read_text())
    assert trial["schema_version"] == SCHEMA_VERSION
    assert summary["schema_version"] == SCHEMA_VERSION


# --- validate --------------------------------------------------------------


def test_a_complete_run_validates_cleanly(tmp_path):
    _write_run(tmp_path)
    assert validate(tmp_path) == []


def test_a_run_with_a_failed_trial_is_still_valid(tmp_path):
    # A failed trial is a real, recorded outcome, not a broken file.
    _write_run(tmp_path, statuses=("ok", "ok", "failed"))
    assert validate(tmp_path) == []


def test_an_empty_folder_is_reported(tmp_path):
    problems = validate(tmp_path)
    assert len(problems) == 1
    assert "no benchmark results" in problems[0]


def test_a_missing_manifest_is_reported(tmp_path):
    row_dir = _write_run(tmp_path)
    (row_dir / "manifest.json").unlink()
    assert any("manifest.json" in p for p in validate(tmp_path))


def test_a_missing_schema_version_is_reported(tmp_path):
    row_dir = _write_run(tmp_path)
    _drop(row_dir / "trial1.json", "schema_version")
    assert any("trial1.json" in p and "schema_version" in p for p in validate(tmp_path))


def test_a_trial_filed_under_the_wrong_row_is_reported(tmp_path):
    row_dir = _write_run(tmp_path)
    _edit(row_dir / "trial0.json", row_id="cifar10-vgg5-spc")
    assert any("trial0.json" in p and "row_id" in p for p in validate(tmp_path))


def test_a_seed_the_manifest_does_not_list_is_reported(tmp_path):
    row_dir = _write_run(tmp_path)
    _edit(row_dir / "trial1.json", seed=42)
    assert any("trial1.json" in p and "seed" in p for p in validate(tmp_path))


def test_a_summary_that_disagrees_with_the_trials_is_reported(tmp_path):
    row_dir = _write_run(tmp_path)
    _edit(row_dir / "summary.json", n_ok=5)
    assert any("summary.json" in p and "n_ok" in p for p in validate(tmp_path))


def test_a_missing_or_stale_trials_csv_is_reported(tmp_path):
    row_dir = _write_run(tmp_path)
    with open(row_dir / "trials.csv") as f:
        header, first = f.readline(), f.readline()
    (row_dir / "trials.csv").write_text(header + first)  # one trial lost
    assert any("trials.csv" in p for p in validate(tmp_path))

    (row_dir / "trials.csv").unlink()
    assert any("trials.csv" in p for p in validate(tmp_path))


def test_trials_csv_and_summary_carry_the_step_memory(tmp_path):
    row_dir = _write_run(tmp_path)
    for i in (0, 1):
        _edit(
            row_dir / f"trial{i}.json",
            step_memory={"total_bytes": 1234, "temp_bytes": 1000},
        )
    summarize_row(tmp_path, ROW.id)
    write_trials_csv(tmp_path, ROW.id)

    with open(row_dir / "trials.csv") as f:
        lines = list(csv.DictReader(f))
    assert lines[0]["step_memory_bytes"] == "1234"
    summary = json.loads((row_dir / "summary.json").read_text())
    assert summary["step_memory"]["total_bytes"] == 1234
