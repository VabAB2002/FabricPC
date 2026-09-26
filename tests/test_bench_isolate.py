"""Tests for running each trial in its own Python process."""

import json
import sys

from fabricpc.bench import isolate
from fabricpc.bench.registry import ROWS


def _stub_child(monkeypatch, code):
    """Replace the real child with a tiny Python script, so tests stay fast."""
    monkeypatch.setattr(
        isolate, "child_command", lambda *a, **k: [sys.executable, "-c", code]
    )


def test_child_command_runs_one_trial_of_one_row(tmp_path):
    cmd = isolate.child_command(
        "mnist-mlp-spc",
        3,
        tmp_path,
        num_epochs=2.0,
        warmup_steps=4,
        timed_steps=6,
        zoo_dir=tmp_path / "zoo",
        curve_batches=7,
    )
    assert cmd[cmd.index("--curve-batches") + 1] == "7"
    assert cmd[:3] == [sys.executable, "-m", "fabricpc.bench"]
    assert cmd[3] == "mnist-mlp-spc"
    assert cmd[cmd.index("--trial") + 1] == "3"
    assert cmd[cmd.index("--out") + 1] == str(tmp_path)
    assert cmd[cmd.index("--epochs") + 1] == "2.0"
    assert cmd[cmd.index("--warmup") + 1] == "4"
    assert cmd[cmd.index("--timed") + 1] == "6"
    assert cmd[cmd.index("--zoo") + 1] == str(tmp_path / "zoo")


def test_child_command_leaves_out_options_that_were_not_given(tmp_path):
    cmd = isolate.child_command(
        "mnist-mlp-spc", 0, tmp_path, num_epochs=None, warmup_steps=5, timed_steps=30
    )
    assert "--epochs" not in cmd
    assert "--zoo" not in cmd


def test_parent_reads_the_result_file_the_child_wrote(tmp_path, monkeypatch):
    row = ROWS["mnist-mlp-spc"]
    path = tmp_path / row.id / "trial0.json"
    written = {
        "row_id": row.id,
        "trial": 0,
        "seed": 0,
        "algorithm": "spc",
        "n_params": 7,
        "metrics": {"accuracy": 0.5},
        "status": "ok",
    }
    _stub_child(
        monkeypatch,
        "import json, pathlib; p = pathlib.Path(%r); "
        "p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(%r))"
        % (str(path), written),
    )

    result = isolate.run_trial_in_child(row, 0, tmp_path)

    assert result.status == "ok"
    assert result.n_params == 7
    assert result.metrics == {"accuracy": 0.5}


def test_a_child_that_dies_without_a_result_is_recorded_as_failed(
    tmp_path, monkeypatch
):
    row = ROWS["mnist-mlp-spc"]
    _stub_child(
        monkeypatch,
        "import os, sys; sys.stderr.write('out of memory\\n'); "
        "sys.stderr.flush(); os._exit(3)",
    )

    result = isolate.run_trial_in_child(row, 1, tmp_path)

    assert result.status == "failed"
    assert result.seed == 1000
    assert "exit code 3" in result.error
    assert "out of memory" in result.error
    on_disk = json.loads((tmp_path / row.id / "trial1.json").read_text())
    assert on_disk["status"] == "failed"


def test_an_old_result_file_is_not_mistaken_for_a_new_one(tmp_path, monkeypatch):
    row = ROWS["mnist-mlp-spc"]
    old = tmp_path / row.id / "trial0.json"
    old.parent.mkdir(parents=True)
    old.write_text(json.dumps({"status": "ok", "row_id": row.id, "trial": 0}))
    _stub_child(monkeypatch, "import os; os._exit(1)")

    result = isolate.run_trial_in_child(row, 0, tmp_path)

    assert result.status == "failed"
