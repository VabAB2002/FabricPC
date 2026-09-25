"""Tests for the ``python -m fabricpc.bench`` command line."""

import json
import subprocess
import sys


def run_cli(*args):
    """Run the bench CLI in a fresh process and return (exit code, stdout, stderr)."""
    proc = subprocess.run(
        [sys.executable, "-m", "fabricpc.bench", *args],
        capture_output=True,
        text=True,
    )
    return proc.returncode, proc.stdout, proc.stderr


def test_list_prints_every_row_id():
    code, out, _ = run_cli("list")
    assert code == 0
    for algo in ("spc", "epc", "backprop"):
        assert f"mnist-mlp-{algo}" in out


def test_dry_run_prints_the_row_config_as_json_without_training():
    code, out, _ = run_cli("mnist-mlp-spc", "--dry-run")
    assert code == 0
    info = json.loads(out)
    assert info["id"] == "mnist-mlp-spc"
    assert info["algorithm"] == "spc"
    assert info["n_trials"] == 5
    assert info["train_config"] == {"num_epochs": 20}


def test_unknown_row_fails_with_a_helpful_message():
    code, out, err = run_cli("no-such-row", "--dry-run")
    assert code != 0
    assert "no-such-row" in err
    assert "list" in err  # points the user at the list command


def test_list_also_names_the_families_that_run_as_one_comparison():
    code, out, _ = run_cli("list")
    assert code == 0
    assert "mnist-mlp " in out or "mnist-mlp\n" in out
    assert "spc vs backprop" in out
