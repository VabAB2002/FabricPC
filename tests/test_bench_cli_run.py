"""Tests for running a row from the command line (in-process, tiny data)."""

import dataclasses
import json


from fabricpc.bench import registry
from fabricpc.bench.__main__ import main as cli_main
from tests.test_bench_runner import fake_mnist_loaders


def main(argv):
    """Run the CLI in this process: the tiny test rows only exist here, so a
    child process would not find them. test_bench_isolate covers the child."""
    return cli_main([*argv, "--in-process"])


def register_tiny_row(monkeypatch, rng_key):
    """A copy of mnist-mlp-spc that loads a tiny random dataset."""
    base = registry.ROWS["mnist-mlp-spc"]
    loaders = fake_mnist_loaders(rng_key)
    tiny = dataclasses.replace(
        base, id="tiny-mlp-spc", dataset="tiny", loader_factory=lambda seed: loaders
    )
    monkeypatch.setitem(registry.ROWS, "tiny-mlp-spc", tiny)
    return tiny


def test_running_a_row_writes_one_file_per_trial(tmp_path, monkeypatch, rng_key):
    register_tiny_row(monkeypatch, rng_key)

    code = main(
        [
            "tiny-mlp-spc",
            "--out",
            str(tmp_path),
            "--trials",
            "2",
            "--epochs",
            "1",
            "--warmup",
            "1",
            "--timed",
            "2",
        ]
    )

    assert code == 0
    files = sorted(p.name for p in (tmp_path / "tiny-mlp-spc").glob("trial*.json"))
    assert files == ["trial0.json", "trial1.json"]
    first = json.loads((tmp_path / "tiny-mlp-spc" / "trial0.json").read_text())
    assert first["status"] == "ok"
    assert first["seed"] == 0

    # A run also leaves a manifest next to the trials.
    manifest = json.loads((tmp_path / "tiny-mlp-spc" / "manifest.json").read_text())
    assert manifest["seeds"] == [0, 1000]
    assert manifest["row"]["id"] == "tiny-mlp-spc"
    assert "tiny-mlp-spc" in manifest["command"]


def test_a_failed_trial_makes_the_command_exit_nonzero(tmp_path, monkeypatch, rng_key):
    tiny = register_tiny_row(monkeypatch, rng_key)

    def broken_loaders(seed):
        raise RuntimeError("dataset is missing")

    broken = dataclasses.replace(
        tiny, id="broken-mlp-spc", loader_factory=broken_loaders
    )
    monkeypatch.setitem(registry.ROWS, "broken-mlp-spc", broken)

    code = main(["broken-mlp-spc", "--out", str(tmp_path), "--trials", "1"])

    assert code != 0
    on_disk = json.loads((tmp_path / "broken-mlp-spc" / "trial0.json").read_text())
    assert on_disk["status"] == "failed"
    assert "dataset is missing" in on_disk["error"]


def test_running_two_or_more_trials_also_writes_a_summary(
    tmp_path, monkeypatch, rng_key
):
    register_tiny_row(monkeypatch, rng_key)
    code = main(
        ["tiny-mlp-spc", "--out", str(tmp_path), "--trials", "2", "--epochs", "1"]
    )
    assert code == 0
    summary = json.loads((tmp_path / "tiny-mlp-spc" / "summary.json").read_text())
    assert summary["n_ok"] == 2
    assert "accuracy" in summary["metrics"]


def test_compare_command_pairs_two_rows(tmp_path, monkeypatch, rng_key):
    tiny = register_tiny_row(monkeypatch, rng_key)
    tiny_bp = dataclasses.replace(
        tiny,
        id="tiny-mlp-backprop",
        algorithm="backprop",
        model_factory=registry.ROWS["mnist-mlp-backprop"].model_factory,
    )
    monkeypatch.setitem(registry.ROWS, "tiny-mlp-backprop", tiny_bp)
    common = ["--out", str(tmp_path), "--trials", "2", "--epochs", "1"]
    assert main(["tiny-mlp-spc", *common]) == 0
    assert main(["tiny-mlp-backprop", *common]) == 0

    code = main(
        ["compare", "tiny-mlp-spc", "tiny-mlp-backprop", "--out", str(tmp_path)]
    )

    assert code == 0
    c = json.loads(
        (tmp_path / "compare-tiny-mlp-spc-vs-tiny-mlp-backprop.json").read_text()
    )
    assert c["metric"] == "accuracy"
    assert c["n"] == 2


def test_smoke_runs_a_short_row_and_checks_an_accuracy_floor(
    tmp_path, monkeypatch, rng_key
):
    register_tiny_row(monkeypatch, rng_key)
    # Floor 0.0 so random data passes; the real smoke uses a real floor.
    code = main(
        ["smoke", "--row", "tiny-mlp-spc", "--floor", "0.0", "--out", str(tmp_path)]
    )
    assert code == 0
    summary = json.loads((tmp_path / "tiny-mlp-spc" / "summary.json").read_text())
    assert summary["n_ok"] >= 2


def test_smoke_fails_when_accuracy_is_below_the_floor(tmp_path, monkeypatch, rng_key):
    register_tiny_row(monkeypatch, rng_key)
    # Random labels cannot reach 99%, so this must fail.
    code = main(
        ["smoke", "--row", "tiny-mlp-spc", "--floor", "0.99", "--out", str(tmp_path)]
    )
    assert code != 0


def _tiny_args(tmp_path, *extra):
    return [
        "tiny-mlp-spc",
        "--out",
        str(tmp_path),
        "--trials",
        "2",
        "--epochs",
        "1",
        "--warmup",
        "1",
        "--timed",
        "2",
        *extra,
    ]


def _count_trial_runs(monkeypatch):
    """Wrap run_trial so a test can see which trials actually trained."""
    from fabricpc.bench import __main__ as cli

    ran = []
    real = cli.run_trial

    def counting(row, trial, *args, **kwargs):
        ran.append(trial)
        return real(row, trial, *args, **kwargs)

    monkeypatch.setattr(cli, "run_trial", counting)
    return ran


def test_resume_skips_trials_that_already_finished(tmp_path, monkeypatch, rng_key):
    register_tiny_row(monkeypatch, rng_key)
    assert main(_tiny_args(tmp_path)) == 0
    # Pretend the session died before trial 1 was written.
    (tmp_path / "tiny-mlp-spc" / "trial1.json").unlink()

    ran = _count_trial_runs(monkeypatch)
    assert main(_tiny_args(tmp_path, "--resume")) == 0

    assert ran == [1]
    summary = json.loads((tmp_path / "tiny-mlp-spc" / "summary.json").read_text())
    assert summary["n_ok"] == 2


def test_resume_reruns_a_failed_trial(tmp_path, monkeypatch, rng_key):
    register_tiny_row(monkeypatch, rng_key)
    assert main(_tiny_args(tmp_path)) == 0
    path = tmp_path / "tiny-mlp-spc" / "trial0.json"
    failed = {**json.loads(path.read_text()), "status": "failed"}
    path.write_text(json.dumps(failed))

    ran = _count_trial_runs(monkeypatch)
    assert main(_tiny_args(tmp_path, "--resume")) == 0

    assert ran == [0]
    assert json.loads(path.read_text())["status"] == "ok"


def test_resume_reruns_a_trial_trained_for_a_different_epoch_count(
    tmp_path, monkeypatch, rng_key
):
    register_tiny_row(monkeypatch, rng_key)
    assert main(_tiny_args(tmp_path)) == 0

    ran = _count_trial_runs(monkeypatch)
    args = _tiny_args(tmp_path, "--resume")
    args[args.index("--epochs") + 1] = "2"
    assert main(args) == 0

    # Old results were for 1 epoch, so neither can be reused for 2.
    assert ran == [0, 1]


def test_without_resume_every_trial_runs_again(tmp_path, monkeypatch, rng_key):
    register_tiny_row(monkeypatch, rng_key)
    assert main(_tiny_args(tmp_path)) == 0

    ran = _count_trial_runs(monkeypatch)
    assert main(_tiny_args(tmp_path)) == 0

    assert ran == [0, 1]


def test_zoo_flag_saves_a_checkpoint_per_trial(tmp_path, monkeypatch, rng_key):
    register_tiny_row(monkeypatch, rng_key)
    zoo = tmp_path / "zoo"

    assert main(_tiny_args(tmp_path, "--zoo", str(zoo))) == 0

    for trial in (0, 1):
        assert (zoo / "tiny-mlp-spc" / f"trial{trial}").is_dir()
        assert (zoo / "tiny-mlp-spc" / f"trial{trial}.json").is_file()
        result = json.loads(
            (tmp_path / "tiny-mlp-spc" / f"trial{trial}.json").read_text()
        )
        assert result["checkpoint"].endswith(f"trial{trial}")


def _with_reference(monkeypatch, rng_key, value):
    """The tiny row, run the full way (2 seeds, 1 epoch), with an expected score."""
    from fabricpc.bench.registry import Reference

    tiny = register_tiny_row(monkeypatch, rng_key)
    ref = Reference(metric="accuracy", value=value, source="test")
    judged = dataclasses.replace(
        tiny, reference=ref, n_trials=2, train_config={"num_epochs": 1}
    )
    monkeypatch.setitem(registry.ROWS, "tiny-mlp-spc", judged)


def test_a_run_far_from_its_expected_score_fails_the_command(
    tmp_path, monkeypatch, rng_key
):
    _with_reference(monkeypatch, rng_key, value=2.0)  # impossible accuracy

    code = main(
        ["tiny-mlp-spc", "--out", str(tmp_path), "--warmup", "1", "--timed", "2"]
    )

    assert code != 0
    summary = json.loads((tmp_path / "tiny-mlp-spc" / "summary.json").read_text())
    assert summary["band"]["status"] == "fail"
    assert summary["band"]["pending_signoff"] is True


def test_a_run_matching_its_expected_score_passes(tmp_path, monkeypatch, rng_key):
    # Run once to learn what the tiny row scores, then expect exactly that.
    register_tiny_row(monkeypatch, rng_key)
    args = ["tiny-mlp-spc", "--out", str(tmp_path), "--trials", "2", "--epochs", "1"]
    assert main([*args, "--warmup", "1", "--timed", "2"]) == 0
    first = json.loads((tmp_path / "tiny-mlp-spc" / "summary.json").read_text())
    _with_reference(monkeypatch, rng_key, value=first["metrics"]["accuracy"]["mean"])

    code = main(
        ["tiny-mlp-spc", "--out", str(tmp_path), "--warmup", "1", "--timed", "2"]
    )

    assert code == 0
    summary = json.loads((tmp_path / "tiny-mlp-spc" / "summary.json").read_text())
    assert summary["band"]["status"] == "pass"


def test_trial_flag_runs_just_that_one_trial(tmp_path, monkeypatch, rng_key):
    register_tiny_row(monkeypatch, rng_key)

    code = main([*_tiny_args(tmp_path), "--trial", "1"])

    assert code == 0
    row_dir = tmp_path / "tiny-mlp-spc"
    assert [p.name for p in row_dir.glob("trial*.json")] == ["trial1.json"]
    assert not (row_dir / "summary.json").exists()


def test_by_default_each_trial_runs_in_its_own_process(tmp_path, monkeypatch, rng_key):
    from fabricpc.bench import __main__ as cli
    from fabricpc.bench.runner import run_trial

    register_tiny_row(monkeypatch, rng_key)
    in_child = []

    def fake_child(row, trial, out, **kwargs):
        in_child.append(trial)
        return run_trial(row, trial, out, **kwargs)  # same work, no real child

    monkeypatch.setattr(cli, "run_trial_in_child", fake_child)
    code = cli_main(_tiny_args(tmp_path))  # no --in-process

    assert code == 0
    assert in_child == [0, 1]


def test_a_run_writes_trials_csv_and_validates(tmp_path, monkeypatch, rng_key):
    register_tiny_row(monkeypatch, rng_key)
    assert main(_tiny_args(tmp_path)) == 0

    assert (tmp_path / "tiny-mlp-spc" / "trials.csv").exists()
    assert main(["validate", str(tmp_path)]) == 0


def test_validate_command_fails_on_a_broken_folder(tmp_path, monkeypatch, rng_key):
    register_tiny_row(monkeypatch, rng_key)
    assert main(_tiny_args(tmp_path)) == 0
    (tmp_path / "tiny-mlp-spc" / "manifest.json").unlink()

    assert main(["validate", str(tmp_path)]) != 0
