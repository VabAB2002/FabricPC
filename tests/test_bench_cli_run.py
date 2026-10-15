"""Tests for running a row from the command line (in-process, tiny data)."""

import dataclasses
import json

import pytest

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


def _trial_files(folder):
    return sorted(p.name for p in folder.glob("trial*.json"))


def test_the_fast_preset_runs_three_seeds(tmp_path, monkeypatch, rng_key):
    register_tiny_row(monkeypatch, rng_key)
    quick = ["--epochs", "0.1", "--warmup", "1", "--timed", "1", "--fast"]

    code = main(["tiny-mlp-spc", "--out", str(tmp_path), "--preset", "fast", *quick])

    assert code == 0
    assert _trial_files(tmp_path / "tiny-mlp-spc") == [
        "trial0.json",
        "trial1.json",
        "trial2.json",
    ]


def test_trials_wins_over_a_preset(tmp_path, monkeypatch, rng_key):
    register_tiny_row(monkeypatch, rng_key)
    quick = ["--epochs", "0.1", "--warmup", "1", "--timed", "1", "--fast"]

    code = main(
        [
            "tiny-mlp-spc",
            "--out",
            str(tmp_path),
            "--preset",
            "standard",
            "--trials",
            "2",
            *quick,
        ]
    )

    assert code == 0
    assert _trial_files(tmp_path / "tiny-mlp-spc") == ["trial0.json", "trial1.json"]


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
    """The tiny row, run the full way (3 seeds, the fewest a verdict needs, and
    1 epoch), with an expected score."""
    from fabricpc.bench.registry import Reference

    tiny = register_tiny_row(monkeypatch, rng_key)
    ref = Reference(metric="accuracy", value=value, source="test")
    judged = dataclasses.replace(
        tiny, reference=ref, n_trials=3, train_config={"num_epochs": 1}
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
    args = ["tiny-mlp-spc", "--out", str(tmp_path), "--trials", "3", "--epochs", "1"]
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


def _trial_names(tmp_path):
    return sorted(p.name for p in (tmp_path / "tiny-mlp-spc").glob("trial*.json"))


def _three_then(tmp_path, *extra):
    """Run 3 trials, then run the same row again into the same folder."""
    args3 = _tiny_args(tmp_path)
    args3[args3.index("--trials") + 1] = "3"
    assert main(args3) == 0
    return main(_tiny_args(tmp_path, *extra))  # 2 trials


def test_a_new_run_clears_the_rows_old_trials(tmp_path, monkeypatch, rng_key):
    register_tiny_row(monkeypatch, rng_key)

    assert _three_then(tmp_path) == 0

    assert _trial_names(tmp_path) == ["trial0.json", "trial1.json"]
    summary = json.loads((tmp_path / "tiny-mlp-spc" / "summary.json").read_text())
    assert summary["n_ok"] == 2
    assert main(["validate", str(tmp_path)]) == 0


def test_an_old_summary_does_not_outlive_a_run_too_small_to_summarize(
    tmp_path, monkeypatch, rng_key
):
    register_tiny_row(monkeypatch, rng_key)
    assert main(_tiny_args(tmp_path)) == 0  # 2 trials -> summary.json
    one = _tiny_args(tmp_path)
    one[one.index("--trials") + 1] = "1"

    assert main(one) == 0

    assert not (tmp_path / "tiny-mlp-spc" / "summary.json").exists()


def test_clearing_leaves_other_rows_alone(tmp_path, monkeypatch, rng_key):
    register_tiny_row(monkeypatch, rng_key)
    other = tmp_path / "some-other-row"
    other.mkdir()
    (other / "trial0.json").write_text("{}")

    assert main(_tiny_args(tmp_path)) == 0

    assert (other / "trial0.json").exists()


def test_compare_command_on_a_family_reports_three_contrasts(tmp_path, capsys):
    from tests.test_bench_summary import write_trials

    write_trials(tmp_path, "mnist-mlp-spc", [0.90, 0.92, 0.94])
    write_trials(tmp_path, "mnist-mlp-epc", [0.92, 0.93, 0.94], algorithm="epc")
    write_trials(
        tmp_path, "mnist-mlp-backprop", [0.91, 0.91, 0.91], algorithm="backprop"
    )

    assert main(["compare", "mnist-mlp", "--out", str(tmp_path)]) == 0

    printed = capsys.readouterr().out
    assert "mnist-mlp-spc - mnist-mlp-backprop" in printed
    assert "mnist-mlp-epc - mnist-mlp-spc" in printed
    assert (tmp_path / "compare-mnist-mlp.json").exists()


def test_running_a_family_runs_all_three_rows_then_compares(
    tmp_path, monkeypatch, rng_key
):
    tiny = register_tiny_row(monkeypatch, rng_key)
    for algo in ("epc", "backprop"):
        base = registry.ROWS[f"mnist-mlp-{algo}"]
        monkeypatch.setitem(
            registry.ROWS,
            f"tiny-mlp-{algo}",
            dataclasses.replace(
                base,
                id=f"tiny-mlp-{algo}",
                dataset="tiny",
                loader_factory=tiny.loader_factory,
            ),
        )
    monkeypatch.setitem(
        registry.COMPARISONS, "tiny-mlp", registry._family_comparison("tiny-mlp")
    )
    args = _tiny_args(tmp_path)
    args[0] = "tiny-mlp"

    assert main(args) == 0

    for algo in ("spc", "epc", "backprop"):
        assert (tmp_path / f"tiny-mlp-{algo}" / "summary.json").exists()
    out = json.loads((tmp_path / "compare-tiny-mlp.json").read_text())
    assert len(out["contrasts"]) == 3


def test_trials_csv_is_kept_up_to_date_if_a_run_is_cut_off(
    tmp_path, monkeypatch, rng_key
):
    from fabricpc.bench import __main__ as cli

    register_tiny_row(monkeypatch, rng_key)
    real = cli.run_trial

    def cut_off_at_trial_1(row, trial, *args, **kwargs):
        if trial == 1:
            raise KeyboardInterrupt  # like a time limit killing the run
        return real(row, trial, *args, **kwargs)

    monkeypatch.setattr(cli, "run_trial", cut_off_at_trial_1)
    try:
        main(_tiny_args(tmp_path))
    except KeyboardInterrupt:
        pass

    with open(tmp_path / "tiny-mlp-spc" / "trials.csv") as f:
        assert len(f.readlines()) == 2  # header + trial 0
    assert main(["validate", str(tmp_path)]) == 0


def test_family_compare_uses_the_familys_metric_unless_told_otherwise(tmp_path, capsys):
    from tests.test_bench_summary import write_trials

    for algo in ("spc", "epc", "backprop"):
        write_trials(
            tmp_path,
            f"tinyshakespeare-transformer-{algo}",
            [0.5, 0.52, 0.54],
            algorithm=algo,
        )
        for i in range(3):
            path = tmp_path / f"tinyshakespeare-transformer-{algo}" / f"trial{i}.json"
            data = json.loads(path.read_text())
            wobble = {"spc": (0.3, 0.1, 0.4), "epc": (0.0, 0.2, 0.1)}.get(algo)
            data["metrics"]["forward_perplexity"] = (
                12.0 + i + (wobble[i] if wobble else 0.0)
            )
            path.write_text(json.dumps(data))

    assert main(["compare", "tinyshakespeare-transformer", "--out", str(tmp_path)]) == 0

    out = json.loads(
        (tmp_path / "compare-tinyshakespeare-transformer.json").read_text()
    )
    assert out["metric"] == "forward_perplexity"
    assert "on forward_perplexity" in capsys.readouterr().out


def _capture_trial_kwargs(monkeypatch):
    from fabricpc.bench import __main__ as cli

    seen = []
    real = cli.run_trial

    def capturing(row, trial, *args, **kwargs):
        seen.append(kwargs)
        return real(row, trial, *args, **kwargs)

    monkeypatch.setattr(cli, "run_trial", capturing)
    return seen


def test_fast_turns_off_the_learning_curve_and_the_diagnostics(
    tmp_path, monkeypatch, rng_key
):
    register_tiny_row(monkeypatch, rng_key)
    seen = _capture_trial_kwargs(monkeypatch)
    assert main(_tiny_args(tmp_path, "--fast")) == 0

    assert all(k["curve_batches"] == 0 for k in seen)
    assert all(k["diagnostics"] is False for k in seen)
    trial = json.loads((tmp_path / "tiny-mlp-spc" / "trial0.json").read_text())
    assert trial["curve"] is None
    assert trial["diagnostics"] is None


def test_without_fast_the_curve_and_diagnostics_stay_on(tmp_path, monkeypatch, rng_key):
    register_tiny_row(monkeypatch, rng_key)
    seen = _capture_trial_kwargs(monkeypatch)
    assert main(_tiny_args(tmp_path)) == 0

    assert all(k.get("diagnostics", True) is True for k in seen)
    assert all(k.get("curve_batches") != 0 for k in seen)


# --- commands that must not train or overwrite by surprise -----------------


def _no_training(monkeypatch):
    import fabricpc.bench.__main__ as cli

    def refuse(*a, **k):
        raise AssertionError("this command must not train anything")

    monkeypatch.setattr(cli, "_run_row", refuse)


def test_dry_run_on_a_family_shows_its_rows_and_trains_nothing(monkeypatch, capsys):
    _no_training(monkeypatch)

    code = cli_main(["mnist-mlp", "--dry-run"])

    assert code == 0
    shown = json.loads(capsys.readouterr().out)
    assert [r["id"] for r in shown] == [
        "mnist-mlp-spc",
        "mnist-mlp-epc",
        "mnist-mlp-backprop",
    ]


def test_one_trial_of_a_family_is_refused(monkeypatch, capsys):
    _no_training(monkeypatch)

    code = cli_main(["mnist-mlp", "--trial", "2"])

    assert code != 0
    assert "row" in capsys.readouterr().err


def test_smoke_without_out_leaves_the_results_folder_alone(
    tmp_path, monkeypatch, rng_key
):
    register_tiny_row(monkeypatch, rng_key)
    monkeypatch.chdir(tmp_path)
    real = tmp_path / "results" / "tiny-mlp-spc"
    real.mkdir(parents=True)
    (real / "trial0.json").write_text("{}")  # stands in for a real run

    main(["smoke", "--row", "tiny-mlp-spc", "--epochs", "0.1", "--floor", "0"])

    assert (real / "trial0.json").read_text() == "{}"
    assert (tmp_path / "smoke-results" / "tiny-mlp-spc").is_dir()


def test_regress_without_out_writes_to_its_own_folder(monkeypatch):
    from fabricpc.bench import regress

    seen = {}
    monkeypatch.setattr(
        regress,
        "run_regress",
        lambda out, run_row, checks: seen.setdefault("out", out) and 0,
    )

    cli_main(["regress"])

    assert seen["out"] == "regress-results"


def test_a_capped_evaluation_is_recorded_on_the_trial(tmp_path, monkeypatch, rng_key):
    # Rehearsals score only a few test batches; the trial says so, so the
    # number is never mistaken for a real test-set score.
    register_tiny_row(monkeypatch, rng_key)
    quick = ["--epochs", "0.1", "--warmup", "1", "--timed", "1", "--fast"]

    code = main(
        [
            "tiny-mlp-spc",
            "--out",
            str(tmp_path),
            "--trials",
            "2",
            "--max-eval-batches",
            "1",
            *quick,
        ]
    )

    assert code == 0
    trial = json.loads((tmp_path / "tiny-mlp-spc" / "trial0.json").read_text())
    assert trial["eval_batches_capped"] == 1


def test_resume_with_fewer_seeds_than_finished_is_refused(
    tmp_path, monkeypatch, rng_key
):
    # Resuming a 3-seed run with --trials 2 used to delete trial 2 silently.
    register_tiny_row(monkeypatch, rng_key)
    quick = ["--epochs", "0.1", "--warmup", "1", "--timed", "1", "--fast"]
    out = ["tiny-mlp-spc", "--out", str(tmp_path)]
    assert main([*out, "--trials", "3", *quick]) == 0

    code = main([*out, "--trials", "2", "--resume", *quick])

    assert code != 0
    assert (tmp_path / "tiny-mlp-spc" / "trial2.json").exists()


def test_a_pc_trial_file_is_strict_json(tmp_path, monkeypatch, rng_key):
    # The safety net logs lambda_min as NaN at every sPC probe; NaN is not
    # JSON, so tools other than Python cannot read the file.
    register_tiny_row(monkeypatch, rng_key)
    quick = ["--epochs", "0.1", "--warmup", "1", "--timed", "1", "--fast"]

    assert main(["tiny-mlp-spc", "--out", str(tmp_path), "--trials", "2", *quick]) == 0

    text = (tmp_path / "tiny-mlp-spc" / "trial0.json").read_text()
    assert json.loads(text)["rate_control"] is not None
    assert "NaN" not in text


def test_max_train_steps_trains_exactly_that_many_steps(tmp_path, monkeypatch, rng_key):
    # A rehearsal must really train: a tiny --epochs rounds to 0 steps on
    # any loader under 5,000 batches, so the cap is in steps instead.
    from fabricpc.training import trainer

    register_tiny_row(monkeypatch, rng_key)
    steps = []
    real_make = trainer._make_step

    def counting(*args, **kwargs):
        step = real_make(*args, **kwargs)

        def wrapped(*a, **k):
            steps.append(1)
            return step(*a, **k)

        return wrapped

    monkeypatch.setattr(trainer, "_make_step", counting)
    quick = ["--warmup", "1", "--timed", "1", "--fast", "--trials", "2"]

    code = main(
        ["tiny-mlp-spc", "--out", str(tmp_path), "--max-train-steps", "2", *quick]
    )

    assert code == 0
    trial = json.loads((tmp_path / "tiny-mlp-spc" / "trial0.json").read_text())
    assert trial["train_steps_capped"] == 2
    assert trial["num_epochs"] == pytest.approx(2 / 3)  # the tiny loader has 3
    assert len(steps) == 2 * 2  # 2 steps in each of the 2 trials


def test_the_printed_lines_show_the_rows_own_score(
    tmp_path, monkeypatch, rng_key, capsys
):
    # A transformer row is scored on forward_perplexity, not accuracy; the
    # lines printed after each trial and after the row must say so.
    tiny = register_tiny_row(monkeypatch, rng_key)
    monkeypatch.setitem(
        registry.ROWS, "tiny-mlp-spc", dataclasses.replace(tiny, metric="energy")
    )

    code = main(_tiny_args(tmp_path))

    assert code == 0
    lines = capsys.readouterr().out.splitlines()
    trial_lines = [x for x in lines if x.startswith("tiny-mlp-spc trial ")]
    assert trial_lines and all("energy=" in x for x in trial_lines)
    assert not any("accuracy=" in x for x in trial_lines)
    assert any(x.startswith("tiny-mlp-spc: energy ") for x in lines)


def test_resume_skips_a_finished_trial_of_a_shortened_run(
    tmp_path, monkeypatch, rng_key
):
    # A shortened trial records the epochs it really trained (here 2 steps,
    # a fraction of an epoch), so resume compared that with the epochs asked
    # for and trained every finished trial again (seen on a Colab rehearsal).
    register_tiny_row(monkeypatch, rng_key)
    caps = ["--max-train-steps", "2", "--max-eval-batches", "1"]
    assert main(_tiny_args(tmp_path, *caps)) == 0
    (tmp_path / "tiny-mlp-spc" / "trial1.json").unlink()

    ran = _count_trial_runs(monkeypatch)
    assert main(_tiny_args(tmp_path, "--resume", *caps)) == 0

    assert ran == [1]


def test_resume_does_not_reuse_a_trial_scored_on_part_of_the_test_set(
    tmp_path, monkeypatch, rng_key
):
    register_tiny_row(monkeypatch, rng_key)
    assert main(_tiny_args(tmp_path, "--max-eval-batches", "1")) == 0

    ran = _count_trial_runs(monkeypatch)
    assert main(_tiny_args(tmp_path, "--resume")) == 0

    # Same epochs, but the old scores came from one test batch only.
    assert ran == [0, 1]
