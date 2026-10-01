"""Tests for the run queue (fabricpc.bench.queue).

Most tests hand the queue a fake "run one job" function, so they are fast
and only check the queue's own bookkeeping. One test runs a tiny MNIST-like
row for real, twice, to check that a restarted queue skips finished trials.
"""

import json
from pathlib import Path

import pytest

from fabricpc.bench import queue
from fabricpc.bench.__main__ import main as cli_main
from tests.test_bench_cli_run import register_tiny_row

# Families that were added at the same time as the queue; all are in now.
IN_FLIGHT = {
    "tinyimagenet-vgg5",
    "tinyshakespeare-bpe-transformer",
    "cifar10-resnet18lean",
    "mnist-fcresnet8",
    "mnist-fcresnet32",
    "mnist-fcresnet128",
}


def write_plan(tmp_path, jobs, **extra):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({"name": "test-plan", "jobs": jobs, **extra}))
    return path


def read_status(out):
    return json.loads((Path(out) / "queue-status.json").read_text())


class FakeRun:
    """Stands in for the normal command line. Records every call and what
    the status file said at that moment."""

    def __init__(self, out, codes=None, interrupt_on=None):
        self.out = Path(out)
        self.codes = codes or {}
        self.interrupt_on = interrupt_on
        self.calls = []
        self.status_seen = []

    def __call__(self, argv):
        job_id = argv[0]
        self.calls.append(list(argv))
        self.status_seen.append(read_status(self.out))
        if job_id == self.interrupt_on:
            raise KeyboardInterrupt
        return self.codes.get(job_id, 0)


# --- reading a plan ---------------------------------------------------------


def test_a_plan_is_read_in_order_with_its_options(tmp_path):
    path = write_plan(
        tmp_path,
        [
            {"id": "mnist-mlp", "trials": 5, "zoo": True, "note": "re-measure"},
            {"id": "mnist-mlp-spc", "epochs": 0.5},
        ],
    )

    plan = queue.load_plan(path)

    assert plan.name == "test-plan"
    assert [j.id for j in plan.jobs] == ["mnist-mlp", "mnist-mlp-spc"]
    first, second = plan.jobs
    assert (first.trials, first.epochs, first.zoo, first.note) == (
        5,
        None,
        True,
        "re-measure",
    )
    # Left out: the row's own seed count and epochs, and no zoo.
    assert (second.trials, second.epochs, second.zoo) == (None, 0.5, False)


def test_plan_defaults_apply_to_every_job_unless_a_job_says_otherwise(tmp_path):
    path = write_plan(
        tmp_path,
        [{"id": "mnist-mlp"}, {"id": "fashionmnist-mlp", "zoo": False}],
        defaults={"trials": 3, "zoo": True},
    )

    a, b = queue.load_plan(path).jobs

    assert (a.trials, a.zoo) == (3, True)
    assert (b.trials, b.zoo) == (3, False)


@pytest.mark.parametrize(
    "jobs, words",
    [
        ([], "no jobs"),
        ([{"trials": 5}], "id"),
        ([{"id": "mnist-mlp", "trials": 0}], "trials"),
        ([{"id": "mnist-mlp", "trials": "five"}], "trials"),
        ([{"id": "mnist-mlp", "epochs": -1}], "epochs"),
        ([{"id": "mnist-mlp", "zoo": "yes"}], "zoo"),
        ([{"id": "mnist-mlp", "epoch": 3}], "epoch"),  # a typo, not ignored
        ([{"id": "mnist-mlp"}, {"id": "mnist-mlp"}], "twice"),
    ],
)
def test_a_broken_plan_is_refused_with_a_clear_message(tmp_path, jobs, words):
    path = write_plan(tmp_path, jobs)

    with pytest.raises(queue.PlanError, match=words):
        queue.load_plan(path)


def test_a_plan_that_is_not_json_is_refused(tmp_path):
    path = tmp_path / "plan.json"
    path.write_text("{not json")

    with pytest.raises(queue.PlanError, match="JSON"):
        queue.load_plan(path)


def test_unknown_ids_are_listed_and_known_ones_expand_to_rows(tmp_path):
    plan = queue.load_plan(
        write_plan(
            tmp_path,
            [{"id": "mnist-mlp"}, {"id": "no-such-row"}, {"id": "mnist-mlp-spc"}],
        )
    )

    assert queue.unknown_ids(plan) == ["no-such-row"]
    assert queue.job_rows("mnist-mlp") == (
        "mnist-mlp-spc",
        "mnist-mlp-epc",
        "mnist-mlp-backprop",
    )
    assert queue.job_rows("mnist-mlp-spc") == ("mnist-mlp-spc",)
    assert queue.job_rows("no-such-row") is None


def test_validate_plan_names_unknown_ids_and_exits_nonzero(tmp_path, capsys):
    path = write_plan(tmp_path, [{"id": "mnist-mlp"}, {"id": "no-such-row"}])

    code = cli_main(["queue", str(path), "--validate-plan"])

    assert code == 1
    assert "no-such-row" in capsys.readouterr().out


def test_validate_plan_passes_a_plan_with_only_known_ids(tmp_path, capsys):
    path = write_plan(tmp_path, [{"id": "mnist-mlp"}])

    assert cli_main(["queue", str(path), "--validate-plan"]) == 0


def test_the_shipped_gpu_campaign_plan_reads_and_starts_with_vgg5():
    plan = queue.load_plan(queue.PLANS_DIR / "gpu_campaign.json")

    ids = [j.id for j in plan.jobs]
    assert ids[0] == "cifar10-vgg5"
    assert plan.jobs[0].trials == 5
    assert {"mnist-mlp", "fashionmnist-mlp", "mnist-autoencoder"} <= set(ids)
    assert ids.index("cifar100-vgg5") < ids.index("tinyshakespeare-transformer")
    shakespeare = plan.jobs[ids.index("tinyshakespeare-transformer")]
    assert shakespeare.epochs == 5
    # Every family in the plan is in the registry now, so nothing is skipped.
    assert queue.unknown_ids(plan) == []
    assert IN_FLIGHT <= set(ids)


def test_the_gpu_campaign_runs_the_cheap_sponsor_rows_before_resnet18():
    # ResNet-18 is the most expensive job, so it goes last, and the plain and
    # lean layouts run on the same, shorter budget so the two can be compared.
    plan = queue.load_plan(queue.PLANS_DIR / "gpu_campaign.json")
    ids = [j.id for j in plan.jobs]
    plain = plan.jobs[ids.index("cifar10-resnet18")]
    lean = plan.jobs[ids.index("cifar10-resnet18lean")]
    assert plain.epochs is not None and plain.epochs < 100
    assert (plain.epochs, plain.trials) == (lean.epochs, lean.trials)
    for cheap in (
        "tinyimagenet-vgg5",
        "mnist-fcresnet8",
        "mnist-fcresnet32",
        "mnist-fcresnet128",
    ):
        assert ids.index(cheap) < min(ids.index(plain.id), ids.index(lean.id))


def test_the_deeper_vggs_wait_in_their_own_plan_behind_an_spc_probe():
    # VGG-7/9 layouts still need the sponsor's sign-off, and sPC's settling
    # should be probed before paying for a full run, so they are not in the
    # main campaign. Their own plan starts each with a short sPC-only probe.
    main = queue.load_plan(queue.PLANS_DIR / "gpu_campaign.json")
    assert not {"cifar10-vgg7", "cifar10-vgg9"} & {j.id for j in main.jobs}

    plan = queue.load_plan(queue.PLANS_DIR / "gpu_deep_vgg.json")
    ids = [j.id for j in plan.jobs]
    assert queue.unknown_ids(plan) == []
    for depth in (7, 9):
        probe = plan.jobs[ids.index(f"cifar10-vgg{depth}-spc")]
        assert probe.trials == 1 and probe.epochs is not None and probe.epochs <= 5
        assert ids.index(probe.id) < ids.index(f"cifar10-vgg{depth}")
        assert "sponsor" in plan.jobs[ids.index(f"cifar10-vgg{depth}")].note


# --- running a plan (with a fake run) --------------------------------------


def test_each_job_runs_through_the_normal_command_with_resume(tmp_path):
    out = tmp_path / "out"
    plan = queue.load_plan(
        write_plan(
            tmp_path,
            [
                {"id": "mnist-mlp", "trials": 5, "zoo": True},
                {"id": "mnist-mlp-spc", "epochs": 0.5},
            ],
        )
    )
    fake = FakeRun(out)

    code = queue.run_queue(plan, out, run_main=fake, finish=False)

    assert code == 0
    first, second = fake.calls
    assert first[0] == "mnist-mlp"
    assert "--resume" in first
    assert first[first.index("--out") + 1] == str(out / "mnist-mlp")
    assert first[first.index("--trials") + 1] == "5"
    assert first[first.index("--zoo") + 1] == str(out / "zoo")
    assert "--epochs" not in first
    assert second[second.index("--epochs") + 1] == "0.5"
    assert "--zoo" not in second and "--trials" not in second


def test_the_status_file_is_written_before_and_after_every_job(tmp_path):
    out = tmp_path / "out"
    plan = queue.load_plan(
        write_plan(tmp_path, [{"id": "mnist-mlp"}, {"id": "fashionmnist-mlp"}])
    )
    fake = FakeRun(out)

    queue.run_queue(plan, out, run_main=fake, finish=False)

    # While job 2 ran, job 1 was done and job 2 was marked running.
    during = fake.status_seen[1]["jobs"]
    assert [j["status"] for j in during] == ["done", "running"]
    assert during[0]["started"] and during[0]["ended"]
    assert during[1]["started"] and during[1]["ended"] is None

    status = read_status(out)
    assert [j["status"] for j in status["jobs"]] == ["done", "done"]
    assert status["state"] == "finished"
    assert "git_sha" in status and status["device"]["platform"]
    assert status["jobs"][0]["rows"] == [
        "mnist-mlp-spc",
        "mnist-mlp-epc",
        "mnist-mlp-backprop",
    ]


def test_an_unknown_id_is_skipped_with_a_message_not_a_crash(tmp_path, capsys):
    out = tmp_path / "out"
    plan = queue.load_plan(
        write_plan(tmp_path, [{"id": "not-built-yet"}, {"id": "mnist-mlp"}])
    )
    fake = FakeRun(out)

    code = queue.run_queue(plan, out, run_main=fake, finish=False)

    assert code == 0
    assert [c[0] for c in fake.calls] == ["mnist-mlp"]
    skipped = read_status(out)["jobs"][0]
    assert skipped["status"] == "skipped"
    assert "not-built-yet" in capsys.readouterr().out
    assert "not in the registry" in skipped["message"]


def test_a_failed_job_stops_the_queue_and_leaves_the_rest_pending(tmp_path):
    out = tmp_path / "out"
    plan = queue.load_plan(
        write_plan(
            tmp_path,
            [{"id": "mnist-mlp"}, {"id": "fashionmnist-mlp"}, {"id": "mnist-mlp-spc"}],
        )
    )
    fake = FakeRun(out, codes={"mnist-mlp": 1})

    code = queue.run_queue(plan, out, run_main=fake, finish=False)

    assert code == 1
    assert len(fake.calls) == 1
    status = read_status(out)
    assert [j["status"] for j in status["jobs"]] == ["failed", "pending", "pending"]
    assert status["state"] == "stopped"


def test_continue_on_failure_runs_the_rest_but_still_exits_nonzero(tmp_path):
    out = tmp_path / "out"
    plan = queue.load_plan(
        write_plan(tmp_path, [{"id": "mnist-mlp"}, {"id": "fashionmnist-mlp"}])
    )
    fake = FakeRun(out, codes={"mnist-mlp": 1})

    code = queue.run_queue(
        plan, out, run_main=fake, continue_on_failure=True, finish=False
    )

    assert code == 1
    assert [j["status"] for j in read_status(out)["jobs"]] == ["failed", "done"]


def test_a_nonzero_exit_with_every_trial_finished_counts_as_done(tmp_path):
    """A row that misses its expected score exits 1 but has all its trials;
    that is a result to look at, not a reason to stop the queue."""
    out = tmp_path / "out"
    plan = queue.load_plan(
        write_plan(tmp_path, [{"id": "mnist-mlp-spc", "trials": 2, "epochs": 1}])
    )

    def band_fail(argv):
        row_dir = out / "mnist-mlp-spc" / "mnist-mlp-spc"
        row_dir.mkdir(parents=True, exist_ok=True)
        for t in range(2):
            (row_dir / f"trial{t}.json").write_text(
                json.dumps({"status": "ok", "num_epochs": 1.0})
            )
        return 1

    code = queue.run_queue(plan, out, run_main=band_fail, finish=False)

    assert code == 0
    job = read_status(out)["jobs"][0]
    assert job["status"] == "done"
    assert job["exit_code"] == 1
    assert "band" in job["message"]


def test_only_runs_just_the_named_jobs(tmp_path):
    out = tmp_path / "out"
    plan = queue.load_plan(
        write_plan(
            tmp_path,
            [{"id": "mnist-mlp"}, {"id": "fashionmnist-mlp"}, {"id": "mnist-mlp-spc"}],
        )
    )
    fake = FakeRun(out)

    queue.run_queue(plan, out, run_main=fake, only=["fashionmnist-mlp"], finish=False)

    assert [c[0] for c in fake.calls] == ["fashionmnist-mlp"]
    statuses = [j["status"] for j in read_status(out)["jobs"]]
    assert statuses == ["not selected", "done", "not selected"]


def test_only_with_an_id_not_in_the_plan_is_an_error(tmp_path, capsys):
    path = write_plan(tmp_path, [{"id": "mnist-mlp"}])

    code = cli_main(
        ["queue", str(path), "--out", str(tmp_path / "o"), "--only", "cifar10-vgg5"]
    )

    assert code == 2
    assert "cifar10-vgg5" in capsys.readouterr().err


def test_an_interrupted_queue_says_so_and_a_restart_finishes_it(tmp_path):
    out = tmp_path / "out"
    plan = queue.load_plan(
        write_plan(tmp_path, [{"id": "mnist-mlp"}, {"id": "fashionmnist-mlp"}])
    )

    with pytest.raises(KeyboardInterrupt):
        queue.run_queue(
            plan,
            out,
            run_main=FakeRun(out, interrupt_on="fashionmnist-mlp"),
            finish=False,
        )
    status = read_status(out)
    assert [j["status"] for j in status["jobs"]] == ["done", "interrupted"]
    assert status["state"] == "interrupted"

    fake = FakeRun(out)
    assert queue.run_queue(plan, out, run_main=fake, finish=False) == 0
    # Both jobs run again with --resume, which skips what already finished.
    assert [c[0] for c in fake.calls] == ["mnist-mlp", "fashionmnist-mlp"]
    status = read_status(out)
    assert [j["status"] for j in status["jobs"]] == ["done", "done"]
    assert status["restarts"] == 1


# --- dry run ----------------------------------------------------------------


def test_dry_run_prints_the_plan_without_running_anything(tmp_path, capsys):
    path = write_plan(
        tmp_path,
        [{"id": "mnist-mlp", "trials": 5, "note": "re-measure"}, {"id": "nope"}],
    )
    out = tmp_path / "out"

    code = cli_main(["queue", str(path), "--out", str(out), "--dry-run"])

    text = capsys.readouterr().out
    assert code == 0
    assert "mnist-mlp" in text and "re-measure" in text
    assert "nope" in text and "unknown" in text
    assert not out.exists()


def test_dry_run_estimates_hours_from_a_probe_file(tmp_path, capsys):
    probe_dir = tmp_path / "probes"
    probe_dir.mkdir()
    rows = [
        {
            "row_id": f"mnist-mlp-{a}",
            "epochs": 20.0,
            "train_s_per_seed": 3600.0,
            "eval_s_per_seed": 0.0,
        }
        for a in ("spc", "epc", "backprop")
    ]
    (probe_dir / "probe-mnist-mlp.json").write_text(
        json.dumps({"target": "mnist-mlp", "device": {"kind": "T4"}, "rows": rows})
    )
    path = write_plan(tmp_path, [{"id": "mnist-mlp", "trials": 2, "epochs": 10}])

    code = cli_main(
        ["queue", str(path), "--out", str(tmp_path / "o"), "--dry-run"]
        + ["--probe", str(probe_dir)]
    )

    text = capsys.readouterr().out
    assert code == 0
    # 3 rows x 2 seeds x half the probed epochs of a 1-hour seed = 3 hours.
    assert "3.0 h" in text
    assert "T4" in text


def test_estimate_is_missing_for_rows_without_a_probe():
    est = queue.estimate_hours(queue.Job(id="mnist-mlp-spc", trials=1), probes={})

    assert est is None


# --- a real, tiny run -------------------------------------------------------


def test_a_real_restart_skips_trials_that_already_finished(
    tmp_path, monkeypatch, rng_key, capsys
):
    register_tiny_row(monkeypatch, rng_key)
    out = tmp_path / "out"
    common = ["--warmup", "1", "--timed", "2", "--in-process"]

    # First run: one seed (as if the queue was killed after trial 0).
    first = write_plan(tmp_path, [{"id": "tiny-mlp-spc", "trials": 1, "epochs": 1}])
    assert cli_main(["queue", str(first), "--out", str(out), *common]) == 0
    trial0 = out / "tiny-mlp-spc" / "tiny-mlp-spc" / "trial0.json"
    before = trial0.read_text()

    # Restart asking for two seeds: trial 0 is kept, only trial 1 trains.
    second = write_plan(tmp_path, [{"id": "tiny-mlp-spc", "trials": 2, "epochs": 1}])
    capsys.readouterr()
    assert cli_main(["queue", str(second), "--out", str(out), *common]) == 0

    text = capsys.readouterr().out
    assert "trial 0: already done, skipping" in text
    assert trial0.read_text() == before
    assert (out / "tiny-mlp-spc" / "tiny-mlp-spc" / "trial1.json").exists()
    status = read_status(out)
    assert status["jobs"][0]["status"] == "done"
    assert status["validate"]["ok"] is True
    assert (out / "report.md").exists()
