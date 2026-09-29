"""Tests for ``fabricpc.bench.refresh``: fixing old results without retraining.

The fake "old" folders here are written the way an older version of the
suite wrote them: a FLOP count that is too small, an absolute checkpoint
path from the cloud machine, and no ``step_memory``.
"""

import dataclasses
import hashlib
import json
import subprocess
import sys

import jax
import pytest

from fabricpc.bench.compute import count_compute
from fabricpc.bench.manifest import write_manifest
from fabricpc.bench.refresh import refresh_results
from fabricpc.bench.registry import ROWS
from fabricpc.bench.runner import TrialResult
from fabricpc.bench.summary import summarize_row
from fabricpc.bench.writer import validate, write_trials_csv

MLP = ROWS["mnist-mlp-spc"]
CLOUD = "/teamspace/studios/this_studio/runs/old-run"
OLD_SHA = "a" * 40


def _write_old_row(
    root,
    row,
    *,
    n=2,
    infer_steps=3,
    flops_per_pass=1000,
    weighted_edges=None,
    step_time_ms=10.0,
    zoo=True,
    statuses=None,
    n_params=None,
):
    """An old results folder for one row, with a zoo next to it."""
    statuses = statuses or ["ok"] * n
    seeds = [i * 1000 for i in range(n)]
    manifest = write_manifest(root, row, seeds=seeds, command=["python", "old"])
    data = json.loads(manifest.read_text())
    data["git_sha"] = OLD_SHA
    manifest.write_text(json.dumps(data))
    params, structure = row.model_factory(jax.random.PRNGKey(0))
    if weighted_edges is None:
        weighted_edges = count_compute(
            params, structure, batch_size=row.batch_size, algorithm=row.algorithm
        ).weighted_edges
    if n_params is None:
        n_params = int(sum(p.size for p in jax.tree_util.tree_leaves(params)))
    factor = 3 if row.algorithm == "backprop" else 2 * infer_steps + 1
    compute = {
        "weighted_edges": weighted_edges,
        "infer_steps": infer_steps,
        "matmuls_per_update": factor * weighted_edges,
        "matmuls_per_update_backprop": 3 * weighted_edges,
        "pc_to_backprop_ratio": factor / 3,
        "flops_per_pass": flops_per_pass,
        "flops_per_update": factor * flops_per_pass,
    }
    for i, status in enumerate(statuses):
        ok = status == "ok"
        if zoo:
            ckpt_dir = root / "zoo" / row.id / f"trial{i}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            (ckpt_dir / "weights").write_text("pretend weights")
        result = TrialResult(
            row_id=row.id,
            trial=i,
            seed=seeds[i],
            algorithm=row.algorithm,
            n_params=n_params,
            metrics={"accuracy": 0.9 + i / 100} if ok else {},
            step_time_ms=step_time_ms if ok else 0.0,
            num_epochs=1.0,
            peak_memory_bytes=1_869_973_760 if ok else None,
            compute=compute if ok else {},
            achieved_tflops=(
                factor * flops_per_pass / (step_time_ms / 1000) / 1e12 if ok else 0.0
            ),
            checkpoint=f"{CLOUD}/zoo/{row.id}/trial{i}" if ok else None,
            status=status,
            error=None if ok else "Traceback...\nRuntimeError: boom",
        )
        data = dataclasses.asdict(result)
        del data["step_memory"]  # older code did not write it at all
        (root / row.id / f"trial{i}.json").write_text(json.dumps(data))
    if statuses.count("ok") >= 2:
        summarize_row(root, row.id)
    write_trials_csv(root, row.id)
    return root / row.id


def _fingerprint(folder):
    """Every file under a folder with a hash of its bytes."""
    return {
        str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(folder.rglob("*"))
        if p.is_file()
    }


def _trial(out, row_id, i=0):
    return json.loads((out / row_id / f"trial{i}.json").read_text())


@pytest.fixture
def old_mlp(tmp_path):
    src = tmp_path / "old"
    _write_old_row(src, MLP)
    return src


# --- the FLOP count --------------------------------------------------------


def test_flops_are_recounted_with_the_current_counter(old_mlp, tmp_path):
    out = tmp_path / "new"
    refresh_results(old_mlp, out)

    params, structure = MLP.model_factory(jax.random.PRNGKey(0))
    want = count_compute(
        params, structure, batch_size=MLP.batch_size, algorithm="spc", infer_steps=3
    )
    t = _trial(out, MLP.id)
    assert t["compute"] == dataclasses.asdict(want)
    assert t["achieved_tflops"] == pytest.approx(
        want.flops_per_update / (10.0 / 1000) / 1e12
    )


def test_the_saved_settling_steps_are_kept(old_mlp, tmp_path):
    # The row's solver today settles for 20 steps; the old run used 3. The
    # recount must describe the run that happened, not today's row.
    out = tmp_path / "new"
    refresh_results(old_mlp, out)
    assert _trial(out, MLP.id)["compute"]["infer_steps"] == 3


def test_step_time_and_metrics_are_left_alone(old_mlp, tmp_path):
    out = tmp_path / "new"
    refresh_results(old_mlp, out)
    old = json.loads((old_mlp / MLP.id / "trial1.json").read_text())
    new = _trial(out, MLP.id, 1)
    for key in ("metrics", "step_time_ms", "train_time_s", "seed", "n_params"):
        assert new[key] == old[key]


def test_a_changed_graph_is_refused(tmp_path):
    src = tmp_path / "old"
    _write_old_row(src, MLP, weighted_edges=99)
    with pytest.raises(ValueError, match="weighted edges"):
        refresh_results(src, tmp_path / "new")


def test_vgg5_fused_conv_flops_come_out_about_four_times_larger(tmp_path):
    # The real bug: the old counter sized each fused conv at its pooled
    # output. These are the numbers the old ePC trial 0 recorded.
    row = ROWS["cifar10-vgg5-epc"]
    src = tmp_path / "old"
    _write_old_row(
        src,
        row,
        n=2,
        infer_steps=5,
        flops_per_pass=12_311_330_816,
        weighted_edges=5,
        step_time_ms=98.30763649999597,
        zoo=False,
    )
    out = tmp_path / "new"
    refresh_results(src, out)
    t = _trial(out, row.id)
    assert t["compute"]["flops_per_pass"] == 49_229_594_624
    assert t["achieved_tflops"] == pytest.approx(5.508, abs=0.01)


# --- checkpoints, memory and the record of what changed --------------------


def test_a_cloud_checkpoint_path_becomes_relative_to_the_new_folder(old_mlp, tmp_path):
    out = tmp_path / "new"
    refresh_results(old_mlp, out)
    ckpt = _trial(out, MLP.id)["checkpoint"]
    assert not ckpt.startswith("/")
    # Like a fresh run, the path is relative to the results folder.
    assert (out / ckpt).resolve() == (old_mlp / "zoo" / MLP.id / "trial0").resolve()


def test_a_zoo_somewhere_else_can_be_named(tmp_path):
    src = tmp_path / "old"
    _write_old_row(src, MLP)
    moved = tmp_path / "elsewhere" / "zoo"
    moved.parent.mkdir()
    (src / "zoo").rename(moved)
    out = tmp_path / "new"
    refresh_results(src, out, zoo=moved)
    ckpt = _trial(out, MLP.id)["checkpoint"]
    assert (out / ckpt).resolve() == (moved / MLP.id / "trial0").resolve()


def test_a_checkpoint_with_no_zoo_on_disk_is_kept_and_flagged(tmp_path):
    src = tmp_path / "old"
    _write_old_row(src, MLP, zoo=False)
    out = tmp_path / "new"
    refresh_results(src, out)
    t = _trial(out, MLP.id)
    assert t["checkpoint"] == f"{CLOUD}/zoo/{MLP.id}/trial0"
    fields = [c["field"] for c in t["refresh"][-1]["changes"]]
    assert "checkpoint" not in fields
    assert any("checkpoint" in note for note in t["refresh"][-1]["notes"])


def test_peak_memory_is_kept_but_marked_not_comparable(old_mlp, tmp_path):
    out = tmp_path / "new"
    refresh_results(old_mlp, out)
    t = _trial(out, MLP.id)
    assert t["peak_memory_bytes"] == 1_869_973_760
    assert t["peak_memory_comparable"] is False
    # step_memory was never measured, so it must not appear out of nowhere.
    assert t.get("step_memory") is None


def test_each_trial_records_what_changed_and_which_code_did_it(old_mlp, tmp_path):
    out = tmp_path / "new"
    refresh_results(old_mlp, out)
    t = _trial(out, MLP.id)
    (record,) = t["refresh"]
    assert len(record["git_sha"] or "") in (0, 40)
    assert "refreshed_utc" in record
    assert record["source"].endswith(f"{MLP.id}/trial0.json")
    changes = {c["field"]: c for c in record["changes"]}
    assert set(changes) >= {"compute", "achieved_tflops", "checkpoint"}
    assert changes["achieved_tflops"]["new"] == pytest.approx(t["achieved_tflops"])
    assert changes["achieved_tflops"]["old"] != changes["achieved_tflops"]["new"]


def test_the_manifest_keeps_the_training_commit_and_adds_the_refresh(old_mlp, tmp_path):
    out = tmp_path / "new"
    refresh_results(old_mlp, out, command=["python", "-m", "fabricpc.bench"])
    manifest = json.loads((out / MLP.id / "manifest.json").read_text())
    assert manifest["git_sha"] == OLD_SHA  # the code that trained
    (record,) = manifest["refresh"]
    assert record["command"] == "python -m fabricpc.bench"
    assert record["source"] == str((old_mlp / MLP.id).resolve())
    assert record["changed_fields"]


def test_refreshing_twice_adds_a_second_record(old_mlp, tmp_path):
    once = tmp_path / "once"
    twice = tmp_path / "twice"
    refresh_results(old_mlp, once)
    refresh_results(once, twice)
    t = _trial(twice, MLP.id)
    assert len(t["refresh"]) == 2
    assert (twice / t["checkpoint"]).resolve() == (
        old_mlp / "zoo" / MLP.id / "trial0"
    ).resolve()
    manifest = json.loads((twice / MLP.id / "manifest.json").read_text())
    assert len(manifest["refresh"]) == 2


# --- the folder as a whole -------------------------------------------------


def test_the_input_folder_is_never_touched(old_mlp, tmp_path):
    before = _fingerprint(old_mlp)
    refresh_results(old_mlp, tmp_path / "new")
    assert _fingerprint(old_mlp) == before


def test_the_output_passes_validate_and_the_summary_is_rebuilt(old_mlp, tmp_path):
    out = tmp_path / "new"
    refresh_results(old_mlp, out)
    assert validate(out) == []
    summary = json.loads((out / MLP.id / "summary.json").read_text())
    tflops = [_trial(out, MLP.id, i)["achieved_tflops"] for i in range(2)]
    assert summary["timing"]["achieved_tflops"]["mean"] == pytest.approx(
        sum(tflops) / 2
    )
    assert summary["compute"]["infer_steps"] == 3
    assert summary["band"]["status"] == "not_comparable"  # 1 epoch, not 20
    with open(out / MLP.id / "trials.csv") as f:
        assert len(f.read().strip().splitlines()) == 3
    assert not (out / "zoo").exists()  # the zoo is referenced, not copied


def test_a_single_row_folder_works_too(old_mlp, tmp_path):
    out = tmp_path / "new"
    refresh_results(old_mlp / MLP.id, out)
    assert validate(out) == []
    ckpt = _trial(out, MLP.id)["checkpoint"]
    assert (out / ckpt).resolve() == (old_mlp / "zoo" / MLP.id / "trial0").resolve()


def test_a_failed_trial_is_copied_as_it_was(tmp_path):
    src = tmp_path / "old"
    _write_old_row(src, MLP, n=3, statuses=["ok", "ok", "failed"])
    out = tmp_path / "new"
    refresh_results(src, out)
    t = _trial(out, MLP.id, 2)
    assert t["status"] == "failed"
    assert t["compute"] == {}
    assert validate(out) == []


def test_writing_into_the_input_folder_is_refused(old_mlp):
    with pytest.raises(ValueError, match="new folder"):
        refresh_results(old_mlp, old_mlp)
    with pytest.raises(ValueError, match="new folder"):
        refresh_results(old_mlp, old_mlp / "inside")


def test_a_family_gets_its_comparison_rebuilt(tmp_path):
    src = tmp_path / "old"
    for algo in ("spc", "epc", "backprop"):
        _write_old_row(src, ROWS[f"mnist-mlp-{algo}"])
    out = tmp_path / "new"
    refresh_results(src, out)
    assert (out / "compare-mnist-mlp.json").exists()
    assert validate(out) == []


# --- command line ----------------------------------------------------------


def test_the_refresh_command_writes_a_valid_folder(old_mlp, tmp_path):
    out = tmp_path / "new"
    proc = subprocess.run(
        [sys.executable, "-m", "fabricpc.bench", "refresh", str(old_mlp)]
        + ["--out", str(out)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "TFLOP/s" in proc.stdout
    assert validate(out) == []


def test_the_refresh_command_needs_a_source_folder(tmp_path):
    proc = subprocess.run(
        [sys.executable, "-m", "fabricpc.bench", "refresh", "--out", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "refresh" in proc.stderr


# --- review fixes: guards and files that must not be lost -------------------


def test_refresh_without_out_is_refused_and_writes_nothing(
    old_mlp, tmp_path, monkeypatch
):
    # --out has a default ("results") for the run commands. For refresh it
    # must be typed, or a real run in ./results would be wiped.
    from fabricpc.bench.__main__ import main

    work = tmp_path / "work"
    (work / "results" / MLP.id).mkdir(parents=True)
    keep = work / "results" / MLP.id / "trial7.json"
    keep.write_text("{}")
    monkeypatch.chdir(work)
    assert main(["refresh", str(old_mlp)]) == 2
    assert keep.exists()
    assert sorted(p.name for p in (work / "results" / MLP.id).iterdir()) == [
        "trial7.json"
    ]


def test_an_out_folder_that_already_holds_results_is_refused(old_mlp, tmp_path):
    out = tmp_path / "new"
    (out / MLP.id).mkdir(parents=True)
    keep = out / MLP.id / "trial7.json"
    keep.write_text("{}")
    with pytest.raises(ValueError, match="already holds"):
        refresh_results(old_mlp, out)
    assert keep.exists()


def test_an_unknown_row_is_refused_before_anything_is_written(tmp_path):
    src = tmp_path / "old"
    _write_old_row(src, MLP)
    # A one-off row with no manifest, like the probe-* folders.
    odd = src / "probe-vgg-backprop"
    odd.mkdir()
    trial = json.loads((src / MLP.id / "trial0.json").read_text())
    trial["row_id"] = "probe-vgg-backprop"
    (odd / "trial0.json").write_text(json.dumps(trial))
    out = tmp_path / "new"
    with pytest.raises(ValueError, match="probe-vgg-backprop"):
        refresh_results(src, out)
    assert not out.exists() or not any(out.rglob("trial*.json"))


def test_a_row_whose_widths_changed_is_refused(tmp_path):
    # Same number of weighted edges, different parameter count: the widths
    # changed, so today's FLOP count would describe a model never run.
    src = tmp_path / "old"
    _write_old_row(src, MLP, n_params=12345)
    with pytest.raises(ValueError, match="parameters"):
        refresh_results(src, tmp_path / "new")


def test_a_pairwise_compare_is_rebuilt(tmp_path):
    # Like cifar10-vgg5-plain-layout: two rows of a family and a pairwise file.
    from fabricpc.bench.summary import compare_rows

    src = tmp_path / "old"
    for algo in ("epc", "backprop"):
        _write_old_row(src, ROWS[f"mnist-mlp-{algo}"])
    compare_rows(src, "mnist-mlp-epc", "mnist-mlp-backprop", metric="accuracy")
    out = tmp_path / "new"
    refresh_results(src, out)
    rebuilt = json.loads(
        (out / "compare-mnist-mlp-epc-vs-mnist-mlp-backprop.json").read_text()
    )
    assert rebuilt["n"] == 2
    assert "ci95_low" in rebuilt


def test_a_family_compare_on_another_metric_is_rebuilt(tmp_path):
    # Like hopfield-mac-check's compare-patterns64-hopfield-exact_recall_p20.
    from fabricpc.bench.compare import compare_family
    from fabricpc.bench.registry import COMPARISONS

    src = tmp_path / "old"
    for algo in ("spc", "epc", "backprop"):
        row_dir = _write_old_row(src, ROWS[f"mnist-mlp-{algo}"])
        for path in row_dir.glob("trial*.json"):
            t = json.loads(path.read_text())
            t["metrics"]["top5"] = t["metrics"]["accuracy"] + 0.05
            path.write_text(json.dumps(t))
    compare_family(src, COMPARISONS["mnist-mlp"], metric="top5")
    out = tmp_path / "new"
    refresh_results(src, out)
    rebuilt = json.loads((out / "compare-mnist-mlp-top5.json").read_text())
    assert rebuilt["metric"] == "top5"
    assert "ci95_low" in rebuilt["contrasts"][0]
    assert (out / "compare-mnist-mlp.json").exists()
    assert not (out / "NOTES.md").exists()


def test_a_compare_that_cannot_be_rebuilt_is_copied_and_noted(tmp_path):
    src = tmp_path / "old"
    _write_old_row(src, MLP)
    stale = {"row_a": "gone-row", "row_b": MLP.id, "metric": "accuracy", "n": 3}
    (src / "compare-gone-row-vs-mnist-mlp-spc.json").write_text(json.dumps(stale))
    out = tmp_path / "new"
    refresh_results(src, out)
    copied = out / "compare-gone-row-vs-mnist-mlp-spc.json"
    assert json.loads(copied.read_text()) == stale
    notes = (out / "NOTES.md").read_text()
    assert "compare-gone-row-vs-mnist-mlp-spc.json" in notes
    assert "copied unchanged" in notes


def test_notes_are_copied_with_a_warning_on_top(tmp_path):
    src = tmp_path / "old"
    _write_old_row(src, MLP)
    (src / "NOTES.md").write_text("3 vs 5 seeds; ePC here is default ePC.\n")
    out = tmp_path / "new"
    refresh_results(src, out)
    notes = (out / "NOTES.md").read_text()
    assert "3 vs 5 seeds; ePC here is default ePC." in notes
    assert notes.index("refresh") < notes.index("3 vs 5 seeds")


def test_a_changed_band_verdict_is_recorded(old_mlp, tmp_path):
    summary_path = old_mlp / MLP.id / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["band"] = {"status": "pass"}
    summary_path.write_text(json.dumps(summary))
    out = tmp_path / "new"
    refresh_results(old_mlp, out)
    manifest = json.loads((out / MLP.id / "manifest.json").read_text())
    record = manifest["refresh"][-1]
    assert "band" in record["changed_fields"]
    assert any("pass" in n and "not_comparable" in n for n in record["notes"])
