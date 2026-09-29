"""Tests for ``python -m fabricpc.bench report``.

Each test builds a small, fake results folder in tmp_path that looks like
the real ones (row folders with summary.json and manifest.json, compare
files at the family level, a NOTES.md, and some junk), then checks what
the report page says about it.
"""

import json

from fabricpc.bench.__main__ import main
from fabricpc.bench.report import build_report, render


def _stats(mean, se, n, **extra):
    return {"mean": mean, "std": se * n**0.5, "se": se, "n": n, **extra}


def _write_row(
    family_dir,
    row_id,
    algorithm,
    *,
    acc,
    se,
    n=3,
    step_ms=10.0,
    band="no_reference",
    sha="abc1234def",
    device="cuda:0",
    metric_extra=None,
):
    row_dir = family_dir / row_id
    row_dir.mkdir(parents=True)
    summary = {
        "row_id": row_id,
        "algorithm": algorithm,
        "n_ok": n,
        "n_failed": 0,
        "seeds": [1000 * i for i in range(n)],
        "metrics": {"accuracy": _stats(acc, se, n, **(metric_extra or {}))},
        "timing": {"step_time_ms": _stats(step_ms, 0.1, n)},
        "band": {"status": band, "reason": f"{band} because of a test"},
        "schema_version": 2,
    }
    (row_dir / "summary.json").write_text(json.dumps(summary))
    manifest = {
        "schema_version": 2,
        "created_utc": "2026-09-25T20:55:53Z",
        "git_sha": sha,
        "devices": [device],
        "platform": {"system": "Linux", "machine": "x86_64"},
    }
    (row_dir / "manifest.json").write_text(json.dumps(manifest))
    return row_dir


def _family_compare(family_dir, name="cifar10-vgg5"):
    data = {
        "comparison": name,
        "metric": "accuracy",
        "seeds": [0, 1000, 2000],
        "n_trials": 3,
        "unpaired_seeds": {},
        "contrasts": [
            {
                "arm_a": f"{name}-spc",
                "arm_b": f"{name}-backprop",
                "mean_diff": -0.0229,
                "se_diff": 0.00095,
                "p_value": 0.0017,
                "significant_at_05": True,
                "n": 3,
            },
            {
                "arm_a": f"{name}-epc",
                "arm_b": f"{name}-backprop",
                "mean_diff": 0.0006,
                "se_diff": 0.0003,
                "p_value": 0.1978,
                "significant_at_05": False,
                "n": 3,
            },
        ],
    }
    (family_dir / f"compare-{name}.json").write_text(json.dumps(data))


def _good_family(root):
    fam = root / "cifar10-vgg5-fused"
    _write_row(fam, "cifar10-vgg5-backprop", "backprop", acc=0.8673, se=0.0006)
    _write_row(fam, "cifar10-vgg5-spc", "spc", acc=0.8444, se=0.0008, step_ms=25.0)
    _write_row(fam, "cifar10-vgg5-epc", "epc", acc=0.8680, se=0.0008, step_ms=20.0)
    _family_compare(fam)
    (fam / "NOTES.md").write_text("# Notes\n\n- tflops are too low by 4x here.\n")
    return fam


def test_family_table_has_each_row_with_mean_se_and_slowdown(tmp_path):
    _good_family(tmp_path)
    text = render(build_report(tmp_path), "md")
    assert "cifar10-vgg5-fused" in text
    for algo in ("backprop", "spc", "epc"):
        assert f"| {algo} |" in text
    assert "0.8673 ± 0.0006" in text
    assert "25.00" in text  # step time of spc in ms
    assert "2.50x" in text  # spc is 25 ms vs backprop's 10 ms
    assert "1.00x" in text  # backprop against itself
    assert "no reference" in text  # band verdict in plain words


def test_paired_comparisons_are_said_in_plain_words(tmp_path):
    _good_family(tmp_path)
    text = render(build_report(tmp_path), "md")
    assert "spc - backprop" in text
    assert "-0.0229" in text
    assert "0.0017" in text
    # Three seeds and three contrasts: p < 0.05 is not proof, so the page
    # says what the test says and no more.
    assert "yes (p < 0.05, uncorrected)" in text
    assert "no, could be seed noise" in text
    assert "a real difference" not in text


def test_provenance_and_notes_are_included(tmp_path):
    _good_family(tmp_path)
    text = render(build_report(tmp_path), "md")
    assert "abc1234" in text  # short git sha
    assert "cuda:0" in text
    assert "2026-09-25" in text
    assert "tflops are too low by 4x" in text  # NOTES.md is quoted
    assert "NOTES.md" in text  # and linked


def test_confidence_interval_is_shown_when_the_summary_has_one(tmp_path):
    fam = tmp_path / "fam"
    _write_row(
        fam,
        "mnist-mlp-backprop",
        "backprop",
        acc=0.98,
        se=0.001,
        metric_extra={"ci95": [0.977, 0.983]},
    )
    _write_row(
        fam,
        "mnist-mlp-spc",
        "spc",
        acc=0.97,
        se=0.001,
        metric_extra={"ci_low": 0.967, "ci_high": 0.973},
    )
    text = render(build_report(tmp_path), "md")
    assert "[0.9770, 0.9830]" in text
    assert "[0.9670, 0.9730]" in text


def test_old_single_pair_compare_file_is_read_too(tmp_path):
    fam = tmp_path / "plain"
    _write_row(fam, "cifar10-vgg5-backprop", "backprop", acc=0.86, se=0.001)
    _write_row(fam, "cifar10-vgg5-epc", "epc", acc=0.85, se=0.001)
    old = {
        "row_a": "cifar10-vgg5-epc",
        "row_b": "cifar10-vgg5-backprop",
        "metric": "accuracy",
        "n": 5,
        "mean_diff": -0.00352,
        "p_value": 0.1455,
        "significant_at_05": False,
    }
    (fam / "compare-cifar10-vgg5-epc-vs-cifar10-vgg5-backprop.json").write_text(
        json.dumps(old)
    )
    text = render(build_report(tmp_path), "md")
    assert "epc - backprop" in text
    assert "-0.0035" in text
    assert "p-value" in text and "0.145" in text


def test_row_without_summary_falls_back_to_trials_csv(tmp_path):
    fam = tmp_path / "untuned"
    row = fam / "cifar10-vgg5-spc"
    row.mkdir(parents=True)
    (row / "trials.csv").write_text(
        "trial,seed,status,num_epochs,accuracy,step_time_ms,error\n"
        "0,0,ok,50.0,0.5,30.0,\n"
    )
    text = render(build_report(tmp_path), "md")
    assert "| spc |" in text
    assert "0.5000" in text
    assert "no summary.json" in text  # says where the numbers came from


def test_broken_and_foreign_folders_are_skipped_with_a_note(tmp_path):
    _good_family(tmp_path)
    logs = tmp_path / "diagnosis-2026-09-27"
    logs.mkdir()
    (logs / "tf.log").write_text("some log\n")
    (logs / "probe.py").write_text("print(1)\n")
    bad = tmp_path / "broken" / "mnist-mlp-spc"
    bad.mkdir(parents=True)
    (bad / "summary.json").write_text("{not json")
    (tmp_path / "broken" / "compare-x.json").write_text("[1, 2")
    text = render(build_report(tmp_path), "md")
    assert "diagnosis-2026-09-27" in text  # listed as not benchmark results
    assert "could not read" in text
    assert "cifar10-vgg5-fused" in text  # the good family still shows


def test_zoo_folders_are_not_treated_as_rows(tmp_path):
    fam = _good_family(tmp_path)
    (fam / "zoo" / "cifar10-vgg5-spc" / "trial0").mkdir(parents=True)
    text = render(build_report(tmp_path), "md")
    assert "| zoo" not in text


def test_nested_family_folders_are_found(tmp_path):
    _good_family(tmp_path / "l4-2026-09-27")
    text = render(build_report(tmp_path), "md")
    assert "l4-2026-09-27/cifar10-vgg5-fused" in text


def test_missing_root_does_not_crash(tmp_path):
    text = render(build_report(tmp_path / "nope"), "md")
    assert "not found" in text


def test_html_format_is_a_whole_page(tmp_path):
    _good_family(tmp_path)
    page = render(build_report(tmp_path), "html")
    assert page.startswith("<!doctype html>")
    assert "<table>" in page
    assert "cifar10-vgg5-fused" in page
    assert "&lt;" not in page.split("<body>")[0]  # head is not escaped junk


def test_cli_writes_the_report_file(tmp_path):
    root = tmp_path / "results"
    _good_family(root)
    out = tmp_path / "report.md"
    assert main(["report", str(root), "--out", str(out)]) == 0
    text = out.read_text()
    assert text.startswith("# ")
    assert "cifar10-vgg5-fused" in text


def test_cli_html_format_from_flag(tmp_path, capsys):
    root = tmp_path / "results"
    _good_family(root)
    out = tmp_path / "page.out"
    assert main(["report", str(root), "--out", str(out), "--format", "html"]) == 0
    assert out.read_text().startswith("<!doctype html>")


def test_cli_without_out_prints_the_report(tmp_path, capsys):
    root = tmp_path / "results"
    _good_family(root)
    assert main(["report", str(root)]) == 0
    assert "cifar10-vgg5-fused" in capsys.readouterr().out


def test_cli_needs_a_results_root(capsys):
    assert main(["report"]) == 2


def test_folder_with_two_families_compares_each_to_its_own_backprop(tmp_path):
    fam = tmp_path / "l4-mlp"
    _write_row(fam, "mnist-mlp-backprop", "backprop", acc=0.98, se=0.001, step_ms=1.0)
    _write_row(fam, "mnist-mlp-spc", "spc", acc=0.98, se=0.001, step_ms=3.0)
    _write_row(fam, "cifar10-vgg5-backprop", "backprop", acc=0.8, se=0.01, step_ms=20)
    _write_row(fam, "cifar10-vgg5-epc-follow", "epc", acc=0.8, se=0.01, step_ms=40)
    _family_compare(fam, name="mnist-mlp")
    text = render(build_report(tmp_path), "md")
    assert "3.00x" in text  # mnist spc vs mnist backprop
    assert "2.00x" in text  # the vgg follow-up vs vgg backprop, not vs mnist
    assert "60.00x" not in text and "40.00x" not in text
    assert "spc - backprop (mnist-mlp)" in text  # which family the pair is from


def test_big_differences_stay_readable(tmp_path):
    fam = tmp_path / "tf"
    _write_row(fam, "x-backprop", "backprop", acc=0.5, se=0.01)
    data = {
        "comparison": "x",
        "metric": "perplexity",
        "contrasts": [
            {"arm_a": "x-epc", "arm_b": "x-backprop", "mean_diff": 699215.29361},
        ],
    }
    (fam / "compare-x.json").write_text(json.dumps(data))
    text = render(build_report(tmp_path), "md")
    assert "+699,215.3" in text


def test_loose_files_next_to_families_are_mentioned(tmp_path):
    job = tmp_path / "l4-2026-09-27"
    _good_family(job)
    (job / "vgg-epc-follow-result.json").write_text("{}")
    text = render(build_report(tmp_path), "md")
    assert "vgg-epc-follow-result.json" in text


def test_the_interval_summary_py_writes_is_shown(tmp_path):
    # summary._stats writes ci95_low / ci95_high; use exactly that shape.
    from fabricpc.bench.summary import _stats as summary_stats

    stats = summary_stats([0.97, 0.98, 0.99])
    fam = tmp_path / "fam"
    extra = {k: stats[k] for k in ("ci95_low", "ci95_high")}
    _write_row(
        fam, "mnist-mlp-backprop", "backprop", acc=0.98, se=0.0058, metric_extra=extra
    )
    text = render(build_report(tmp_path), "md")
    assert f"[{stats['ci95_low']:.4f}, {stats['ci95_high']:.4f}]" in text


def test_a_one_seed_row_with_no_interval_does_not_drop_the_family(tmp_path):
    # With one good trial summary._stats writes ci95_low = ci95_high = None.
    from fabricpc.bench.summary import _stats as summary_stats

    stats = summary_stats([0.97])
    assert stats["ci95_low"] is None
    fam = tmp_path / "fam"
    extra = {k: stats[k] for k in ("ci95_low", "ci95_high")}
    _write_row(
        fam, "mnist-mlp-backprop", "backprop", acc=0.97, se=0.0, n=1, metric_extra=extra
    )
    text = render(build_report(tmp_path), "md")
    assert "could not read it" not in text
    assert "| backprop |" in text


def test_paired_comparisons_show_the_interval_of_the_difference(tmp_path):
    fam = _good_family(tmp_path)
    path = fam / "compare-cifar10-vgg5.json"
    data = json.loads(path.read_text())
    data["contrasts"][0].update(ci95_low=-0.0270, ci95_high=-0.0188)
    path.write_text(json.dumps(data))
    text = render(build_report(tmp_path), "md")
    assert "95% CI of diff" in text
    assert "[-0.0270, -0.0188]" in text
