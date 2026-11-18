"""Edge cases found in the 11/18 review: stray trial file names, refresh of
rows that are no longer in the registry, and listing rows without tensorflow."""

import json
import subprocess
import sys

from fabricpc.bench.results.refresh import refresh_results
from fabricpc.bench.results.writer import validate, write_trials_csv
from fabricpc.bench.run.manifest import SCHEMA_VERSION, trial_files, trial_number
from fabricpc.bench.stats.summary import load_trials, summarize_row


def make_row(root, row_id, n=3):
    row = root / row_id
    row.mkdir(parents=True)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "row": {"id": row_id, "batch_size": 200},
        "seeds": list(range(n)),
    }
    (row / "manifest.json").write_text(json.dumps(manifest))
    for i in range(n):
        trial = {
            "schema_version": SCHEMA_VERSION,
            "row_id": row_id,
            "trial": i,
            "seed": i,
            "status": "ok",
            "algorithm": "spc",
            "metrics": {"accuracy": 0.9 + 0.01 * i},
            "step_time_ms": 1.0,
            "compile_time_s": 1.0,
            "train_time_s": 1.0,
            "achieved_tflops": 0.1,
            "n_params": 10,
            "num_epochs": 20,
        }
        (row / f"trial{i}.json").write_text(json.dumps(trial))
    return row


def test_trial_number_only_reads_real_trial_names():
    assert trial_number("trial12.json") == 12
    assert trial_number("trial0 copy.json") is None
    assert trial_number("trial0-old.json") is None
    assert trial_number("trials.json") is None


def test_a_stray_trial_file_does_not_crash_reading_or_validating(tmp_path):
    row = make_row(tmp_path, "oldsweep-mlp-spc")
    (row / "trial0 copy.json").write_text("{}")

    assert [p.name for p in trial_files(row)] == [
        "trial0.json",
        "trial1.json",
        "trial2.json",
    ]
    assert len(load_trials(tmp_path, "oldsweep-mlp-spc")) == 3
    problems = validate(tmp_path)
    assert any("trial0 copy.json" in p for p in problems)


def test_refresh_keeps_the_summary_of_a_row_not_in_the_registry(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    make_row(src, "oldsweep-mlp-spc")
    summarize_row(src, "oldsweep-mlp-spc")
    write_trials_csv(src, "oldsweep-mlp-spc")

    refresh_results(src, out)

    assert (out / "oldsweep-mlp-spc" / "summary.json").exists()


def test_the_registry_loads_without_tensorflow_datasets():
    code = (
        "import sys; sys.modules['tensorflow_datasets'] = None\n"
        "from fabricpc.bench.definitions.registry import ROWS\n"
        "assert 'mnist-mlp-spc' in ROWS\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
