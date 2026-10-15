"""Every trial records the solver it trained with, and ePC rows report their
settled fraction under that name (sponsor issue #59 and the 2026-10-13
mentor meeting)."""

import csv
import dataclasses
import json

import pytest

from fabricpc.bench.compare import compare_family
from fabricpc.bench.registry import COMPARISONS, ROWS
from fabricpc.bench.runner import TrialResult, run_trial
from fabricpc.bench.summary import summarize_row
from fabricpc.bench.writer import write_trials_csv
from tests.test_bench_runner import fake_mnist_loaders

_KW = dict(num_epochs=1, warmup_steps=1, timed_steps=2)


@pytest.mark.parametrize(
    "algorithm, solver_class, eta, steps",
    [("spc", "InferenceSGD", 0.05, 20), ("epc", "EPCInference", 1e-3, 5)],
)
def test_pc_trials_record_the_solver_they_trained_with(
    tmp_path, rng_key, algorithm, solver_class, eta, steps
):
    row = ROWS[f"mnist-mlp-{algorithm}"]

    result = run_trial(row, 0, tmp_path, loaders=fake_mnist_loaders(rng_key), **_KW)

    assert result.status == "ok", result.error
    assert result.solver["class"] == solver_class
    assert result.solver["eta_infer"] == pytest.approx(eta)
    assert result.solver["infer_steps"] == steps
    saved = json.loads((tmp_path / row.id / "trial0.json").read_text())
    assert saved["solver"] == result.solver


def test_backprop_trials_record_no_solver(tmp_path, rng_key):
    # Backprop's graph carries a solver object only so it can be built; it
    # never settles, so recording one would mislead.
    row = ROWS["mnist-mlp-backprop"]
    result = run_trial(row, 0, tmp_path, loaders=fake_mnist_loaders(rng_key), **_KW)
    assert result.status == "ok", result.error
    assert result.solver is None


def _write(out_dir, row_id, algorithm, fractions, accuracies=None):
    (out_dir / row_id).mkdir(parents=True, exist_ok=True)
    accuracies = accuracies or [0.9 + 0.01 * i for i in range(len(fractions))]
    for i, (f, acc) in enumerate(zip(fractions, accuracies)):
        regime = None
        if f is not None:
            regime = {
                "init": {"band": "backprop-like", "f_weighted": f / 2},
                "final": {"band": "partially relaxed", "f_weighted": f},
            }
        r = TrialResult(
            row_id=row_id,
            trial=i,
            seed=i * 1000,
            algorithm=algorithm,
            n_params=10,
            metrics={"accuracy": acc},
            step_time_ms=10.0,
            num_epochs=1.0,
            epc_regime=regime,
            solver={"class": "EPCInference", "eta_infer": 0.001, "infer_steps": 5},
        )
        (out_dir / row_id / f"trial{i}.json").write_text(
            json.dumps(dataclasses.asdict(r))
        )


def test_the_summary_reports_the_settled_fraction_at_the_end(tmp_path):
    _write(tmp_path, "mnist-mlp-epc", "epc", [0.2, 0.4, 0.6])

    s = summarize_row(tmp_path, "mnist-mlp-epc")

    assert s.settled_fraction["mean"] == pytest.approx(0.4)
    assert s.settled_fraction["n"] == 3
    on_disk = json.loads((tmp_path / "mnist-mlp-epc" / "summary.json").read_text())
    assert on_disk["settled_fraction"]["mean"] == pytest.approx(0.4)
    assert on_disk["solver"]["class"] == "EPCInference"


def test_rows_that_do_not_settle_errors_have_no_settled_fraction(tmp_path):
    _write(tmp_path, "mnist-mlp-backprop", "backprop", [None, None])
    assert summarize_row(tmp_path, "mnist-mlp-backprop").settled_fraction is None


def test_the_csv_has_a_settled_fraction_column(tmp_path):
    _write(tmp_path, "mnist-mlp-epc", "epc", [0.25, 0.5])

    path = write_trials_csv(tmp_path, "mnist-mlp-epc")

    rows = list(csv.DictReader(path.open()))
    assert [float(r["settled_fraction"]) for r in rows] == [0.25, 0.5]
    assert rows[0]["solver"] == "EPCInference"
    assert float(rows[0]["eta_infer"]) == pytest.approx(0.001)
    assert int(rows[0]["infer_steps"]) == 5


def test_a_family_comparison_says_how_settled_the_epc_row_was(tmp_path):
    for algo in ("spc", "backprop"):
        _write(tmp_path, f"mnist-mlp-{algo}", algo, [None, None, None])
    _write(tmp_path, "mnist-mlp-epc", "epc", [0.05, 0.07, 0.09])

    out = compare_family(tmp_path, COMPARISONS["mnist-mlp"], metric="accuracy")

    assert out["settled_fraction"]["mnist-mlp-epc"] == pytest.approx(0.07)
    assert out["settled_fraction"]["mnist-mlp-spc"] is None
    assert out["settled_fraction"]["mnist-mlp-backprop"] is None


def test_the_printed_line_names_the_settled_fraction(tmp_path, capsys):
    from fabricpc.bench.__main__ import _print_pc_likeness

    _write(tmp_path, "mnist-mlp-epc", "epc", [0.2, 0.4, 0.6])

    _print_pc_likeness("mnist-mlp-epc", summarize_row(tmp_path, "mnist-mlp-epc"))

    out = capsys.readouterr().out
    assert "settled fraction 0.40" in out
