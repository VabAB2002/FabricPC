"""Tests for how PC-like a row's runs were, in its summary and the report.

Each ePC trial records its regime at the start and end of training
(backprop-like, partially relaxed, near PC equilibrium), and each PC trial
records how closely its weight update follows backprop's. The summary and
the report gather those up so nobody has to open the trial files.
"""

import dataclasses
import json

from fabricpc.bench.report import build_report, render
from fabricpc.bench.runner import TrialResult
from fabricpc.bench.summary import regime_summary, summarize_row


def _regime(band, f, unstable=False):
    return {
        "band": band,
        "f_weighted": f,
        "f_max": f,
        "unstable": unstable,
        "eta_lambda_max": 0.01,
        "lambda_max": 10.0,
        "lambda_min": 0.5,
        "label": f"... {band}",
    }


def _diag(mean_cos):
    return {
        "backprop_alignment": {
            "layers": {},
            "weakest": "hidden1",
            "min_cos": mean_cos - 0.1,
            "mean_cos": mean_cos,
        },
        "stiffness": {"lambda_max": 2.0},
    }


def _trial(row_id, i, algorithm, *, final_band=None, f=None, cos=None):
    epc = None
    if final_band is not None:
        epc = {
            "init": _regime("backprop-like", 0.01),
            "final": _regime(final_band, f, unstable=final_band == "unstable"),
        }
    diag = None if cos is None else {"init": _diag(1.0), "final": _diag(cos)}
    return TrialResult(
        row_id=row_id,
        trial=i,
        seed=1000 * i,
        algorithm=algorithm,
        n_params=10,
        metrics={"accuracy": 0.9},
        step_time_ms=1.0,
        num_epochs=1.0,
        epc_regime=epc,
        diagnostics=diag,
    )


def _write(root, row_id, trials):
    (root / row_id).mkdir(parents=True, exist_ok=True)
    for t in trials:
        path = root / row_id / f"trial{t.trial}.json"
        path.write_text(json.dumps(dataclasses.asdict(t)))


def _epc_trials(row_id="mnist-mlp-epc"):
    return [
        _trial(row_id, 0, "epc", final_band="backprop-like", f=0.05, cos=0.99),
        _trial(row_id, 1, "epc", final_band="backprop-like", f=0.07, cos=0.98),
        _trial(row_id, 2, "epc", final_band="partially relaxed", f=0.30, cos=0.90),
    ]


def test_regime_summary_counts_the_bands_at_the_start_and_end():
    trials = [dataclasses.asdict(t) for t in _epc_trials()]

    s = regime_summary(trials)

    final = s["epc_regime"]["final"]
    assert final["bands"] == {"backprop-like": 2, "partially relaxed": 1}
    assert abs(final["f_weighted"]["mean"] - 0.14) < 1e-9
    assert final["unstable"] == 0
    assert s["epc_regime"]["init"]["bands"] == {"backprop-like": 3}


def test_regime_summary_averages_how_closely_pc_follows_backprop():
    trials = [dataclasses.asdict(t) for t in _epc_trials()]

    s = regime_summary(trials)

    final = s["backprop_alignment"]["final"]
    assert abs(final["mean_cos"]["mean"] - (0.99 + 0.98 + 0.90) / 3) < 1e-9
    assert final["mean_cos"]["n"] == 3


def test_a_backprop_row_has_nothing_to_report():
    trials = [
        dataclasses.asdict(_trial("mnist-mlp-backprop", i, "backprop"))
        for i in range(3)
    ]

    assert regime_summary(trials) == {"epc_regime": None, "backprop_alignment": None}


def test_the_summary_file_carries_the_regime(tmp_path):
    _write(tmp_path, "mnist-mlp-epc", _epc_trials())

    summarize_row(tmp_path, "mnist-mlp-epc")

    summary = json.loads((tmp_path / "mnist-mlp-epc" / "summary.json").read_text())
    assert summary["epc_regime"]["final"]["bands"]["backprop-like"] == 2
    assert summary["backprop_alignment"]["final"]["mean_cos"]["n"] == 3


def test_the_report_says_how_pc_like_each_row_was(tmp_path):
    family = tmp_path / "mnist-mlp"
    _write(family, "mnist-mlp-epc", _epc_trials())
    _write(
        family,
        "mnist-mlp-spc",
        [_trial("mnist-mlp-spc", i, "spc", cos=0.68) for i in range(3)],
    )

    page = render(build_report(tmp_path), "md")

    assert "PC-likeness" in page
    assert "backprop-like 2/3" in page
    assert "partially relaxed 1/3" in page
    assert "cos 0.68" in page
