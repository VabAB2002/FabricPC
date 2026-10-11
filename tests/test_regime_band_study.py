"""Tests for scripts/regime_band_study.py's summary table."""

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "regime_band_study.py"


def _load():
    spec = importlib.util.spec_from_file_location("regime_band_study", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(seed, eta_lambda_end, unstable, f_end):
    return {
        "eta": 0.3,
        "seed": seed,
        "accuracy": 0.94 + seed / 1e5,
        "eta_lambda_max_init": 0.43,
        "eta_lambda_max_final": eta_lambda_end,
        "unstable_final": unstable,
        "f_bar_init": 0.889,
        "f_bar_init_batch2": 0.890,
        "f_max_init": 0.94,
        "f_max_init_batch2": 0.94,
        "f_bar_final": f_end,
        "f_max_final": f_end,
    }


def test_summary_uses_medians_and_counts_unstable_seeds():
    study = _load()
    rows = [
        _row(0, 2.371, True, 9.0),
        _row(1000, 1.874, False, 1.2),
        _row(2000, 2.195, True, 50000.0),  # a diverging seed must not swamp it
    ]

    table = study.summary(rows)

    line = table.splitlines()[-1]
    assert "2 of 3" in line
    assert "2.37, 1.87, 2.19" in line
    assert "| 9 |" in line  # the median f, not the mean (about 16,670)
