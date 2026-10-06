"""Tests for the pcx-style summary."""

import json

import pytest

from experiments.nudging.summarize import summarize


def _write(tmp_path, name, bests, lasts, target=88.83):
    d = tmp_path / name
    d.mkdir()
    for i, (b, last) in enumerate(zip(bests, lasts)):
        (d / f"seed{i}.json").write_text(
            json.dumps({"best_acc": b, "last_acc": last, "config": {"target": target}})
        )


def test_seven_seeds_give_pcx_top_five(tmp_path):
    bests = [0.80, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86]
    _write(tmp_path, "c10-nn", bests, [b - 0.01 for b in bests])
    s = summarize(tmp_path, "c10-nn")
    assert s["n"] == 7
    assert s["top5_mean"] == pytest.approx(0.84)
    assert s["best_mean"] == pytest.approx(0.83)
    assert s["last_mean"] == pytest.approx(0.82)
    assert s["target"] == 88.83
    assert s["gap_top5"] == pytest.approx(84.0 - 88.83)


def test_fewer_than_seven_seeds_have_no_top_five(tmp_path):
    _write(tmp_path, "c10-se", [0.80, 0.82, 0.84], [0.79, 0.81, 0.83], target=87.98)
    s = summarize(tmp_path, "c10-se")
    assert s["n"] == 3
    assert s["top5_mean"] is None and s["gap_top5"] is None
    assert s["best_std"] == pytest.approx(0.016329931, rel=1e-6)  # numpy default ddof=0
