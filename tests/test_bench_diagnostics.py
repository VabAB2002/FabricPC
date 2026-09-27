"""Tests for the per-trial diagnostics: how backprop-like a PC run's weight
updates are, layer by layer, and how stiff the network is."""

import jax
import pytest

from fabricpc.bench.diagnostics import backprop_alignment, stiffness
from fabricpc.bench.registry import ROWS
from fabricpc.core import EPCInference

BATCH = 8


def _batch(seed=1):
    kx, ky = jax.random.split(jax.random.PRNGKey(seed))
    x = jax.random.normal(kx, (BATCH, 784))
    y = jax.nn.one_hot(jax.random.randint(ky, (BATCH,), 0, 10), 10)
    return {"x": x, "y": y}


def _build(algo, inference=None):
    params, structure = ROWS[f"mnist-mlp-{algo}"].model_factory(jax.random.PRNGKey(0))
    if inference is not None:
        structure = structure._replace(
            config={**structure.config, "inference": inference}
        )
    return params, structure


def _scaled(params, factor):
    return params._replace(
        nodes={
            name: p._replace(weights={k: w * factor for k, w in p.weights.items()})
            for name, p in params.nodes.items()
        }
    )


class TestBackpropAlignment:
    def test_backprop_rows_have_nothing_to_compare(self):
        params, structure = _build("backprop")
        assert (
            backprop_alignment(
                params, structure, _batch(), jax.random.PRNGKey(2), "backprop"
            )
            is None
        )

    def test_one_small_epc_step_learns_exactly_like_backprop(self):
        # One ePC step from zero error is backprop's activation gradient, so
        # every layer's update points the same way as backprop's.
        params, structure = _build("epc", EPCInference(eta_infer=1e-4, infer_steps=1))
        out = backprop_alignment(
            params, structure, _batch(), jax.random.PRNGKey(2), "epc"
        )
        assert set(out["layers"]) == {"hidden1", "hidden2", "class"}
        for layer in out["layers"].values():
            assert layer["cos"] == pytest.approx(1.0, abs=1e-3)
        assert out["min_cos"] == pytest.approx(1.0, abs=1e-3)

    def test_reports_each_layer_and_the_weakest_one(self):
        params, structure = _build("spc")
        out = backprop_alignment(
            params, structure, _batch(), jax.random.PRNGKey(2), "spc"
        )
        cosines = {n: v["cos"] for n, v in out["layers"].items()}
        for layer in out["layers"].values():
            assert -1.0 <= layer["cos"] <= 1.0
            assert layer["size_ratio"] >= 0.0
        assert out["weakest"] == min(cosines, key=cosines.get)
        assert out["min_cos"] == cosines[out["weakest"]]
        assert out["mean_cos"] == pytest.approx(sum(cosines.values()) / 3)


class TestStiffness:
    def test_every_row_gets_the_same_comparable_number(self):
        # The error-coordinate lambda_max is a property of the weights, so
        # the same weights give the same number whatever trained them.
        values = []
        for algo in ("spc", "epc", "backprop"):
            params, structure = _build(algo)
            out = stiffness(params, structure, _batch(), jax.random.PRNGKey(2), algo)
            values.append(out["lambda_max"])
        assert values[0] == pytest.approx(values[1]) == pytest.approx(values[2])
        assert values[0] > 1.0

    def test_stiffer_weights_measure_larger(self):
        params, structure = _build("backprop")
        soft = stiffness(params, structure, _batch(), jax.random.PRNGKey(2), "backprop")
        stiff = stiffness(
            _scaled(params, 3.0), structure, _batch(), jax.random.PRNGKey(2), "backprop"
        )
        assert stiff["lambda_max"] > soft["lambda_max"]

    def test_spc_rows_also_record_their_settles_margin(self):
        params, structure = _build("spc")
        out = stiffness(params, structure, _batch(), jax.random.PRNGKey(2), "spc")
        eta = float(structure.config["inference"].config["eta_infer"])
        assert out["settle_stiffness"] > 0
        assert out["eta_times_settle_stiffness"] == pytest.approx(
            eta * out["settle_stiffness"]
        )
        backprop = stiffness(
            *_build("backprop"), _batch(), jax.random.PRNGKey(2), "backprop"
        )
        assert "settle_stiffness" not in backprop


def test_a_trial_records_both_at_the_start_and_the_end(tmp_path, rng_key):
    from tests.test_bench_runner import fake_mnist_loaders

    from fabricpc.bench.runner import run_trial

    kw = dict(
        loaders=fake_mnist_loaders(rng_key),
        num_epochs=1,
        warmup_steps=1,
        timed_steps=2,
    )
    spc = run_trial(ROWS["mnist-mlp-spc"], 0, tmp_path, **kw)
    assert set(spc.diagnostics) == {"init", "final"}
    for when in ("init", "final"):
        assert spc.diagnostics[when]["stiffness"]["lambda_max"] > 0
        assert "min_cos" in spc.diagnostics[when]["backprop_alignment"]

    bp = run_trial(ROWS["mnist-mlp-backprop"], 0, tmp_path, **kw)
    assert bp.diagnostics["final"]["backprop_alignment"] is None
    assert bp.diagnostics["final"]["stiffness"]["lambda_max"] > 0


def test_diagnostics_can_be_turned_off(tmp_path, rng_key):
    from tests.test_bench_runner import fake_mnist_loaders

    from fabricpc.bench.runner import run_trial

    result = run_trial(
        ROWS["mnist-mlp-spc"],
        0,
        tmp_path,
        loaders=fake_mnist_loaders(rng_key),
        num_epochs=1,
        warmup_steps=1,
        timed_steps=2,
        diagnostics=False,
    )
    assert result.diagnostics is None


def test_trials_csv_shows_the_end_of_run_diagnostics(tmp_path):
    import csv
    import json

    from fabricpc.bench.writer import write_trials_csv

    row_dir = tmp_path / "r"
    row_dir.mkdir()
    trial = {
        "trial": 0,
        "seed": 0,
        "status": "ok",
        "num_epochs": 1.0,
        "metrics": {"accuracy": 0.9},
        "diagnostics": {
            "init": {"stiffness": {"lambda_max": 6.5}, "backprop_alignment": None},
            "final": {
                "stiffness": {"lambda_max": 140.0},
                "backprop_alignment": {
                    "min_cos": 0.4,
                    "mean_cos": 0.8,
                    "weakest": "conv1",
                },
            },
        },
    }
    (row_dir / "trial0.json").write_text(json.dumps(trial))
    (line,) = list(csv.DictReader(open(write_trials_csv(tmp_path, "r"))))
    assert float(line["stiffness_init"]) == 6.5
    assert float(line["stiffness_final"]) == 140.0
    assert float(line["bp_cos_min"]) == 0.4
    assert line["bp_cos_weakest"] == "conv1"
