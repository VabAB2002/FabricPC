"""Tests for changing a row's settings and tuning them with Optuna."""

import dataclasses
import json

import jax
import jax.numpy as jnp
import optax
import pytest

from fabricpc.bench import registry
from fabricpc.bench.isolate import child_command
from fabricpc.bench.registry import ROWS
from fabricpc.bench.runner import run_trial
from fabricpc.bench.settings import apply_settings, load_settings, write_settings
from fabricpc.bench.tune import Range, parse_space, tune_row
from tests.test_bench_cli_run import main
from tests.test_bench_runner import fake_mnist_loaders

_FAST = dict(num_epochs=1, warmup_steps=1, timed_steps=2)


def _solver(row):
    _, structure = row.model_factory(jax.random.PRNGKey(0))
    return structure.config["inference"]


# --- settings ---------------------------------------------------------------


def test_solver_settings_rebuild_the_rows_own_solver():
    row = apply_settings(ROWS["mnist-mlp-spc"], {"eta_infer": 0.01, "infer_steps": 7})

    solver = _solver(row)
    assert type(solver).__name__ == "InferenceSGD"
    assert solver.config["eta_infer"] == pytest.approx(0.01)
    assert solver.config["infer_steps"] == 7
    assert row.id == "mnist-mlp-spc"
    assert row.settings == {"eta_infer": 0.01, "infer_steps": 7.0}
    # The registry's row is left alone.
    assert _solver(ROWS["mnist-mlp-spc"]).config["infer_steps"] == 20


def test_lr_scale_is_the_same_as_scaling_the_learning_rate():
    # AdamW with weight decay, so the decay term is checked too.
    base = dataclasses.replace(
        ROWS["mnist-mlp-backprop"],
        optimizer_factory=lambda steps: optax.adamw(1e-3, weight_decay=0.1),
    )
    scaled = apply_settings(base, {"lr_scale": 2.5}).optimizer_factory(10)
    reference = optax.adamw(2.5e-3, weight_decay=0.1)

    params = {"w": jnp.arange(6.0).reshape(2, 3)}
    key = jax.random.PRNGKey(1)
    s1, s2 = scaled.init(params), reference.init(params)
    for i in range(3):
        grads = {"w": jax.random.normal(jax.random.fold_in(key, i), (2, 3))}
        u1, s1 = scaled.update(grads, s1, params)
        u2, s2 = reference.update(grads, s2, params)
        assert jnp.allclose(u1["w"], u2["w"], rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize(
    "row_id, settings, message",
    [
        ("mnist-mlp-backprop", {"eta_infer": 0.1}, "does not settle"),
        ("mnist-mlp-spc", {"momentum": 0.9}, "unknown setting"),
        ("mnist-mlp-spc", {"lr_scale": 0.0}, "above 0"),
    ],
)
def test_settings_that_do_not_apply_are_refused(row_id, settings, message):
    with pytest.raises(ValueError, match=message):
        apply_settings(ROWS[row_id], settings)


def test_settings_files_round_trip_and_a_folder_is_looked_up_by_row(tmp_path):
    write_settings(tmp_path / "mnist-mlp-spc.json", "mnist-mlp-spc", {"lr_scale": 2})

    assert load_settings(tmp_path / "mnist-mlp-spc.json", "mnist-mlp-spc") == {
        "lr_scale": 2.0
    }
    assert load_settings(tmp_path, "mnist-mlp-spc") == {"lr_scale": 2.0}
    assert load_settings(tmp_path, "mnist-mlp-epc") == {}  # no file: none
    with pytest.raises(ValueError, match="not mnist-mlp-epc"):
        load_settings(tmp_path / "mnist-mlp-spc.json", "mnist-mlp-epc")


def test_parse_space():
    space = parse_space(["eta_infer=1e-4:1e-1", "infer_steps=2:40:lin"])
    assert space["eta_infer"] == Range(1e-4, 1e-1, log=True)
    assert space["infer_steps"] == Range(2, 40, log=False, integer=True)
    with pytest.raises(ValueError):
        parse_space(["eta_infer=0.1:0.01"])


# --- a trial that only trains and scores --------------------------------------


def test_an_unmeasured_trial_trains_and_scores_but_times_nothing(tmp_path, rng_key):
    row = apply_settings(ROWS["mnist-mlp-epc"], {"lr_scale": 0.5})

    result = run_trial(
        row,
        0,
        tmp_path,
        loaders=fake_mnist_loaders(rng_key),
        num_epochs=1,
        measure=False,
        settled=True,
    )

    assert result.status == "ok", result.error
    assert "accuracy" in result.metrics
    assert result.step_time_ms == 0.0 and result.compute == {}
    assert result.epc_regime["init"] is None
    assert 0.0 <= result.epc_regime["final"]["f_weighted"] <= 2.0
    assert result.settings == {"lr_scale": 0.5}


# --- the command line ---------------------------------------------------------


def register_tiny_rows(monkeypatch, rng_key):
    """Tiny copies of the MNIST MLP rows, with a held-out split to tune on."""
    loaders = fake_mnist_loaders(rng_key)
    for algo in ("spc", "epc"):
        base = registry.ROWS[f"mnist-mlp-{algo}"]
        tiny = dataclasses.replace(
            base,
            id=f"tiny-mlp-{algo}",
            dataset="tiny",
            loader_factory=lambda seed: loaders,
            tune_loader_factory=lambda seed: loaders,
        )
        monkeypatch.setitem(registry.ROWS, tiny.id, tiny)


def _run_args(tmp_path, *extra):
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
        "--fast",
        *extra,
    ]


def test_a_run_with_set_records_and_uses_the_settings(tmp_path, monkeypatch, rng_key):
    register_tiny_rows(monkeypatch, rng_key)

    assert main(_run_args(tmp_path, "--set", "eta_infer=0.01")) == 0

    trial = json.loads((tmp_path / "tiny-mlp-spc" / "trial0.json").read_text())
    assert trial["solver"]["eta_infer"] == pytest.approx(0.01)
    assert trial["settings"] == {"eta_infer": 0.01}
    manifest = json.loads((tmp_path / "tiny-mlp-spc" / "manifest.json").read_text())
    assert manifest["row"]["settings"] == {"eta_infer": 0.01}


def test_resume_never_reuses_a_trial_run_with_other_settings(
    tmp_path, monkeypatch, rng_key
):
    register_tiny_rows(monkeypatch, rng_key)
    assert main(_run_args(tmp_path, "--set", "lr_scale=2")) == 0

    assert main(_run_args(tmp_path, "--resume")) == 0

    trial = json.loads((tmp_path / "tiny-mlp-spc" / "trial0.json").read_text())
    assert trial["settings"] is None  # trained again with the row's own


def test_a_child_trial_gets_the_rows_settings():
    row = apply_settings(ROWS["mnist-mlp-spc"], {"eta_infer": 0.0123})
    cmd = child_command(row.id, 0, "out", settings=row.settings)
    assert cmd[cmd.index("--set") + 1] == "eta_infer=0.0123"


# --- tuning -------------------------------------------------------------------


def test_tune_tries_the_rows_own_settings_first_and_saves_the_best(
    tmp_path, monkeypatch, rng_key
):
    register_tiny_rows(monkeypatch, rng_key)
    row = registry.ROWS["tiny-mlp-spc"]

    best = tune_row(row, tmp_path, budget=3, epochs=1)

    assert set(best["settings"]) == {"lr_scale", "eta_infer", "infer_steps"}
    assert best["completed"] == 3 and best["metric"] == "accuracy"
    assert "never the test set" in best["scored_on"]
    tries = json.loads((tmp_path / "tiny-mlp-spc" / "study.json").read_text())
    assert tries[0]["settings"] == {
        "lr_scale": 1.0,
        "eta_infer": 0.05,
        "infer_steps": 20,
    }
    assert best["value"] >= tries[0]["value"]
    saved = json.loads((tmp_path / "tiny-mlp-spc.json").read_text())
    assert saved["settings"] == best["settings"]
    # What tune wrote is what a run reads.
    assert load_settings(tmp_path, "tiny-mlp-spc") == pytest.approx(best["settings"])


def test_tune_refuses_a_row_with_no_held_out_split():
    row = ROWS["patterns64-hopfield-spc"]
    with pytest.raises(ValueError, match="no held-out split"):
        tune_row(row, "unused", budget=1)


def test_min_settled_only_applies_to_epc(tmp_path, monkeypatch, rng_key):
    register_tiny_rows(monkeypatch, rng_key)
    with pytest.raises(ValueError, match="ePC"):
        tune_row(registry.ROWS["tiny-mlp-spc"], tmp_path, budget=1, min_settled=0.5)


def test_min_settled_records_each_tries_settled_fraction(
    tmp_path, monkeypatch, rng_key
):
    register_tiny_rows(monkeypatch, rng_key)

    try:
        tune_row(
            registry.ROWS["tiny-mlp-epc"], tmp_path, budget=2, epochs=1, min_settled=0.0
        )
    except RuntimeError:
        pass  # every try may fail to settle on random data; the record is the point
    tries = json.loads((tmp_path / "tiny-mlp-epc" / "study.json").read_text())
    assert all("settled_fraction" in t or "error" in t for t in tries)


def test_the_tune_command_then_a_run_with_its_settings(tmp_path, monkeypatch, rng_key):
    register_tiny_rows(monkeypatch, rng_key)
    tuned = tmp_path / "tuned"

    code = main(
        ["tune", "tiny-mlp-spc", "--budget", "2", "--epochs", "1", "--out", str(tuned)]
    )

    assert code == 0
    best = json.loads((tuned / "tiny-mlp-spc.json").read_text())["settings"]
    assert main(_run_args(tmp_path / "run", "--settings", str(tuned))) == 0
    trial = json.loads((tmp_path / "run" / "tiny-mlp-spc" / "trial0.json").read_text())
    assert trial["settings"] == pytest.approx(best)
