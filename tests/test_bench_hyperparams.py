"""Rows store their tuned settings as defaults; --lr and the other setting
flags change them for one run (Nov 17 sponsor feedback)."""

import json

import jax
import jax.numpy as jnp
import pytest

from fabricpc.bench.__main__ import _other_settings_in_folder, main
from fabricpc.bench.cli import explicit
from fabricpc.bench.definitions.registry import ROWS, with_overrides
from fabricpc.bench.run.isolate import child_command
from fabricpc.bench.run.manifest import describe_row
from fabricpc.bench.stats.band import check_band
from fabricpc.bench.stats.summary import RowSummary

REBUILDABLE = [r for r in ROWS.values() if r.rebuild is not None]


def test_the_mlp_rows_store_their_settings():
    assert {r.id for r in REBUILDABLE} >= {"mnist-mlp-spc", "fashionmnist-mlp-epc"}
    assert ROWS["mnist-mlp-spc"].hyperparams == {
        "learning_rate": 0.001,
        "weight_decay": 0.1,
        "batch_size": 200,
        "eta_infer": 0.05,
        "infer_steps": 20,
    }
    assert "eta_infer" not in ROWS["mnist-mlp-backprop"].hyperparams


@pytest.mark.parametrize("row", REBUILDABLE, ids=lambda r: r.id)
def test_rebuilding_a_row_from_its_own_defaults_changes_nothing(row):
    again = row.rebuild(dict(row.hyperparams))
    assert describe_row(again) == describe_row(row)
    # Same optimizer: one update on the same gradient is identical.
    params = {"w": jnp.arange(4.0)}
    grads = {"w": jnp.array([0.5, -1.0, 2.0, 0.1])}
    a, b = row.optimizer_factory(100), again.optimizer_factory(100)
    ua, _ = a.update(grads, a.init(params), params)
    ub, _ = b.update(grads, b.init(params), params)
    assert jnp.array_equal(ua["w"], ub["w"])
    # Same graph, including the settling solver's settings.
    _, s1 = row.model_factory(jax.random.PRNGKey(0))
    _, s2 = again.model_factory(jax.random.PRNGKey(0))
    assert dict(s1.config["inference"].config) == dict(s2.config["inference"].config)


def test_an_override_changes_the_optimizer_and_the_solver():
    row = with_overrides(
        ROWS["mnist-mlp-spc"], {"learning_rate": 0.01, "infer_steps": 5}
    )
    assert row.overrides == {"learning_rate": 0.01, "infer_steps": 5}
    assert row.hyperparams["learning_rate"] == 0.01
    _, structure = row.model_factory(jax.random.PRNGKey(0))
    assert structure.config["inference"].config["infer_steps"] == 5
    assert describe_row(row)["overrides"] == row.overrides


def test_bad_overrides_say_what_the_row_has():
    with pytest.raises(ValueError, match="its settings: learning_rate"):
        with_overrides(ROWS["mnist-mlp-epc"], {"eta_infer": 0.1})
    with pytest.raises(ValueError, match="cannot change its settings"):
        with_overrides(ROWS["cifar10-vgg5-spc"], {"learning_rate": 0.1})


def test_each_trial_process_gets_the_overrides():
    row = with_overrides(ROWS["mnist-mlp-spc"], {"learning_rate": 0.01})
    cmd = child_command(row.id, 0, "out", overrides=dict(row.overrides))
    assert cmd[cmd.index("--lr") + 1] == "0.01"


def test_changed_settings_get_no_pass_or_fail():
    row = with_overrides(ROWS["mnist-mlp-spc"], {"learning_rate": 0.01})
    summary = RowSummary(
        row_id=row.id,
        algorithm="spc",
        n_ok=5,
        n_failed=0,
        seeds=[0, 1, 2, 3, 4],
        metrics={"accuracy": {"mean": 0.98, "se": 0.001, "n": 5}},
        num_epochs=20.0,
    )
    verdict = check_band(row, summary)
    assert verdict.status == "not_comparable"
    assert "settings changed" in verdict.reason


def _fake_run(folder, overrides):
    folder.mkdir(parents=True)
    (folder / "trial0.json").write_text("{}")
    manifest = {"row": {"id": "mnist-mlp-spc", "overrides": overrides}}
    (folder / "manifest.json").write_text(json.dumps(manifest))


def test_a_run_with_other_settings_never_wipes_the_folder(tmp_path):
    default, changed = ROWS["mnist-mlp-spc"], with_overrides(
        ROWS["mnist-mlp-spc"], {"learning_rate": 0.01}
    )
    _fake_run(tmp_path / "a" / "mnist-mlp-spc", {})
    assert _other_settings_in_folder(tmp_path / "a", changed)
    assert _other_settings_in_folder(tmp_path / "a", default) is None
    _fake_run(tmp_path / "b" / "mnist-mlp-spc", {"learning_rate": 0.01})
    assert _other_settings_in_folder(tmp_path / "b", default)
    assert _other_settings_in_folder(tmp_path / "b", changed) is None


def test_run_passes_the_setting_flags_on():
    seen = []
    explicit.run_cli(
        ["--dataset", "mnist", "--architecture", "mlp", "--method", "spc"]
        + ["--lr", "0.01", "--batch-size", "100"],
        run_main=lambda argv: seen.append(list(argv)) or 0,
    )
    assert seen == [["mnist-mlp-spc", "--lr", "0.01", "--batch-size", "100"]]


def test_setting_flags_are_refused_for_other_commands(capsys):
    with pytest.raises(SystemExit) as e:
        main(["smoke", "--lr", "0.1"])
    assert e.value.code == 2
    assert "nothing was run" in capsys.readouterr().err
