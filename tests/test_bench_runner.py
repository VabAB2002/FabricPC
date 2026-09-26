"""Tests for running one benchmark trial."""

import json

import jax
import pytest

from fabricpc.bench.registry import ROWS
from fabricpc.bench.runner import run_trial
from tests.conftest import ListLoader


def fake_mnist_loaders(key, n_batches=3, batch_size=8):
    """Tiny random stand-in for MNIST so the runner test stays fast."""
    batches = []
    for i in range(n_batches):
        kx, ky = jax.random.split(jax.random.fold_in(key, i))
        x = jax.random.normal(kx, (batch_size, 784))
        y = jax.nn.one_hot(jax.random.randint(ky, (batch_size,), 0, 10), 10)
        batches.append((x, y))
    return ListLoader(batches), ListLoader(batches[:1])


@pytest.mark.parametrize("algorithm", ["spc", "epc", "backprop"])
def test_run_trial_trains_measures_and_writes_a_json_file(tmp_path, rng_key, algorithm):
    row = ROWS[f"mnist-mlp-{algorithm}"]
    loaders = fake_mnist_loaders(rng_key)

    result = run_trial(
        row,
        trial=0,
        out_dir=tmp_path,
        loaders=loaders,
        num_epochs=1,
        warmup_steps=1,
        timed_steps=3,
    )

    # What every trial must report.
    assert result.row_id == row.id
    assert result.trial == 0
    assert result.algorithm == algorithm
    assert result.n_params == 218_058  # 784*256+256 + 256*64+64 + 64*10+10
    assert 0.0 <= result.metrics["accuracy"] <= 1.0
    assert result.compile_time_s > 0
    assert result.step_time_ms > 0
    assert result.train_time_s > 0
    assert result.status == "ok"

    # And it must land on disk as JSON with the same numbers.
    path = tmp_path / row.id / "trial0.json"
    assert path.exists()
    on_disk = json.loads(path.read_text())
    assert on_disk["row_id"] == row.id
    assert on_disk["metrics"]["accuracy"] == result.metrics["accuracy"]
    # Both memory numbers are always present (None on a CPU that cannot say).
    assert "memory_bytes" in on_disk
    assert "peak_memory_bytes" in on_disk
    assert on_disk["step_memory"]["total_bytes"] > 0


def test_each_trial_gets_its_own_seed(tmp_path, rng_key):
    # Trial i uses seed_offset + i * 1000, the same rule the experiment
    # framework uses, so results line up across arms.
    row = ROWS["mnist-mlp-spc"]
    loaders = fake_mnist_loaders(rng_key)
    r0 = run_trial(
        row, 0, tmp_path, loaders=loaders, num_epochs=1, warmup_steps=1, timed_steps=2
    )
    r1 = run_trial(
        row, 1, tmp_path, loaders=loaders, num_epochs=1, warmup_steps=1, timed_steps=2
    )
    assert r0.seed == 0
    assert r1.seed == 1000


def test_trial_result_includes_the_compute_count(tmp_path, rng_key):
    # Every trial records how many matmuls one update costs, and how fast
    # the hardware actually got through them.
    row = ROWS["mnist-mlp-spc"]
    loaders = fake_mnist_loaders(rng_key)
    result = run_trial(
        row, 0, tmp_path, loaders=loaders, num_epochs=1, warmup_steps=1, timed_steps=2
    )
    assert result.compute["weighted_edges"] == 3
    assert result.compute["matmuls_per_update"] == 2 * 3 * 20 + 3
    assert result.compute["pc_to_backprop_ratio"] > 1
    assert result.achieved_tflops > 0

    on_disk = json.loads((tmp_path / row.id / "trial0.json").read_text())
    assert on_disk["compute"]["matmuls_per_update"] == 123


def test_trial_can_save_its_trained_params_to_the_zoo(tmp_path, rng_key):
    # The model zoo: trained weights that load back and give the same score.
    from fabricpc.bench.zoo import load_params
    from fabricpc.training import evaluate

    row = ROWS["mnist-mlp-spc"]
    loaders = fake_mnist_loaders(rng_key)
    zoo = tmp_path / "zoo"
    result = run_trial(
        row,
        0,
        tmp_path,
        loaders=loaders,
        num_epochs=1,
        warmup_steps=1,
        timed_steps=2,
        zoo_dir=zoo,
    )
    # Recorded relative to the results folder, so it stays right when the
    # folder is copied elsewhere (e.g. from a cloud job to a laptop).
    assert result.checkpoint == f"zoo/{row.id}/trial0"
    assert (zoo / row.id / "trial0.json").exists()

    params, structure = load_params(zoo, row, trial=0)
    _, test_loader = loaders
    again = evaluate(
        params, structure, test_loader, {"num_epochs": 1}, rng_key, algorithm="pc"
    )
    assert float(again["accuracy"]) == result.metrics["accuracy"]


def test_no_zoo_dir_means_no_checkpoint(tmp_path, rng_key):
    row = ROWS["mnist-mlp-spc"]
    result = run_trial(
        row,
        0,
        tmp_path,
        loaders=fake_mnist_loaders(rng_key),
        num_epochs=1,
        warmup_steps=1,
        timed_steps=2,
    )
    assert result.checkpoint is None


def test_epc_trials_record_their_regime_at_the_start_and_the_end(tmp_path, rng_key):
    loaders = fake_mnist_loaders(rng_key)
    kw = dict(loaders=loaders, num_epochs=1, warmup_steps=1, timed_steps=2)

    epc = run_trial(ROWS["mnist-mlp-epc"], 0, tmp_path, **kw)
    assert set(epc.epc_regime) == {"init", "final"}
    assert "band" in epc.epc_regime["final"]

    spc = run_trial(ROWS["mnist-mlp-spc"], 0, tmp_path, **kw)
    assert spc.epc_regime is None


def test_trial_records_a_learning_curve_one_point_per_epoch(tmp_path, rng_key):
    loaders = fake_mnist_loaders(rng_key)
    result = run_trial(
        ROWS["mnist-mlp-spc"],
        0,
        tmp_path,
        loaders=loaders,
        num_epochs=2,
        warmup_steps=1,
        timed_steps=2,
        curve_batches=1,
    )
    assert [p["epoch"] for p in result.curve] == [1, 2]
    for point in result.curve:
        assert 0.0 <= point["accuracy"] <= 1.0
        assert "train_energy" in point


def test_curve_batches_zero_turns_the_curve_off(tmp_path, rng_key):
    result = run_trial(
        ROWS["mnist-mlp-backprop"],
        0,
        tmp_path,
        loaders=fake_mnist_loaders(rng_key),
        num_epochs=1,
        warmup_steps=1,
        timed_steps=2,
        curve_batches=0,
    )
    assert result.curve is None


def test_pc_trials_also_score_the_trained_model_with_a_plain_forward_pass(
    tmp_path, rng_key
):
    # PC evaluation lets the free output keep settling, which can make it
    # over-confident. So PC rows also report the one-pass score, under
    # forward_<name>, computed from the same trained weights.
    from fabricpc.bench.zoo import load_params
    from fabricpc.training import evaluate

    row = ROWS["mnist-mlp-spc"]
    loaders = fake_mnist_loaders(rng_key)
    zoo = tmp_path / "zoo"
    result = run_trial(
        row,
        0,
        tmp_path,
        loaders=loaders,
        num_epochs=1,
        warmup_steps=1,
        timed_steps=2,
        zoo_dir=zoo,
    )

    params, structure = load_params(zoo, row, 0)
    _, test_loader = loaders
    config = {**row.train_config, "num_epochs": 1.0}
    master_key = jax.random.PRNGKey(result.seed)
    _, _, eval_key = jax.random.split(master_key, 3)
    forward = evaluate(
        params, structure, test_loader, config, eval_key, algorithm="backprop"
    )

    assert "accuracy" in result.metrics and "energy" in result.metrics
    for name, value in forward.items():
        assert result.metrics[f"forward_{name}"] == pytest.approx(float(value))
    # A forward pass has no settling, so no PC energy to report.
    assert "forward_energy" not in result.metrics


def test_backprop_trials_report_forward_scores_equal_to_their_own(tmp_path, rng_key):
    # Backprop's evaluation already is the forward pass, so its forward_*
    # numbers are just a copy. Having them on every row lets a family be
    # compared on forward_accuracy across all three methods.
    result = run_trial(
        ROWS["mnist-mlp-backprop"],
        0,
        tmp_path,
        loaders=fake_mnist_loaders(rng_key),
        num_epochs=1,
        warmup_steps=1,
        timed_steps=2,
    )
    plain = {k: v for k, v in result.metrics.items() if not k.startswith("forward_")}
    assert plain
    for name, value in plain.items():
        assert result.metrics[f"forward_{name}"] == value
