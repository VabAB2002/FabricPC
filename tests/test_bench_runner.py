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
    assert result.checkpoint == str(zoo / row.id / "trial0")
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
