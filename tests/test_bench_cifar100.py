"""Tests for the CIFAR-100 VGG-5 rows (fabricpc.bench.cifar100)."""

import jax
import jax.numpy as jnp
import pytest

from fabricpc.bench.registry import ALGORITHMS, COMPARISONS, ROWS

FAMILY = "cifar100-vgg5"


def _n_params(params):
    return sum(p.size for p in jax.tree_util.tree_leaves(params))


def test_cifar100_vgg5_rows_are_registered():
    for algo in ALGORITHMS:
        row = ROWS[f"{FAMILY}-{algo}"]
        assert row.id == f"{FAMILY}-{algo}"
        assert row.dataset == "cifar100"
        assert row.model == "vgg5"
        assert row.algorithm == algo
        assert row.batch_size == 128  # pcx's batch size
        assert row.train_config["num_epochs"] == 50  # pcx's epoch count
        assert row.tier == 1


def test_cifar100_vgg5_is_a_family_we_compare():
    comparison = COMPARISONS[FAMILY]
    assert comparison.rows == tuple(f"{FAMILY}-{algo}" for algo in ALGORITHMS)
    assert comparison.metric == "accuracy"


def test_cifar100_rows_have_no_expected_score_until_our_first_full_run():
    # pcx's 67.19% is a nudged method we do not run, so it stays a note.
    for algo in ALGORITHMS:
        assert ROWS[f"{FAMILY}-{algo}"].reference is None


def test_cifar100_factory_builds_a_hundred_class_fused_vgg5(rng_key):
    for algo in ALGORITHMS:
        _, structure = ROWS[f"{FAMILY}-{algo}"].model_factory(rng_key)
        assert tuple(structure.nodes["input"].node_info.shape) == (32, 32, 3)
        assert tuple(structure.nodes["class"].node_info.shape) == (100,)
        assert len([n for n in structure.nodes if n.startswith("conv")]) == 4
        assert not [n for n in structure.nodes if n.startswith("pool")]
        assert type(structure.nodes["conv4"]).__name__ == "ConvPoolNode"


def test_cifar100_vgg5_only_differs_from_cifar10_in_the_output_layer(rng_key):
    # Same convs; the last layer reads 2x2x512 = 2048 features and now has
    # 90 more outputs, each with 2048 weights and a bias.
    p100, _ = ROWS[f"{FAMILY}-spc"].model_factory(rng_key)
    p10, _ = ROWS["cifar10-vgg5-spc"].model_factory(rng_key)
    assert _n_params(p100) - _n_params(p10) == 90 * (2 * 2 * 512 + 1)
    # VGG-5 with pcx's channels is a few million weights, not more.
    assert 4_000_000 < _n_params(p100) < 5_000_000


def test_cifar100_rows_use_pcx_pc_ce_settings(rng_key):
    # pcx VGG5_PCN_CE_cifar100.yaml: T = 12 settling steps, hard tanh.
    from fabricpc.bench import cifar100

    solver = cifar100._solver_for("spc")
    assert solver.config["infer_steps"] == 12
    assert type(cifar100._solver_for("epc")).__name__ == "EPCInference"
    for algo in ALGORITHMS:
        _, structure = ROWS[f"{FAMILY}-{algo}"].model_factory(rng_key)
        act = structure.nodes["conv1"].node_info.activation
        assert type(act).__name__ == "HardTanhActivation"


def test_cifar100_learning_rate_warms_up_then_decays_to_zero():
    from fabricpc.bench import cifar100

    lr = cifar100._lr_schedule(1000, cifar100._PC_PEAK_LR)
    assert float(lr(0)) == 0.0
    assert float(lr(100)) == pytest.approx(cifar100._PC_PEAK_LR)
    assert float(lr(1000)) == pytest.approx(0.0, abs=1e-12)


def test_cifar100_rows_let_adam_scale_up_tiny_pc_gradients():
    # Same reason as the CIFAR-10 VGG-5 rows: sPC's first-layer gradients
    # are 1e-8 to 1e-10, so Adam's epsilon is 1e-12.
    def last_step(opt, grad):
        params = {"w": jnp.zeros(4)}
        state = opt.init(params)
        for _ in range(200):
            updates, state = opt.update({"w": jnp.full(4, grad)}, state, params)
        return float(jnp.abs(updates["w"]).max())

    for algo in ALGORITHMS:
        make = ROWS[f"{FAMILY}-{algo}"].optimizer_factory
        assert last_step(make(1000), 1e-10) > 0.5 * last_step(make(1000), 1e-2)


def test_cifar100_pc_rows_have_the_safety_net():
    assert ROWS[f"{FAMILY}-spc"].rate_control.target == 1.8
    assert ROWS[f"{FAMILY}-epc"].rate_control.target == 1.0
    assert ROWS[f"{FAMILY}-backprop"].rate_control is None


def test_cifar100_loaders_augment_training_images_only(monkeypatch):
    # Swap in a fake loader so the test does not need the dataset on disk.
    from fabricpc.bench import cifar100
    from fabricpc.utils.data import dataloader

    made = []

    class FakeCifar100:
        def __init__(self, split, batch_size, shuffle=True, seed=None):
            made.append((split, batch_size, shuffle, seed))

    monkeypatch.setattr(dataloader, "Cifar100Loader", FakeCifar100)
    train, test = cifar100._loaders(128)(7)
    assert made == [("train", 128, True, 7), ("test", 128, False, None)]
    assert isinstance(train, dataloader.AugmentedImageLoader)
    assert isinstance(test, FakeCifar100)


def test_cifar100_backprop_row_uses_pcx_backprop_weight_settings():
    # pcx tuned backprop on its own for CIFAR-100 (VGG5_BP_CE_cifar100.yaml):
    # a much smaller weight decay than its PC config, and a slightly lower rate.
    from fabricpc.bench import cifar100

    assert cifar100._weight_settings("backprop") == (2.1e-4, 2.2e-5)
    assert cifar100._weight_settings("spc") == (2.2e-4, 7.6e-3)
    assert cifar100._weight_settings("epc") == (2.2e-4, 7.6e-3)


def test_cifar100_backprop_optimizer_decays_weights_less_than_pc():
    import jax.numpy as jnp

    def decay_step(algo):
        opt = ROWS[f"{FAMILY}-{algo}"].optimizer_factory(1000)
        params = {"w": jnp.ones(4)}
        state = opt.init(params)
        # No gradient, so after warmup the update is only the weight decay.
        for _ in range(200):
            updates, state = opt.update({"w": jnp.zeros(4)}, state, params)
        return float(jnp.abs(updates["w"]).max())

    assert decay_step("backprop") < 0.01 * decay_step("spc")
