"""Tests for the Tiny-ImageNet VGG-5 rows (fabricpc.bench.tinyimagenet)."""

import jax
import jax.numpy as jnp
import pytest

from fabricpc.bench.registry import ALGORITHMS, COMPARISONS, ROWS

FAMILY = "tinyimagenet-vgg5"


def _n_params(params):
    return sum(p.size for p in jax.tree_util.tree_leaves(params))


def test_tinyimagenet_vgg5_rows_are_registered():
    for algo in ALGORITHMS:
        row = ROWS[f"{FAMILY}-{algo}"]
        assert row.id == f"{FAMILY}-{algo}"
        assert row.dataset == "tinyimagenet"
        assert row.model == "vgg5"
        assert row.algorithm == algo
        assert row.batch_size == 128  # pcx's batch size
        assert row.train_config["num_epochs"] == 50  # pcx's epoch count
        assert row.tier == 1


def test_tinyimagenet_vgg5_is_a_family_we_compare():
    comparison = COMPARISONS[FAMILY]
    assert comparison.rows == tuple(f"{FAMILY}-{algo}" for algo in ALGORITHMS)
    assert comparison.metric == "accuracy"


def test_tinyimagenet_rows_have_no_expected_score_until_our_first_full_run():
    # pcx's 46.40% is negative nudging, a method we do not run.
    for algo in ALGORITHMS:
        assert ROWS[f"{FAMILY}-{algo}"].reference is None


def test_tinyimagenet_factory_builds_a_200_class_fused_vgg5_on_56px(rng_key):
    for algo in ALGORITHMS:
        _, structure = ROWS[f"{FAMILY}-{algo}"].model_factory(rng_key)
        assert tuple(structure.nodes["input"].node_info.shape) == (56, 56, 3)
        assert tuple(structure.nodes["class"].node_info.shape) == (200,)
        assert len([n for n in structure.nodes if n.startswith("conv")]) == 4
        assert not [n for n in structure.nodes if n.startswith("pool")]
        assert type(structure.nodes["conv4"]).__name__ == "ConvPoolNode"
        # pcx's vode shapes: 28, 14, 7, then 3 (7 // 2) before the linear layer.
        assert tuple(structure.nodes["conv4"].node_info.shape) == (3, 3, 512)


def test_tinyimagenet_vgg5_matches_pcx_parameter_count(rng_key):
    # Same convs as CIFAR (3->128->256->512->512, 3x3), then Linear(512*3*3, 200).
    p_tin, _ = ROWS[f"{FAMILY}-spc"].model_factory(rng_key)
    p_c10, _ = ROWS["cifar10-vgg5-spc"].model_factory(rng_key)
    head_tin = 3 * 3 * 512 * 200 + 200
    head_c10 = 2 * 2 * 512 * 10 + 10
    assert _n_params(p_tin) - _n_params(p_c10) == head_tin - head_c10


def test_tinyimagenet_forward_pass_runs_on_a_56px_batch(rng_key):
    from fabricpc.graph_initialization import initialize_graph_state

    params, structure = ROWS[f"{FAMILY}-backprop"].model_factory(rng_key)
    x = jnp.zeros((2, 56, 56, 3))
    y = jnp.zeros((2, 200)).at[:, 0].set(1.0)
    state = initialize_graph_state(
        structure, 2, rng_key, clamps={"input": x, "class": y}, params=params
    )
    assert state.nodes["class"].z_latent.shape == (2, 200)


def test_tinyimagenet_rows_use_pcx_pc_ce_settings(rng_key):
    # pcx VGG5_PCN_CE_tinyimagenet.yaml: T = 7 settling steps, hard tanh, and
    # state SGD at 0.01004 with momentum 0.05, about 0.0106 without momentum.
    from fabricpc.bench import tinyimagenet

    solver = tinyimagenet._solver_for("spc")
    assert solver.config["infer_steps"] == 7
    assert solver.config["eta_infer"] == pytest.approx(0.0106)
    assert type(tinyimagenet._solver_for("epc")).__name__ == "EPCInference"
    for algo in ALGORITHMS:
        _, structure = ROWS[f"{FAMILY}-{algo}"].model_factory(rng_key)
        act = structure.nodes["conv1"].node_info.activation
        assert type(act).__name__ == "HardTanhActivation"


def test_tinyimagenet_weight_settings_follow_pcx_per_method():
    # PC: VGG5_PCN_CE_tinyimagenet.yaml; backprop: VGG5_BP_CE_tinyimagenet.yaml.
    from fabricpc.bench import tinyimagenet

    assert tinyimagenet._weight_settings("spc") == (4.5e-5, 1.4e-3)
    assert tinyimagenet._weight_settings("epc") == (4.5e-5, 1.4e-3)
    assert tinyimagenet._weight_settings("backprop") == (8.7e-5, 2.1e-5)


def test_tinyimagenet_learning_rate_warms_up_then_decays_to_zero():
    from fabricpc.bench import tinyimagenet

    lr = tinyimagenet._lr_schedule(1000, 4.5e-5)
    assert float(lr(0)) == 0.0
    assert float(lr(100)) == pytest.approx(4.5e-5)
    assert float(lr(1000)) == pytest.approx(0.0, abs=1e-12)


def test_tinyimagenet_rows_let_adam_scale_up_tiny_pc_gradients():
    def last_step(opt, grad):
        params = {"w": jnp.zeros(4)}
        state = opt.init(params)
        for _ in range(200):
            updates, state = opt.update({"w": jnp.full(4, grad)}, state, params)
        return float(jnp.abs(updates["w"]).max())

    for algo in ALGORITHMS:
        make = ROWS[f"{FAMILY}-{algo}"].optimizer_factory
        assert last_step(make(1000), 1e-10) > 0.5 * last_step(make(1000), 1e-2)


def test_tinyimagenet_backprop_optimizer_decays_weights_less_than_pc():
    def decay_step(algo):
        opt = ROWS[f"{FAMILY}-{algo}"].optimizer_factory(1000)
        params = {"w": jnp.ones(4)}
        state = opt.init(params)
        for _ in range(200):
            updates, state = opt.update({"w": jnp.zeros(4)}, state, params)
        return float(jnp.abs(updates["w"]).max())

    assert decay_step("backprop") < 0.1 * decay_step("spc")


def test_tinyimagenet_pc_rows_have_the_safety_net():
    assert ROWS[f"{FAMILY}-spc"].rate_control.target == 1.8
    assert ROWS[f"{FAMILY}-epc"].rate_control.target == 1.0
    assert ROWS[f"{FAMILY}-backprop"].rate_control is None


def test_tinyimagenet_loaders_crop_like_pcx(monkeypatch):
    # Training: random flip plus a random 56x56 window, no padding.
    # Testing: the val split (the test split has no labels), centre 56x56.
    from fabricpc.bench import tinyimagenet
    from fabricpc.utils.data import dataloader

    made = []

    class FakeTinyImageNet:
        def __init__(self, split, batch_size, shuffle=True, seed=None, **kw):
            made.append((split, batch_size, shuffle, seed, kw.get("center_crop")))

    monkeypatch.setattr(dataloader, "TinyImageNetLoader", FakeTinyImageNet)
    train, test = tinyimagenet._loaders(128)(7)
    assert made == [("train", 128, True, 7, None), ("val", 128, False, None, 56)]
    assert isinstance(train, dataloader.AugmentedImageLoader)
    assert train.flip_prob == 0.5
    assert train.crop_pad == 0
    assert train.crop_size == 56
    assert isinstance(test, FakeTinyImageNet)
