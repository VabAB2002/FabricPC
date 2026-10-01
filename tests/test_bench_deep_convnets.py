"""Tests for the deeper CIFAR-10 conv rows (fabricpc.bench.rows_deep_convnets):
cifar10-vgg7, cifar10-vgg9 and cifar10-resnet18lean."""

import jax
import optax
import pytest

from fabricpc.bench import registry
from fabricpc.bench.registry import ALGORITHMS, COMPARISONS, ROWS

VGG_FAMILIES = {"cifar10-vgg7": 7, "cifar10-vgg9": 9}


def _n_params(params):
    return sum(p.size for p in jax.tree_util.tree_leaves(params))


@pytest.mark.parametrize("family", sorted(VGG_FAMILIES))
def test_deep_vgg_rows_follow_the_vgg5_recipe(family):
    for algo in ALGORITHMS:
        row = ROWS[f"{family}-{algo}"]
        vgg5 = ROWS[f"cifar10-vgg5-{algo}"]
        assert row.dataset == "cifar10"
        assert row.model == family.split("-")[1]
        assert row.algorithm == algo
        assert row.batch_size == vgg5.batch_size == 128
        assert row.train_config == vgg5.train_config
        assert row.rate_control == vgg5.rate_control
        assert row.reference is None
        assert row.tier == 2


@pytest.mark.parametrize("family", sorted(VGG_FAMILIES))
def test_deep_vgg_uses_one_pc_node_per_conv(family, rng_key):
    depth = VGG_FAMILIES[family]
    for algo in ALGORITHMS:
        _, structure = ROWS[f"{family}-{algo}"].model_factory(rng_key)
        convs = [n for n in structure.nodes if n.startswith("conv")]
        assert len(convs) == depth - 1
        assert not [n for n in structure.nodes if n.startswith("pool")]
        assert tuple(structure.nodes["class"].node_info.shape) == (10,)


@pytest.mark.parametrize("family", sorted(VGG_FAMILIES))
def test_deep_vgg_solver_matches_vgg5(family, rng_key):
    _, deep = ROWS[f"{family}-spc"].model_factory(rng_key)
    _, vgg5 = ROWS["cifar10-vgg5-spc"].model_factory(rng_key)
    a, b = deep.config["inference"].config, vgg5.config["inference"].config
    assert a["infer_steps"] == b["infer_steps"]
    assert a["eta_infer"] == b["eta_infer"]
    _, epc = ROWS[f"{family}-epc"].model_factory(rng_key)
    assert type(epc.config["inference"]).__name__ == "EPCInference"


@pytest.mark.parametrize("family", sorted(VGG_FAMILIES))
def test_deep_vgg_optimizer_uses_the_deep_adam_eps(family):
    # Same schedule as VGG-5: zero at the start, peak after a tenth.
    opt = ROWS[f"{family}-spc"].optimizer_factory(1000)
    ref = ROWS["cifar10-vgg5-spc"].optimizer_factory(1000)
    params = {"w": jax.numpy.ones(3)}
    grads = {"w": jax.numpy.full(3, 1e-10)}
    s, r = opt.init(params), ref.init(params)
    for _ in range(150):
        u, s = opt.update(grads, s, params)
        v, r = ref.update(grads, r, params)
    assert jax.numpy.allclose(u["w"], v["w"])
    assert float(abs(u["w"][0])) > 1e-5  # the 1e-12 eps lets tiny grads move


def test_resnet18lean_rows_mirror_the_plain_rows(rng_key):
    for algo in ALGORITHMS:
        row = ROWS[f"cifar10-resnet18lean-{algo}"]
        plain = ROWS[f"cifar10-resnet18-{algo}"]
        assert row.model == "resnet18lean"
        assert row.dataset == "cifar10"
        assert row.batch_size == plain.batch_size
        assert row.train_config == plain.train_config
        assert row.rate_control == plain.rate_control
        assert row.reference is None
        assert row.tier == 2


def test_resnet18lean_builds_the_lean_graph_with_the_plain_solver(rng_key):
    for algo in ALGORITHMS:
        lean_params, lean = ROWS[f"cifar10-resnet18lean-{algo}"].model_factory(rng_key)
        plain_params, plain = ROWS[f"cifar10-resnet18-{algo}"].model_factory(rng_key)
        assert _n_params(lean_params) == _n_params(plain_params)
        assert "s1b1_res" in lean.nodes and "avgpool" not in lean.nodes
        assert "s1b1_skip_sum" in plain.nodes  # the registered rows stay plain
        assert lean.config["inference"].config == plain.config["inference"].config


def test_resnet18lean_optimizer_matches_plain():
    lean = ROWS["cifar10-resnet18lean-spc"].optimizer_factory(100)
    plain = ROWS["cifar10-resnet18-spc"].optimizer_factory(100)
    assert isinstance(lean, optax.GradientTransformation)
    params = {"w": jax.numpy.ones(2)}
    grads = {"w": jax.numpy.full(2, 0.5)}
    u, _ = lean.update(grads, lean.init(params), params)
    v, _ = plain.update(grads, plain.init(params), params)
    assert jax.numpy.allclose(u["w"], v["w"])


@pytest.mark.parametrize(
    "family", ["cifar10-vgg7", "cifar10-vgg9", "cifar10-resnet18lean"]
)
def test_new_families_are_compared(family):
    comparison = COMPARISONS[family]
    assert comparison.rows == tuple(f"{family}-{a}" for a in ALGORITHMS)
    assert comparison.metric == "accuracy"


def test_existing_rows_are_untouched():
    for family in ("cifar10-vgg5", "cifar10-resnet18", "cifar100-vgg5"):
        for algo in ALGORITHMS:
            assert f"{family}-{algo}" in ROWS
    assert registry.ROWS["cifar10-resnet18-spc"].model == "resnet18"
