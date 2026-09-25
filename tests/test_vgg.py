"""Tests for the VGG graph builder in fabricpc.models."""

import jax
import jax.numpy as jnp
import pytest

from fabricpc.core.inference import InferenceSGD
from fabricpc.graph_initialization import initialize_params
from fabricpc.models import create_vgg
from fabricpc.training import make_train_step
import optax


def build(depth, input_shape=(32, 32, 3), num_classes=10):
    structure = create_vgg(
        depth,
        input_shape=input_shape,
        num_classes=num_classes,
        inference=InferenceSGD(eta_infer=0.05, infer_steps=2),
    )
    params = initialize_params(structure, jax.random.PRNGKey(0))
    return params, structure


def test_vgg5_is_four_conv_blocks_and_one_classifier():
    # pcx's VGG-5: conv channels 128, 256, 512, 512, each followed by a
    # 2x2 max pool, then one linear layer. In FabricPC every conv and pool
    # is its own node, plus the input and the classifier.
    _, structure = build(5)
    names = list(structure.nodes)
    assert names[0] == "input"
    assert names[-1] == "class"
    convs = [n for n in names if n.startswith("conv")]
    pools = [n for n in names if n.startswith("pool")]
    assert len(convs) == 4
    assert len(pools) == 4
    assert len(structure.edges) == 9  # a chain of 10 nodes


def test_vgg5_shapes_halve_at_every_pool():
    _, structure = build(5)
    shape = lambda n: tuple(structure.nodes[n].node_info.shape)  # noqa: E731
    assert shape("conv1") == (32, 32, 128)
    assert shape("pool1") == (16, 16, 128)
    assert shape("conv2") == (16, 16, 256)
    assert shape("pool2") == (8, 8, 256)
    assert shape("conv3") == (8, 8, 512)
    assert shape("pool3") == (4, 4, 512)
    assert shape("conv4") == (4, 4, 512)
    assert shape("pool4") == (2, 2, 512)
    assert shape("class") == (10,)


def test_vgg5_parameter_count_matches_the_hand_count():
    params, _ = build(5)
    n = sum(p.size for p in jax.tree_util.tree_leaves(params))
    convs = (3 * 128 + 128 * 256 + 256 * 512 + 512 * 512) * 9  # 3x3 kernels
    conv_bias = 128 + 256 + 512 + 512
    classifier = 2 * 2 * 512 * 10 + 10
    assert n == convs + conv_bias + classifier


def test_vgg7_has_six_conv_blocks():
    # pcx's VGG-7: 128, 128, 256, 256, 512, 512.
    _, structure = build(7)
    assert len([n for n in structure.nodes if n.startswith("conv")]) == 6


def test_unsupported_depth_is_rejected():
    with pytest.raises(ValueError, match="depth"):
        build(6)


@pytest.mark.parametrize("algorithm", ["pc", "backprop"])
def test_vgg5_takes_one_training_step_under_both_algorithms(algorithm):
    params, structure = build(5)
    step = make_train_step(structure, optax.adam(1e-3), algorithm=algorithm)
    opt_state = optax.adam(1e-3).init(params)
    x = jnp.zeros((2, 32, 32, 3))
    y = jax.nn.one_hot(jnp.array([1, 7]), 10)
    new_params, _, metrics, _ = step(
        params, opt_state, {"x": x, "y": y}, jax.random.PRNGKey(1)
    )
    assert jnp.isfinite(metrics["energy"])


# --- fuse_pool: pcx's layout, one PC node per conv + activation + pool block ---


def build_fused(depth, **kw):
    structure = create_vgg(
        depth,
        input_shape=(32, 32, 3),
        num_classes=10,
        inference=InferenceSGD(eta_infer=0.05, infer_steps=2),
        fuse_pool=True,
        **kw,
    )
    return initialize_params(structure, jax.random.PRNGKey(0)), structure


def test_fused_vgg5_has_one_node_per_block_and_no_pool_nodes():
    _, structure = build_fused(5)
    names = list(structure.nodes)
    assert not [n for n in names if n.startswith("pool")]
    shapes = [tuple(structure.nodes[f"conv{i}"].node_info.shape) for i in range(1, 5)]
    assert shapes == [(16, 16, 128), (8, 8, 256), (4, 4, 512), (2, 2, 512)]


def test_fused_vgg5_has_the_same_weights_as_the_plain_layout():
    fused, _ = build_fused(5)
    plain, _ = build(5)

    def count(p):
        return sum(x.size for x in jax.tree_util.tree_leaves(p))

    assert count(fused) == count(plain)


def test_fused_block_computes_conv_activation_then_max_pool():
    # Same weights: fused conv1 must equal plain conv1 followed by a 2x2 max pool.
    from types import SimpleNamespace

    plain_params, plain = build(5)
    _, fused = build_fused(5)
    x = jax.random.normal(jax.random.PRNGKey(3), (2, 32, 32, 3))
    edge = next(iter(plain_params.nodes["conv1"].weights))
    inputs = {edge: x}

    node = plain.nodes["conv1"]
    unpooled, _ = type(node).predict(
        plain_params.nodes["conv1"],
        inputs,
        SimpleNamespace(z_latent=jnp.zeros((2, 32, 32, 128))),
        node.node_info,
    )
    expected = jax.lax.reduce_window(
        unpooled, -jnp.inf, jax.lax.max, (1, 2, 2, 1), (1, 2, 2, 1), "VALID"
    )

    fnode = fused.nodes["conv1"]
    got, _ = type(fnode).predict(
        plain_params.nodes["conv1"],
        inputs,
        SimpleNamespace(z_latent=jnp.zeros((2, 16, 16, 128))),
        fnode.node_info,
    )
    assert got.shape == (2, 16, 16, 128)
    assert jnp.allclose(got, expected, atol=1e-5)


def test_fused_vgg7_pools_only_after_the_last_conv_of_a_block():
    _, structure = build_fused(7)
    kinds = {n: type(node).__name__ for n, node in structure.nodes.items()}
    convs = [n for n in kinds if n.startswith("conv")]
    assert len(convs) == 6
    assert sorted(set(kinds[n] for n in convs)) == ["ConvNode", "ConvPoolNode"]


@pytest.mark.parametrize("algorithm", ["pc", "backprop"])
def test_fused_vgg5_takes_one_training_step(algorithm):
    params, structure = build_fused(5)
    step = make_train_step(structure, optax.adam(1e-3), algorithm=algorithm)
    opt_state = optax.adam(1e-3).init(params)
    x = jnp.zeros((2, 32, 32, 3))
    y = jax.nn.one_hot(jnp.array([1, 7]), 10)
    _, _, metrics, _ = step(params, opt_state, {"x": x, "y": y}, jax.random.PRNGKey(1))
    assert jnp.isfinite(metrics["energy"])
