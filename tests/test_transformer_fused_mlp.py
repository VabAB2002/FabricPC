"""The fused MLP block: LayerNorm -> ff1 -> GELU -> ff2 -> + residual in one node.

The plain v2 transformer gives the MLP two PC nodes (LnMlp1 with a wide
hidden latent, then Mlp2Residual). Each extra node is one more hop a PC
error has to travel, like the separate max-pool nodes that broke sPC on
VGG-5. fuse_mlp=True keeps the same weights and math in one node.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import optax
import pytest

from fabricpc.core.inference import InferenceSGDNormClip
from fabricpc.graph_initialization import initialize_params
from fabricpc.models import create_deep_transformer
from fabricpc.nodes import MlpResidualNode
from fabricpc.training import make_train_step

SEQ, EMBED, FF, VOCAB = 8, 16, 32, 10


def _build(depth=2, fuse_mlp=False):
    structure = create_deep_transformer(
        depth=depth,
        embed_dim=EMBED,
        num_heads=2,
        mlp_dim=FF,
        seq_len=SEQ,
        vocab_size=VOCAB,
        inference=InferenceSGDNormClip(eta_infer=0.1, infer_steps=3, max_norm=5.0),
        fuse_mlp=fuse_mlp,
    )
    return initialize_params(structure, jax.random.PRNGKey(0)), structure


def _count(params):
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


def test_fused_depth2_has_one_mlp_node_per_block():
    _, structure = _build(fuse_mlp=True)
    hidden = [n for n in structure.nodes if n not in ("input_ids", "logits")]
    assert hidden == ["embed", "L0_mha", "L0_mlp", "L1_mha", "L1_mlp"]
    assert isinstance(structure.nodes["L0_mlp"], MlpResidualNode)
    assert tuple(structure.nodes["L0_mlp"].node_info.shape) == (SEQ, EMBED)


def test_plain_layout_is_unchanged_by_default():
    _, structure = _build()
    assert "L0_mlp1" in structure.nodes and "L0_mlp2" in structure.nodes
    assert "L0_mlp" not in structure.nodes


def test_fused_layout_has_the_same_number_of_weights():
    fused, _ = _build(fuse_mlp=True)
    plain, _ = _build()
    assert _count(fused) == _count(plain)


def test_fused_skip_slot_is_a_residual_merge_like_mlp2():
    _, structure = _build(fuse_mlp=True)
    slots = structure.nodes["L0_mlp"].node_info.slots
    assert slots["skip"].is_skip_connection
    assert not slots["skip"].is_variance_scalable
    assert slots["in"].is_variance_scalable


def _edge_key(structure, node, slot):
    return next(
        k for k, e in structure.edges.items() if e.target == node and e.slot == slot
    )


def test_fused_block_computes_the_same_as_mlp1_then_mlp2():
    # Give the fused node the plain layout's weights; its prediction must
    # equal mlp2(residual=x, in=mlp1(x)).
    plain_params, plain = _build(depth=1)
    _, fused = _build(depth=1, fuse_mlp=True)
    x = jax.random.normal(jax.random.PRNGKey(3), (2, SEQ, EMBED))

    p1 = plain_params.nodes["L0_mlp1"]
    p2 = plain_params.nodes["L0_mlp2"]
    n1, n2 = plain.nodes["L0_mlp1"], plain.nodes["L0_mlp2"]
    h, _ = type(n1).predict(
        p1, {_edge_key(plain, "L0_mlp1", "in"): x}, None, n1.node_info
    )
    expected, _ = type(n2).predict(
        p2,
        {
            _edge_key(plain, "L0_mlp2", "in"): h,
            _edge_key(plain, "L0_mlp2", "residual"): x,
        },
        None,
        n2.node_info,
    )

    fnode = fused.nodes["L0_mlp"]
    fparams = type(p1)(
        weights={**p1.weights, **p2.weights}, biases={**p1.biases, **p2.biases}
    )
    got, _ = type(fnode).predict(
        fparams,
        {
            _edge_key(fused, "L0_mlp", "in"): x,
            _edge_key(fused, "L0_mlp", "skip"): x,
        },
        None,
        fnode.node_info,
    )
    assert got.shape == (2, SEQ, EMBED)
    assert jnp.allclose(got, expected, atol=1e-5)


@pytest.mark.parametrize("algorithm", ["pc", "backprop"])
def test_fused_transformer_takes_one_training_step(algorithm):
    params, structure = _build(fuse_mlp=True)
    step = make_train_step(structure, optax.adam(1e-3), algorithm=algorithm)
    opt_state = optax.adam(1e-3).init(params)
    tokens = jax.random.randint(jax.random.PRNGKey(1), (2, SEQ), 0, VOCAB)
    y = jax.nn.one_hot(jnp.roll(tokens, -1, axis=1), VOCAB)
    new_params, _, metrics, _ = step(
        params, opt_state, {"x": tokens, "y": y}, jax.random.PRNGKey(2)
    )
    assert jnp.isfinite(metrics["energy"])
    # The fused node's weights actually learn.
    before = params.nodes["L0_mlp"].weights["W_ff1"]
    after = new_params.nodes["L0_mlp"].weights["W_ff1"]
    assert not jnp.allclose(before, after)


def test_mupc_can_be_turned_off():
    # With muPC on, the 0.02-style init is scaled down a second time; turning
    # it off lets the init alone set the scale (sponsor issue #16).
    structure = create_deep_transformer(
        depth=1,
        embed_dim=EMBED,
        num_heads=2,
        mlp_dim=FF,
        seq_len=SEQ,
        vocab_size=VOCAB,
        inference=InferenceSGDNormClip(eta_infer=0.1, infer_steps=3, max_norm=5.0),
        fuse_mlp=True,
        use_mupc=False,
    )
    assert all(n.node_info.scaling_config is None for n in structure.nodes.values())
    _, with_mupc = _build(depth=1, fuse_mlp=True)
    assert with_mupc.nodes["L0_mlp"].node_info.scaling_config is not None
