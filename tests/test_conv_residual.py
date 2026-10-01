"""Tests for ConvResidualNode and the lean ResNet-18 layout.

The lean layout folds each residual block's add into the block's second
conv node, and the final average pool into the last block, so the graph has
no PC latents without parameters. Backprop must compute exactly the same
function as the plain layout; only the PC latents change.
"""

import jax
import jax.numpy as jnp
import optax
import pytest

from fabricpc.core.activations import GeluActivation, ReLUActivation
from fabricpc.core.inference import InferenceSGD
from fabricpc.core.initializers import MuPCInitializer, XavierInitializer
from fabricpc.core.mupc import MuPCConfig
from fabricpc.core.topology import Edge
from fabricpc.core.types import GraphParams, NodeParams
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import FeedforwardStateInit, initialize_params
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.models import build_resnet18
from fabricpc.nodes import ConvNode, ConvResidualNode, IdentityNode, Linear
from fabricpc.training import make_train_step


def _n_params(params):
    return sum(p.size for p in jax.tree_util.tree_leaves(params))


def _latents(structure):
    """Nodes that hold a PC latent: everything but the clamped input."""
    return [n for n, node in structure.nodes.items() if node.node_info.in_degree > 0]


def _param_free_latents(structure, params):
    return [
        n
        for n in _latents(structure)
        if not params.nodes[n].weights and not params.nodes[n].biases
    ]


def _resnet(lean, scaling=True, activation=None):
    return build_resnet18(
        weight_init=MuPCInitializer() if scaling else XavierInitializer(),
        inference=InferenceSGD(eta_infer=0.05, infer_steps=2),
        scaling=MuPCConfig(include_output=False) if scaling else None,
        output_weight_init=XavierInitializer(),
        activation=activation or GeluActivation(),
        lean=lean,
    )


def _forward_output(structure, params, x, key):
    state = initialize_graph_state(
        structure,
        x.shape[0],
        key,
        clamps={"input": x},
        state_init=FeedforwardStateInit(),
        params=params,
    )
    return state.nodes["output"].z_mu


def _lean_params_from_plain(plain_params, lean_structure):
    """Copy the plain layout's weights into the lean layout.

    Every lean node with weights has a plain twin: ``<block>_res`` is the
    plain ``<block>_conv_b``, and every other node keeps its name. Each of
    these nodes has exactly one weight array, so it is moved across under
    the lean graph's edge key.
    """
    nodes = {}
    for name, node in lean_structure.nodes.items():
        twin = name[: -len("_res")] + "_conv_b" if name.endswith("_res") else name
        plain = plain_params.nodes[twin]
        in_edges = [e for e in node.node_info.in_edges if e.endswith(":in")]
        assert len(plain.weights) == len(in_edges)
        weights = {in_edges[0]: next(iter(plain.weights.values()))} if in_edges else {}
        nodes[name] = NodeParams(weights=weights, biases=dict(plain.biases))
    return GraphParams(nodes=nodes)


# ---------------------------------------------------------------------------
# The node on its own
# ---------------------------------------------------------------------------


def _small_block(global_pool=False, **scales):
    """input(8,8,4) -> conv_a(8,8,4) -> res(conv + add of input) -> out(3)."""
    pixels = IdentityNode(shape=(8, 8, 4), name="pixels")
    conv_a = ConvNode(shape=(8, 8, 4), name="conv_a", kernel_size=(3, 3))
    res = ConvResidualNode(
        shape=(4,) if global_pool else (8, 8, 4),
        name="res",
        kernel_size=(3, 3),
        global_pool=global_pool,
        map_shape=(8, 8, 4) if global_pool else None,
        activation=ReLUActivation(),
        **scales,
    )
    out = Linear(shape=(3,), name="out", flatten_input=True)
    structure = graph(
        nodes=[pixels, conv_a, res, out],
        edges=[
            Edge(source=pixels, target=conv_a.slot("in")),
            Edge(source=conv_a, target=res.slot("in")),
            Edge(source=pixels, target=res.slot("skip")),
            Edge(source=res, target=out.slot("in")),
        ],
        task_map=TaskMap(x=pixels, y=out),
        inference=InferenceSGD(eta_infer=0.05, infer_steps=3),
    )
    return structure


def _res_mu(structure, params, x, key):
    state = initialize_graph_state(
        structure,
        x.shape[0],
        key,
        clamps={"pixels": x},
        state_init=FeedforwardStateInit(),
        params=params,
    )
    return state.nodes["conv_a"].z_mu, state.nodes["res"].z_mu


class TestConvResidualNode:
    def test_only_the_in_edge_gets_a_kernel(self, rng_key):
        structure = _small_block()
        params = initialize_params(structure, rng_key)
        res = params.nodes["res"]
        assert list(res.weights) == ["conv_a->res:in"]
        assert res.weights["conv_a->res:in"].shape == (3, 3, 4, 4)
        assert res.biases["b"].shape == (1, 1, 1, 4)

    def test_output_is_conv_activation_plus_skip(self, rng_key):
        structure = _small_block()
        params = initialize_params(structure, rng_key)
        x = jax.random.normal(rng_key, (2, 8, 8, 4))
        h, z = _res_mu(structure, params, x, rng_key)
        w = params.nodes["res"].weights["conv_a->res:in"]
        conv = jax.lax.conv_general_dilated(
            h, w, (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC")
        )
        expected = jax.nn.relu(conv + params.nodes["res"].biases["b"]) + x
        assert jnp.allclose(z, expected, atol=1e-5)

    def test_scales_go_where_mupc_would_put_them(self, rng_key):
        structure = _small_block(in_scale=0.5, branch_scale=0.25)
        params = initialize_params(structure, rng_key)
        x = jax.random.normal(rng_key, (2, 8, 8, 4))
        h, z = _res_mu(structure, params, x, rng_key)
        w = params.nodes["res"].weights["conv_a->res:in"]
        conv = jax.lax.conv_general_dilated(
            0.5 * h, w, (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC")
        )
        expected = 0.25 * jax.nn.relu(conv + params.nodes["res"].biases["b"]) + x
        assert jnp.allclose(z, expected, atol=1e-5)

    def test_global_pool_averages_the_block_output(self, rng_key):
        plain = _small_block()
        pooled = _small_block(global_pool=True, pool_scale=2.0)
        params = initialize_params(plain, rng_key)
        pooled_params = initialize_params(pooled, rng_key)
        # Same block weights, different head (the head's input size differs).
        pooled_params = GraphParams(
            nodes={**pooled_params.nodes, "res": params.nodes["res"]}
        )
        x = jax.random.normal(rng_key, (2, 8, 8, 4))
        _, z_map = _res_mu(plain, params, x, rng_key)
        _, z_vec = _res_mu(pooled, pooled_params, x, rng_key)
        assert z_vec.shape == (2, 4)
        assert jnp.allclose(z_vec, 2.0 * z_map.mean(axis=(1, 2)), atol=1e-5)

    def test_skip_edge_is_required(self):
        pixels = IdentityNode(shape=(8, 8, 4), name="pixels")
        res = ConvResidualNode(shape=(8, 8, 4), name="res", kernel_size=(3, 3))
        out = Linear(shape=(3,), name="out", flatten_input=True)
        with pytest.raises(ValueError):
            graph(
                nodes=[pixels, res, out],
                edges=[
                    Edge(source=pixels, target=res.slot("in")),
                    Edge(source=res, target=out.slot("in")),
                ],
                task_map=TaskMap(x=pixels, y=out),
                inference=InferenceSGD(eta_infer=0.05, infer_steps=3),
            )

    def test_global_pool_needs_the_map_shape(self):
        with pytest.raises(ValueError, match="map_shape"):
            ConvResidualNode(
                shape=(4,), name="res", kernel_size=(3, 3), global_pool=True
            )

    def test_a_small_block_trains(self, rng_key):
        structure = _small_block()
        params = initialize_params(structure, rng_key)
        optimizer = optax.adam(1e-3)
        step = make_train_step(structure, optimizer)
        batch = {
            "x": jax.random.normal(rng_key, (4, 8, 8, 4)),
            "y": jax.nn.one_hot(jnp.array([0, 1, 2, 0]), 3),
        }
        p, _, metrics, _ = step(params, optimizer.init(params), batch, rng_key)
        assert jnp.isfinite(metrics["energy"])
        before = params.nodes["res"].weights["conv_a->res:in"]
        assert not jnp.allclose(before, p.nodes["res"].weights["conv_a->res:in"])


# ---------------------------------------------------------------------------
# The lean ResNet-18
# ---------------------------------------------------------------------------


class TestLeanResNet18:
    def test_the_plain_layout_has_nine_param_free_latents(self, rng_key):
        # The pattern the lean layout removes: 8 SkipConnection + 1 AvgPool.
        structure = _resnet(lean=False)
        params = initialize_params(structure, rng_key)
        assert len(_latents(structure)) == 30  # 29 hidden + the output
        assert len(_param_free_latents(structure, params)) == 9

    def test_lean_layout_has_no_param_free_latents(self, rng_key):
        structure = _resnet(lean=True)
        params = initialize_params(structure, rng_key)
        assert _param_free_latents(structure, params) == []
        assert len(_latents(structure)) == 21  # 30 minus the 9 folded nodes

    def test_same_parameter_count_as_the_plain_layout(self, rng_key):
        plain = initialize_params(_resnet(lean=False), rng_key)
        lean = initialize_params(_resnet(lean=True), rng_key)
        assert _n_params(lean) == _n_params(plain) == 2_795_210

    def test_last_block_holds_the_pooled_features(self):
        structure = _resnet(lean=True)
        assert tuple(structure.nodes["s4b2_res"].node_info.shape) == (256,)
        assert tuple(structure.nodes["output"].node_info.shape) == (10,)

    @pytest.mark.parametrize("scaling", [True, False], ids=["mupc", "no-scaling"])
    def test_backprop_forward_matches_the_plain_layout(self, rng_key, scaling):
        # With the same weights both layouts compute the same function, so
        # the backprop rows of the two families train the same network.
        plain_structure = _resnet(lean=False, scaling=scaling)
        lean_structure = _resnet(lean=True, scaling=scaling)
        plain_params = initialize_params(plain_structure, rng_key)
        lean_params = _lean_params_from_plain(plain_params, lean_structure)
        x = jax.random.normal(jax.random.PRNGKey(7), (3, 32, 32, 3))
        y_plain = _forward_output(plain_structure, plain_params, x, rng_key)
        y_lean = _forward_output(lean_structure, lean_params, x, rng_key)
        assert jnp.allclose(y_plain, y_lean, atol=1e-5, rtol=1e-4)

    def test_mupc_sees_the_same_residual_depth(self):
        # Eight merge nodes on the longest path in both layouts, so muPC's
        # 1/sqrt(L) damping means the same thing in each.
        from fabricpc.core.mupc import _count_skip_connections_depth

        for lean in (False, True):
            s = _resnet(lean=lean)
            assert _count_skip_connections_depth(s.nodes, s.edges, s.node_order) == 8

    def test_plain_is_still_the_default(self):
        structure = build_resnet18(
            weight_init=XavierInitializer(),
            inference=InferenceSGD(eta_infer=0.05, infer_steps=2),
        )
        assert "s1b1_skip_sum" in structure.nodes
        assert "avgpool" in structure.nodes
