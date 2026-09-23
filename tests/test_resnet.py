"""Tests for the ResNet-18 builder in fabricpc.models."""

import jax

from fabricpc.core.inference import InferenceSGD
from fabricpc.core.initializers import MuPCInitializer, XavierInitializer
from fabricpc.core.mupc import MuPCConfig
from fabricpc.graph_initialization import initialize_params
from fabricpc.models import build_resnet18


def test_resnet18_matches_the_demo_graph_size():
    # The demo's docstring: 31 nodes, 38 edges, 2,795,210 parameters.
    structure = build_resnet18(
        weight_init=MuPCInitializer(),
        inference=InferenceSGD(eta_infer=0.05, infer_steps=2),
        scaling=MuPCConfig(include_output=False),
        output_weight_init=XavierInitializer(),
    )
    params = initialize_params(structure, jax.random.PRNGKey(0))
    assert len(structure.nodes) == 31
    assert len(structure.edges) == 38
    assert sum(p.size for p in jax.tree_util.tree_leaves(params)) == 2_795_210


def test_resnet18_input_and_output_shapes():
    structure = build_resnet18(
        weight_init=XavierInitializer(),
        inference=InferenceSGD(eta_infer=0.05, infer_steps=2),
    )
    assert tuple(structure.nodes["input"].node_info.shape) == (32, 32, 3)
    assert tuple(structure.nodes["output"].node_info.shape) == (10,)
