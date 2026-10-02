"""Tests for the hint solver used in the hint experiment.

Run with: python -m pytest experiments/hint -q
"""

import jax
import jax.numpy as jnp
import numpy as np

from experiments.hint.hint_inference import InferenceSGDHint, hint_matrix
from experiments.hint.models import deep_mlp
from fabricpc.core.inference import InferenceSGD
from fabricpc.core.inference import run_inference
from fabricpc.graph_initialization.state_initializer import initialize_graph_state


def _settle(solver, seed=0):
    params, structure = deep_mlp(
        depth=3, width=16, solver=solver, key=jax.random.PRNGKey(seed)
    )
    rng = np.random.default_rng(seed)
    x = jnp.asarray(rng.normal(size=(4, 784)), dtype=jnp.float32)
    y = jax.nn.one_hot(jnp.asarray(rng.integers(0, 10, size=4)), 10)
    clamps = {"pixels": x, "class": y}
    state = initialize_graph_state(
        structure, 4, jax.random.PRNGKey(1), clamps=clamps, params=params
    )
    return run_inference(params, state, clamps, structure), structure


def test_zero_hint_is_exactly_plain_spc():
    plain, _ = _settle(InferenceSGD(eta_infer=0.1, infer_steps=5))
    hinted, _ = _settle(
        InferenceSGDHint(eta_infer=0.1, infer_steps=5, hint_strength=0.0)
    )
    for name in plain.nodes:
        np.testing.assert_allclose(
            plain.nodes[name].z_latent, hinted.nodes[name].z_latent, atol=1e-6
        )


def test_hint_changes_the_hidden_latents():
    plain, _ = _settle(InferenceSGD(eta_infer=0.1, infer_steps=5))
    hinted, _ = _settle(
        InferenceSGDHint(eta_infer=0.1, infer_steps=5, hint_strength=1.0)
    )
    moved = float(
        jnp.abs(plain.nodes["h0"].z_latent - hinted.nodes["h0"].z_latent).max()
    )
    assert moved > 1e-4


def test_no_chain_mode_moves_the_first_layer_after_one_step():
    # With the whisper chain switched off, only the hint moves hidden latents,
    # so even the first layer moves after a single step. With the chain on and
    # no hint, one step only reaches the layer next to the output.
    chain_only, _ = _settle(InferenceSGD(eta_infer=0.1, infer_steps=1))
    hint_only, _ = _settle(
        InferenceSGDHint(eta_infer=0.1, infer_steps=1, hint_strength=1.0, chain=False)
    )
    h0_chain = chain_only.nodes["h0"]
    h0_hint = hint_only.nodes["h0"]
    assert float(jnp.abs(h0_chain.z_latent - h0_chain.z_mu).max()) < 1e-7
    assert float(jnp.abs(h0_hint.z_latent - h0_hint.z_mu).max()) > 1e-4


def test_hint_matrix_is_fixed_for_a_node_and_differs_between_nodes():
    a1 = hint_matrix("h0", 16, 10, seed=0)
    a2 = hint_matrix("h0", 16, 10, seed=0)
    b = hint_matrix("h1", 16, 10, seed=0)
    np.testing.assert_array_equal(a1, a2)
    assert a1.shape == (10, 16)
    assert not np.allclose(a1, b)
