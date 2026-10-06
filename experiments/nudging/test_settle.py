"""Tests for the momentum settle (pcx's state optimizer).

Run with: PYTHONPATH=. python -m pytest experiments/nudging -q
"""

import jax
import jax.numpy as jnp
import numpy as np

from experiments.nudging.settle import settle
from fabricpc.core.inference import InferenceSGD, run_inference
from fabricpc.graph_initialization import initialize_params
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.models.vgg import create_vgg


def _tiny(steps):
    s = create_vgg(
        5,
        input_shape=(16, 16, 3),
        num_classes=4,
        inference=InferenceSGD(eta_infer=0.05, infer_steps=steps),
        fuse_pool=True,
        head="se",
    )
    p = initialize_params(s, jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 16, 16, 3))
    y = jax.nn.one_hot(jnp.array([1, 3]), 4)
    clamps = {"input": x, "class": y}
    st = initialize_graph_state(s, 2, jax.random.PRNGKey(2), clamps=clamps, params=p)
    return s, p, st, clamps


def _grad_at(p, st, clamps, s, node):
    st = InferenceSGD.zero_grads(p, st, clamps, s)
    st = InferenceSGD.forward_value_and_grad(p, st, clamps, s)
    return st.nodes[node].latent_grad


def test_momentum_zero_equals_fabricpc_sgd():
    s, p, st, clamps = _tiny(5)
    ref = run_inference(p, st, clamps, s)
    out = settle(p, st, clamps, s, lr_x=0.05, momentum=0.0, steps=5)
    for n in s.nodes:
        np.testing.assert_allclose(
            out.nodes[n].z_latent, ref.nodes[n].z_latent, atol=1e-6
        )


def test_momentum_changes_the_path_but_not_the_clamps():
    s, p, st, clamps = _tiny(5)
    a = settle(p, st, clamps, s, lr_x=0.05, momentum=0.0, steps=5)
    b = settle(p, st, clamps, s, lr_x=0.05, momentum=0.5, steps=5)
    moved = jnp.abs(a.nodes["conv4"].z_latent - b.nodes["conv4"].z_latent).max()
    assert float(moved) > 1e-7
    np.testing.assert_array_equal(b.nodes["class"].z_latent, clamps["class"])


def test_two_momentum_steps_match_hand_computation_on_one_node():
    # v1 = g1, z1 = z0 - lr*g1 ; v2 = m*v1 + g2, z2 = z1 - lr*v2
    s, p, st, clamps = _tiny(2)
    lr, m = 0.05, 0.5
    one = settle(p, st, clamps, s, lr_x=lr, momentum=m, steps=1)
    g1 = _grad_at(p, st, clamps, s, "conv4")
    np.testing.assert_allclose(
        one.nodes["conv4"].z_latent, st.nodes["conv4"].z_latent - lr * g1, atol=1e-6
    )
    g2 = _grad_at(p, one, clamps, s, "conv4")
    two = settle(p, st, clamps, s, lr_x=lr, momentum=m, steps=2)
    expected = one.nodes["conv4"].z_latent - lr * (m * g1 + g2)
    np.testing.assert_allclose(two.nodes["conv4"].z_latent, expected, atol=1e-5)
