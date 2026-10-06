"""Tests for nudged batch gradients."""

import jax
import jax.numpy as jnp
import numpy as np

from experiments.nudging.nudge import nudged_grads
from fabricpc.core.inference import InferenceSGD
from fabricpc.graph_initialization import initialize_params
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.models.vgg import create_vgg
from fabricpc.training.trainer import _batch_grads, build_clamps


def _setup(steps=3):
    s = create_vgg(
        5,
        input_shape=(16, 16, 3),
        num_classes=4,
        inference=InferenceSGD(eta_infer=0.05, infer_steps=steps),
        fuse_pool=True,
        head="se",
    )
    p = initialize_params(s, jax.random.PRNGKey(0))
    batch = {
        "x": jax.random.normal(jax.random.PRNGKey(1), (4, 16, 16, 3)),
        "y": jax.nn.one_hot(jnp.array([0, 1, 2, 3]), 4),
    }
    return s, p, batch


def _flat(g):
    return jnp.concatenate(
        [w.ravel() for n in g.nodes.values() for w in n.weights.values()]
    )


def _close(a, b, tol):
    """Whole-vector check: same direction and nearly the same size."""
    cos = float(a @ b / (jnp.linalg.norm(a) * jnp.linalg.norm(b)))
    rel = float(jnp.linalg.norm(a - b) / jnp.linalg.norm(b))
    assert cos > 1 - tol and rel < tol, (cos, rel)


def _grads(beta, steps=3, momentum=0.0):
    s, p, b = _setup(steps)
    g, _ = nudged_grads(
        p,
        b,
        s,
        jax.random.PRNGKey(5),
        beta=jnp.float32(beta),
        lr_x=0.05,
        momentum=momentum,
        steps=steps,
    )
    return _flat(g)


def test_beta_one_equals_plain_clamped_pc():
    s, p, b = _setup()
    ref = _batch_grads(p, b, s, jax.random.PRNGKey(5), algorithm="pc")[0]
    np.testing.assert_allclose(_grads(1.0), _flat(ref), rtol=1e-5, atol=1e-7)


def test_output_stays_at_the_nudged_target():
    s, p, b = _setup()
    beta = jnp.float32(0.3)
    _, _, final = nudged_grads(
        p,
        b,
        s,
        jax.random.PRNGKey(5),
        beta=beta,
        lr_x=0.05,
        momentum=0.0,
        steps=3,
        return_state=True,
    )
    clamps = build_clamps(b, s, clamp_target=True)
    init = initialize_graph_state(s, 4, jax.random.PRNGKey(5), clamps=clamps, params=p)
    u = init.nodes["class"].z_mu
    h = u - beta * (u - b["y"])
    np.testing.assert_allclose(final.nodes["class"].z_latent, h, atol=1e-6)


def test_negative_beta_flips_and_scales_at_zero_steps():
    # With no settling, the raw output error is -beta*(u - y), so the 1/beta
    # scaling makes the +beta and -beta gradients identical.
    _close(_grads(0.2, steps=0), _grads(-0.2, steps=0), 1e-5)


def test_negative_beta_gradient_is_not_the_raw_gradient():
    # Guard against a missing 1/beta: the raw nudged gradient at -0.2 points
    # the opposite way from the one at +0.2.
    g = _grads(-0.2, steps=0)
    assert float(jnp.dot(g, _grads(0.2, steps=0))) > 0


def test_small_beta_limit_is_finite():
    g1, g2 = _grads(1e-3), _grads(2e-3)
    assert bool(jnp.isfinite(g1).all())
    # Dividing by a tiny beta amplifies float rounding in near-zero entries,
    # so compare the whole vectors, not entry by entry.
    _close(g1, g2, 2e-2)
