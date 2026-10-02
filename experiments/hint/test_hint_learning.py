"""Tests for adding the hint to the learning signal (not to the latents)."""

import jax
import jax.numpy as jnp
import numpy as np

from experiments.hint.hint_learning import batch_grads, dfa_grads, hidden_names
from experiments.hint.models import deep_mlp
from fabricpc.core.inference import InferenceSGD
from fabricpc.training.trainer import _batch_grads


def _setup(seed=0):
    params, structure = deep_mlp(
        3, 16, InferenceSGD(eta_infer=0.1, infer_steps=4), jax.random.PRNGKey(seed)
    )
    rng = np.random.default_rng(seed)
    x = jnp.asarray(rng.normal(size=(8, 784)), dtype=jnp.float32)
    y = jax.nn.one_hot(jnp.asarray(rng.integers(0, 10, size=8)), 10)
    return params, structure, {"x": x, "y": y}


def _flat(g, name):
    return jnp.concatenate([w.ravel() for w in g.nodes[name].weights.values()])


def test_mode_none_is_exactly_fabricpc_pc_gradients():
    params, structure, batch = _setup()
    ours = batch_grads(params, structure, batch, jax.random.PRNGKey(1), mode="none")
    theirs = _batch_grads(
        params, batch, structure, jax.random.PRNGKey(1), algorithm="pc"
    )[0]
    for name in hidden_names(structure) + ["class"]:
        np.testing.assert_allclose(
            _flat(ours, name), _flat(theirs, name), rtol=1e-5, atol=1e-8
        )


def test_pure_mode_gives_textbook_dfa_on_hidden_layers():
    params, structure, batch = _setup()
    ours = batch_grads(params, structure, batch, jax.random.PRNGKey(1), mode="pure")
    ref = dfa_grads(params, structure, batch)
    for name in hidden_names(structure):
        a, b = _flat(ours, name), _flat(ref, name)
        cos = float(a @ b / (jnp.linalg.norm(a) * jnp.linalg.norm(b)))
        assert cos > 0.999


def test_far_mode_leaves_a_layer_alone_when_its_whisper_is_loud():
    params, structure, batch = _setup()
    plain = batch_grads(params, structure, batch, jax.random.PRNGKey(1), mode="none")
    far = batch_grads(
        params, structure, batch, jax.random.PRNGKey(1), mode="far", threshold=0.0
    )
    # threshold 0: no layer counts as faded, so nothing changes
    for name in hidden_names(structure):
        np.testing.assert_allclose(_flat(far, name), _flat(plain, name), rtol=1e-6)
