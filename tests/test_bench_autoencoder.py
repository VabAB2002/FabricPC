"""The MNIST autoencoder rows: reconstruction error and hidden sparsity.

The faculty advisor's user story: train an autoencoder under backprop and
under PC and compare them on something other than accuracy.
"""

import jax
import jax.numpy as jnp
import pytest

from fabricpc.bench.metrics import reconstruction_mse, sparsity
from fabricpc.bench.registry import COMPARISONS, ROWS
from tests.conftest import ListLoader

AE_ROWS = [f"mnist-autoencoder-{a}" for a in ("spc", "epc", "backprop")]


def fake_image_loaders(key, n_batches=3, batch_size=8):
    """Random images in [0, 1], with the image itself as the target."""
    batches = []
    for i in range(n_batches):
        x = jax.random.uniform(jax.random.fold_in(key, i), (batch_size, 784))
        batches.append((x, x))
    return ListLoader(batches), ListLoader(batches[:1])


class _Node:
    def __init__(self, z_mu):
        self.z_mu = z_mu


class _State:
    def __init__(self, nodes):
        self.nodes = nodes


def test_reconstruction_mse_is_the_mean_squared_pixel_error():
    import types

    y = jnp.array([[0.0, 1.0, 0.5, 0.5], [1.0, 1.0, 0.0, 0.0]])
    out = jnp.array([[0.0, 0.0, 0.5, 1.0], [1.0, 1.0, 0.0, 0.0]])
    structure = types.SimpleNamespace(task_map={"x": "pixels", "y": "recon"})
    value, weight = reconstruction_mse.fn(
        _State({"recon": _Node(out)}), {"y": y}, structure
    )
    assert jnp.allclose(value, jnp.array([(1.0 + 0.25) / 4, 0.0]))
    assert jnp.allclose(weight, 1.0)


def test_sparsity_is_the_fraction_of_silent_units():
    metric = sparsity(("a", "b"))
    state = _State(
        {
            "a": _Node(jnp.array([[0.0, 2.0], [0.0, 0.0]])),
            "b": _Node(jnp.array([[1.0, 1.0, 0.0, 3.0], [0.0, 0.0, 0.0, 0.0]])),
        }
    )
    value, weight = metric.fn(state, {}, None)
    # sample 0: 1 of 2 and 1 of 4 silent -> 2 / 6; sample 1: all 6 silent
    assert jnp.allclose(value, jnp.array([2 / 6, 1.0]))
    assert jnp.allclose(weight, 1.0)


def test_autoencoder_rows_are_registered_with_their_own_metric():
    for row_id in AE_ROWS:
        row = ROWS[row_id]
        assert row.metric == "reconstruction_mse"
        assert set(row.eval_metrics) >= {
            "reconstruction_mse",
            "code_sparsity",
            "hidden_sparsity",
        }
    assert COMPARISONS["mnist-autoencoder"].metric == "reconstruction_mse"


def test_autoencoder_is_784_128_32_128_784():
    _, structure = ROWS["mnist-autoencoder-spc"].model_factory(jax.random.PRNGKey(0))
    shapes = [tuple(structure.nodes[n].node_info.shape) for n in structure.node_order]
    assert shapes == [(784,), (128,), (32,), (128,), (784,)]
    assert structure.task_map == {"x": "pixels", "y": "recon"}


def test_the_loader_gives_the_image_back_as_its_own_target():
    from fabricpc.bench.registry import _AsReconstruction

    inner = ListLoader([(jnp.ones((2, 784)) * 0.3, jnp.eye(10)[:2])])
    (x, y), *_ = list(_AsReconstruction(inner))
    assert len(_AsReconstruction(inner)) == 1
    assert jnp.array_equal(x, y)


@pytest.mark.parametrize("row_id", AE_ROWS)
def test_a_trial_reports_reconstruction_and_sparsity(tmp_path, rng_key, row_id):
    from fabricpc.bench.runner import run_trial

    result = run_trial(
        ROWS[row_id],
        0,
        tmp_path,
        loaders=fake_image_loaders(rng_key),
        num_epochs=1,
        warmup_steps=1,
        timed_steps=2,
    )
    assert result.status == "ok", result.error
    m = result.metrics
    assert m["reconstruction_mse"] >= 0.0
    for name in ("code_sparsity", "hidden_sparsity"):
        assert 0.0 <= m[name] <= 1.0
        assert 0.0 <= m[f"forward_{name}"] <= 1.0
    # Accuracy means nothing for a reconstruction.
    assert "accuracy" not in m
    assert ("energy" in m) == (row_id != "mnist-autoencoder-backprop")
    assert result.curve[0]["reconstruction_mse"] >= 0.0
