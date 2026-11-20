"""Scores for rows that are not classifiers, as ``EvalMetric``s.

The autoencoder rows (the faculty advisor's user story) are judged on how
well they rebuild the input and on how many hidden units stay silent.

Sparsity is read from each unit's prediction z_mu, its activation. In a
settled PC state the latent is z_mu plus a small leftover error, so reading
the latent would make every PC unit look slightly active and PC look less
sparse than its activations are. On the forward pass the two are equal.
"""

from typing import Sequence

import jax.numpy as jnp

from fabricpc.training.metrics import EvalMetric

# A ReLU unit is silent when its activation is exactly zero; this leaves room
# for float rounding and nothing more.
SILENT_BELOW = 1e-6


def _reconstruction_mse_fn(state, batch, structure):
    """Per sample, the mean over pixels of (target - output)^2; weight 1."""
    z_mu = state.nodes[structure.task_map["y"]].z_mu
    y = jnp.asarray(batch["y"], dtype=z_mu.dtype).reshape(z_mu.shape)
    axes = tuple(range(1, z_mu.ndim))
    value = jnp.mean((y - z_mu) ** 2, axis=axes)
    return value, jnp.ones_like(value)


reconstruction_mse = EvalMetric(fn=_reconstruction_mse_fn)


def sparsity(node_names: Sequence[str], silent_below: float = SILENT_BELOW):
    """Fraction of the named nodes' units with |activation| < silent_below,
    per sample, pooled over the nodes."""
    names = tuple(node_names)

    def fn(state, batch, structure):
        silent = 0.0
        total = 0
        for name in names:
            z = state.nodes[name].z_mu
            axes = tuple(range(1, z.ndim))
            silent = silent + jnp.sum(jnp.abs(z) < silent_below, axis=axes)
            total += int(jnp.size(z) // z.shape[0])
        value = silent / total
        return value, jnp.ones_like(value)

    return EvalMetric(fn=fn)


# One plain line per headline metric, printed under a run's result so a
# reader knows what the number means without opening the code.
_MEANING = {
    "accuracy": "share of test examples the model labels correctly (higher is better)",
    "perplexity": "how unsure the model is about the next token; 1 is perfect "
    "(lower is better)",
    "reconstruction_mse": "mean squared error between each test image and the "
    "model's rebuilt copy (lower is better)",
}


def explain_metric(name: str) -> str:
    """What a metric means, in one plain line ('' if we have no wording)."""
    if name in _MEANING:
        return _MEANING[name]
    kind, _, noise = name.rpartition("_p")
    if kind in ("bit_accuracy", "exact_recall", "nearest_correct") and noise.isdigit():
        what = {
            "bit_accuracy": "share of pattern bits recalled correctly",
            "exact_recall": "share of patterns recalled with every bit right",
            "nearest_correct": "share of recalls closest to the right pattern",
        }[kind]
        return f"{what}, starting from {noise}% of the bits flipped (higher is better)"
    return ""
