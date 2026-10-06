"""Nudged predictive-coding gradients, following pcx.

Normal PC training clamps the output to the label y. Nudging clamps it to a
point between the network's own prediction u and the label:

    h = u - beta * (u - y)

beta = 1 is the normal clamp, 0 < beta < 1 is positive nudging, and
beta < 0 (negative nudging) pushes the target away from the label. pcx sets
h once, right after the feedforward pass, keeps it fixed while the hidden
latents settle, and then multiplies the weight gradients by 1/beta. For
negative beta that flips the update back to the learning direction.
"""

import jax
import jax.numpy as jnp

from experiments.nudging.settle import settle
from fabricpc.core.energy import graph_energy
from fabricpc.core.state_ops import set_latents_to_clamps
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.training.trainer import (
    build_clamps,
    grad_denominator,
    pc_weight_gradients,
)


def nudged_grads(
    params,
    batch,
    structure,
    key,
    *,
    beta,
    lr_x,
    momentum,
    steps,
    return_state=False,
):
    """Weight gradients for one batch with the output nudged by ``beta``.

    ``beta`` may be a traced scalar, so changing it between epochs does not
    recompile a jitted caller.
    """
    clamps = build_clamps(batch, structure, clamp_target=True)
    out = structure.task_map["y"]
    batch_size = clamps[out].shape[0]
    state = initialize_graph_state(
        structure, batch_size, key, clamps=clamps, params=params
    )
    # The feedforward init still computes the output's prediction u even
    # though the output is clamped.
    u = state.nodes[out].z_mu
    y = clamps[out]
    # beta == 1 uses y itself, so it matches plain clamped PC bit for bit.
    target = jnp.where(beta == 1.0, y, u - beta * (u - y))
    nudged = {**clamps, out: target}
    state = set_latents_to_clamps(state, nudged)
    state = settle(
        params, state, nudged, structure, lr_x=lr_x, momentum=momentum, steps=steps
    )
    grads = pc_weight_gradients(params, state, structure, nudged)
    grads = jax.tree_util.tree_map(lambda g: g / beta, grads)
    metrics = {
        "energy": graph_energy(state, structure) / grad_denominator(structure, nudged)
    }
    if return_state:
        return grads, metrics, state
    return grads, metrics
