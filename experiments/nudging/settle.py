"""Settling with momentum, the way pcx moves its latent states.

pcx updates the latents with ``optax.sgd(lr_x, momentum)`` and resets the
optimizer at the start of every batch. FabricPC's InferenceSGD has no
momentum, so this runs the same per-step mechanics (zero the gradients,
forward pass, accumulate latent gradients) and then takes a heavy-ball step:

    v = momentum * v + grad
    z = z - lr_x * v

with v starting at zero for each call, so each batch starts fresh. With
momentum 0 this is exactly InferenceSGD's step.
"""

import jax
import jax.numpy as jnp

from fabricpc.core.inference import InferenceSGD
from fabricpc.core.state_ops import update_node_in_state


def settle(params, state, clamps, structure, *, lr_x, momentum, steps):
    """Run ``steps`` momentum steps on every unclamped latent. Returns the state."""
    free = [n for n in structure.nodes if n not in clamps]
    velocity = {n: jnp.zeros_like(state.nodes[n].z_latent) for n in free}

    def body(_, carry):
        state, velocity = carry
        state = InferenceSGD.zero_grads(params, state, clamps, structure)
        state = InferenceSGD.forward_value_and_grad(params, state, clamps, structure)
        new_velocity = {}
        for n in free:
            node = state.nodes[n]
            v = momentum * velocity[n] + node.latent_grad
            new_velocity[n] = v
            state = update_node_in_state(state, n, z_latent=node.z_latent - lr_x * v)
        return state, new_velocity

    state, _ = jax.lax.fori_loop(0, steps, body, (state, velocity))
    return state
