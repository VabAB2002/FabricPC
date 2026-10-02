"""sPC with a direct "hint" from the output error.

Plain sPC passes the output error back one layer per settling step, and it
gets weaker at every hop. Here every hidden layer also hears a rough copy of
the output error straight away, through its own fixed random matrix B (the
idea behind direct feedback alignment). The usual layer-to-layer passing still
runs on top, unless ``chain=False``.

    hint_strength = 0              -> exactly plain sPC
    chain=False, infer_steps = 1   -> direct feedback alignment (hint only)
    anything in between            -> the hybrid

Weights still learn from each layer's own local error, as in normal PC.
"""

import zlib

import jax.numpy as jnp
import numpy as np

from fabricpc.core.inference import InferenceSGD
from fabricpc.core.state_ops import update_node_in_state


def hint_matrix(node_name: str, width: int, out_dim: int, seed: int) -> np.ndarray:
    """The fixed random (out_dim, width) matrix that carries the hint to one node.

    Seeded from the node name so it never changes during training.
    """
    rng = np.random.default_rng([seed, zlib.crc32(node_name.encode())])
    return rng.normal(0.0, 1.0 / np.sqrt(out_dim), size=(out_dim, width)).astype(
        np.float32
    )


class InferenceSGDHint(InferenceSGD):
    """InferenceSGD plus a direct hint to every hidden node."""

    def __init__(
        self,
        eta_infer=0.1,
        infer_steps=20,
        latent_decay=0.0,
        hint_strength=1.0,
        chain=True,
        output_node="class",
        hint_seed=0,
    ):
        # Skip InferenceSGD.__init__ so the extra settings land in the config.
        super(InferenceSGD, self).__init__(
            eta_infer=eta_infer,
            infer_steps=infer_steps,
            latent_decay=latent_decay,
            hint_strength=hint_strength,
            chain=chain,
            output_node=output_node,
            hint_seed=hint_seed,
        )

    @classmethod
    def inference_step(cls, params, state, clamps, structure, config):
        state = cls.zero_grads(params, state, clamps, structure)
        state = cls.forward_value_and_grad(params, state, clamps, structure)

        out = config["output_node"]
        # The output's prediction error (target - prediction when clamped).
        out_error = state.nodes[out].error
        strength = config["hint_strength"]

        for name in structure.nodes:
            if name in clamps or name == out:
                continue
            node_state = state.nodes[name]
            width = node_state.z_latent.shape[-1]
            b = jnp.asarray(
                hint_matrix(name, width, out_error.shape[-1], config["hint_seed"])
            )
            # Same sign as the true gradient -W^T e that the chain would bring.
            hint = -(out_error @ b)
            grad = node_state.latent_grad
            if not config["chain"]:
                grad = jnp.zeros_like(grad)
            state = update_node_in_state(
                state, name, latent_grad=grad + strength * hint
            )

        return cls.update_latents(params, state, clamps, structure, config)
