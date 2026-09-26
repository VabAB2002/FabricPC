"""The stiffness a state-based settle sees, and so its largest stable rate.

State-based solvers (``InferenceSGD`` and its variants) step every free
latent by -eta * g(z), where g is the latent gradient the solver itself
accumulates (``zero_grads`` then ``forward_value_and_grad``, muPC top-down
scales included). Near a state that step is z <- z - eta * A dz with
A = dg/dz (plus latent_decay on the diagonal), so the settle is stable only
while eta * rho(A) < 2, rho the largest eigenvalue magnitude. This is the
state-coordinate counterpart of ``epsilon_spectrum`` for ePC.

Why it matters. Training can make a network stiffer (on the character
transformer, sPC's settle at eta 0.0175 was stable at init and diverged
after training; its norm clip hid the divergence and turned the weight
updates into noise). A rate that is safe at init has no lasting margin, the
same as for ePC. ``fabricpc.training.InferenceRateController`` uses this
measure to keep a state-based rate below the bound.

A is not symmetric under muPC (the top-down scales precondition the
gradient), so this uses power iteration on Jacobian-vector products of g,
which finds the eigenvalue of largest magnitude without needing symmetry.
"""

from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import lax

from fabricpc.core.state_ops import update_node_in_state
from fabricpc.core.types import GraphParams, GraphState, GraphStructure


def _latent_grad_fn(
    params: GraphParams,
    state: GraphState,
    clamps: Mapping[str, Any],
    structure: GraphStructure,
) -> Tuple[Callable, Dict[str, jnp.ndarray]]:
    """``(g, z0)``: g maps the free latents to the solver's latent gradient
    (plus latent_decay * z, the decay term of the update), z0 the state's."""
    solver = structure.config["inference"]
    cls = type(solver)
    decay = float(solver.config.get("latent_decay", 0.0))
    free = tuple(name for name in structure.nodes if name not in clamps)

    def g(latents):
        st = state
        for name in free:
            st = update_node_in_state(st, name, z_latent=latents[name])
        st = cls.zero_grads(params, st, clamps, structure)
        st = cls.forward_value_and_grad(params, st, clamps, structure)
        return {
            name: st.nodes[name].latent_grad + decay * latents[name] for name in free
        }

    return g, {name: state.nodes[name].z_latent for name in free}


def _norm(tree) -> jnp.ndarray:
    return jnp.sqrt(sum(jnp.sum(x * x) for x in jax.tree_util.tree_leaves(tree)))


def make_latent_curvature(
    structure: GraphStructure, iters: int = 30
) -> Callable[[GraphParams, GraphState, Mapping[str, Any], jax.Array], jnp.ndarray]:
    """Compile ``(params, state, clamps, key) -> rho(A)`` for one graph.

    Reuse the returned function across probes; it compiles once.
    """

    def run(params, state, clamps, key):
        g, z0 = _latent_grad_fn(params, state, clamps, structure)

        def av(v):
            return jax.jvp(g, (z0,), (v,))[1]

        leaves, treedef = jax.tree_util.tree_flatten(z0)
        keys = jax.random.split(key, len(leaves))
        v0 = jax.tree_util.tree_unflatten(
            treedef,
            [jax.random.normal(k, x.shape, jnp.float32) for k, x in zip(keys, leaves)],
        )
        v0 = jax.tree_util.tree_map(lambda x: x / _norm(v0), v0)

        def body(_, carry):
            v, _ = carry
            w = av(v)
            size = _norm(w)
            return jax.tree_util.tree_map(lambda x: x / (size + 1e-30), w), size

        _, size = lax.fori_loop(0, iters, body, (v0, jnp.asarray(0.0)))
        return size

    return jax.jit(run)


def latent_curvature(
    params: GraphParams,
    state: GraphState,
    clamps: Mapping[str, Any],
    structure: GraphStructure,
    iters: int = 30,
    key: Optional[jax.Array] = None,
) -> float:
    """rho(A) at the state, as a float (one compile per call; use
    :func:`make_latent_curvature` for repeated probes)."""
    key = jax.random.PRNGKey(0) if key is None else key
    return float(make_latent_curvature(structure, iters)(params, state, clamps, key))
