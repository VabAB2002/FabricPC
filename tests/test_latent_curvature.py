"""The stiffness a state-based (sPC) settle sees, and its stability bound.

sPC steps every free latent z by -eta * g(z), with g the solver's own latent
gradient (muPC top-down scales included). Near a state the step is
z <- z - eta * A (z - z*), A = dg/dz, so the settle is stable only while
eta * rho(A) < 2. ``make_latent_curvature`` measures rho(A) by power
iteration on Jacobian-vector products of g.
"""

import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np
import pytest

from fabricpc.core.activations import SoftmaxActivation, TanhActivation
from fabricpc.core.energy import CrossEntropyEnergy, graph_energy
from fabricpc.core.inference import InferenceSGD
from fabricpc.core.initializers import NormalInitializer
from fabricpc.core.latent_curvature import latent_curvature, make_latent_curvature
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_graph_state, initialize_params
from fabricpc.nodes import Linear
from fabricpc.nodes.identity import IdentityNode
from fabricpc.training import build_clamps

BATCH = 4


def _mlp(std, eta=0.1, steps=1):
    x = IdentityNode(shape=(5,), name="x")
    h1 = Linear(
        shape=(7,),
        activation=TanhActivation(),
        name="h1",
        weight_init=NormalInitializer(std=std),
    )
    h2 = Linear(
        shape=(6,),
        activation=TanhActivation(),
        name="h2",
        weight_init=NormalInitializer(std=std),
    )
    y = Linear(
        shape=(3,),
        activation=SoftmaxActivation(),
        energy=CrossEntropyEnergy(),
        name="y",
        weight_init=NormalInitializer(std=std),
    )
    return graph(
        nodes=[x, h1, h2, y],
        edges=[
            Edge(source=x, target=h1.slot("in")),
            Edge(source=h1, target=h2.slot("in")),
            Edge(source=h2, target=y.slot("in")),
        ],
        task_map=TaskMap(x=x, y=y),
        inference=InferenceSGD(eta_infer=eta, infer_steps=steps),
    )


def _setup(std=1.5):
    s = _mlp(std)
    params = initialize_params(s, jax.random.PRNGKey(0))
    kx, ky = jax.random.split(jax.random.PRNGKey(1))
    clamps = build_clamps(
        {
            "x": jax.random.normal(kx, (BATCH, 5)),
            "y": jax.nn.one_hot(jax.random.randint(ky, (BATCH,), 0, 3), 3),
        },
        s,
        clamp_target=True,
    )
    state = initialize_graph_state(
        s, BATCH, jax.random.PRNGKey(2), clamps=clamps, params=params
    )
    return s, params, clamps, state


def _dense_operator(s, params, clamps, state):
    """A = dg/dz written out column by column, for checking."""
    from fabricpc.core.latent_curvature import _latent_grad_fn

    g, z0 = _latent_grad_fn(params, state, clamps, s)
    flat0, unravel = jax.flatten_util.ravel_pytree(z0)
    cols = []
    for i in range(flat0.size):
        e = jnp.zeros_like(flat0).at[i].set(1.0)
        cols.append(
            jax.flatten_util.ravel_pytree(jax.jvp(g, (z0,), (unravel(e),))[1])[0]
        )
    return np.asarray(jnp.stack(cols, axis=1))


def test_power_iteration_finds_the_largest_eigenvalue():
    s, params, clamps, state = _setup()
    dense = _dense_operator(s, params, clamps, state)
    want = float(np.max(np.abs(np.linalg.eigvals(dense))))
    got = latent_curvature(params, state, clamps, s, iters=100)
    assert got == pytest.approx(want, rel=1e-3)


def test_the_measured_value_is_the_settles_stability_limit():
    # On a linear graph the energy is exactly quadratic, so the bound is
    # exact: just under 2 / rho the settle converges, just over it the
    # stiffest direction grows every step and the energy runs away. (On a
    # nonlinear graph the curvature falls away from the state and the bound
    # is only local.)
    from fabricpc.core.activations import IdentityActivation
    from fabricpc.core.inference import run_inference

    x = IdentityNode(shape=(5,), name="x")
    hs = [
        Linear(
            shape=(6,),
            activation=IdentityActivation(),
            name=f"h{i}",
            weight_init=NormalInitializer(std=0.8),
        )
        for i in range(3)
    ]
    y = Linear(shape=(3,), name="y", weight_init=NormalInitializer(std=0.8))
    chain = [x, *hs, y]
    s = graph(
        nodes=chain,
        edges=[Edge(source=a, target=b.slot("in")) for a, b in zip(chain, chain[1:])],
        task_map=TaskMap(x=x, y=y),
        inference=InferenceSGD(eta_infer=0.1, infer_steps=1),
    )
    params = initialize_params(s, jax.random.PRNGKey(0))
    clamps = build_clamps(
        {
            "x": jax.random.normal(jax.random.PRNGKey(1), (BATCH, 5)),
            "y": jax.random.normal(jax.random.PRNGKey(2), (BATCH, 3)),
        },
        s,
        clamp_target=True,
    )
    state = initialize_graph_state(
        s, BATCH, jax.random.PRNGKey(3), clamps=clamps, params=params
    )
    rho = latent_curvature(params, state, clamps, s, iters=200)

    def energy_after(eta, steps=400):
        st = s._replace(
            config={
                **s.config,
                "inference": InferenceSGD(eta_infer=eta, infer_steps=steps),
            }
        )
        return float(graph_energy(run_inference(params, state, clamps, st), st))

    start = float(graph_energy(state, s))
    assert energy_after(0.95 * 2 / rho) < start
    assert energy_after(1.05 * 2 / rho) > 100 * start


def test_a_stiffer_network_measures_larger():
    def measure(std):
        s, params, clamps, state = _setup(std)
        return latent_curvature(params, state, clamps, s, iters=100)

    assert measure(2.5) > measure(0.5)


def test_latent_decay_adds_to_the_measured_value():
    s, params, clamps, state = _setup()
    base = latent_curvature(params, state, clamps, s, iters=100)
    decayed = s._replace(
        config={
            **s.config,
            "inference": InferenceSGD(eta_infer=0.1, infer_steps=1, latent_decay=3.0),
        }
    )
    assert latent_curvature(params, state, clamps, decayed, iters=100) == pytest.approx(
        base + 3.0, rel=1e-3
    )


def test_the_compiled_version_can_be_reused():
    s, params, clamps, state = _setup()
    fn = make_latent_curvature(s, iters=60)
    a = float(fn(params, state, clamps, jax.random.PRNGKey(0)))
    b = float(fn(params, state, clamps, jax.random.PRNGKey(0)))
    assert a == b > 0
