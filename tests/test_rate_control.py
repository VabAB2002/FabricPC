"""Tests for the ePC rate that follows lambda_max (issue #72).

Two pieces: ``train(..., structure_callback=...)``, which lets a run swap its
graph structure between batches, and ``InferenceRateController``, which uses that
hook to keep eta_infer * lambda_max at a chosen value as lambda_max changes.
"""

import math
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import optax
import pytest

from conftest import ListLoader
from fabricpc.core import EPCInference
from fabricpc.core.activations import SoftmaxActivation, TanhActivation
from fabricpc.core.energy import CrossEntropyEnergy
from fabricpc.core.epsilon_spectrum import make_epsilon_spectrum
from fabricpc.core.inference import InferenceSGD
from fabricpc.core.initializers import NormalInitializer
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_graph_state, initialize_params
from fabricpc.nodes import Linear
from fabricpc.nodes.identity import IdentityNode
from fabricpc.training import (
    InferenceRateController,
    build_clamps,
    make_train_step,
    train,
)

BATCH = 8


def _mlp(inference, std=0.5, width=12):
    x = IdentityNode(shape=(6,), name="x")
    h1 = Linear(
        shape=(width,),
        activation=TanhActivation(),
        name="h1",
        weight_init=NormalInitializer(std=std),
    )
    h2 = Linear(
        shape=(width,),
        activation=TanhActivation(),
        name="h2",
        weight_init=NormalInitializer(std=std),
    )
    y = Linear(
        shape=(4,),
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
        inference=inference,
    )


def _batches(n=6, seed=0):
    out = []
    for i in range(n):
        kx, ky = jax.random.split(jax.random.PRNGKey(seed * 100 + i))
        xs = jax.random.normal(kx, (BATCH, 6))
        ys = jax.nn.one_hot(jax.random.randint(ky, (BATCH,), 0, 4), 4)
        out.append((xs, ys))
    return out


def _clamps(structure, batch):
    xs, ys = batch
    return build_clamps({"x": xs, "y": ys}, structure, clamp_target=True)


def _lambda_max(structure, params, clamps):
    state = initialize_graph_state(
        structure, BATCH, jax.random.PRNGKey(1), clamps=clamps, params=params
    )
    fn = make_epsilon_spectrum(structure, 20)
    return float(fn(params, state, clamps, jax.random.PRNGKey(2)).host().lambda_max)


def _eta(structure):
    return float(structure.config["inference"].config["eta_infer"])


def _scaled(params, factor):
    """Every weight times ``factor``: a stiffer network with a larger lambda_max."""
    return params._replace(
        nodes={
            name: p._replace(weights={k: w * factor for k, w in p.weights.items()})
            for name, p in params.nodes.items()
        }
    )


# ---------------------------------------------------------------------------
# train(..., structure_callback=...)
# ---------------------------------------------------------------------------


class TestStructureCallback:
    def test_a_returned_structure_is_used_from_the_next_batch_on(self):
        s = _mlp(EPCInference(eta_infer=0.01, infer_steps=3))
        faster = s._replace(
            config={
                **s.config,
                "inference": EPCInference(eta_infer=0.05, infer_steps=3),
            }
        )
        params = initialize_params(s, jax.random.PRNGKey(0))
        seen = []

        def on_iter(ctx):
            seen.append(_eta(ctx.structure))

        def swap(ctx):
            return faster if ctx.step == 2 else None

        epochs = []
        result = train(
            params,
            s,
            ListLoader(_batches()),
            optax.adam(1e-3),
            {"num_epochs": 1},
            jax.random.PRNGKey(3),
            verbose=False,
            iter_callback=on_iter,
            structure_callback=swap,
            epoch_callback=lambda ctx: epochs.append(_eta(ctx.structure)),
        )
        assert seen == [0.01, 0.01, 0.05, 0.05, 0.05, 0.05]
        assert epochs == [0.05]
        assert result.step == 6

    def test_swapping_matches_stepping_each_structure_by_hand(self):
        # Batch 0 on s, then batch 1 on other, with the trainer's own keys.
        s = _mlp(EPCInference(eta_infer=0.01, infer_steps=3))
        other = s._replace(
            config={
                **s.config,
                "inference": EPCInference(eta_infer=0.05, infer_steps=3),
            }
        )
        params = initialize_params(s, jax.random.PRNGKey(0))
        batches = _batches(2)
        opt = optax.adam(1e-3)
        rng = jax.random.PRNGKey(3)
        swapped = train(
            params,
            s,
            ListLoader(batches),
            opt,
            {"num_epochs": 1},
            rng,
            verbose=False,
            structure_callback=lambda ctx: other,
        )

        epoch_key = jax.random.fold_in(rng, 0)
        p, o = params, opt.init(params)
        for i, st in enumerate((s, other)):
            xs, ys = batches[i]
            p, o, _, _ = make_train_step(st, opt)(
                p, o, {"x": xs, "y": ys}, jax.random.fold_in(epoch_key, i)
            )
        for a, b in zip(
            jax.tree_util.tree_leaves(swapped.params), jax.tree_util.tree_leaves(p)
        ):
            assert jnp.allclose(a, b, atol=1e-6)

    def test_a_structure_with_other_nodes_is_rejected(self):
        s = _mlp(EPCInference(eta_infer=0.01, infer_steps=3))
        wider = _mlp(EPCInference(eta_infer=0.01, infer_steps=3), width=13)
        x = IdentityNode(shape=(6,), name="x")
        y = Linear(shape=(4,), name="y")
        smaller = graph(
            nodes=[x, y],
            edges=[Edge(source=x, target=y.slot("in"))],
            task_map=TaskMap(x=x, y=y),
            inference=EPCInference(),
        )
        params = initialize_params(s, jax.random.PRNGKey(0))
        for bad in (smaller, wider):
            with pytest.raises(ValueError, match="structure_callback"):
                train(
                    params,
                    s,
                    ListLoader(_batches(2)),
                    optax.adam(1e-3),
                    {"num_epochs": 1},
                    jax.random.PRNGKey(3),
                    verbose=False,
                    structure_callback=lambda ctx, bad=bad: bad,
                )


# ---------------------------------------------------------------------------
# InferenceRateController
# ---------------------------------------------------------------------------


def _ctx(step, params, epoch=0):
    return SimpleNamespace(
        step=step, params=params, epoch_idx=epoch, metrics={"energy": 1.0}
    )


class TestInferenceRateController:
    def _setup(self, kappa=0.2, **kw):
        s = _mlp(EPCInference(eta_infer=0.5, infer_steps=4))
        params = initialize_params(s, jax.random.PRNGKey(0))
        clamps = _clamps(s, _batches(1)[0])
        ctl = InferenceRateController(
            s, clamps, target=kappa, every=5, key=jax.random.PRNGKey(2), **kw
        )
        return s, params, clamps, ctl

    def test_start_picks_the_largest_power_of_two_step_under_the_target(self):
        s, params, clamps, ctl = self._setup(kappa=0.2)
        lam = _lambda_max(s, params, clamps)
        start = ctl.start(params)
        eta = _eta(start)
        # eta = 0.5 / 2**k, the largest such value with eta * lam <= 0.2
        assert eta * lam <= 0.2 + 1e-9
        assert (2 * eta) * lam > 0.2 or eta == 0.5
        assert math.log2(0.5 / eta) == pytest.approx(round(math.log2(0.5 / eta)))
        assert ctl.structure is start
        # Everything but the rate is the configured solver's.
        assert start.config["inference"].config["infer_steps"] == 4
        assert start.nodes is s.nodes

    def test_never_goes_above_the_configured_rate(self):
        s, params, clamps, ctl = self._setup(kappa=1.9)
        start = ctl.start(_scaled(params, 0.01))  # almost flat: lambda_max ~ 1
        assert _eta(start) == 0.5

    def test_only_probes_every_n_updates(self):
        s, params, clamps, ctl = self._setup()
        ctl.start(params)
        for step in (1, 2, 3, 4, 6, 7):
            assert ctl.on_iter(_ctx(step, params)) is None
        assert [r["update"] for r in ctl.history] == [0]

    def test_a_stiffer_network_gets_a_smaller_step(self):
        s, params, clamps, ctl = self._setup(kappa=0.2)
        ctl.start(params)
        eta0 = _eta(ctl.structure)
        stiff = _scaled(params, 3.0)
        new = ctl.on_iter(_ctx(5, stiff))
        assert new is not None and _eta(new) < eta0
        assert _eta(new) * _lambda_max(s, stiff, clamps) <= 0.2 + 1e-9
        assert ctl.structure is new

    def test_keeps_the_step_low_while_a_spike_is_in_the_window(self):
        # One high reading holds the rate down for `window` probes, so a
        # stiffness that jumps up and down does not flip the rate every probe.
        s, params, clamps, ctl = self._setup(kappa=0.2, window=3)
        ctl.start(params)
        eta0 = _eta(ctl.structure)
        ctl.on_iter(_ctx(5, _scaled(params, 3.0)))
        low = _eta(ctl.structure)
        assert low < eta0
        ctl.on_iter(_ctx(10, params))  # back to normal, spike still in window
        ctl.on_iter(_ctx(15, params))
        assert _eta(ctl.structure) == low
        ctl.on_iter(_ctx(20, params))  # the spike has left the window
        assert _eta(ctl.structure) == eta0

    def test_history_records_each_probe(self):
        s, params, clamps, ctl = self._setup(kappa=0.2)
        ctl.start(params)
        ctl.on_iter(_ctx(5, _scaled(params, 3.0), epoch=1))
        row = ctl.history[-1]
        assert row["update"] == 5 and row["epoch"] == 1
        for key in (
            "lambda_max",
            "lambda_used",
            "eta_before",
            "eta_after",
            "eta_lambda_max",
            "crossed",
            "band",
            "f_weighted",
            "f_max",
        ):
            assert key in row
        assert row["eta_after"] < row["eta_before"]
        assert row["eta_lambda_max"] == pytest.approx(
            row["eta_before"] * row["lambda_max"]
        )

    def test_flags_a_crossing_that_happened_between_probes(self):
        s, params, clamps, ctl = self._setup(kappa=1.9)
        ctl.start(params)
        ctl.on_iter(_ctx(5, _scaled(params, 6.0)))
        row = ctl.history[-1]
        assert row["crossed"] == (row["eta_lambda_max"] > 2.0)
        assert row["crossed"]
        assert ctl.crossings == 1

    def test_needs_a_single_solver(self):
        from fabricpc.core.inference import InferenceSchedule

        s = _mlp(
            InferenceSchedule(
                EPCInference(eta_infer=0.01, infer_steps=2),
                InferenceSGD(eta_infer=0.1, infer_steps=2),
            )
        )
        with pytest.raises(ValueError, match="one solver"):
            InferenceRateController(
                s,
                _clamps(s, _batches(1)[0]),
                target=0.2,
                every=5,
                key=jax.random.PRNGKey(2),
            )

    def test_rejects_a_target_outside_the_stable_range(self):
        s = _mlp(EPCInference(eta_infer=0.5, infer_steps=4))
        clamps = _clamps(s, _batches(1)[0])
        for bad in (0.0, 2.0, -1.0):
            with pytest.raises(ValueError, match="target"):
                InferenceRateController(
                    s, clamps, target=bad, every=5, key=jax.random.PRNGKey(2)
                )

    def test_drives_a_real_training_run(self):
        s, params, clamps, ctl = self._setup(kappa=0.2)
        start = ctl.start(params)
        result = train(
            params,
            start,
            ListLoader(_batches(10)),
            optax.adam(1e-2),
            {"num_epochs": 2},
            jax.random.PRNGKey(3),
            verbose=False,
            structure_callback=ctl.on_iter,
        )
        assert result.step == 20
        assert [r["update"] for r in ctl.history] == [0, 5, 10, 15, 20]
        for row in ctl.history:
            assert row["eta_after"] * row["lambda_used"] <= 0.2 + 1e-9
        assert ctl.summary()["probes"] == 5


class TestStateBasedRate:
    """The same controller on sPC, measured with ``latent_curvature``."""

    def _setup(self, kappa=0.2):
        from fabricpc.core.inference import InferenceSGDNormClip

        s = _mlp(InferenceSGDNormClip(eta_infer=0.5, infer_steps=4, max_norm=5.0))
        params = initialize_params(s, jax.random.PRNGKey(0))
        clamps = _clamps(s, _batches(1)[0])
        ctl = InferenceRateController(
            s, clamps, target=kappa, every=5, key=jax.random.PRNGKey(2)
        )
        return s, params, clamps, ctl

    def _rho(self, s, params, clamps):
        from fabricpc.core.latent_curvature import latent_curvature

        state = initialize_graph_state(
            s, BATCH, jax.random.PRNGKey(2), clamps=clamps, params=params
        )
        return latent_curvature(params, state, clamps, s, iters=20)

    def test_keeps_the_solver_and_its_other_settings(self):
        s, params, clamps, ctl = self._setup()
        start = ctl.start(params)
        solver = start.config["inference"]
        assert type(solver).__name__ == "InferenceSGDNormClip"
        assert solver.config["max_norm"] == 5.0
        assert solver.config["infer_steps"] == 4
        assert _eta(start) * self._rho(s, params, clamps) <= 0.2 + 1e-6

    def test_a_stiffer_network_gets_a_smaller_step(self):
        s, params, clamps, ctl = self._setup()
        ctl.start(params)
        eta0 = _eta(ctl.structure)
        new = ctl.on_iter(_ctx(5, _scaled(params, 4.0)))
        assert new is not None and _eta(new) < eta0
        row = ctl.history[-1]
        assert row["band"] is None  # the regime label is ePC's
        assert ctl.summary()["solver"] == "InferenceSGDNormClip"

    def test_drives_a_real_training_run(self):
        s, params, clamps, ctl = self._setup()
        result = train(
            params,
            ctl.start(params),
            ListLoader(_batches(10)),
            optax.adam(1e-2),
            {"num_epochs": 1},
            jax.random.PRNGKey(3),
            verbose=False,
            structure_callback=ctl.on_iter,
        )
        assert result.step == 10
        assert [r["update"] for r in ctl.history] == [0, 5, 10]
