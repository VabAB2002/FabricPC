"""Small library bugs found while reviewing the code for the benchmark suite."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest

from fabricpc.core.activations import GeluActivation, ReLUActivation
from fabricpc.core.epsilon_spectrum import EpsilonSpectrum
from fabricpc.core.inference import InferenceSGD
from fabricpc.core.inference_epc import EPCInference
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.nodes import (
    LnMlp1Node,
    Linear,
    Mlp2ResidualNode,
    MlpResidualNode,
)
from fabricpc.nodes.identity import IdentityNode
from fabricpc.nodes.skip_connection import SkipConnection

# ---------------------------------------------------------------------------
# EPCInference.regime and latent_decay
# ---------------------------------------------------------------------------


def _shifted(spectrum, d):
    return EpsilonSpectrum.from_modes(
        [float(t) + d for t in spectrum.ritz_values], list(spectrum.ritz_weights)
    )


def test_regime_counts_latent_decay():
    # The update is eps <- eps*(1 - eta*d) - eta*grad, so every mode relaxes
    # as if its curvature were lambda + d. The verdict has to see that.
    spectrum = EpsilonSpectrum.from_modes([2.0, 10.0, 40.0], [0.5, 0.3, 0.2])
    d = 20.0
    with_decay = EPCInference(eta_infer=0.02, infer_steps=6, latent_decay=d)
    same_rate = EPCInference(eta_infer=0.02, infer_steps=6)

    got = with_decay.regime(spectrum)
    want = same_rate.regime(_shifted(spectrum, d))
    assert got.f_weighted == pytest.approx(want.f_weighted)
    assert got.f_max == pytest.approx(want.f_max)
    assert got.eta_lambda_max == pytest.approx(0.02 * (40.0 + d))
    assert got.f_weighted > same_rate.regime(spectrum).f_weighted
    # The measured extremes are still reported as measured.
    assert got.lambda_max == 40.0


def test_decay_can_push_a_stable_rate_past_the_limit():
    spectrum = EpsilonSpectrum.from_modes([90.0], [1.0])
    assert not EPCInference(eta_infer=0.02, infer_steps=4).regime(spectrum).unstable
    decayed = EPCInference(eta_infer=0.02, infer_steps=4, latent_decay=20.0)
    assert decayed.regime(spectrum).unstable  # 0.02 * (90 + 20) = 2.2


def test_zero_decay_regime_is_unchanged():
    spectrum = EpsilonSpectrum.from_modes([10.0, 12.0, 16.4], [0.3, 0.3, 0.4])
    a = EPCInference(eta_infer=0.02, infer_steps=5).regime(spectrum)
    b = EPCInference(eta_infer=0.02, infer_steps=5, latent_decay=0.0).regime(spectrum)
    assert a == b


# ---------------------------------------------------------------------------
# Weightless nodes silently dropped the activation they were given
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("node_cls", [IdentityNode, SkipConnection])
def test_weightless_nodes_reject_an_activation_they_would_ignore(node_cls):
    with pytest.raises(ValueError, match="activation"):
        node_cls(shape=(4,), name="n", activation=ReLUActivation())


@pytest.mark.parametrize("node_cls", [IdentityNode, SkipConnection])
def test_weightless_nodes_still_build_with_the_default(node_cls):
    node_cls(shape=(4,), name="n")


# ---------------------------------------------------------------------------
# Residual slots that the node cannot work without
# ---------------------------------------------------------------------------


def _mlp_graph(mlp2_residual: bool):
    x = Linear(shape=(4, 8), name="x")
    h = LnMlp1Node(
        shape=(4, 16), name="h", embed_dim=8, ff_dim=16, activation=GeluActivation()
    )
    out = Mlp2ResidualNode(shape=(4, 8), name="out", embed_dim=8, ff_dim=16)
    y = Linear(shape=(4, 8), name="y")
    edges = [
        Edge(source=x, target=h.slot("in")),
        Edge(source=h, target=out.slot("in")),
        Edge(source=out, target=y.slot("in")),
    ]
    if mlp2_residual:
        edges.append(Edge(source=x, target=out.slot("residual")))
    return graph(
        nodes=[x, h, out, y],
        edges=edges,
        task_map=TaskMap(x=x, y=y),
        inference=InferenceSGD(eta_infer=0.1, infer_steps=2),
    )


def test_mlp2_residual_without_its_residual_edge_fails_at_build_time():
    # predict() needs the residual input; without it the node used to fail
    # later with a bare StopIteration.
    with pytest.raises(ValueError, match="requires at least one edge into slot"):
        _mlp_graph(mlp2_residual=False)
    _mlp_graph(mlp2_residual=True)


def test_fused_mlp_node_needs_its_skip_edge():
    x = Linear(shape=(4, 8), name="x")
    mlp = MlpResidualNode(shape=(4, 8), name="mlp", embed_dim=8, ff_dim=16)
    y = Linear(shape=(4, 8), name="y")
    with pytest.raises(ValueError, match="requires at least one edge into slot"):
        graph(
            nodes=[x, mlp, y],
            edges=[
                Edge(source=x, target=mlp.slot("in")),
                Edge(source=mlp, target=y.slot("in")),
            ],
            task_map=TaskMap(x=x, y=y),
            inference=InferenceSGD(eta_infer=0.1, infer_steps=2),
        )
