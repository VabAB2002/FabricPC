"""The Hopfield recall rows: get a stored pattern back from a noisy copy.

Checks the pattern and probe generators, the recall scores, the Storkey
learning rule used as a fixed reference, and that each of the three rows
runs through the normal trial runner.
"""

import types

import jax.numpy as jnp
import numpy as np
import pytest

from fabricpc.bench import rows_hopfield as rh
from fabricpc.bench.registry import COMPARISONS, ROWS

FAMILY = "patterns64-hopfield"
HOP_ROWS = [f"{FAMILY}-{a}" for a in ("spc", "epc", "backprop")]


class _Node:
    def __init__(self, z_latent):
        self.z_latent = z_latent


def _state(z):
    return types.SimpleNamespace(nodes={"hopfield": _Node(jnp.asarray(z))})


# --- patterns and probes ---------------------------------------------------


def test_patterns_are_plus_minus_one_and_fixed_by_the_seed():
    a = rh.make_patterns(0, n_patterns=7, dim=64)
    b = rh.make_patterns(0, n_patterns=7, dim=64)
    c = rh.make_patterns(1000, n_patterns=7, dim=64)
    assert a.shape == (7, 64)
    assert set(np.unique(a)) <= {-1.0, 1.0}
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_flip_changes_about_the_asked_fraction_of_bits():
    rng = np.random.default_rng(0)
    clean = np.ones((200, 64), dtype=np.float32)
    noisy = rh.flip_bits(clean, 0.2, rng)
    assert set(np.unique(noisy)) <= {-1.0, 1.0}
    assert abs(np.mean(noisy == -1.0) - 0.2) < 0.02
    assert np.array_equal(rh.flip_bits(clean, 0.0, rng), clean)


def test_train_loader_gives_noisy_clean_pairs_and_a_new_draw_each_epoch():
    patterns = rh.make_patterns(0, n_patterns=3, dim=16)
    loader = rh.RecallTrainLoader(patterns, noise=0.15, copies=20, batch_size=8, seed=5)
    assert len(loader) == (3 * 20) // 8
    first = list(loader)
    second = list(loader)
    assert len(first) == len(loader)
    x, y = first[0]
    assert x.shape == y.shape == (8, 16)
    # every target is one of the stored patterns
    for row in y:
        assert any(np.array_equal(row, p) for p in patterns)
    assert not np.array_equal(first[0][0], second[0][0])
    # a fresh loader with the same seed replays the same data
    again = list(rh.RecallTrainLoader(patterns, 0.15, 20, 8, seed=5))
    assert np.array_equal(again[0][0], first[0][0])


def test_probe_loader_covers_every_noise_level_and_is_repeatable():
    patterns = rh.make_patterns(0, n_patterns=4, dim=16)
    probes = rh.ProbeLoader(patterns, probes_per_pattern=5, batch_size=16, seed=7)
    batches = list(probes)
    assert len(batches) == len(probes)
    level = np.concatenate([b["noise_pct"] for b in batches])
    assert len(level) == 4 * 5 * len(rh.NOISE_LEVELS)
    for pct in rh.NOISE_PCTS:
        assert np.sum(level == pct) == 4 * 5
    b = batches[0]
    assert set(b) == {"x", "y", "noise_pct", "pattern", "memories"}
    assert b["memories"].shape == (16, 4, 16)
    for i in range(len(b["x"])):
        assert np.array_equal(b["y"][i], patterns[b["pattern"][i]])
    # a probe at 0% noise is the clean pattern itself
    clean = [
        (x, y)
        for bb in batches
        for x, y, p in zip(bb["x"], bb["y"], bb["noise_pct"])
        if p == 0
    ]
    assert all(np.array_equal(x, y) for x, y in clean)
    again = list(rh.ProbeLoader(patterns, 5, 16, seed=7))
    assert np.array_equal(again[0]["x"], batches[0]["x"])


# --- recall scores ---------------------------------------------------------


def _batch(y, z_pattern, pct, memories):
    y = jnp.asarray(y, dtype=jnp.float32)
    n = y.shape[0]
    return {
        "x": y,
        "y": y,
        "noise_pct": jnp.asarray(pct),
        "pattern": jnp.asarray(z_pattern),
        "memories": jnp.broadcast_to(jnp.asarray(memories), (n,) + memories.shape),
    }


def test_bit_accuracy_exact_and_nearest_count_only_their_noise_level():
    memories = np.array([[1, 1, 1, 1], [-1, -1, -1, -1]], dtype=np.float32)
    y = memories[[0, 0, 1]]
    batch = _batch(y, [0, 0, 1], [20, 20, 10], memories)
    # sample 0: perfect; sample 1: one wrong bit (and a zero, which counts as +1);
    # sample 2 is at another noise level and must get weight 0
    z = np.array([[0.9, 0.3, 2.0, 0.1], [0.0, -0.5, 1.0, 1.0], [-1, -1, -1, -1]])
    metrics = rh.recall_metrics((20,))
    v, w = metrics["bit_accuracy_p20"].fn(_state(z), batch, None)
    assert np.allclose(w, [1, 1, 0])
    assert np.allclose(v * w, [1.0, 0.75, 0.0])
    v, w = metrics["exact_recall_p20"].fn(_state(z), batch, None)
    assert np.allclose(v * w, [1.0, 0.0, 0.0])
    v, w = metrics["nearest_correct_p20"].fn(_state(z), batch, None)
    assert np.allclose(v * w, [1.0, 1.0, 0.0])


def test_nearest_correct_is_false_when_the_output_is_closer_to_another_pattern():
    memories = np.array([[1, 1, 1, 1], [-1, -1, -1, 1]], dtype=np.float32)
    batch = _batch(memories[[0]], [0], [30], memories)
    z_near = np.array([[-1.0, 1.0, 1.0, 1.0]])  # 1 bit from pattern 0, 2 from 1
    z_tie = np.array([[-1.0, 1.0, -1.0, 1.0]])  # 2 bits from both: not a win
    z_far = np.array([[-1.0, -1.0, -1.0, 1.0]])  # this is pattern 1
    metric = rh.recall_metrics((30,))["nearest_correct_p30"]
    assert float(metric.fn(_state(z_near), batch, None)[0][0]) == 1.0
    assert float(metric.fn(_state(z_tie), batch, None)[0][0]) == 0.0
    assert float(metric.fn(_state(z_far), batch, None)[0][0]) == 0.0


def test_recall_metric_names_cover_each_level_and_the_reference():
    names = set(rh.recall_metrics(rh.NOISE_PCTS))
    for pct in rh.NOISE_PCTS:
        for kind in ("bit_accuracy", "exact_recall", "nearest_correct"):
            assert f"{kind}_p{pct:02d}" in names
        assert f"storkey_rule_bit_accuracy_p{pct:02d}" in names
        assert f"storkey_rule_exact_recall_p{pct:02d}" in names


def test_the_do_nothing_baseline_scores_the_probe_itself():
    # Handing back the noisy probe unchanged is the score a model that
    # learned nothing gets. Every row reports it, and the lift over it.
    memories = np.array([[1, 1, 1, 1], [-1, -1, -1, -1]], dtype=np.float32)
    y = memories[[0, 0, 1]]
    batch = _batch(y, [0, 0, 1], [20, 20, 20], memories)
    batch["x"] = jnp.asarray(
        [[1, 1, 1, -1], [1, 1, 1, 1], [1, -1, -1, -1]], dtype=jnp.float32
    )
    z = np.array([[1.0, 1, 1, 1], [1, 1, 1, 1], [1, -1, -1, -1]])
    metrics = rh.recall_metrics((20,))
    v, w = metrics["probe_bit_accuracy_p20"].fn(_state(z), batch, None)
    assert np.allclose(v * w, [0.75, 1.0, 0.75])
    v, w = metrics["probe_exact_recall_p20"].fn(_state(z), batch, None)
    assert np.allclose(v * w, [0.0, 1.0, 0.0])
    # the model fixed sample 0, kept sample 1 and did nothing for sample 2
    v, w = metrics["bit_accuracy_lift_p20"].fn(_state(z), batch, None)
    assert np.allclose(v * w, [0.25, 0.0, 0.0])


def test_recall_metric_names_include_the_baseline_and_the_lift():
    names = set(rh.recall_metrics(rh.NOISE_PCTS))
    for pct in rh.NOISE_PCTS:
        assert f"probe_bit_accuracy_p{pct:02d}" in names
        assert f"probe_exact_recall_p{pct:02d}" in names
        assert f"bit_accuracy_lift_p{pct:02d}" in names


# --- the Storkey learning rule, as a fixed reference -----------------------


def _storkey_by_hand(patterns):
    """Storkey (1997), written out one weight at a time."""
    n = patterns.shape[1]
    w = np.zeros((n, n))
    for xi in patterns:
        h = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                h[i, j] = sum(w[i, k] * xi[k] for k in range(n) if k not in (i, j))
        new = w.copy()
        for i in range(n):
            for j in range(n):
                new[i, j] += (xi[i] * xi[j] - xi[i] * h[j, i] - h[i, j] * xi[j]) / n
        np.fill_diagonal(new, 0.0)
        w = new
    return w


def test_storkey_weights_match_the_rule_written_out_by_hand():
    patterns = rh.make_patterns(3, n_patterns=3, dim=6)
    w = np.asarray(rh.storkey_weights(jnp.asarray(patterns)))
    assert np.allclose(w, _storkey_by_hand(patterns), atol=1e-5)
    assert np.allclose(w, w.T, atol=1e-6)


def test_storkey_network_holds_its_patterns_and_cleans_up_noise():
    patterns = rh.make_patterns(0, n_patterns=7, dim=64)
    w = rh.storkey_weights(jnp.asarray(patterns))
    for p in patterns:
        assert np.array_equal(np.asarray(rh.storkey_recall(w, jnp.asarray(p))), p)
    noisy = rh.flip_bits(patterns, 0.1, np.random.default_rng(1))
    out = np.stack([np.asarray(rh.storkey_recall(w, jnp.asarray(x))) for x in noisy])
    assert np.mean(out == patterns) > 0.95


# --- rows ------------------------------------------------------------------


def test_rows_and_family_are_registered_on_bit_accuracy():
    for row_id in HOP_ROWS:
        row = ROWS[row_id]
        assert row.model == "hopfield"
        assert row.metric == rh.MAIN_METRIC
        assert rh.MAIN_METRIC in row.eval_metrics
    family = COMPARISONS[FAMILY]
    assert family.rows == tuple(HOP_ROWS)
    assert family.metric == rh.MAIN_METRIC


def test_graph_is_probe_hopfield_output():
    import jax

    _, structure = ROWS[HOP_ROWS[0]].model_factory(jax.random.PRNGKey(0))
    shapes = [tuple(structure.nodes[n].node_info.shape) for n in structure.node_order]
    assert shapes == [(rh.DIM,)] * 3
    assert structure.task_map == {"x": "probe", "y": "output"}


def _small_loaders(seed):
    patterns = rh.make_patterns(seed, n_patterns=3, dim=rh.DIM)
    train = rh.RecallTrainLoader(patterns, 0.15, copies=16, batch_size=16, seed=1)
    test = rh.ProbeLoader(patterns, probes_per_pattern=2, batch_size=16, seed=2)
    return train, test


@pytest.mark.parametrize("row_id", HOP_ROWS)
def test_a_trial_reports_recall_scores(tmp_path, row_id):
    from fabricpc.bench.runner import run_trial

    result = run_trial(
        ROWS[row_id],
        0,
        tmp_path,
        loaders=_small_loaders(0),
        num_epochs=1,
        warmup_steps=1,
        timed_steps=2,
        curve_batches=1,
    )
    assert result.status == "ok", result.error
    m = result.metrics
    assert 0.0 <= m[rh.MAIN_METRIC] <= 1.0
    assert 0.0 <= m[f"forward_{rh.MAIN_METRIC}"] <= 1.0
    # the Storkey rule does not depend on training, so it is solid at 0%
    assert m["storkey_rule_exact_recall_p00"] == 1.0
    assert "accuracy" not in m
    assert ("energy" in m) == (not row_id.endswith("backprop"))


@pytest.mark.parametrize("row_id", HOP_ROWS)
def test_every_row_beats_handing_back_the_probe(tmp_path, row_id):
    # The ePC row once scored exactly the noisy probe (its weights never
    # learned to denoise) and still looked like a normal row. A short run
    # on the real data must already beat the do-nothing baseline.
    from fabricpc.bench.runner import run_trial

    result = run_trial(
        ROWS[row_id],
        0,
        tmp_path,
        num_epochs=10,
        warmup_steps=1,
        timed_steps=2,
        curve_batches=0,
        diagnostics=False,
    )
    assert result.status == "ok", result.error
    m = result.metrics
    assert m["bit_accuracy_lift_p20"] > 0.03
    assert m["bit_accuracy_p20"] > m["probe_bit_accuracy_p20"] + 0.03
