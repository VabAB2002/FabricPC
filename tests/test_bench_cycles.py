"""Tests for the loop measurement: refine with sPC, or unroll? (issue #64)"""

import json

import jax

from fabricpc.bench import cycles
from fabricpc.core import InferenceSGD
from fabricpc.training.trainer import build_clamps, convert_batch
from tests.test_bench_runner import fake_mnist_loaders


def _build(loop=2, unroll=1, steps=10):
    return cycles.cyclic_factory(
        loop, unroll=unroll, inference=InferenceSGD(eta_infer=0.15, infer_steps=steps)
    )(jax.random.PRNGKey(0))


def test_the_loops_have_the_nodes_and_edges_they_say():
    _, s2 = _build(loop=2)
    _, s4 = _build(loop=4)
    assert set(s2.nodes) == {"pixels", "hidden1", "hidden2", "back", "class"}
    assert {"hidden1", "hidden4", "back"} <= set(s4.nodes)
    assert "hidden2->back:in" in s2.edges and "back->hidden1:in" in s2.edges


def test_unrolling_visits_the_loop_more_often():
    _, once = _build(unroll=1)
    _, four = _build(unroll=4)
    assert len(four.schedule) > len(once.schedule)
    assert four.schedule.count("hidden1") == 4 * once.schedule.count("hidden1")


def test_matmuls_by_hand_for_the_demo_loop():
    # Weighted edges: pixels->h1, back->h1, h1->h2, h2->class, h2->back = 5.
    params, structure = _build(unroll=1)
    visits = sum(
        cycles._weighted_in_edges(structure, params, n) for n in structure.schedule
    )
    spc = cycles.Arm("spc", 1, 0, 10)
    epc = cycles.Arm("unroll", 1, 5, 0)

    assert cycles.matmuls_per_sample(spc, structure, params) == visits + 2 * 10 * 5
    assert cycles.matmuls_per_sample(epc, structure, params) == visits * (1 + 2 * 5 + 2)


def test_settling_longer_lowers_the_true_energy():
    params, _ = _build()
    batch = convert_batch(next(iter(fake_mnist_loaders(jax.random.PRNGKey(1))[0])))
    energies = []
    for steps in (0, 20, 200):
        _, structure = _build(steps=steps)
        clamps = build_clamps(batch, structure, clamp_target=True)
        state = cycles._settler(structure)(params, clamps, jax.random.PRNGKey(2))
        energies.append(float(cycles.true_energy(params, state, clamps, structure)))
    assert energies[0] > energies[1] > energies[2]


def test_the_cheapest_setting_that_lands_is_picked():
    rows = [
        {"arm": "anchor", "energy": 1.00, "matmuls_per_sample": 9999},
        {"arm": "unroll", "energy": 1.30, "matmuls_per_sample": 10},  # misses
        {"arm": "unroll", "energy": 1.005, "matmuls_per_sample": 40},
        {"arm": "unroll", "energy": 1.001, "matmuls_per_sample": 80},
        {"arm": "refine", "energy": 1.50, "matmuls_per_sample": 20},
    ]
    best = cycles.cheapest_landing(rows, anchor=1.00)
    assert best["unroll"]["matmuls_per_sample"] == 40
    assert best["refine"] is None  # none of its settings lands


def test_a_small_run_writes_the_comparison(tmp_path, monkeypatch, rng_key):
    monkeypatch.setattr(cycles, "UNROLLS", (1, 2))
    monkeypatch.setattr(cycles, "EPC_SWEEP", (5,))
    monkeypatch.setattr(cycles, "REFINES", (5,))
    monkeypatch.setattr(cycles, "SPC_STEPS", (10,))
    monkeypatch.setattr(cycles, "ANCHOR_STEPS", 50)

    report = cycles.run(
        tmp_path,
        loader_factory=lambda seed: fake_mnist_loaders(rng_key, n_batches=3),
        epochs=0.5,
        n_batches=2,
        loops=["loop2"],
    )

    data = json.loads((tmp_path / "cycles.json").read_text())
    assert data == json.loads(json.dumps(report))
    rows = data["loops"]["loop2"]["rows"]
    assert [r["arm"] for r in rows] == [
        "anchor",
        "unroll",
        "unroll",
        "refine",
        "unroll2+refine",
        "spc",
    ]
    for r in rows:
        assert r["matmuls_per_sample"] > 0 and r["settle_ms"] > 0
        assert 0.0 <= r["accuracy"] <= 1.0
    assert set(data["loops"]["loop2"]["cheapest_landing"]) == {
        "unroll",
        "refine",
        "unroll2+refine",
        "spc",
    }
    assert "Cheapest setting that lands" in (tmp_path / "cycles.md").read_text()
