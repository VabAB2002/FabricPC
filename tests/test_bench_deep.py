"""Tests for the deep FC-ResNet rows (fabricpc.bench.deep)."""

import jax
import pytest

from fabricpc.bench.registry import ALGORITHMS, COMPARISONS, ROWS

DEPTHS = (8, 16, 32, 64, 128)
WIDTH = 64  # the demo's hidden width


def _n_params(params):
    return sum(p.size for p in jax.tree_util.tree_leaves(params))


def _expected_params(depth):
    # stem 784 -> 64 with bias, one 64 x 64 matrix and bias per block,
    # readout 64 -> 10 with bias. The skips have no weights.
    stem = 784 * WIDTH + WIDTH
    block = WIDTH * WIDTH + WIDTH
    readout = WIDTH * 10 + 10
    return stem + depth * block + readout


@pytest.mark.parametrize("depth", DEPTHS)
def test_every_depth_has_three_rows_and_a_family(depth):
    family = f"mnist-fcresnet{depth}"
    for algo in ALGORITHMS:
        row = ROWS[f"{family}-{algo}"]
        assert row.dataset == "mnist"
        assert row.model == f"fcresnet{depth}"
        assert row.algorithm == algo
        assert row.depth == depth
        assert row.batch_size == 256  # the demo's batch size
        assert row.train_config["num_epochs"] == 3  # the demo's epoch count
    comparison = COMPARISONS[family]
    assert comparison.rows == tuple(f"{family}-{a}" for a in ALGORITHMS)
    assert comparison.metric == "accuracy"


def test_other_rows_have_no_depth():
    assert ROWS["mnist-mlp-spc"].depth is None


def test_shallow_depths_are_tier_one_and_deep_ones_tier_two():
    assert ROWS["mnist-fcresnet8-spc"].tier == 1
    assert ROWS["mnist-fcresnet16-spc"].tier == 1
    assert ROWS["mnist-fcresnet32-spc"].tier == 2
    assert ROWS["mnist-fcresnet128-backprop"].tier == 2


@pytest.mark.parametrize("depth", (8, 32))
def test_graph_has_one_pc_node_per_block(rng_key, depth):
    params, structure = ROWS[f"mnist-fcresnet{depth}-spc"].model_factory(rng_key)
    blocks = [n for n in structure.nodes if n.startswith("block")]
    assert len(blocks) == depth
    for name in blocks:
        assert type(structure.nodes[name]).__name__ == "LinearResidual"
    # input, stem, the blocks and the readout; every one but the input is a
    # latent that PC settles.
    assert len(structure.nodes) == depth + 3
    assert _n_params(params) == _expected_params(depth)


def test_all_three_rows_build_the_same_graph(rng_key):
    sizes = set()
    for algo in ALGORITHMS:
        params, structure = ROWS[f"mnist-fcresnet8-{algo}"].model_factory(rng_key)
        sizes.add((_n_params(params), len(structure.nodes), len(structure.edges)))
    assert len(sizes) == 1


def test_mupc_is_on_with_the_depth_factor(rng_key):
    # A block's weight path is scaled by 1/sqrt(fan_in * depth), its skip
    # path not at all, and the softmax readout is left out, as in the demo.
    _, structure = ROWS["mnist-fcresnet32-spc"].model_factory(rng_key)
    block = structure.nodes["block5/res"].node_info
    scale = block.scaling_config.forward_scale
    in_edges = [k for k in scale if ":in" in k]
    assert len(in_edges) == 1
    assert not [k for k in scale if ":skip" in k]
    gain = type(block.activation).variance_gain(block.activation.config)
    assert scale[in_edges[0]] == pytest.approx(gain / (WIDTH * 32) ** 0.5)
    assert structure.nodes["output"].node_info.scaling_config is None
    stem = structure.nodes["stem"].node_info.scaling_config.forward_scale
    assert list(stem.values())[0] == pytest.approx(1 / 784**0.5)


def test_rows_use_the_demo_solver_settings(rng_key):
    from fabricpc.bench import deep

    assert deep.infer_steps(8) == 30
    assert deep.infer_steps(128) == 390
    _, structure = ROWS["mnist-fcresnet8-spc"].model_factory(rng_key)
    assert structure.config["inference"].config["eta_infer"] == 0.1
    assert structure.config["inference"].config["infer_steps"] == 30
    _, structure = ROWS["mnist-fcresnet8-epc"].model_factory(rng_key)
    assert type(structure.config["inference"]).__name__ == "EPCInference"


def test_pc_rows_have_the_safety_net_and_small_adam_epsilon():
    import jax.numpy as jnp

    assert ROWS["mnist-fcresnet128-spc"].rate_control.target == 1.8
    assert ROWS["mnist-fcresnet128-epc"].rate_control.target == 1.0
    assert ROWS["mnist-fcresnet128-backprop"].rate_control is None

    def last_step(opt, grad):
        params = {"w": jnp.zeros(4)}
        state = opt.init(params)
        for _ in range(200):
            updates, state = opt.update({"w": jnp.full(4, grad)}, state, params)
        return float(jnp.abs(updates["w"]).max())

    make = ROWS["mnist-fcresnet128-spc"].optimizer_factory
    assert last_step(make(100), 1e-10) > 0.5 * last_step(make(100), 1e-2)


def test_depth_of_reads_the_registry_and_falls_back_to_the_id():
    from fabricpc.bench.deep import depth_of

    assert depth_of("mnist-fcresnet64-epc") == 64
    assert depth_of("mnist-fcresnet12-spc") == 12  # not registered, id still says
    assert depth_of("mnist-mlp-spc") is None


def test_depth_table_puts_depths_in_rows_and_methods_in_columns():
    from fabricpc.bench.deep import depth_table

    def row(row_id, algo, mean, se=None):
        return {
            "id": row_id,
            "algorithm": algo,
            "metric": "accuracy",
            "stats": {"accuracy": {"mean": mean, "se": se}},
        }

    loaded = [
        row("mnist-fcresnet128-spc", "spc", 0.80),
        row("mnist-fcresnet8-spc", "spc", 0.92, 0.001),
        row("mnist-fcresnet8-backprop", "backprop", 0.95),
        row("mnist-mlp-spc", "spc", 0.98),  # not a depth row: left out
    ]
    head, body = depth_table(loaded)
    assert head == ["model", "depth", "backprop", "spc", "epc"]
    assert body == [
        ["mnist-fcresnet", "8", "0.9500", "0.9200 ± 0.0010", "-"],
        ["mnist-fcresnet", "128", "-", "0.8000", "-"],
    ]
    assert depth_table([loaded[-1]]) is None


def test_report_shows_accuracy_against_depth(tmp_path):
    import json

    from fabricpc.bench.report import build_report, render

    for depth, acc in ((8, 0.92), (128, 0.81)):
        d = tmp_path / f"mnist-fcresnet{depth}" / f"mnist-fcresnet{depth}-spc"
        d.mkdir(parents=True)
        summary = {
            "algorithm": "spc",
            "n_ok": 1,
            "metrics": {"accuracy": {"mean": acc, "se": None, "n": 1}},
        }
        (d / "summary.json").write_text(json.dumps(summary))
    page = render(build_report(tmp_path))
    assert "Accuracy against depth" in page
    assert "| mnist-fcresnet | 8 | - | 0.9200 (n=1) | - |" in page
    assert "| mnist-fcresnet | 128 | - | 0.8100 (n=1) | - |" in page


def test_depth_table_keeps_runs_from_different_results_folders_apart(tmp_path):
    # A full GPU run and a 1-seed Mac check of the same row must not
    # overwrite each other: each gets its own line, labelled by folder,
    # with its seed count.
    import json

    from fabricpc.bench.report import build_report, render

    for folder, n, acc in (("a-gpu", 5, 0.97), ("b-mac", 1, 0.31)):
        d = tmp_path / folder / "mnist-fcresnet128" / "mnist-fcresnet128-epc"
        d.mkdir(parents=True)
        summary = {
            "algorithm": "epc",
            "n_ok": n,
            "metrics": {"accuracy": {"mean": acc, "se": None, "n": n}},
        }
        (d / "summary.json").write_text(json.dumps(summary))
    page = render(build_report(tmp_path))
    depth_part = page.split("Accuracy against depth", 1)[1]
    assert "| a-gpu | mnist-fcresnet | 128 | - | - | 0.9700 (n=5) |" in depth_part
    assert "| b-mac | mnist-fcresnet | 128 | - | - | 0.3100 (n=1) |" in depth_part


def test_depth_table_rows_from_one_folder_have_no_folder_column():
    from fabricpc.bench.deep import depth_table

    loaded = [
        {
            "id": "mnist-fcresnet8-spc",
            "algorithm": "spc",
            "metric": "accuracy",
            "stats": {"accuracy": {"mean": 0.9, "se": None}},
            "source": "runs",
        }
    ]
    head, body = depth_table(loaded)
    assert head[0] == "model"
    assert body == [["mnist-fcresnet", "8", "-", "0.9000", "-"]]


def test_depth_table_works_when_the_root_is_a_family_folder(tmp_path):
    import json

    from fabricpc.bench.report import build_report, render

    d = tmp_path / "mnist-fcresnet8-spc"
    d.mkdir()
    summary = {
        "algorithm": "spc",
        "n_ok": 2,
        "metrics": {"accuracy": {"mean": 0.9, "se": None, "n": 2}},
    }
    (d / "summary.json").write_text(json.dumps(summary))
    page = render(build_report(tmp_path))
    assert "| mnist-fcresnet | 8 | - | 0.9000 (n=2) | - |" in page
