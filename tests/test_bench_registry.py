"""Tests for the benchmark registry: which rows exist and how they are named."""

import jax
import pytest

from fabricpc.bench.registry import ROWS, ALGORITHMS, BenchmarkRow


def test_mnist_mlp_rows_are_registered():
    # One row per algorithm for the MNIST MLP family.
    for algo in ("spc", "epc", "backprop"):
        assert f"mnist-mlp-{algo}" in ROWS


def test_every_row_id_is_dataset_model_algorithm():
    # Row ids look like "cifar10-vgg5-spc": dataset, model, algorithm.
    for row_id, row in ROWS.items():
        dataset, model, algo = row_id.rsplit("-", 2)[0], *row_id.rsplit("-", 2)[1:]
        assert algo in ALGORITHMS
        assert row.id == row_id
        assert row.dataset == dataset
        assert row.model == model
        assert row.algorithm == algo


def test_rows_are_frozen():
    row = ROWS["mnist-mlp-spc"]
    assert isinstance(row, BenchmarkRow)
    with pytest.raises(Exception):
        row.n_trials = 99  # type: ignore[misc]


def test_model_factory_builds_a_graph_for_each_algorithm(rng_key):
    # The same graph must build under every algorithm; only the solver differs.
    for algo in ("spc", "epc", "backprop"):
        row = ROWS[f"mnist-mlp-{algo}"]
        params, structure = row.model_factory(rng_key)
        assert len(structure.nodes) == 4
        assert len(structure.edges) == 3
        assert sum(p.size for p in jax.tree_util.tree_leaves(params)) > 0


def test_cifar10_vgg5_rows_are_registered():
    for algo in ("spc", "epc", "backprop"):
        row = ROWS[f"cifar10-vgg5-{algo}"]
        assert row.dataset == "cifar10"
        assert row.model == "vgg5"
        assert row.batch_size == 128  # pcx's batch size
        assert row.tier == 1


def test_cifar10_vgg5_factory_builds_a_ten_class_vgg(rng_key):
    row = ROWS["cifar10-vgg5-spc"]
    params, structure = row.model_factory(rng_key)
    assert tuple(structure.nodes["input"].node_info.shape) == (32, 32, 3)
    assert tuple(structure.nodes["class"].node_info.shape) == (10,)
    assert len([n for n in structure.nodes if n.startswith("conv")]) == 4


def test_cifar10_resnet18_rows_are_registered(rng_key):
    for algo in ("spc", "epc", "backprop"):
        row = ROWS[f"cifar10-resnet18-{algo}"]
        assert row.model == "resnet18"
        assert row.tier == 1
    params, structure = ROWS["cifar10-resnet18-spc"].model_factory(rng_key)
    assert len(structure.nodes) == 31


def test_tinyshakespeare_transformer_rows_are_registered(rng_key):
    # Character-level language modeling; the row reports perplexity.
    for algo in ("spc", "epc", "backprop"):
        row = ROWS[f"tinyshakespeare-transformer-{algo}"]
        assert row.dataset == "tinyshakespeare"
        assert row.model == "transformer"
        assert row.tier == 1
    params, structure = ROWS["tinyshakespeare-transformer-spc"].model_factory(rng_key)
    assert "logits" in structure.nodes
    assert tuple(structure.nodes["logits"].node_info.shape) == (128, 65)


def test_fashionmnist_mlp_rows_reuse_the_mnist_graph(rng_key):
    for algo in ("spc", "epc", "backprop"):
        row = ROWS[f"fashionmnist-mlp-{algo}"]
        assert row.dataset == "fashionmnist"
        assert row.model == "mlp"
    _, structure = ROWS["fashionmnist-mlp-spc"].model_factory(rng_key)
    assert len(structure.nodes) == 4  # same 784-256-64-10 MLP as MNIST


def test_mlp_rows_carry_an_expected_score_from_our_own_runs():
    for dataset in ("mnist", "fashionmnist"):
        for algo in ("spc", "epc", "backprop"):
            ref = ROWS[f"{dataset}-mlp-{algo}"].reference
            assert ref is not None
            assert ref.metric == "accuracy"
            assert 0.8 < ref.value < 1.0
            assert "Colab T4" in ref.source


def test_rows_without_a_full_run_yet_have_no_expected_score():
    # They get one from their first full 5-seed GPU run, not from pcx.
    for family in ("cifar10-vgg5", "cifar10-resnet18", "tinyshakespeare-transformer"):
        for algo in ("spc", "epc", "backprop"):
            assert ROWS[f"{family}-{algo}"].reference is None


def test_vgg5_rows_use_one_pc_node_per_conv_block(rng_key):
    # pcx's layout: pooling lives inside each block's conv node. With separate
    # MaxPool nodes the PC chain doubles and sPC stops learning (37% vs ~87%).
    for algo in ("spc", "epc", "backprop"):
        _, structure = ROWS[f"cifar10-vgg5-{algo}"].model_factory(rng_key)
        assert not [n for n in structure.nodes if n.startswith("pool")]
        assert type(structure.nodes["conv4"]).__name__ == "ConvPoolNode"


def test_transformer_rows_match_the_sponsors_tuned_char_demo(rng_key):
    # examples/transformer_v2_demo.py CHAR_DEFAULTS: norm-clip at 5.0 and
    # weights drawn from Normal(std=0.01517).
    from fabricpc.bench import registry

    spc_solver = registry._transformer_solver_for("spc")
    assert spc_solver.config["max_norm"] == 5.0
    assert spc_solver.config["infer_steps"] == 12
    assert spc_solver.config["eta_infer"] == pytest.approx(0.0174852165627398)

    _, structure = ROWS["tinyshakespeare-transformer-spc"].model_factory(rng_key)
    init = structure.nodes["L0_mha"].node_info.weight_init
    assert type(init).__name__ == "NormalInitializer"
    assert init.config["std"] == pytest.approx(0.015166293102182283)


def test_transformer_learning_rate_is_a_cosine_decay_to_a_tenth():
    from fabricpc.bench import registry

    lr = registry._char_transformer_lr(1000)
    peak = registry._CHAR_TRANSFORMER["lr"]
    assert float(lr(0)) == pytest.approx(peak)
    assert float(lr(500)) < peak
    assert float(lr(1000)) == pytest.approx(0.1 * peak)


def test_each_family_compares_on_its_own_metric():
    from fabricpc.bench.registry import COMPARISONS

    assert COMPARISONS["tinyshakespeare-transformer"].metric == "perplexity"
    assert COMPARISONS["cifar10-vgg5"].metric == "accuracy"
    assert COMPARISONS["mnist-mlp"].metric == "accuracy"
