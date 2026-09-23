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
