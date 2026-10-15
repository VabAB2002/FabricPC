"""Adding a model: one method list, a family recipe, and rows of your own."""

import json
import sys
from pathlib import Path

import jax
import optax
import pytest

from fabricpc.bench import registry
from fabricpc.bench.methods import ALGORITHMS, METHODS, split_row_id
from fabricpc.bench.recipe import flat_image_loaders, pc_family
from fabricpc.bench.registry import COMPARISONS, ROWS, register_family
from fabricpc.core.inference import InferenceSGD
from fabricpc.core.inference_epc import EPCInference
from tests.test_bench_cli_run import main
from tests.test_bench_runner import fake_mnist_loaders

EXAMPLE = str(
    Path(__file__).resolve().parent.parent / "examples" / "bench_add_your_model.py"
)


def test_one_method_list_feeds_every_part():
    from fabricpc.bench import regress, report, runner

    assert registry.ALGORITHMS == ALGORITHMS
    assert set(report.ALGORITHMS) == set(ALGORITHMS)
    assert regress.ALGOS == ALGORITHMS
    assert runner._TRAINER_ALGORITHM == {m: METHODS[m].trainer for m in METHODS}


def test_split_row_id():
    assert split_row_id("mnist-wide-mlp-epc") == ("mnist-wide-mlp", "epc")
    assert split_row_id("mnist-mlp") == ("mnist-mlp", None)


def _mlp_graph(solver):
    """The built-in MNIST MLP, rebuilt through the recipe's builder."""
    _, structure = ROWS["mnist-mlp-spc"].model_factory(jax.random.PRNGKey(0))
    return structure._replace(config={**structure.config, "inference": solver})


def test_the_recipe_rebuilds_the_built_in_mlp_family_exactly():
    rows = pc_family(
        dataset="mnist",
        model="mlp",
        build_graph=_mlp_graph,
        solvers={
            "spc": InferenceSGD(eta_infer=0.05, infer_steps=20),
            "epc": EPCInference(),
        },
        loader_factory=flat_image_loaders("mnist", 200),
        optimizer=optax.adamw(0.001, weight_decay=0.1),
        num_epochs=20,
        batch_size=200,
    )

    assert list(rows) == ["mnist-mlp-spc", "mnist-mlp-epc", "mnist-mlp-backprop"]
    key = jax.random.PRNGKey(3)
    for row_id, row in rows.items():
        built = ROWS[row_id]
        mine, mine_s = row.model_factory(key)
        theirs, theirs_s = built.model_factory(key)
        same = jax.tree_util.tree_map(lambda a, b: bool((a == b).all()), mine, theirs)
        assert all(jax.tree_util.tree_leaves(same)), row_id
        assert type(mine_s.config["inference"]) is type(theirs_s.config["inference"])
        assert dict(mine_s.config["inference"].config) == dict(
            theirs_s.config["inference"].config
        )
        assert row.rate_control == built.rate_control
        assert row.train_config == built.train_config
        assert row.tune_loader_factory is None  # none given here


def test_the_recipe_refuses_a_settling_method_without_a_solver():
    with pytest.raises(ValueError, match="no solver given for \\['epc'\\]"):
        pc_family(
            dataset="x",
            model="y",
            build_graph=_mlp_graph,
            solvers={"spc": InferenceSGD()},
            loader_factory=None,
            optimizer=optax.sgd(0.1),
            num_epochs=1,
            batch_size=8,
        )


def _tiny_family(dataset="tiny", methods=ALGORITHMS):
    loaders = fake_mnist_loaders(jax.random.PRNGKey(0))
    return pc_family(
        dataset=dataset,
        model="mlp",
        build_graph=_mlp_graph,
        solvers={
            "spc": InferenceSGD(eta_infer=0.05, infer_steps=20),
            "epc": EPCInference(),
        },
        loader_factory=lambda seed: loaders,
        tune_loader_factory=lambda seed: loaders,
        optimizer=optax.adamw(0.001),
        num_epochs=1,
        batch_size=8,
        methods=methods,
    )


def _forget_example(monkeypatch):
    """Drop the example from the module cache, so this test loads it anew."""
    for key in [k for k in sys.modules if k.startswith("fabricpc_bench_rows_")]:
        monkeypatch.delitem(sys.modules, key)


@pytest.fixture
def clean_registry(monkeypatch):
    """Undo whatever a test registers."""
    monkeypatch.setattr(registry, "ROWS", dict(ROWS))
    monkeypatch.setattr(registry, "COMPARISONS", dict(COMPARISONS))
    from fabricpc.bench import __main__ as cli

    monkeypatch.setattr(cli, "ROWS", registry.ROWS)
    monkeypatch.setattr(cli, "COMPARISONS", registry.COMPARISONS)
    yield registry


def test_a_registered_family_compares_like_a_built_in_one(clean_registry):
    family = register_family(_tiny_family())

    assert family.rows == ("tiny-mlp-spc", "tiny-mlp-epc", "tiny-mlp-backprop")
    assert family.contrasts == COMPARISONS["mnist-mlp"].contrasts[:0] + (
        ("tiny-mlp-spc", "tiny-mlp-backprop"),
        ("tiny-mlp-epc", "tiny-mlp-backprop"),
        ("tiny-mlp-epc", "tiny-mlp-spc"),
    )
    assert clean_registry.COMPARISONS["tiny-mlp"] == family


def test_a_family_of_two_methods_has_one_contrast(clean_registry):
    family = register_family(_tiny_family(methods=("spc", "backprop")))
    assert family.contrasts == (("tiny-mlp-spc", "tiny-mlp-backprop"),)


def test_register_family_refuses_taken_ids_and_mixed_families(clean_registry):
    with pytest.raises(ValueError, match="already registered"):
        register_family(
            {
                "mnist-mlp-spc": ROWS["mnist-mlp-spc"],
                "mnist-mlp-epc": ROWS["mnist-mlp-epc"],
            }
        )
    rows = {**_tiny_family("a"), **_tiny_family("b")}
    with pytest.raises(ValueError, match="one family"):
        register_family(rows)


def test_a_registered_family_runs_and_compares_from_the_command_line(
    clean_registry, tmp_path
):
    register_family(_tiny_family())

    code = main(
        [
            "tiny-mlp",
            "--out",
            str(tmp_path),
            "--trials",
            "2",
            "--fast",
            "--warmup",
            "1",
            "--timed",
            "2",
        ]
    )

    assert code == 0
    compare = json.loads((tmp_path / "compare-tiny-mlp.json").read_text())
    assert len(compare["contrasts"]) == 3


def test_the_worked_example_plugs_in_through_the_rows_module_setting(
    clean_registry, monkeypatch, capsys
):
    _forget_example(monkeypatch)
    monkeypatch.setenv("FABRICPC_BENCH_ROWS", EXAMPLE)

    assert main(["list"]) == 0
    assert main(["list"]) == 0  # loading it twice registers it once

    out = capsys.readouterr().out
    assert "mnist-wide-mlp-epc" in out
    family = clean_registry.COMPARISONS["mnist-wide-mlp"]
    assert family.rows[-1] == "mnist-wide-mlp-backprop"
    row = clean_registry.ROWS["mnist-wide-mlp-spc"]
    assert row.tune_loader_factory is not None  # tune can take it


def test_rows_module_flag_passes_the_module_on_to_child_trials(
    clean_registry, monkeypatch
):
    _forget_example(monkeypatch)
    # Set (empty) through monkeypatch, so the value main writes is undone.
    monkeypatch.setenv("FABRICPC_BENCH_ROWS", "")
    import os

    assert main(["list", "--rows-module", EXAMPLE]) == 0
    _forget_example(monkeypatch)

    # Children inherit the environment, so they find the same rows.
    assert os.environ["FABRICPC_BENCH_ROWS"] == EXAMPLE
    assert "mnist-wide-mlp" in clean_registry.COMPARISONS
