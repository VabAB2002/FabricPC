"""``run --dataset --architecture --method`` and ``list`` (Nov 17 sponsor
feedback: separate arguments instead of one combined name)."""

import json

import pytest

from fabricpc.bench.__main__ import main
from fabricpc.bench.cli import explicit
from fabricpc.bench.definitions.registry import COMPARISONS, ROWS


def test_every_row_and_family_can_be_picked_with_separate_flags():
    for row in ROWS.values():
        assert explicit.target_for(row.dataset, row.model, row.algorithm) == row.id
    for family in COMPARISONS.values():
        first = ROWS[family.rows[0]]
        assert explicit.target_for(first.dataset, first.model) == family.id


def test_run_hands_over_to_the_short_id_with_the_same_flags():
    seen = []
    code = explicit.run_cli(
        [
            "--dataset",
            "mnist",
            "--architecture",
            "mlp",
            "--method",
            "spc",
            "--trials",
            "2",
            "--epochs",
            "0.5",
            "--fast",
            "--out",
            "x",
        ],
        run_main=lambda argv: seen.append(list(argv)) or 0,
    )
    assert code == 0
    assert seen == [
        ["mnist-mlp-spc", "--out", "x", "--trials", "2", "--epochs", "0.5", "--fast"]
    ]


def test_run_without_a_method_runs_the_family():
    seen = []
    explicit.run_cli(
        ["--dataset", "mnist", "--architecture", "mlp"],
        run_main=lambda argv: seen.append(list(argv)) or 0,
    )
    assert seen == [["mnist-mlp"]]


def test_run_dry_run_prints_the_same_json_as_the_short_id(capsys):
    assert main(["mnist-mlp-spc", "--dry-run"]) == 0
    short = json.loads(capsys.readouterr().out)
    assert (
        main(
            [
                "run",
                "--dataset",
                "mnist",
                "--architecture",
                "mlp",
                "--method",
                "spc",
                "--dry-run",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == short


def test_a_combination_that_does_not_exist_names_what_does(capsys):
    with pytest.raises(SystemExit) as e:
        main(["run", "--dataset", "mnist", "--architecture", "vgg5"])
    assert e.value.code == 2
    err = capsys.readouterr().err
    assert "no vgg5 benchmark on mnist" in err
    assert "mlp" in err


def test_list_shows_one_line_per_benchmark_with_its_metric(capsys):
    assert main(["list", "--dataset", "mnist"]) == 0
    out = capsys.readouterr().out
    assert "reconstruction_mse" in out  # the autoencoder row's metric
    assert "spc, epc, backprop" in out
    assert "cifar10" not in out


def test_list_ids_keeps_the_old_output(capsys):
    assert main(["list", "--ids"]) == 0
    out = capsys.readouterr().out
    assert "mnist-mlp-spc" in out
    assert "spc vs backprop" in out
