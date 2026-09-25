"""Tests for comparisons through fabricpc.experiments' PlannedMultiContrastExperiment."""

import json

import numpy as np
import pytest

from fabricpc.bench.compare import arm_for_row, compare_family, planned_results
from fabricpc.bench.registry import COMPARISONS, ROWS
from fabricpc.bench.runner import run_trial
from fabricpc.experiments import PlannedMultiContrastExperiment
from fabricpc.experiments.statistics import paired_ttest
from tests.test_bench_runner import fake_mnist_loaders
from tests.test_bench_summary import write_trials


def test_every_model_family_has_a_three_way_comparison():
    families = {row_id.rsplit("-", 1)[0] for row_id in ROWS}
    assert set(COMPARISONS) == families
    for fam, cmp in COMPARISONS.items():
        assert cmp.rows == (f"{fam}-spc", f"{fam}-epc", f"{fam}-backprop")
        assert cmp.contrasts == (
            (f"{fam}-spc", f"{fam}-backprop"),
            (f"{fam}-epc", f"{fam}-backprop"),
            (f"{fam}-epc", f"{fam}-spc"),
        )


@pytest.mark.parametrize("algorithm", ["spc", "backprop"])
def test_our_trial_trains_exactly_what_the_experiment_framework_trains(
    tmp_path, rng_key, algorithm
):
    # Same seed, same data, same keys: the accuracy must match to the bit.
    row = ROWS[f"mnist-mlp-{algorithm}"]
    loaders = fake_mnist_loaders(rng_key)
    ours = [
        run_trial(
            row,
            t,
            tmp_path,
            loaders=loaders,
            num_epochs=1,
            warmup_steps=1,
            timed_steps=2,
        ).metrics["accuracy"]
        for t in range(2)
    ]

    arm = arm_for_row(row, steps_per_epoch=len(loaders[0]), num_epochs=1)
    theirs = PlannedMultiContrastExperiment(
        arms=[arm],
        contrasts=[],
        metric="accuracy",
        data_loader_factory=lambda seed: loaders,
        n_trials=2,
    ).run()

    assert ours == list(theirs.per_arm_metrics(arm.name))


def test_planned_results_rebuilds_the_frameworks_results_from_trial_files(tmp_path):
    write_trials(tmp_path, "mnist-mlp-spc", [0.90, 0.92, 0.94])
    write_trials(
        tmp_path, "mnist-mlp-backprop", [0.91, 0.91, 0.91], algorithm="backprop"
    )

    res = planned_results(
        tmp_path,
        rows=("mnist-mlp-spc", "mnist-mlp-backprop"),
        contrasts=(("mnist-mlp-spc", "mnist-mlp-backprop"),),
        metric="accuracy",
    )

    assert res.seeds == [0, 1000, 2000]
    assert list(res.per_arm_metrics("mnist-mlp-spc")) == [0.90, 0.92, 0.94]
    (c,) = res.contrast_results()
    expected = paired_ttest(np.array([0.90, 0.92, 0.94]), np.array([0.91] * 3))
    assert c.mean_diff == pytest.approx(0.01)
    assert c.p_value == pytest.approx(expected.p_value)


def test_rows_run_with_different_seeds_cannot_be_compared(tmp_path):
    write_trials(tmp_path, "mnist-mlp-spc", [0.90, 0.92])
    write_trials(tmp_path, "mnist-mlp-backprop", [0.91, 0.91], seed_step=7)
    with pytest.raises(ValueError, match="seed"):
        planned_results(
            tmp_path,
            rows=("mnist-mlp-spc", "mnist-mlp-backprop"),
            contrasts=(),
            metric="accuracy",
        )


def test_failed_trials_are_left_out_of_the_pairing(tmp_path):
    write_trials(tmp_path, "mnist-mlp-spc", [0.90, 0.92, 0.94])
    write_trials(
        tmp_path, "mnist-mlp-backprop", [0.91, 0.91, 0.91], algorithm="backprop"
    )
    broken = tmp_path / "mnist-mlp-spc" / "trial2.json"
    broken.write_text(
        json.dumps({**json.loads(broken.read_text()), "status": "failed"})
    )

    with pytest.raises(ValueError, match="seed"):
        # spc now has 2 good trials, backprop 3: they no longer pair up.
        planned_results(
            tmp_path,
            rows=("mnist-mlp-spc", "mnist-mlp-backprop"),
            contrasts=(),
            metric="accuracy",
        )


def test_compare_family_writes_all_three_contrasts(tmp_path):
    write_trials(tmp_path, "mnist-mlp-spc", [0.90, 0.92, 0.94])
    write_trials(tmp_path, "mnist-mlp-epc", [0.92, 0.93, 0.94], algorithm="epc")
    write_trials(
        tmp_path, "mnist-mlp-backprop", [0.91, 0.91, 0.91], algorithm="backprop"
    )

    out = compare_family(tmp_path, COMPARISONS["mnist-mlp"], metric="accuracy")

    on_disk = json.loads((tmp_path / "compare-mnist-mlp.json").read_text())
    assert on_disk == out
    assert on_disk["metric"] == "accuracy"
    assert on_disk["seeds"] == [0, 1000, 2000]
    pairs = [(c["arm_a"], c["arm_b"]) for c in on_disk["contrasts"]]
    assert pairs == list(COMPARISONS["mnist-mlp"].contrasts)
    assert on_disk["contrasts"][2]["mean_diff"] == pytest.approx(0.01)  # epc - spc
