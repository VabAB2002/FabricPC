"""Comparisons between rows, computed by fabricpc.experiments.

Issue #59 asks that every benchmark go through
``PlannedMultiContrastExperiment``. That runner trains every arm inside one
process and records only the task metric and wall time, while the suite runs
each trial in its own process and records timing, memory, and compute too.
So the suite does the training, and this module hands the finished trials to
the framework's own results class, whose ``contrast_results()`` does the
paired t-tests and effect sizes. The statistics are the framework's, not a
copy of them.

That is only honest if a trial here trains the same model the framework
would. ``arm_for_row`` builds the framework's view of a row, and the tests
check that both give the same accuracy for the same seed.
"""

import functools
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

from fabricpc.bench.registry import BenchmarkRow, Comparison
from fabricpc.bench.runner import _TRAINER_ALGORITHM
from fabricpc.experiments import ExperimentArm, PlannedMultiContrastResults
from fabricpc.experiments.ab_experiment import TrialResult as ArmTrial
from fabricpc.training import evaluate, train

MIN_TRIALS = 2


def arm_for_row(
    row: BenchmarkRow, *, steps_per_epoch: int, num_epochs: Optional[float] = None
) -> ExperimentArm:
    """The row as an arm of PlannedMultiContrastExperiment.

    The framework builds the optimizer before it sees a loader, so the caller
    says how many batches an epoch has; the learning-rate schedule needs it.
    """
    algorithm = _TRAINER_ALGORITHM[row.algorithm]
    epochs = num_epochs if num_epochs is not None else row.train_config["num_epochs"]
    return ExperimentArm(
        name=row.id,
        model_factory=row.model_factory,
        train_fn=functools.partial(train, algorithm=algorithm),
        eval_fn=functools.partial(evaluate, algorithm=algorithm),
        optimizer=row.optimizer_factory(math.ceil(steps_per_epoch * float(epochs))),
        train_config={**row.train_config, "num_epochs": float(epochs)},
    )


def _good_trials(results_dir, row_id: str):
    row_dir = Path(results_dir) / row_id
    files = sorted(row_dir.glob("trial*.json"), key=lambda p: int(p.stem[5:]))
    if not files:
        raise FileNotFoundError(f"no trial files under {row_dir}")
    trials = [json.loads(p.read_text()) for p in files]
    return [t for t in trials if t.get("status") == "ok"]


def planned_results(
    results_dir,
    *,
    rows: Sequence[str],
    contrasts: Sequence[Tuple[str, str]],
    metric: str,
) -> PlannedMultiContrastResults:
    """Load finished trials into the framework's results object.

    Refuses rows whose good trials used different seeds: unpaired numbers
    must not go into a paired test.
    """
    per_arm: Dict[str, list] = {}
    seeds = None
    epochs = 0.0
    for row_id in rows:
        good = _good_trials(results_dir, row_id)
        row_seeds = [t["seed"] for t in good]
        if seeds is None:
            seeds = row_seeds
        elif row_seeds != seeds:
            raise ValueError(
                f"cannot pair {row_id} with {rows[0]}: their good trials used "
                f"different seeds ({row_seeds} vs {seeds})"
            )
        per_arm[row_id] = [
            ArmTrial(
                metric_value=float(t["metrics"][metric]),
                train_time=float(t.get("train_time_s", 0.0)),
                all_metrics=dict(t["metrics"]),
            )
            for t in good
        ]
        epochs = float(good[0].get("num_epochs", 0.0)) if good else epochs

    n = len(seeds or [])
    if n < MIN_TRIALS:
        raise ValueError(f"need at least {MIN_TRIALS} paired trials, found {n}")
    return PlannedMultiContrastResults(
        arm_names=list(rows),
        contrasts=list(contrasts),
        metric=metric,
        n_trials=n,
        per_arm_trials=per_arm,
        seeds=list(seeds),
        total_time=sum(t.train_time for arm in per_arm.values() for t in arm),
        num_epochs=epochs,
    )


def _plain(d: dict) -> dict:
    """numpy scalars (the framework returns some) as plain Python values."""
    return {k: v.item() if hasattr(v, "item") else v for k, v in d.items()}


def compare_family(results_dir, comparison: Comparison, *, metric: str) -> dict:
    """Run a family's planned contrasts and write ``compare-<family>.json``."""
    res = planned_results(
        results_dir,
        rows=comparison.rows,
        contrasts=comparison.contrasts,
        metric=metric,
    )
    out = {
        "comparison": comparison.id,
        "metric": metric,
        "seeds": res.seeds,
        "n_trials": res.n_trials,
        "num_epochs": res.num_epochs,
        "contrasts": [_plain(asdict(c)) for c in res.contrast_results()],
    }
    path = Path(results_dir) / f"compare-{comparison.id}.json"
    path.write_text(json.dumps(out, indent=2))
    return out
