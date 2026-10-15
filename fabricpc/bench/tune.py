"""Tune a row's settings with Optuna, scored on data held out of training.

``python -m fabricpc.bench tune <row or family> --budget N --out DIR`` tries
N settings per row (the row's own first) and keeps the best. Each try is a
normal training run (``run_trial``, the same code a benchmark run uses,
safety net included) on most of the training set, scored with the row's own
metric on the part held back. The test set is never read, so a tuned row's
benchmark score is not inflated by the tuning.

A family is tuned row by row with the same budget, the same seed and the
same held-out data, so no method gets more search than another.

What is searched (``fabricpc.bench.settings``): ``lr_scale`` for every row,
``eta_infer`` and ``infer_steps`` for PC rows. Default ranges are around the
row's own values: lr_scale 0.1 to 10, eta_infer a tenth to ten times the
row's, infer_steps half to twice the row's. ``--space name=low:high`` (add
``:lin`` for a linear scale) replaces a range.

``--min-settled F`` keeps only ePC settings whose settled fraction ends at
F or above, to find an ePC that really settles instead of one that trains
like backprop.

Output in ``DIR``: ``<row id>.json`` with the best settings (pass the folder
to a run as ``--settings DIR``), and ``<row id>/`` with one file per try and
``study.json`` listing them all.
"""

import dataclasses
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional

import jax

from fabricpc.bench.manifest import SCHEMA_VERSION, _git_dirty, _git_sha
from fabricpc.bench.registry import BenchmarkRow
from fabricpc.bench.runner import run_trial, solver_record
from fabricpc.bench.settings import SOLVER_SETTINGS, apply_settings
from fabricpc.bench.summary import trial_settled_fraction
from fabricpc.bench.trend import lower_is_better


@dataclass(frozen=True)
class Range:
    """Where one setting is searched."""

    low: float
    high: float
    log: bool = True
    integer: bool = False


def own_settings(row: BenchmarkRow) -> Dict[str, float]:
    """The row's settings as built: lr_scale 1 and its solver's values."""
    own: Dict[str, float] = {"lr_scale": 1.0}
    if row.algorithm == "backprop":
        return own
    _, structure = row.model_factory(jax.random.PRNGKey(0))
    solver = solver_record(structure, row.algorithm) or {}
    for name in SOLVER_SETTINGS:
        if name in solver:
            own[name] = solver[name]
    return own


def default_space(row: BenchmarkRow, own: Mapping[str, float]) -> Dict[str, Range]:
    """Ranges around the row's own settings (see the module doc)."""
    space = {"lr_scale": Range(0.1, 10.0)}
    if "eta_infer" in own:
        eta = float(own["eta_infer"])
        space["eta_infer"] = Range(eta / 10, eta * 10)
    if "infer_steps" in own:
        steps = int(own["infer_steps"])
        space["infer_steps"] = Range(
            max(1, steps // 2), max(2, steps * 2), log=False, integer=True
        )
    return space


def parse_space(texts: List[str]) -> Dict[str, Range]:
    """``["eta_infer=1e-4:1e-1", "infer_steps=2:40:lin"]`` as ranges."""
    space = {}
    for text in texts:
        try:
            name, rest = text.split("=", 1)
            parts = rest.split(":")
            low, high = float(parts[0]), float(parts[1])
            scale = parts[2] if len(parts) > 2 else "log"
        except (ValueError, IndexError):
            raise ValueError(f"--space {text!r}: expected name=low:high[:lin]")
        if scale not in ("log", "lin") or not 0 < low < high:
            raise ValueError(f"--space {text!r}: need 0 < low < high, scale log or lin")
        integer = name == "infer_steps"
        if integer and (low != int(low) or high != int(high) or low < 1):
            raise ValueError(
                f"--space {text!r}: infer_steps needs whole numbers, 1 or more"
            )
        space[name] = Range(low, high, log=scale == "log", integer=integer)
    return space


def _suggest(trial, name: str, r: Range):
    if r.integer:
        return trial.suggest_int(name, int(r.low), int(r.high), log=r.log)
    return trial.suggest_float(name, r.low, r.high, log=r.log)


def _last_line(text: Optional[str]) -> str:
    lines = (text or "").strip().splitlines()
    return lines[-1] if lines else ""


def tune_row(
    row: BenchmarkRow,
    out_dir,
    *,
    budget: int,
    seed: int = 0,
    epochs: Optional[float] = None,
    max_train_steps: Optional[int] = None,
    space: Optional[Mapping[str, Range]] = None,
    min_settled: Optional[float] = None,
    loaders=None,
    command: str = "",
) -> dict:
    """Search ``budget`` settings for ``row`` and write the best (module doc).

    ``loaders`` replaces the row's held-out split (tests pass tiny data).
    Returns what ``<row id>.json`` holds.
    """
    try:
        import optuna
    except ImportError as e:  # an optional extra
        raise RuntimeError(
            "tune needs optuna: pip install 'fabricpc[experiments]'"
        ) from e
    if loaders is None and row.tune_loader_factory is None:
        raise ValueError(
            f"{row.id} has no held-out split to tune on yet "
            "(its row needs a tune_loader_factory)"
        )
    if budget < 1:
        raise ValueError("budget must be at least 1")
    if min_settled is not None and row.algorithm != "epc":
        raise ValueError("--min-settled only applies to ePC rows")

    own = own_settings(row)
    space = {**default_space(row, own), **(space or {})}
    for name in space:
        if name not in own:
            raise ValueError(f"{row.id}: {name} cannot be tuned on this row")
    lower = lower_is_better(row.metric)
    out_dir = Path(out_dir)
    tries_dir = out_dir / row.id
    if any((tries_dir / name).exists() for name in ("manifest.json", "summary.json")):
        # Tries are written as trial<n>.json and would replace a real run's.
        raise ValueError(
            f"{out_dir} holds benchmark results for {row.id}; "
            "tune into a folder of its own (--out, default 'tuned')"
        )
    tries_dir.mkdir(parents=True, exist_ok=True)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize" if lower else "maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    # The row's own settings go first, so tuning can only keep or beat them.
    first = {name: own[name] for name in space}
    tries_own = all(space[k].low <= v <= space[k].high for k, v in first.items())
    if tries_own:  # a --space that leaves them out cannot try them
        study.enqueue_trial(first)

    def objective(trial):
        settings = {name: _suggest(trial, name, r) for name, r in space.items()}
        tuned = apply_settings(row, settings)
        data = loaders or tuned.tune_loader_factory(seed)
        # Every try trains from the same seed: only the settings differ.
        result = run_trial(
            tuned,
            trial.number,
            out_dir,
            loaders=data,
            num_epochs=epochs,
            seed_offset=seed - trial.number * 1000,
            max_train_steps=max_train_steps,
            curve_batches=0,
            diagnostics=False,
            measure=False,
            settled=min_settled is not None,
        )
        if result.status != "ok":
            trial.set_user_attr("error", _last_line(result.error))
            raise optuna.TrialPruned()
        value = result.metrics.get(row.metric)
        if value is None or not math.isfinite(value):
            trial.set_user_attr("error", f"{row.metric} is {value}")
            raise optuna.TrialPruned()
        if min_settled is not None:
            settled = trial_settled_fraction(dataclasses.asdict(result))
            trial.set_user_attr("settled_fraction", settled)
            if settled is None or settled < min_settled:
                trial.set_user_attr("error", f"settled fraction {settled}")
                raise optuna.TrialPruned()
        return value

    study.optimize(objective, n_trials=budget, catch=(Exception,))

    tries = [
        {
            "number": t.number,
            "settings": t.params,
            "value": t.value,
            "state": t.state.name.lower(),
            **t.user_attrs,
        }
        for t in study.trials
    ]
    (tries_dir / "study.json").write_text(json.dumps(tries, indent=2))
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    about = {
        "schema_version": SCHEMA_VERSION,
        "metric": row.metric,
        "direction": "lower" if lower else "higher",
        "scored_on": "data held out of the training set (never the test set)",
        "budget": budget,
        "completed": len(done),
        "seed": seed,
        "epochs": epochs if epochs is not None else row.train_config["num_epochs"],
        "max_train_steps": max_train_steps,
        "min_settled": min_settled,
        "space": {k: dataclasses.asdict(v) for k, v in space.items()},
        "own_settings": own,
        "own_value": tries[0]["value"] if tries and tries_own else None,
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "command": command,
    }
    if not done:
        about.update(row=row.id, settings=None, value=None)
        (out_dir / f"{row.id}.json").write_text(json.dumps(about, indent=2))
        raise RuntimeError(f"{row.id}: no try finished; see {tries_dir / 'study.json'}")
    best = study.best_trial
    result = {"row": row.id, "settings": dict(best.params), "value": best.value}
    result.update(about)
    (out_dir / f"{row.id}.json").write_text(json.dumps(result, indent=2))
    return result


def describe_result(result: dict) -> str:
    """One line for the terminal."""
    settings = ", ".join(f"{k}={v:.4g}" for k, v in result["settings"].items())
    own = result.get("own_value")
    was = f" (own settings: {own:.4f})" if own is not None else ""
    return (
        f"{result['row']}: best {result['metric']} {result['value']:.4f}{was} "
        f"with {settings}; {result['completed']}/{result['budget']} tries finished"
    )
