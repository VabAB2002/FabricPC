"""Change a row's training settings without editing the row.

Three settings can be changed on any row, which is what ``tune`` searches
and what ``--settings FILE`` applies to a run:

- ``lr_scale``: multiplies the weight update the row's optimizer makes. For
  the optax optimizers the rows use (SGD, Adam, AdamW, with or without a
  schedule) this is the same as multiplying the learning rate, schedule
  and all, since their last step scales the update by the rate.
- ``eta_infer``: the settle's step size (PC rows only).
- ``infer_steps``: how many settle steps each batch gets (PC rows only).

The changed row keeps its id and records the settings in ``row.settings``,
so the manifest and every trial say what was run.
"""

import dataclasses
import json
from pathlib import Path
from typing import Dict, Mapping

import optax

from fabricpc.bench.registry import BenchmarkRow

SOLVER_SETTINGS = ("eta_infer", "infer_steps")
SETTINGS = ("lr_scale",) + SOLVER_SETTINGS


def with_solver_settings(structure, changes: Mapping[str, float]):
    """The graph with its settling solver rebuilt with ``changes``.

    Refuses a solver schedule (several solvers in a row): which one a
    change is meant for is not clear.
    """
    inference = structure.config.get("inference")
    if inference is None or "solvers" in inference.config:
        raise ValueError(
            "this row's graph settles with a schedule of solvers; "
            "eta_infer and infer_steps can only be changed on a single solver"
        )
    missing = [k for k in changes if k not in inference.config]
    if missing:
        raise ValueError(f"{type(inference).__name__} has no setting {missing}")
    config = dict(inference.config)
    for name, value in changes.items():
        config[name] = int(value) if name == "infer_steps" else float(value)
    rebuilt = type(inference)(**config)
    return structure._replace(config={**structure.config, "inference": rebuilt})


def check_settings(row: BenchmarkRow, settings: Mapping[str, float]) -> None:
    """Raise ValueError for a setting that does not exist or does not apply."""
    unknown = sorted(set(settings) - set(SETTINGS))
    if unknown:
        raise ValueError(f"unknown setting {unknown}; known: {list(SETTINGS)}")
    solver = [k for k in settings if k in SOLVER_SETTINGS]
    if row.algorithm == "backprop" and solver:
        raise ValueError(
            f"{row.id}: backprop does not settle, so {solver} do not apply"
        )
    for name, value in settings.items():
        if not float(value) > 0:
            raise ValueError(f"{name} must be above 0, got {value}")


def apply_settings(row: BenchmarkRow, settings: Mapping[str, float]) -> BenchmarkRow:
    """``row`` trained with ``settings`` in place of its own (see module doc)."""
    check_settings(row, settings)
    if not settings:
        return row
    model_factory = row.model_factory
    solver_changes = {k: settings[k] for k in SOLVER_SETTINGS if k in settings}
    if solver_changes:

        def model_factory(rng_key, _build=row.model_factory):
            params, structure = _build(rng_key)
            return params, with_solver_settings(structure, solver_changes)

    optimizer_factory = row.optimizer_factory
    if "lr_scale" in settings:
        scale = float(settings["lr_scale"])

        def optimizer_factory(total_steps, _build=row.optimizer_factory):
            return optax.chain(_build(total_steps), optax.scale(scale))

    return dataclasses.replace(
        row,
        model_factory=model_factory,
        optimizer_factory=optimizer_factory,
        settings={**row.settings, **{k: float(v) for k, v in settings.items()}},
    )


def write_settings(path, row_id: str, settings: Mapping[str, float], **about) -> Path:
    """Save settings for one row; ``about`` says where they came from."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"row": row_id, "settings": dict(settings), **about}
    path.write_text(json.dumps(data, indent=2))
    return path


def load_settings(path, row_id: str) -> Dict[str, float]:
    """The settings saved for ``row_id``.

    ``path`` is a settings file, or a folder holding ``<row id>.json`` files
    (what ``tune`` writes); a row with no file in the folder gets none.
    """
    path = Path(path)
    if path.is_dir():
        path = path / f"{row_id}.json"
        if not path.exists():
            return {}
    data = json.loads(path.read_text())
    if data.get("row") != row_id:
        raise ValueError(f"{path} holds settings for {data.get('row')}, not {row_id}")
    return {k: float(v) for k, v in data["settings"].items()}
