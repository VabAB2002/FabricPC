"""Run one trial of one benchmark row and save the result as JSON."""

import json
import math
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Optional

import jax

from fabricpc.bench.compute import count_compute
from fabricpc.bench.manifest import SCHEMA_VERSION
from fabricpc.bench.measure import memory_snapshot, step_memory, time_steps
from fabricpc.bench.registry import BenchmarkRow
from fabricpc.bench.zoo import save_params
from fabricpc.training import evaluate, train

# The trainer knows two algorithms. Both PC solvers use "pc"; the solver
# itself lives inside the graph the row's model factory built.
_TRAINER_ALGORITHM = {"spc": "pc", "epc": "pc", "backprop": "backprop"}

# Any fixed number works; it just keeps the timing key apart from the three
# keys the experiment framework uses.
_TIMING_STREAM = 7


def total_train_steps(train_loader, num_epochs: float) -> int:
    """Optimizer updates in a run: batches per epoch times epochs, rounded up."""
    return math.ceil(len(train_loader) * float(num_epochs))


def seed_for_trial(trial: int, seed_offset: int = 0) -> int:
    """Same rule as PlannedMultiContrastExperiment, so arms line up."""
    return seed_offset + trial * 1000


@dataclass(frozen=True)
class TrialResult:
    row_id: str
    trial: int
    seed: int
    algorithm: str
    n_params: int
    metrics: Dict[str, float] = field(default_factory=dict)
    compile_time_s: float = 0.0
    step_time_ms: float = 0.0
    train_time_s: float = 0.0
    memory_bytes: Optional[int] = None  # held after warmup
    # Most held by the end of the trial, process-wide. On GPU this includes
    # compile scratch space and can be the same for every algorithm; see
    # step_memory for the compiled training step's own needs.
    peak_memory_bytes: Optional[int] = None
    step_memory: Optional[Dict[str, int]] = None
    num_epochs: float = 0.0
    compute: Dict[str, float] = field(default_factory=dict)
    achieved_tflops: float = 0.0  # flops per update / measured step time
    checkpoint: Optional[str] = None  # zoo path of the trained params
    status: str = "ok"
    error: Optional[str] = None
    schema_version: int = SCHEMA_VERSION


def run_trial(
    row: BenchmarkRow,
    trial: int,
    out_dir,
    *,
    loaders=None,
    num_epochs: Optional[float] = None,
    warmup_steps: int = 5,
    timed_steps: int = 30,
    seed_offset: int = 0,
    zoo_dir=None,
) -> TrialResult:
    """Train the row's model once, measure it, and write ``trial<i>.json``.

    ``loaders`` lets a test pass a small fake dataset. ``num_epochs`` lets a
    caller shorten a run. Both default to what the row says. ``zoo_dir``
    saves the trained params there; None skips saving.
    """
    seed = seed_for_trial(trial, seed_offset)
    out_path = Path(out_dir) / row.id / f"trial{trial}.json"
    algorithm = _TRAINER_ALGORITHM[row.algorithm]
    epochs = float(
        num_epochs if num_epochs is not None else row.train_config["num_epochs"]
    )

    try:
        # Split the seed exactly as PlannedMultiContrastExperiment does, so a
        # trial here trains the same model the experiment framework would.
        # The timing key is drawn off to the side and never touches training.
        master_key = jax.random.PRNGKey(seed)
        graph_key, train_key, eval_key = jax.random.split(master_key, 3)
        timing_key = jax.random.fold_in(master_key, _TIMING_STREAM)

        params, structure = row.model_factory(graph_key)
        n_params = int(sum(p.size for p in jax.tree_util.tree_leaves(params)))
        # Timing reads a few batches, so training gets fresh loaders after it
        # (the framework also builds fresh loaders for every arm).
        timing_loader, _ = loaders or row.loader_factory(seed)
        optimizer = row.optimizer_factory(total_train_steps(timing_loader, epochs))
        timing = time_steps(
            params,
            structure,
            optimizer,
            timing_loader,
            timing_key,
            algorithm=algorithm,
            warmup_steps=warmup_steps,
            timed_steps=timed_steps,
        )
        memory = memory_snapshot().bytes_in_use
        step_mem = step_memory(
            params,
            structure,
            optimizer,
            timing_loader,
            timing_key,
            algorithm=algorithm,
        )
        train_loader, test_loader = loaders or row.loader_factory(seed)

        compute = count_compute(
            params, structure, batch_size=row.batch_size, algorithm=row.algorithm
        )
        # tflops = flops per update / seconds per update / 1e12
        achieved_tflops = (
            compute.flops_per_update / (timing.step_time_ms / 1000.0) / 1e12
        )

        config = {**row.train_config, "num_epochs": epochs}
        t0 = time.perf_counter()
        trained = train(
            params,
            structure,
            train_loader,
            optimizer,
            config,
            train_key,
            algorithm=algorithm,
            verbose=False,
        )
        train_time_s = time.perf_counter() - t0

        raw = evaluate(
            trained.params,
            structure,
            test_loader,
            config,
            eval_key,
            algorithm=algorithm,
        )
        metrics = {k: float(v) for k, v in raw.items()}
        peak_memory = memory_snapshot().peak_bytes

        checkpoint = None
        if zoo_dir is not None:
            path = save_params(
                zoo_dir,
                row,
                trial,
                trained.params,
                meta={"seed": seed, "num_epochs": epochs, "metrics": metrics},
            )
            checkpoint = str(path)

        result = TrialResult(
            row_id=row.id,
            trial=trial,
            seed=seed,
            algorithm=row.algorithm,
            n_params=n_params,
            metrics=metrics,
            compile_time_s=timing.compile_time_s,
            step_time_ms=timing.step_time_ms,
            train_time_s=train_time_s,
            memory_bytes=memory,
            peak_memory_bytes=peak_memory,
            step_memory=step_mem,
            num_epochs=epochs,
            compute=asdict(compute),
            achieved_tflops=achieved_tflops,
            checkpoint=checkpoint,
        )
    except Exception:  # noqa: BLE001 - a failed trial must still be recorded
        result = TrialResult(
            row_id=row.id,
            trial=trial,
            seed=seed,
            algorithm=row.algorithm,
            n_params=0,
            num_epochs=epochs,
            status="failed",
            error=traceback.format_exc(),
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(asdict(result), indent=2))
    return result


def finished_trial(out_dir, row_id: str, trial: int, num_epochs: float) -> bool:
    """True when ``trial<i>.json`` holds a good result for this epoch count.

    Used by ``--resume``. A failed trial, a missing or broken file, or one
    trained for a different number of epochs does not count as finished.
    """
    path = Path(out_dir) / row_id / f"trial{trial}.json"
    try:
        saved = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    return saved.get("status") == "ok" and saved.get("num_epochs") == float(num_epochs)
