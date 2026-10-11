"""Guess how long (and how much) a benchmark run will take, without running it.

    python -m fabricpc.bench probe <row|family> [--trials N] [--epochs E]
                                                [--price-per-hour P]

For each row we build the model and the loaders, compile the training step
and time a handful of steps with ``measure.time_steps`` (the same code a real
trial uses), then multiply out:

    one seed = compile + steps per epoch * epochs * median step + evaluation
    the row  = trials * one seed
    cost     = hours * price per hour

Evaluation is timed too, on a few test batches. ``evaluate`` builds a fresh
jitted function every time it is called, so each call pays a compile. One
call on 1 + k batches is timed, with a mark when it asks for the second
batch: the compile is over by then, so what came before is the compile and
what follows is the batches' time (all 1 + k of them, since JAX runs the
first batch while the loop moves on).

The extras a real trial pays are timed too: the diagnostics at the start
and the end of the trial (and the ePC regime there), and the safety net's
stiffness probe, which runs every ``every`` updates. What is not counted:
starting one Python process per trial, and the recompile each time the
safety net changes the rate. So read the estimate as a floor, not a promise.
The numbers only hold for the hardware they were measured on, so the report
says which device that was.
"""

import itertools
import json
import math
import platform as host
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import jax

from fabricpc.bench import registry
from fabricpc.bench.measure import time_steps
from fabricpc.bench.runner import (
    _TIMING_STREAM,
    _TRAINER_ALGORITHM,
    DEFAULT_CURVE_BATCHES,
    total_train_steps,
)
from fabricpc.bench.diagnostics import diagnose
from fabricpc.bench.measure import epc_regime
from fabricpc.training import evaluate
from fabricpc.training.rate_control import InferenceRateController
from fabricpc.training.trainer import build_clamps, convert_batch

# What a Lightning AI T4 actually cost us per hour. Pass --price-per-hour
# for any other machine.
DEFAULT_PRICE_PER_HOUR = 1.06

DEFAULT_WARMUP = 2
DEFAULT_TIMED = 10
DEFAULT_EVAL_BATCHES = 8
# The timing pass at the start of every real trial (run_trial's defaults):
# one compile step, 5 warmup steps and 30 timed ones.
TRIAL_TIMING_STEPS = 1 + 5 + 30
# A training step slower than this marks the row as slow to probe.
SLOW_STEP_MS = 500.0

NOT_COUNTED = (
    "starting one Python process per trial",
    "the recompile each time the safety net changes the inference rate",
)


@dataclass(frozen=True)
class RowProbe:
    row_id: str
    trials: int
    epochs: float
    steps_per_epoch: int
    compile_time_s: float
    step_time_ms: float
    timed_steps: int
    test_batches: int
    eval_batch_ms: Optional[float]  # one test batch, compile not included
    eval_compile_s: Optional[float]  # paid on every evaluate() call
    train_s_per_seed: float
    eval_s_per_seed: float
    # Diagnostics (and the ePC regime) at the start and the end of a trial.
    diagnostics_s_per_seed: float
    # The safety net's stiffness probes: how many, and their time in total.
    safety_net_probes: int
    safety_net_s_per_seed: float
    seconds_per_seed: float
    total_seconds: float
    cost_usd: float


class _Batches:
    """A fixed list of batches, so evaluate() sees exactly these. Notes the
    moment the second batch is asked for: evaluate() has compiled by then."""

    def __init__(self, batches):
        self._batches = list(batches)
        self.second_batch_at = None

    def __iter__(self):
        for i, batch in enumerate(self._batches):
            if i == 1:
                self.second_batch_at = time.perf_counter()
            yield batch

    def __len__(self):
        return len(self._batches)


def device_info() -> Dict[str, object]:
    """Which device the timings came from, so nobody reads a CPU number as
    a GPU one."""
    device = jax.devices()[0]
    return {
        "platform": device.platform,
        "kind": device.device_kind,
        "count": jax.device_count(),
        "machine": host.machine(),
        "processor": host.processor(),
    }


def _time_eval(row, params, structure, test_loader, key, algorithm, k, slow):
    """(ms per test batch, compile seconds per evaluate call), or Nones.

    ``slow`` means a training step already takes a while; then we time only
    2 extra batches, so the probe itself stays cheap.
    """
    if slow:
        k = min(k, 2)
    if k < 1:
        return None, None
    first = list(itertools.islice(test_loader, k + 1))
    if not first:
        return None, None
    batches = _Batches(first[i % len(first)] for i in range(k + 1))

    # One call on 1 + k batches. evaluate() compiles on its first batch, so
    # by the time it asks for the second the compile is over: what comes
    # before is the compile, what comes after is the batches. JAX does not
    # wait for a batch to finish before asking for the next, so the first
    # batch mostly runs after the mark too: 1 + k batches. Timing two calls
    # and subtracting them did not work: the compile wobbled by more than
    # the batches between the calls (the transformer read 0.0 ms per batch
    # on a T4).
    t0 = time.perf_counter()
    evaluate(
        params,
        structure,
        batches,
        dict(row.train_config),
        key,
        algorithm=algorithm,
        metrics=row.metrics_for(algorithm),
    )
    end = time.perf_counter()
    per_batch = (end - batches.second_batch_at) / (k + 1)
    return per_batch * 1000.0, batches.second_batch_at - t0


def _seconds(fn) -> float:
    """Wall-clock of one call, waiting for the device to finish."""
    t0 = time.perf_counter()
    jax.block_until_ready(fn())
    return time.perf_counter() - t0


def _safety_net_probe_s(row, params, structure, probe_batch, key) -> float:
    """Seconds for one stiffness probe once compiled, the way a trial's
    InferenceRateController runs it (the first probe pays the compile)."""
    controller = InferenceRateController(
        structure,
        build_clamps(convert_batch(probe_batch), structure, clamp_target=True),
        target=row.rate_control.target,
        every=row.rate_control.every,
        window=row.rate_control.window,
        key=key,
    )
    _seconds(lambda: controller.start(params))  # compile
    return _seconds(lambda: controller.start(params))


def probe_row(
    row,
    *,
    trials: Optional[int] = None,
    epochs: Optional[float] = None,
    warmup: int = DEFAULT_WARMUP,
    timed: int = DEFAULT_TIMED,
    eval_batches: int = DEFAULT_EVAL_BATCHES,
    price_per_hour: float = DEFAULT_PRICE_PER_HOUR,
    seed: int = 0,
    diagnostics: bool = True,
) -> RowProbe:
    """Time a few steps of one row and work out what a full run would cost."""
    trials = int(trials if trials is not None else row.n_trials)
    epochs = float(epochs if epochs is not None else row.train_config["num_epochs"])
    algorithm = _TRAINER_ALGORITHM[row.algorithm]

    master_key = jax.random.PRNGKey(seed)
    graph_key, _, eval_key = jax.random.split(master_key, 3)
    timing_key = jax.random.fold_in(master_key, _TIMING_STREAM)

    params, structure = row.model_factory(graph_key)
    train_loader, test_loader = row.loader_factory(seed)
    steps_per_epoch = len(train_loader)
    total_steps = total_train_steps(train_loader, epochs)
    timing = time_steps(
        params,
        structure,
        row.optimizer_factory(total_steps),
        train_loader,
        timing_key,
        algorithm=algorithm,
        warmup_steps=warmup,
        timed_steps=timed,
    )
    # A real trial compiles the step twice: once in its timing pass and again
    # inside train(), which builds its own step. The timing pass also runs
    # its own steps before training starts.
    steps = total_steps + TRIAL_TIMING_STEPS
    train_s = steps * timing.step_time_ms / 1000.0 + 2 * timing.compile_time_s

    test_batches = len(test_loader)
    eval_ms, eval_compile = _time_eval(
        row,
        params,
        structure,
        test_loader,
        eval_key,
        algorithm,
        eval_batches,
        slow=timing.step_time_ms > SLOW_STEP_MS,
    )
    eval_s = 0.0
    if eval_ms is not None:
        # A trial scores the whole test set once, PC rows once more with a
        # plain forward pass, and a small learning-curve slice every epoch.
        passes = 1 if algorithm == "backprop" else 2
        curve_calls = math.ceil(epochs)
        curve_batches = min(DEFAULT_CURVE_BATCHES, test_batches)
        calls = passes + curve_calls
        batches = passes * test_batches + curve_calls * curve_batches
        eval_s = calls * eval_compile + batches * eval_ms / 1000.0

    probe_batch = next(iter(train_loader))
    diag_s = 0.0
    if diagnostics:
        # Each is read twice a trial, and each call compiles its own function.
        diag_s = 2 * _seconds(
            lambda: diagnose(params, structure, probe_batch, timing_key, row.algorithm)
        )
    if row.algorithm == "epc":  # the regime is read on ePC rows only
        diag_s += 2 * _seconds(
            lambda: epc_regime(params, structure, probe_batch, timing_key)
        )
    n_probes, net_s = 0, 0.0
    if row.rate_control is not None:
        n_probes = total_steps // row.rate_control.every + 1
        net_s = n_probes * _safety_net_probe_s(
            row, params, structure, probe_batch, timing_key
        )

    per_seed = train_s + eval_s + diag_s + net_s
    total = trials * per_seed
    return RowProbe(
        row_id=row.id,
        trials=trials,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        compile_time_s=timing.compile_time_s,
        step_time_ms=timing.step_time_ms,
        timed_steps=timing.timed_steps,
        test_batches=test_batches,
        eval_batch_ms=eval_ms,
        eval_compile_s=eval_compile,
        train_s_per_seed=train_s,
        eval_s_per_seed=eval_s,
        diagnostics_s_per_seed=diag_s,
        safety_net_probes=n_probes,
        safety_net_s_per_seed=net_s,
        seconds_per_seed=per_seed,
        total_seconds=total,
        cost_usd=total / 3600.0 * price_per_hour,
    )


def probe(
    target: str,
    *,
    price_per_hour: float = DEFAULT_PRICE_PER_HOUR,
    **kwargs,
) -> Dict[str, object]:
    """Probe one row or every row of a family. Raises KeyError if unknown."""
    if target in registry.ROWS:
        row_ids = [target]
    elif target in registry.COMPARISONS:
        row_ids = list(registry.COMPARISONS[target].rows)
    else:
        raise KeyError(target)
    rows = [
        asdict(probe_row(registry.ROWS[r], price_per_hour=price_per_hour, **kwargs))
        for r in row_ids
    ]
    total = sum(r["total_seconds"] for r in rows)
    return {
        "target": target,
        "device": device_info(),
        "price_per_hour": price_per_hour,
        "rows": rows,
        "total_seconds": total,
        "total_hours": total / 3600.0,
        "cost_usd": total / 3600.0 * price_per_hour,
        "not_counted": list(NOT_COUNTED),
    }


def _hms(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rest = divmod(seconds, 3600)
    m, s = divmod(rest, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def format_probe(report: Dict[str, object]) -> str:
    """The estimate as a few readable lines."""
    d = report["device"]
    lines = [
        f"Measured on {d['platform']} ({d['kind']}, {d['count']} device(s)), "
        f"priced at ${report['price_per_hour']:.2f}/hour"
    ]
    for r in report["rows"]:
        eval_ms = r["eval_batch_ms"]
        eval_part = f", eval {eval_ms:.1f}ms/batch" if eval_ms is not None else ""
        lines.append(
            f"{r['row_id']}: step {r['step_time_ms']:.2f}ms, compile "
            f"{r['compile_time_s']:.1f}s{eval_part}, {r['steps_per_epoch']} "
            f"steps/epoch x {r['epochs']:g} epochs -> {_hms(r['seconds_per_seed'])} "
            f"per seed, {r['trials']} seeds {_hms(r['total_seconds'])} "
            f"(${r['cost_usd']:.2f})"
        )
    if len(report["rows"]) > 1:
        lines.append(
            f"{report['target']} total: {_hms(report['total_seconds'])} "
            f"({report['total_hours']:.2f} h), ${report['cost_usd']:.2f}"
        )
    lines.append("Not counted: " + "; ".join(report["not_counted"]) + ".")
    return "\n".join(lines)


def write_probe(out_dir, report: Dict[str, object]) -> Path:
    """Save the report as ``probe-<target>.json`` in ``out_dir``."""
    path = Path(out_dir) / f"probe-{report['target']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    return path
